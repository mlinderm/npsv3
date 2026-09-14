"""Megatron-Energon polylithic loader for packed pileup images.

Energon-specific, so that the packing core in [loaders.py](loaders.py) -- `select_packs`,
`build_packed_batch`, `PackedBatch` -- stays importable without the dependency and survives a loader
swap.

The shape of the thing: a region's metadata (`r`, `g`, geometry, label) lives in a tiny JSON shard,
its pixels in a separate gzipped shard. The packing buffer therefore decides a pack from metadata
alone and only then fetches the bytes for the groups it actually selected, which is what makes a
512-group lookahead affordable. Compared with the greedy `PackedImageDataModule`:

* every rank takes the same number of steps, because `limit_train_batches` defines a virtual epoch
  over an indefinitely-repeating dataset rather than the shard slice deciding it;
* the loader's position rides in the Lightning checkpoint, so a run resumes mid-epoch exactly;
* images arrive as `uint8` and are normalized on the GPU, roughly a 4x cut in host->device bytes.
"""

import gzip
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from megatron.energon import (
    Cooker,
    FileStore,
    FileStoreCachePool,
    Lazy,
    Sample,
    SkipSample,
    TaskEncoder,
    WorkerConfig,
    basic_sample_keys,
    cooker,
    edataclass,
    get_savable_loader,
    get_train_dataset,
    get_val_dataset,
    stateless,
)
from torchvision.transforms import v2 as transforms

from npsv3.models.loaders import (
    LABEL_PADDING_VALUE,
    ImageGroup,
    PackedBatch,
    build_packed_batch,
    select_packs,
    to_tensor,
)

REAL_SUFFIX = "real.bin.gz"
SIMS_SUFFIX = "sims.bin.gz"


@edataclass
class GroupMeta(Sample):
    """One region's metadata, with lazy handles for its pixels.

    Cheap enough to hold thousands of, which is the point: `packing_buffer_size` groups of these are
    what `select_packs` sees.
    """

    #: Replicates, genotypes, and the image geometry (height, width, channels)
    r: int
    g: int
    h: int
    w: int
    c: int
    #: The canonical positive: an index into the genotype axis. None for unlabeled examples.
    label: int | None
    #: The derived per-genotype rank view of `label`, recorded by the writer rather than recomputed.
    label_rank: list[int] | None
    region: str
    #: Lazy handles into the pixel tree; no bytes are read until `.get()`. `sims` is replaced by a
    # duck-typed `_SubsampledLazy` when genotype subsampling is on.
    real: Lazy
    sims: Any = None
    #: Image slots this group occupies in a pack. Filled in by `preencode_sample`.
    cost: int = 0


@cooker(need_cache=True)
@stateless
def cook_group(sample: dict, px: FileStore, cache) -> GroupMeta:
    """Turn a crude JSON sample into a `GroupMeta` with lazy pixel handles.

    `basic_sample_keys` is mandatory, not decorative: without `__key__` / `__restore_key__` the
    dataset is not restorable, which is half the reason for this loader.
    """
    meta = sample["json"]
    if isinstance(meta, (bytes, str)):
        meta = json.loads(meta)

    key = sample["__key__"]
    return GroupMeta(
        **basic_sample_keys(sample),
        r=int(meta["r"]),
        g=int(meta["g"]),
        h=int(meta["h"]),
        w=int(meta["w"]),
        c=int(meta["c"]),
        label=None if meta.get("label") is None else int(meta["label"]),
        label_rank=None if meta.get("label_rank") is None else [int(rank) for rank in meta["label_rank"]],
        region=meta.get("region", key),
        real=cache.get_lazy(px, f"{key}.{REAL_SUFFIX}"),
        sims=cache.get_lazy(px, f"{key}.{SIMS_SUFFIX}") if int(meta["g"]) > 0 else None,
    )


GZIP_MAGIC = b"\x1f\x8b"


def _fetch_array(lazy: Lazy, sample: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Fetch, gunzip and reshape one lazy pixel member.

    The aux store energon hands the cooker is wrapped in a `DecodeFileStore`, whose default decoder
    includes webdataset's `gzfilter` -- so a `.bin.gz` member usually arrives already decompressed.
    That is decoder configuration, not a guarantee, so check the magic rather than assume either way.

    `bytearray` rather than `bytes` so the array is writable: `torch.from_numpy` warns on a read-only
    buffer, and `pytest.ini` turns warnings into errors.
    """
    raw = lazy.get(sample)
    if raw[:2] == GZIP_MAGIC:
        raw = gzip.decompress(raw)
    return np.frombuffer(bytearray(raw), dtype=np.uint8).reshape(shape)


class PackedGroupTaskEncoder(TaskEncoder[GroupMeta, GroupMeta, PackedBatch, PackedBatch]):
    """Packs whole regions into fixed-budget `PackedBatch`es.

    A region's real image and all of its simulated candidates are one atomic unit: the metric-learning
    objective scores the real image against every simulation for that region, so a group can never be
    split across ranks or batches. That is what makes this a packing problem rather than a batching one.
    """

    cookers = (Cooker(cook_group),)

    def __init__(
        self,
        *,
        budget: int = 256,
        w_bucket: int | None = None,
        w_max: int | None = None,
        g_max: int | None = None,
        r_max: int | None = None,
        g_train: int | None = None,
        pad: bool = True,
        padding_value: int = LABEL_PADDING_VALUE,
    ):
        """
        Args:
            budget: Image slots per pack, the same meaning as `batch_size` in `PackedImageDataModule`.
            w_bucket: Width bucket size. Groups only share a pack if they share a bucket.
            w_max: Reject groups wider than this.
            g_max: Reject groups with more genotypes than this.
            r_max: Reject groups with more replicates than this.
            g_train: If set, subsample to this many genotypes per group per step. Must retain the positive.
            pad: Pad each pack out to `budget` rows with zeros.
            padding_value: Label value for padding rows and unlabeled groups.
        """
        super().__init__()
        self.budget = budget
        self.w_bucket = w_bucket
        self.w_max = w_max
        self.g_max = g_max
        self.r_max = r_max
        self.g_train = g_train
        self.pad = pad
        self.padding_value = padding_value

    def _cost(self, sample: GroupMeta) -> int:
        slots = 1 + sample.r * sample.g
        if self.w_bucket:
            slots *= math.ceil(sample.w / self.w_bucket)
        return slots

    @stateless
    def preencode_sample(self, sample: GroupMeta) -> GroupMeta:
        """Compute the pack cost and reject groups that cannot be packed. Touches no pixels."""
        if sample.g < 1 or sample.r < 1:
            # No simulations, so nothing for the metric-learning objective to score against
            raise SkipSample
        if self.g_max is not None and sample.g > self.g_max:
            raise SkipSample
        if self.r_max is not None and sample.r > self.r_max:
            raise SkipSample
        if self.w_max is not None and sample.w > self.w_max:
            raise SkipSample

        sample.cost = self._cost(sample)
        if sample.cost > self.budget:
            # Never give an oversized group its own bin; it would blow the fixed pack shape
            raise SkipSample
        return sample

    def select_samples_to_pack(self, samples: list[GroupMeta]) -> list[list[GroupMeta]]:
        packs = select_packs(
            [(sample.cost, sample.w // self.w_bucket if self.w_bucket else 0) for sample in samples],
            budget=self.budget,
        )
        return [[samples[index] for index in pack] for pack in packs]

    @stateless(restore_seeds=True)
    def postencode_sample(self, sample: GroupMeta) -> GroupMeta:
        """Optionally subsample the genotype axis, off by default (`g_train=None`).

        Subsampling thins the G axis only, never the R axis, so every retained genotype keeps all of
        its replicates. The retained set always includes the reference genotype (index 0, the
        homozygous-reference candidate, and the anchor for the third-rank rule) and the positive --
        drop the positive and the group has nothing for the loss to pull towards. `label` is then
        remapped to its new position and `label_rank` is *sliced* rather than recomputed, so each
        retained genotype keeps exactly the rank the generator recorded for it.
        """
        if self.g_train is None or sample.g <= self.g_train:
            return sample

        required = {0}
        if sample.label is not None:
            required.add(sample.label)
        assert len(required) <= self.g_train, f"g_train={self.g_train} cannot retain both genotype 0 and the positive"

        optional = [index for index in range(sample.g) if index not in required]
        keep = sorted(required | set(random.sample(optional, self.g_train - len(required))))

        sample.sims = _SubsampledLazy(sample.sims, keep=keep, shape=(sample.r, sample.g, sample.h, sample.w, sample.c))
        if sample.label_rank is not None:
            sample.label_rank = [sample.label_rank[index] for index in keep]
        if sample.label is not None:
            sample.label = keep.index(sample.label)
        sample.g = len(keep)
        sample.cost = self._cost(sample)
        return sample

    @stateless
    def pack_selected_samples(self, samples: list[GroupMeta]) -> "PackedGroupBatch":
        groups = []
        for sample in samples:
            real = _fetch_array(sample.real, sample, (sample.h, sample.w, sample.c))
            if isinstance(sample.sims, _SubsampledLazy):
                sims = sample.sims.get(sample)
            else:
                sims = _fetch_array(sample.sims, sample, (sample.r, sample.g, sample.h, sample.w, sample.c))

            label_rank = (
                torch.tensor(sample.label_rank, dtype=torch.long)
                if sample.label_rank is not None
                else torch.zeros(sample.g, dtype=torch.long)
            )
            groups.append(
                ImageGroup(
                    query=to_tensor(real),
                    support=to_tensor(sims),
                    label_rank=label_rank,
                    genotype_label=sample.label,
                    meta=sample.region,
                )
            )

        batch = build_packed_batch(
            groups,
            budget=self.budget,
            pad=self.pad,
            width_bucket=self.w_bucket,
            padding_value=self.padding_value,
        )
        return PackedGroupBatch(
            images=batch.images,
            labels=batch.labels,
            offsets=batch.offsets,
            widths=batch.widths,
            genotype_label=batch.genotype_label,
            num_genotypes=batch.num_genotypes,
            meta=batch.meta,
        )

    @stateless
    def batch(self, samples: list["PackedGroupBatch"]) -> "PackedGroupBatch":
        # The pack *is* the batch, so `batch_size=1` and this is an unwrap, not a collation
        assert len(samples) == 1, f"Expected batch_size=1 over packs, got {len(samples)}"
        return samples[0]


@dataclass
class PackedGroupBatch(PackedBatch):
    """`PackedBatch` plus the restore key energon needs to replay a batch.

    Note `__restore_key__` is *not* name-mangled: mangling applies to names with at most one
    trailing underscore.
    """

    __restore_key__: tuple = ()


class _SubsampledLazy:
    """Wraps a sims `Lazy` so the fetch slices the genotype axis on arrival.

    The whole `(R, G, ...)` block is one gzip member, so subsampling cannot avoid decompressing it;
    it only avoids carrying the dropped genotypes into the pack.
    """

    def __init__(self, inner: Lazy, *, keep: list[int], shape: tuple[int, ...]):
        self._inner = inner
        self._keep = keep
        self._shape = shape

    def get(self, sample: Any = None) -> np.ndarray:
        return _fetch_array(self._inner, sample, self._shape)[:, self._keep]


class EnergonPackedImageDataModule(L.LightningDataModule):
    """Lightning DataModule over the polylithic energon trees.

    Args:
        metadataset_path (str): Path to the `metadataset.yaml` written by `scripts/prepare_energon.sh`.
        cache_dir (str | None, optional): Node-local scratch for the cache pool. `/storage` is NFS and
            must not be used. Defaults to None, i.e. `$TMPDIR`.
        batch_size (int, optional): Image-slot budget per pack. Defaults to 256.
        w_bucket (int | None, optional): Width bucket size. Defaults to None (single bucket).
        w_max / g_max / r_max (optional): Reject groups exceeding these. Defaults to None.
        g_train (int | None, optional): Subsample genotypes per step; must retain the positive. Defaults to None.
        packing_buffer_size (int, optional): Groups of lookahead given to `select_packs`. Defaults to 512.
        shuffle_buffer_size (int, optional): Sample shuffle buffer. Defaults to 4096.
        max_samples_per_sequence (int | None, optional): Limit samples drawn per shard sequence. Defaults to 64.
        num_workers (int, optional): DataLoader workers per rank. Defaults to 1.
        cache_workers (int, optional): Threads prefetching pixels into the cache pool. Defaults to 4.
        max_cache_size_gbytes (float, optional): Cap on the cache pool. The failure mode of an uncapped
            pool is a confusing ENOSPC deep inside a worker. Defaults to 64.
        limit_val_batches (int | None, optional): Fixed validation length, so ranks agree on the step count.
        pad (bool, optional): Pad packs to batch_size. Defaults to True.
    """

    def __init__(
        self,
        *,
        metadataset_path: str,
        cache_dir: str | None = None,
        batch_size: int = 256,
        w_bucket: int | None = None,
        w_max: int | None = None,
        g_max: int | None = None,
        r_max: int | None = None,
        g_train: int | None = None,
        packing_buffer_size: int = 512,
        shuffle_buffer_size: int = 4096,
        max_samples_per_sequence: int | None = 64,
        num_workers: int = 1,
        cache_workers: int = 4,
        max_cache_size_gbytes: float = 64,
        limit_val_batches: int | None = 32,
        pad: bool = True,
        mean=(0.5,),
        std=(0.5,),
        seed_offset: int = 0,
        **_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Normalization moves to the GPU (`on_after_batch_transfer`), which is what buys the transfer
        # reduction: the loader ships uint8 rather than float32.
        self.transforms = transforms.Compose(
            [
                transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input
                transforms.Normalize(mean=self.hparams.mean, std=self.hparams.std),
            ]
        )

        self._loader = None
        self._cache_pool = None

    # -- construction ------------------------------------------------------------------------

    def _task_encoder(self) -> PackedGroupTaskEncoder:
        return PackedGroupTaskEncoder(
            budget=self.hparams.batch_size,
            w_bucket=self.hparams.w_bucket,
            w_max=self.hparams.w_max,
            g_max=self.hparams.g_max,
            r_max=self.hparams.r_max,
            g_train=self.hparams.g_train,
            pad=self.hparams.pad,
        )

    def _worker_config(self) -> WorkerConfig:
        # Lightning owns process-group initialization, so rank comes from the trainer rather than
        # from torch.distributed directly.
        rank, world_size = 0, 1
        if self.trainer is not None:
            rank, world_size = self.trainer.global_rank, self.trainer.world_size
        return WorkerConfig(
            rank=rank,
            world_size=world_size,
            num_workers=self.hparams.num_workers,
            seed_offset=self.hparams.seed_offset,
        )

    def _make_cache_pool(self) -> FileStoreCachePool:
        if self._cache_pool is None:
            cache_dir = self.hparams.cache_dir or os.environ.get("TMPDIR") or "/tmp"  # noqa: S108
            if cache_dir.startswith("/storage"):
                msg = f"cache_dir {cache_dir!r} is on NFS; the cache pool must live on node-local scratch"
                raise ValueError(msg)
            os.makedirs(cache_dir, exist_ok=True)
            logging.info("Energon cache pool in %s (capped at %.1f GB)", cache_dir, self.hparams.max_cache_size_gbytes)
            self._cache_pool = FileStoreCachePool(
                # A Path, not a str: energon joins onto it with `/`
                parent_cache_dir=Path(cache_dir),
                num_workers=self.hparams.cache_workers,
                max_cache_size_gbytes=self.hparams.max_cache_size_gbytes,
            )
        return self._cache_pool

    def train_dataloader(self):
        if self.hparams.num_workers < 1:
            # With workers, energon keeps one iterator alive across `iter()` calls, which is what
            # makes `limit_train_batches` a *virtual* epoch that advances through the data. Without
            # them, every epoch restarts the underlying iteration from the top.
            logging.warning("num_workers=0: each virtual epoch will restart the dataset rather than continue it")

        dataset = get_train_dataset(
            self.hparams.metadataset_path,
            split_part="train",
            worker_config=self._worker_config(),
            # The pack *is* the batch; this is not a placeholder
            batch_size=1,
            packing_buffer_size=self.hparams.packing_buffer_size,
            shuffle_buffer_size=self.hparams.shuffle_buffer_size,
            max_samples_per_sequence=self.hparams.max_samples_per_sequence,
            task_encoder=self._task_encoder(),
        )
        self._loader = get_savable_loader(dataset, cache_pool=self._make_cache_pool())
        return self._loader

    def val_dataloader(self):
        dataset = get_val_dataset(
            self.hparams.metadataset_path,
            split_part="val",
            worker_config=self._worker_config(),
            batch_size=1,
            packing_buffer_size=self.hparams.packing_buffer_size,
            # A fixed length so every rank agrees on the validation step count too
            limit=self.hparams.limit_val_batches,
            task_encoder=self._task_encoder(),
        )
        return get_savable_loader(dataset, cache_pool=self._make_cache_pool())

    # -- Lightning hooks ---------------------------------------------------------------------

    def on_after_batch_transfer(self, batch, dataloader_idx):  # noqa: ARG002
        # Normalize on the device, after the uint8 transfer
        batch.images = self.transforms(batch.images)
        return batch

    def state_dict(self):
        if self._loader is None:
            return {}
        # A collective: every rank must call it even though only rank 0 holds the result. This is
        # what matches Lightning's single-file checkpoint; `save_state_rank` would not.
        return {"loader": self._loader.save_state_global(0)}

    def load_state_dict(self, state_dict):
        if self._loader is None or "loader" not in state_dict:
            return
        self._loader.restore_state_global(state_dict["loader"])

    def teardown(self, stage=None):  # noqa: ARG002
        if self._cache_pool is not None:
            self._cache_pool.close()
            self._cache_pool = None
