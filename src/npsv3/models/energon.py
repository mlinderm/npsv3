"""Megatron-Energon polylithic loader for packed pileup images.

An example's (region's) metadata (`R`, `G`, geometry, label) lives in a small JSON record, with the images themselves
in a separate gzipped tar. Determining the packing only requires the metadata, enabling a larger lookahead buffer. The
actual pixel data is only fetched after packing has been determined.
"""

import contextlib
import gzip
import io
import json
import logging
import math
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
import yaml
from megatron.energon import (
    Cooker,
    CrudeJsonlDatasetFactory,
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
from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory, CrudeWebdataset
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME
from torchvision.transforms import v2 as transforms

from npsv3.models.loaders import (
    LABEL_PADDING_VALUE,
    REAL_SUFFIX,
    SIMS_SUFFIX,
    ImageGroup,
    PackedBatch,
    build_packed_batch,
    ntuple,
    select_packs,
    to_tensor,
)


@edataclass
class GroupMeta(Sample):
    """One region's metadata, with lazy handles for the actual images"""
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
    #: Lazy handles into the pixel tree; no bytes are read until `.get()`.
    real: Lazy
    sims: Any = None
    #: Image slots this group occupies in a pack. Filled in by `preencode_sample`.
    cost: int = 0


@cooker(need_cache=True)
@stateless
def cook_group(sample: dict, imgs: FileStore, cache) -> GroupMeta:
    """Return `GroupMeta` with lazy pixel handles from a crude JSON sample

    Note `basic_sample_keys` is mandatory for restorability.
    """
    meta = sample["json"]
    if isinstance(meta, (bytes, str)):
        meta = json.loads(meta)

    # `sample["__key__"]` is the jsonl row number for the CrudeJsonlDatasetFactory meta_ds (only
    # `basic_sample_keys` needs it, for restorability) -- it is not the pixel-tar member key, which
    # `EnergonExampleWriter` carries as an explicit `key` field in the JSON payload instead.
    key = meta["key"]
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
        real=cache.get_lazy(imgs, f"{key}.{REAL_SUFFIX}"),
        # `sims` pixel member only exists in the tar when the writer actually generated `sim.images`
        # (`r > 0`); `g` can now be nonzero (genotypes considered) even without simulated images.
        sims=cache.get_lazy(imgs, f"{key}.{SIMS_SUFFIX}") if int(meta["r"]) > 0 else None,
    )


def _group_cost(r: int, g: int, w: int, w_bucket: int | None) -> int:
    """Image slots one group occupies in a pack: ``1 + r*g``, times ``ceil(w / w_bucket)`` if bucketed."""
    slots = 1 + r * g
    if w_bucket:
        slots *= math.ceil(w / w_bucket)
    return slots


def _group_accepted(
    r: int, g: int, w: int, *, label: int | None, g_max: int | None, r_max: int | None, w_max: int | None
) -> bool:
    """Same reject rules as `PackedGroupTaskEncoder.preencode_sample`, without needing a `GroupMeta`."""
    if g < 1 or r < 1:
        # No simulations, so nothing for the metric-learning objective to score against
        return False
    if label is None:
        # No resolvable truth genotype (e.g. the true haplotype wasn't found even after population
        # top-up), so `label_rank` is all zeros -- InfoNCE.forward's per-variant softmax has no positive
        # to normalize against and torch.mean over the empty positives tensor is NaN, poisoning the
        # whole batch's loss. Reject rather than packing a group that can't be scored.
        return False
    if g_max is not None and g > g_max:
        return False
    if r_max is not None and r > r_max:
        return False
    if w_max is not None and w > w_max:
        return False
    return True


GZIP_MAGIC = b"\x1f\x8b"


def _fetch_array(lazy: Lazy, sample: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Return ndarray with valid shape from lazy pixel member (including fetching and any necessary decompression)"""
    raw = lazy.get(sample)
    # The aux store energon hands the cooker is wrapped in a `DecodeFileStore`, whose default decoder
    # includes webdataset's `gzfilter` -- so a `.bin.gz` member usually arrives already decompressed.
    # That is decoder configuration, not a guarantee, so check the magic rather than assume either way.
    if raw[:2] == GZIP_MAGIC:
        raw = gzip.decompress(raw)

    # `bytearray` rather than `bytes` so the array is writable: `torch.from_numpy` warns on a read-only
    # buffer, and `pytest.ini` turns warnings into errors.
    return np.frombuffer(bytearray(raw), dtype=np.uint8).reshape(shape)

# The base `TaskEncoder`'s `T_encoded_sample` type parameter is `Any` here rather than a concrete type:
# `select_samples_to_pack` uses it for a flat `GroupMeta` (pre-pack metadata), while `pack_selected_samples`/`batch`
# use it for the materialized `PackedGroupBatch` (post-pack), and no single type can honor both -- packing genuinely
# changes representation, which `TaskEncoder`'s generic contract does not have a slot for.
class PackedGroupTaskEncoder(TaskEncoder[GroupMeta, Any, PackedBatch, PackedBatch]):
    """Pack one or more whole regions into fixed-budget `PackedBatch`es.

    A region's real image and a complete set of its simulated candidates are an atomic unit (the loss function needs to
    score the real image vs. all possible simulated genotypes).
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
        pad: bool = True,
        padding_value: int = LABEL_PADDING_VALUE,
    ):
        """
        Args:
            budget: Image slots per pack, i.e., batch size
            w_bucket: Width bucket size. Groups only share a pack if they share a bucket.
            w_max: Reject groups wider than this.
            g_max: Reject groups with more genotypes than this.
            r_max: Reject groups with more replicates than this.
            pad: Pad each pack out to `budget` rows with zeros.
            padding_value: Label value for padding rows. Groups without a resolvable truth genotype are
                rejected outright (`_group_accepted`) rather than packed with this value, since the
                metric-learning loss has no positive to score them against.
        """
        super().__init__()
        self.budget = budget
        self.w_bucket = w_bucket
        self.w_max = w_max
        self.g_max = g_max
        self.r_max = r_max
        self.pad = pad
        self.padding_value = padding_value

    def _cost(self, sample: GroupMeta) -> int:
        return _group_cost(sample.r, sample.g, sample.w, self.w_bucket)

    @stateless
    def preencode_sample(self, sample: GroupMeta) -> GroupMeta:
        """Compute the pack cost and reject groups that cannot be packed without needing images."""
        if not _group_accepted(
            sample.r, sample.g, sample.w, label=sample.label, g_max=self.g_max, r_max=self.r_max, w_max=self.w_max
        ):
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

    @stateless
    def pack_selected_samples(self, samples: list[GroupMeta]) -> "PackedGroupBatch":
        groups = []
        for sample in samples:
            real = _fetch_array(sample.real, sample, (sample.h, sample.w, sample.c))
            sims = _fetch_array(sample.sims, sample, (sample.r, sample.g, sample.h, sample.w, sample.c))

            # `preencode_sample` already rejected unlabeled groups (`_group_accepted`), so `label_rank` is
            # always defined here.
            assert sample.label_rank is not None, "Unlabeled group reached pack_selected_samples"
            groups.append(
                ImageGroup(
                    query=to_tensor(real),
                    support=to_tensor(sims),
                    label_rank=torch.tensor(sample.label_rank, dtype=torch.long),
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


def _prepare_imgs_tree(imgs_dir: str, *, num_workers: int) -> None:
    """Index `imgs_ds`'s tar shards as a prepared CRUDE energon dataset.

    `imgs_ds` always gets one `train` split covering every shard: it is only ever read by key via `FileStore` as an aux
    store, never filtered by split, however some split is still needed for energon requires to treat `imgs_ds` as a
    prepared, directory-referenceable dataset.
    """
    tree = EPath(imgs_dir)
    all_tars = sorted(str(p.relative_to(Path(imgs_dir))) for p in Path(imgs_dir).glob("**/*.tar"))
    if not all_tars:
        msg = f"No imgs_ds/imgs-*.tar shards found under {imgs_dir!r}"
        raise ValueError(msg)

    # Swallow the Energon's unconditional prints
    with contextlib.redirect_stdout(io.StringIO()):
        BaseWebdatasetFactory.prepare_dataset(
            tree,
            all_tars,
            split_parts_ratio=[("train", 1.0)],
            shuffle_seed=None,
            progress_fn=lambda els, length=None: els,  # noqa: ARG005
            workers=num_workers,
        )
    dataset_yaml = {"__module__": "megatron.energon", "__class__": CrudeWebdataset.__name__}
    with (Path(imgs_dir) / MAIN_FOLDER_NAME / "dataset.yaml").open("w") as dataset_yaml_file:
        yaml.dump(dataset_yaml, dataset_yaml_file, sort_keys=False)


def _merge_meta_shards(meta_dir: str) -> str:
    """Return path to concatenated `meta-*.jsonl` fragments in sorted filename order prepared as Energon dataset"""
    fragments = sorted(Path(meta_dir).glob("meta-*.jsonl"))
    if not fragments:
        msg = f"No meta_ds/meta-*.jsonl fragments found under {meta_dir!r}"
        raise ValueError(msg)

    merged_path = Path(meta_dir) / "train.jsonl"
    with merged_path.open("wb") as merged_file:
        for fragment in fragments:
            with fragment.open("rb") as fragment_file:
                shutil.copyfileobj(fragment_file, merged_file)
    for fragment in fragments:
        fragment.unlink()

    CrudeJsonlDatasetFactory.prepare_dataset(EPath(merged_path))
    return str(merged_path)


def prepare_energon_dataset(output_dir: str, *, num_workers: int = 2) -> str:
    """Prepare the polylithic energon trees saved under `output_dir` as a single train split

    1. `imgs_ds` (the aux pixel tree) is indexed as a single-split prepared dataset, unshuffled, so it can be referenced
       by directory.
    2. Every actor's metadata fragment is merged, in plain sorted (actor) order,
       into one plain `meta_ds/train.jsonl` file that matches `imgs_ds`'s own order. `dataset.yaml` then references it
       directly (a single `DatasetReference` with `aux: {imgs: ...}`, no blend) -- so a train dataset built with
       `repeat=False` (see `EnergonPackedImageDataModule`) terminates after one true pass over that file rather than
       looping.

    Args:
        output_dir: The directory `EnergonExampleWriter` wrote `meta_ds`/`imgs_ds` under.
        num_workers: Parallelism for indexing `imgs_ds`'s tar shards.

    Returns:
        The path to the `dataset.yaml` written.
    """
    imgs_dir = os.path.join(output_dir, "imgs_ds")
    _prepare_imgs_tree(imgs_dir, num_workers=num_workers)

    meta_path = _merge_meta_shards(os.path.join(output_dir, "meta_ds"))

    dataset = {
        "__module__": "megatron.energon",
        "__class__": "MetadatasetV2",
        "splits": {"train": {"path": f"./meta_ds/{Path(meta_path).name}", "aux": {"imgs": "./imgs_ds"}}},
    }
    dataset_yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(dataset_yaml_path, "w") as dataset_yaml_file:
        yaml.safe_dump(dataset, dataset_yaml_file, sort_keys=False)
    return dataset_yaml_path


def _read_split(dataset_yaml_path: str | Path, split_part: str | None = None) -> tuple[Path, dict]:
    """Teturn the directory plus one split's raw `{path, aux}` mapping (paths still relative to that directory) from an
    Energon `datasert_yaml_path` file.

    If `split_part` is present, that split is used. Otherwise the file is assumed to be a `prepare_energon_dataset`-written tree,
    which always writes exactly one split conventionally named "train", and that split is used regardless of name.
    """
    root = Path(dataset_yaml_path).parent
    with open(dataset_yaml_path) as dataset_yaml_file:
        dataset_yaml = yaml.safe_load(dataset_yaml_file)
    splits = dataset_yaml["splits"]
    if split_part is not None and split_part in splits:
        return root, splits[split_part]
    (split,) = splits.values()
    return root, split


def available_splits(metadataset_path: str) -> set[str]:
    """Return the splits defined in `metadataset_path`'s Energon `dataset.yaml`"""
    with open(metadataset_path) as metadataset_file:
        return set(yaml.safe_load(metadataset_file)["splits"].keys())


def _leaf_dataset_reference(dataset_yaml_path: str | Path) -> dict:
    """Read one `prepare_energon_dataset` tree's split and resolve its `path`/`aux` to absolute paths,
    so it can be embedded as a leaf inside another, synthesized metadataset (see `_build_metadataset`).

    Absolute, because that synthesized metadataset is handed to energon as an in-memory dict rather
    than a file on disk: `load_dataset` resolves a dict metadataset's relative paths against "/", not
    against `dataset_yaml_path`'s own directory, so a relative path here would silently resolve wrong.
    """
    root, split = _read_split(dataset_yaml_path)
    return {
        "path": str(root / split["path"]),
        "aux": {name: str(root / aux_path) for name, aux_path in split["aux"].items()},
    }


def _build_metadataset(split_paths: dict[str, str | Sequence[str] | None]) -> dict:
    """Assemble an in-memory `MetadatasetV2` spec from one or more `prepare_energon_dataset` trees per
    split (e.g. `{"train": [...], "val": [...], "test": [...]}`), without writing anything to disk --
    `load_dataset` (megatron.energon) accepts a plain dict directly, in place of a `dataset.yaml` path.

    A split with no paths (`None` or an empty sequence) is omitted entirely, so
    `EnergonPackedImageDataModule.val_dataloader`/`test_dataloader` fail loudly (a `KeyError` on the
    split name) if called without one, rather than silently reading the wrong data. A single path
    becomes a plain leaf reference; more than one is blended with equal weight, so e.g. several
    samples' independently-prepared trees can serve as one split.
    """
    splits = {}
    for split_name, paths in split_paths.items():
        if not paths:
            continue
        paths = [paths] if isinstance(paths, str) else list(paths)
        leaves = [_leaf_dataset_reference(path) for path in paths]
        splits[split_name] = leaves[0] if len(leaves) == 1 else {"blend": leaves}

    return {"__module__": "megatron.energon", "__class__": "MetadatasetV2", "splits": splits}


def estimate_epoch_batches(
    metadataset_paths: str | Sequence[str],
    *,
    budget: int,
    split_part: str = "train",
    w_bucket: int | None = None,
    w_max: int | None = None,
    g_max: int | None = None,
    r_max: int | None = None,
    world_size: int = 1,
    occupancy: float = 0.9,
) -> int:
    """Estimate the number of packed batches in one pass over one or more `prepare_energon_dataset` trees.

    Streams each tree's metadata jsonl file directly (`key, r, g, h, w, c, label, label_rank, region,
    sample` per group, one small flat record per line) rather than going through the energon pipeline
    itself, so this touches no pixels and needs no packing buffer -- cheap enough to run before
    launching training.

    Sums each accepted group's pack cost (`_group_cost`, applying the same reject rules
    `PackedGroupTaskEncoder.preencode_sample` would via `_group_accepted`) across all given trees'
    jsonl files, divides by `budget * occupancy` -- a bin-packing lower bound corrected for the
    sub-100% occupancy packing actually achieves in practice (see `test_occupancy`, typically >90%) --
    then by `world_size`, since `wds.split_by_node`-style sharding gives each DDP rank about
    `1/world_size` of the shards.

    This is an approximation, not an exact count: packing is greedy and each rank's shard slice packs
    independently, so no single number is the *exact* one-pass length for every rank.
    `EnergonPackedImageDataModule`'s train datasets repeat indefinitely and every rank is given the
    same `limit_train_batches` regardless, specifically so the DDP all-reduce is never reached an
    unequal number of times -- see `test_two_rank_ddp`.

    Args:
        metadataset_paths: One or more paths to `dataset.yaml` files, summed together -- matching how
            `EnergonPackedImageDataModule`/`_build_metadataset` blend multiple `prepare_energon_dataset`
            trees into one split. A path may also be a single pre-combined multi-split `dataset.yaml`
            (see `EnergonPackedImageDataModule`'s `metadataset_path` argument), in which case
            `split_part` selects the relevant split.
        budget: Image slots per pack, i.e., `PackedGroupTaskEncoder.budget` / `batch_size`.
        split_part: Which split to read from a pre-combined multi-split `dataset.yaml`. Ignored for a
            plain `prepare_energon_dataset` tree, which has only the one split regardless of name.
        w_bucket, w_max, g_max, r_max: Same meaning as on `PackedGroupTaskEncoder`.
        world_size: Number of DDP ranks the estimate should be divided across. Defaults to 1 (the
            whole split, single-process).
        occupancy: Expected fraction of each pack's budget that packing actually fills. Defaults to
            0.9, a conservative reading of `test_occupancy`'s >0.90 sanity threshold.

    Returns:
        A batch count of at least 1.
    """
    paths = [metadataset_paths] if isinstance(metadataset_paths, str) else list(metadataset_paths)

    total_cost = 0
    for metadataset_path in paths:
        root, split = _read_split(metadataset_path, split_part)
        with (root / split["path"]).open() as jsonl_file:
            for line in jsonl_file:
                if not line.strip():
                    continue
                row = json.loads(line)
                r, g, w = int(row["r"]), int(row["g"]), int(row["w"])
                label = row.get("label")
                if _group_accepted(r, g, w, label=label, g_max=g_max, r_max=r_max, w_max=w_max):
                    total_cost += _group_cost(r, g, w, w_bucket)

    packs = total_cost / (budget * occupancy) / world_size
    logging.info(f"Estimated {packs:.1f} packs given {total_cost} images, a batch size of {budget}, a world size of {world_size}, and an estiomated occupancy of {occupancy}")
    return max(1, math.ceil(packs))


@contextlib.contextmanager
def _quiet_energon(verbose: bool):
    """`get_train_dataset`/`get_val_dataset` unconditionally `print()` a per-worker shard-slice summary
    while building the dataset (energon's `_print_shard_slices`, with no verbosity flag of its own) --
    redirected here, and only around those two calls, so nothing else's output is touched.
    """
    if verbose:
        yield
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            yield


class _OpaqueIterable:
    """Hides a `DataLoader` from Lightning, which otherwise re-instantiates every predict `DataLoader`
    to inject an index-tracking batch sampler (see `_DataConnector._prepare_dataloader`). Energon's
    loaders can't be re-instantiated that way, and have no sampler to wrap regardless. The cost is that
    `BasePredictionWriter` callbacks receive no `batch_indices`."""

    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        return iter(self.loader)


class EnergonPackedImageDataModule(L.LightningDataModule):
    """Lightning DataModule over the polylithic energon trees.

    The data can be specified either as:

    - `metadataset_path`: a single, pre-combined `dataset.yaml` that already defines its own
      train/val/test splits -- e.g. hand-built, or prepared once and reused as-is.
    - `train_metadataset_paths` (+ optional `val_metadataset_paths` / `test_metadataset_paths`): one
      or more `dataset.yaml` paths *per split*, each written by `prepare_energon_dataset` (which
      always writes a single, unsplit tree -- see its docstring). These are combined in memory into
      one metadataset (`_build_metadataset`), no file written to disk -- the common case, e.g.
      training across several samples' independently-prepared trees.

    Args:
        metadataset_path (str | None, optional): A pre-combined `dataset.yaml` with real train/val/test
            splits. Mutually exclusive with `train_metadataset_paths`; exactly one of the two is
            required.
        train_metadataset_paths (str | Sequence[str] | None, optional): One or more `dataset.yaml`
            paths written by `prepare_energon_dataset` (called automatically by `vcf_to_graph_examples`
            when writing an energon dataset), combined into the train split. More than one is blended
            with equal weight.
        val_metadataset_paths / test_metadataset_paths (str | Sequence[str] | None, optional): Same,
            for the validation/test splits. Either may be omitted (None), in which case
            `val_dataloader`/`test_dataloader` must not be called.
        cache_dir (str | None, optional): Node-local scratch for the cache pool. `/storage` is NFS and
            must not be used. Defaults to None, i.e. `$TMPDIR`.
        batch_size (int, optional): Image-slot budget per pack. Defaults to 256.
        w_bucket (int | None, optional): Width bucket size. Defaults to None (single bucket).
        w_max / g_max / r_max (optional): Reject groups exceeding these. Defaults to None.
        packing_buffer_size (int, optional): Groups of lookahead given to `select_packs`. Defaults to 512.
        shuffle_buffer_size (int, optional): Sample shuffle buffer. Defaults to 4096.
        max_samples_per_sequence (int | None, optional): Limit samples drawn per shard sequence. Defaults to 64.
        num_workers (int, optional): DataLoader workers per rank. Defaults to 1.
        cache_workers (int, optional): Threads prefetching pixels into the cache pool. Defaults to 4.
        max_cache_size_gbytes (float, optional): Cap on the cache pool. The failure mode of an uncapped
            pool is a confusing ENOSPC deep inside a worker. Defaults to 64.
        limit_val_batches (int | None, optional): Fixed validation length, so ranks agree on the step
            count. Applied per rank (`LimitDataset` truncates each rank's own shard slice to at most
            this many batches, not the split as a whole), so this only equalizes ranks in practice --
            it assumes every rank's local slice has at least this many batches to begin with.
        limit_test_batches (int | None, optional): Same as `limit_val_batches`, for the test split.
        predict_split (str, optional): Split `predict_dataloader` makes a single, unshuffled pass over.
            Defaults to "train", since `prepare_energon_dataset` writes only that split.
        limit_predict_batches (int | None, optional): Cap on the prediction pass, in packed batches.
            Defaults to None, i.e. the entire split.
        pad (bool, optional): Pad packs to batch_size. Defaults to True.
        repeat (bool, optional): Repeat the train dataset indefinitely, so `limit_train_batches` defines
            a virtual epoch (see `train_dataloader`). Defaults to True; pass False for a single finite
            pass over the train split, e.g. in tests that need the dataloader to exhaust.
        verbose (bool, optional): Let energon's own per-worker shard-slice summary print to stdout
            (see `_quiet_energon`). Defaults to False.
    """

    def __init__(
        self,
        *,
        metadataset_path: str | None = None,
        train_metadataset_paths: str | Sequence[str] | None = None,
        val_metadataset_paths: str | Sequence[str] | None = None,
        test_metadataset_paths: str | Sequence[str] | None = None,
        cache_dir: str | None = None,
        batch_size: int = 256,
        w_bucket: int | None = None,
        w_max: int | None = None,
        g_max: int | None = None,
        r_max: int | None = None,
        packing_buffer_size: int = 512,
        shuffle_buffer_size: int = 4096,
        max_samples_per_sequence: int | None = 64,
        num_workers: int = 1,
        cache_workers: int = 4,
        max_cache_size_gbytes: float = 64,
        limit_val_batches: int | None = 32,
        limit_test_batches: int | None = 32,
        predict_split: str = "train",
        limit_predict_batches: int | None = None,
        pad: bool = True,
        mean=(0.5,),
        std=(0.5,),
        seed_offset: int = 0,
        repeat: bool = True,
        verbose: bool = False,
        **_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # `get_train_dataset`/`get_val_dataset` accept either a path or a dict, even though their type stubs only
        # declare `str | EPath | Path` requiring `# type: ignore`s below.
        self._metadataset: str | dict
        if metadataset_path is not None:
            _mutually_exclusive_msg = (
                "Pass either `metadataset_path` (a pre-combined dataset.yaml with its own train/val/test "
                "splits) or the per-split `*_metadataset_paths`, not both"
            )
            assert train_metadataset_paths is None, _mutually_exclusive_msg
            assert val_metadataset_paths is None, _mutually_exclusive_msg
            assert test_metadataset_paths is None, _mutually_exclusive_msg
            # Used as-is: `get_train_dataset`/`get_val_dataset` accept a path directly, same as a dict.
            self._metadataset = metadataset_path
        else:
            assert train_metadataset_paths is not None, (
                "Must pass either `metadataset_path` or `train_metadataset_paths`"
            )
            # Combined once, in-memory (no dataset.yaml written to disk) -- see `_build_metadataset`. A
            # split with no paths given is simply absent, so its dataloader fails loudly if ever called.
            self._metadataset = _build_metadataset({
                "train": train_metadataset_paths,
                "val": val_metadataset_paths,
                "test": test_metadataset_paths,
            })

        # Normalization moves to the GPU (`on_after_batch_transfer`) to reduce the size of the data transferred (uint8 vs float32).
        self.transforms = transforms.Compose(
            [
                transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input
                transforms.Normalize(mean=self.hparams["mean"], std=self.hparams["std"]),
            ]
        )

        self._loader = None
        self._cache_pool = None

    def _task_encoder(self) -> PackedGroupTaskEncoder:
        return PackedGroupTaskEncoder(
            budget=self.hparams["batch_size"],
            w_bucket=self.hparams["w_bucket"],
            w_max=self.hparams["w_max"],
            g_max=self.hparams["g_max"],
            r_max=self.hparams["r_max"],
            pad=self.hparams["pad"],
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
            num_workers=self.hparams["num_workers"],
            seed_offset=self.hparams["seed_offset"],
        )

    def _make_cache_pool(self) -> FileStoreCachePool:
        if self._cache_pool is None:
            cache_dir = self.hparams["cache_dir"] or tempfile.gettempdir()
            os.makedirs(cache_dir, exist_ok=True)
            logging.info(
                "Energon cache pool in %s (capped at %.1f GB)", cache_dir, self.hparams["max_cache_size_gbytes"]
            )
            self._cache_pool = FileStoreCachePool(
                # A Path, not a str: energon joins onto it with `/`
                parent_cache_dir=Path(cache_dir),
                num_workers=self.hparams["cache_workers"],
                max_cache_size_gbytes=self.hparams["max_cache_size_gbytes"],
            )
        return self._cache_pool

    def train_dataloader(self):
        if self.hparams["repeat"] and self.hparams["num_workers"] < 1:
            # With workers, energon keeps one iterator alive across `iter()` calls, which is what
            # makes `limit_train_batches` a *virtual* epoch that advances through the data. Without
            # them, every epoch restarts the underlying iteration from the top.
            logging.warning("num_workers=0: each virtual epoch will restart the dataset rather than continue it")

        with _quiet_energon(self.hparams["verbose"]):
            dataset = get_train_dataset(
                self._metadataset,  # type: ignore
                split_part="train",
                worker_config=self._worker_config(),
                # The pack *is* the batch; this is not a placeholder
                batch_size=1,
                packing_buffer_size=self.hparams["packing_buffer_size"],
                shuffle_buffer_size=self.hparams["shuffle_buffer_size"],
                max_samples_per_sequence=self.hparams["max_samples_per_sequence"],
                task_encoder=self._task_encoder(),
                repeat=self.hparams["repeat"],
            )
        self._loader = get_savable_loader(dataset, cache_pool=self._make_cache_pool())
        return self._loader

    def val_dataloader(self):
        with _quiet_energon(self.hparams["verbose"]):
            dataset = get_val_dataset(
                self._metadataset,  # type: ignore
                split_part="val",
                worker_config=self._worker_config(),
                batch_size=1,
                packing_buffer_size=self.hparams["packing_buffer_size"],
                # A fixed length so every rank agrees on the validation step count too
                limit=self.hparams["limit_val_batches"],
                task_encoder=self._task_encoder(),
            )
        return get_savable_loader(dataset, cache_pool=self._make_cache_pool())

    def test_dataloader(self):
        with _quiet_energon(self.hparams["verbose"]):
            dataset = get_val_dataset(
                self._metadataset,  # type: ignore
                split_part="test",
                worker_config=self._worker_config(),
                batch_size=1,
                packing_buffer_size=self.hparams["packing_buffer_size"],
                # A fixed length so every rank agrees on the test step count too
                limit=self.hparams["limit_test_batches"],
                task_encoder=self._task_encoder(),
            )
        return get_savable_loader(dataset, cache_pool=self._make_cache_pool())

    def predict_dataloader(self):
        # A single, finite, unshuffled pass (unlike `train_dataloader`, which repeats indefinitely),
        # defaulting to the `train` split since that is what `prepare_energon_dataset` writes.
        with _quiet_energon(self.hparams["verbose"]):
            dataset = get_val_dataset(
                self._metadataset,  # type: ignore
                split_part=self.hparams["predict_split"],
                worker_config=self._worker_config(),
                batch_size=1,
                packing_buffer_size=self.hparams["packing_buffer_size"],
                limit=self.hparams["limit_predict_batches"],
                task_encoder=self._task_encoder(),
            )
        return _OpaqueIterable(get_savable_loader(dataset, cache_pool=self._make_cache_pool()))

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


def _random_patch_mask(batch_size: int, num_patches: int, *, mask_fraction: float, device=None) -> torch.Tensor:
    """A `(batch_size, num_patches)` bool mask, `mask_fraction` of each row's patches True, chosen
    independently per row."""
    num_masked = int(mask_fraction * num_patches)
    selected = torch.rand(batch_size, num_patches, device=device).argsort(dim=1)[:, :num_masked]
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    mask.scatter_(1, selected, True)
    return mask


class EnergonMaskedImageDataModule(EnergonPackedImageDataModule):
    """Energon-backed data module for masked-image pretraining (`MaskedImageModeling` in `transformer.py`).

    Reuses `EnergonPackedImageDataModule`'s packing/loading pipeline unchanged -- masked pretraining
    needs no group/genotype/label structure, just a steady stream of same-sized images -- and replaces
    the `PackedBatch` it would otherwise hand to the training step with `(pixel_values, bool_masked_pos)`,
    matching `MaskedImageModeling.forward`. Since it still goes through `PackedGroupTaskEncoder`, a
    region without any simulated genotypes (rejected by `_group_accepted`) contributes no images here
    either, even though masked pretraining itself has no use for the simulations.

    Args:
        patch_size (int | tuple[int, int], optional): Patch size the mask is defined over. Defaults to 16.
        mask_fraction (float, optional): Fraction of each image's patches masked. Defaults to 0.5.
        (all other args, see `EnergonPackedImageDataModule`)
    """

    def __init__(self, *, patch_size: int | tuple[int, int] = 16, mask_fraction: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters({"patch_size": ntuple(patch_size, 2), "mask_fraction": mask_fraction})

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch = super().on_after_batch_transfer(batch, dataloader_idx)
        patch_h, patch_w = self.hparams["patch_size"]
        b, _c, h, w = batch.images.shape
        num_patches = (h // patch_h) * (w // patch_w)
        mask = _random_patch_mask(b, num_patches, mask_fraction=self.hparams["mask_fraction"], device=batch.images.device)
        return batch.images, mask
