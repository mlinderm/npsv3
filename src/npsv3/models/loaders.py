import math
from abc import ABC, abstractmethod
from collections.abc import Generator, Hashable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from itertools import repeat
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from torch.utils import data
from torchvision.transforms import v2 as transforms

LABEL_PADDING_VALUE = -100
"""Label value for padding rows and for groups without a defined genotype label."""


# Adapted from timm library
# https://github.com/huggingface/pytorch-image-models/blob/954613a470652e4a113ff45b62dbd15c4e229218/timm/layers/helpers.py#L10C1-L15C17
def ntuple(val: Iterable | Any, n: int) -> tuple:
    """Convert a val to a tuple of n elements."""
    if isinstance(val, Iterable) and not isinstance(val, str):
        as_tuple= tuple(val)
        assert len(as_tuple) == n, f"Expected {n} values, got {len(as_tuple)}"
    else:
        as_tuple = tuple(repeat(val, n))
    return as_tuple


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert *HWC numpy array to a *CHW tensor"""
    return torch.from_numpy(img).movedim(-1, -3).contiguous()


class EmptyDataset(data.Dataset):
    def __init__(self):
        super(EmptyDataset).__init__()

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


class BaseDataModule(L.LightningDataModule, ABC):
    def __init__(
        self,
        *,
        train_urls: str | None = None,
        validate_urls: str | None = None,
        test_urls: str | None = None,
        predict_urls: str | None = None,
        batch_size=256,
        pad=False,
        num_workers=1,
        shuffle_size=1000,
        mean=(0.5,),
        std=(0.5,),
        legacy_axis_order=False,
        **_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["training_urls", "validation_urls", "prediction_urls", "test_urls"])

        self.train_urls = train_urls
        self.validate_urls = validate_urls
        self.predict_urls = predict_urls
        self.test_urls = test_urls

        self.transforms = transforms.Compose(
            [
                transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input
                transforms.Normalize(mean=self.hparams.mean, std=self.hparams.std),
            ]
        )

    def _make_base_dataset(self, urls, mode="train", resampled=False):
        legacy_axis_order = self.hparams.legacy_axis_order

        def to_tuple(data):
            query_image = to_tensor(data["image.npy.gz"])
            support_images = to_tensor(data["sim.images.npy.gz"])
            if legacy_axis_order:
                # Shards written before the replicates-major convention are genotypes-major (G, R, H, W, C).
                # With R=1 (the whole existing corpus) this transpose is free.
                support_images = support_images.transpose(0, 1)
            num_genotypes = support_images.size(1)

            # `label` is the canonical positive and is absent entirely for unlabeled examples
            label = data.get("label.cls")
            if (label_rank := data.get("label.rank.npy")) is not None:
                label_rank = torch.from_numpy(label_rank)
            elif label is not None:
                # Synthesize ranked labels from the integer label if not present
                label_rank = F.one_hot(torch.tensor(label, dtype=torch.long), num_genotypes)
                if label > 0:
                    # The same "presence" for non-reference genotypes, i.e., non-reference concordant, are considered "third-rank" positives
                    label_rank[(label_rank == 0) & (torch.arange(num_genotypes) > 0)] = 3
            else:
                label_rank = torch.zeros(num_genotypes, dtype=torch.long)
            return query_image, support_images, label_rank, label, data.get("region.txt", data["__key__"])

        dataset = wds.WebDataset(urls, nodesplitter=wds.split_by_node, shardshuffle=100 if mode == "train" and not resampled else False, resampled=resampled)
        if mode == "train":
            dataset = dataset.shuffle(self.hparams.shuffle_size)
        return dataset.decode().map(to_tuple)

    @abstractmethod
    def make_loader(self, urls, mode):
        pass

    def train_dataloader(self):
        return self.make_loader(self.train_urls, mode="train")

    def val_dataloader(self):
        # Make sure to return a valid dataloader, even if validation data is not available since Lightning still calls this method with zero validation steps
        if self.validate_urls:
            return self.make_loader(self.validate_urls or [], mode="val")
        return data.DataLoader(EmptyDataset())

    def test_dataloader(self):
        # Make sure to return a valid dataloader, even if test data is not available since Lightning still calls this method with zero validation steps
        if self.test_urls:
            return self.make_loader(self.test_urls or [], mode="test")
        return data.DataLoader(EmptyDataset())

    def predict_dataloader(self):
        return self.make_loader(self.predict_urls, mode="predict")


@dataclass
class ImageGroup:
    """One region's real (query) image plus the grid of simulated (support) images scored against it.

    The support grid is **replicates-major** ``(R, G, C, H, W)``: flattening it yields
    ``[r0g0 ... r0g_{G-1}, r1g0 ...]``, which is why the length-``G`` rank vector is *tiled* R times
    rather than interleaved. ``genotype_label`` is the canonical single positive (an index into the
    genotype axis); ``label_rank`` is its derived view.
    """

    query: torch.Tensor  # (C, H, W)
    support: torch.Tensor  # (R, G, C, H, W)
    label_rank: torch.Tensor  # (G,) int64 ranks
    genotype_label: int | None = None
    meta: Any = None

    @property
    def num_replicates(self) -> int:
        return self.support.shape[0]

    @property
    def num_genotypes(self) -> int:
        return self.support.shape[1]

    @property
    def width(self) -> int:
        return self.support.shape[-1]

    def cost(self, *, width_bucket: int | None = None) -> int:
        """Number of image slots this group occupies in a pack, ``(1 + R*G) * ceil(w / w_bucket)``."""
        slots = 1 + self.num_replicates * self.num_genotypes
        if width_bucket:
            slots *= math.ceil(self.width / width_bucket)
        return slots

    def flat_support(self) -> torch.Tensor:
        """Support images flattened replicates-major to ``(R*G, C, H, W)``."""
        return self.support.reshape(-1, *self.support.shape[2:])

    def flat_labels(self) -> torch.Tensor:
        """Rank labels tiled to match :meth:`flat_support`, i.e. ``(R*G,)``."""
        return torch.tile(self.label_rank, (self.num_replicates,))


@dataclass
class PackedBatch:
    """A packed batch of images: all groups' support images first, then one query image per group.

    Images are laid out as ``[s_00 ... s_0n, s_10 ... s_1m, ..., q_0 ... q_v]`` with ``offsets``
    indexing the *support* region only, which is what
    :func:`torch.nested.nested_tensor_from_jagged` consumes downstream.
    """

    images: torch.Tensor  # (B, C, H, W_bucket), supports [0:offsets[-1]) then queries
    labels: torch.Tensor  # (B,) or (total_support,) int64 ranks, tiled R times
    offsets: torch.Tensor  # (n_groups+1,) int64, into the support region
    widths: torch.Tensor  # (B,) int32, true width before bucket padding
    genotype_label: torch.Tensor  # (n_groups,) int64, the single positive's genotype index
    num_genotypes: torch.Tensor  # (n_groups,) int32, G_b; with offsets this recovers R_b
    meta: list = field(default_factory=list)  # per-group region strings / keys

    # Deliberately un-annotated, so it is a class attribute and not a dataclass field
    _TENSOR_FIELDS = ("images", "labels", "offsets", "widths", "genotype_label", "num_genotypes")

    @property
    def num_groups(self) -> int:
        return self.offsets.numel() - 1

    @property
    def num_images(self) -> int:
        """Number of real (non-padding) image slots."""
        return int(self.offsets[-1]) + self.num_groups

    def positive_indices(self) -> torch.Tensor:
        """Indices into ``images``/``labels`` of every replicate's true-genotype image.

        There are ``R_b`` positives per group (the same genotype simulated ``R_b`` times), at
        ``offsets[b] + r*G_b + genotype_label[b]``. Unlabeled groups contribute nothing.
        """
        indices = []
        lengths = torch.diff(self.offsets)
        for group in range(self.num_groups):
            label = int(self.genotype_label[group])
            if label < 0:
                continue
            num_genotypes = int(self.num_genotypes[group])
            num_replicates = int(lengths[group]) // num_genotypes
            start = int(self.offsets[group])
            indices.extend(start + replicate * num_genotypes + label for replicate in range(num_replicates))
        return torch.tensor(indices, dtype=torch.long)

    def pin_memory(self) -> "PackedBatch":
        return replace(self, **{name: getattr(self, name).pin_memory() for name in self._TENSOR_FIELDS})

    def to(self, device, *args, **kwargs) -> "PackedBatch":
        return replace(self, **{name: getattr(self, name).to(device, *args, **kwargs) for name in self._TENSOR_FIELDS})


def select_packs(costs: Sequence[int | tuple[int, Hashable]], *, budget: int) -> list[list[int]]:
    """Bin pack groups into fixed-budget packs, returning the original indices assigned to each pack.

    Groups are bucketed by width bucket first (so a pack never mixes geometries), then packed
    first-fit-decreasing within each bucket. Pure and metadata-only: no pixels are touched. The
    result is deterministic given the input order.

    Args:
        costs: Per-group cost in image slots, either as a plain int or as a ``(cost, bucket)`` pair.
            Buckets are emitted in first-appearance order.
        budget: Maximum number of image slots per pack.

    Returns:
        list[list[int]]: Indices into ``costs``, ascending within each pack.

    Raises:
        ValueError: If any single group's cost exceeds ``budget``. Such groups must be rejected by
            the caller rather than given their own oversized bin.
    """
    buckets: dict[Hashable, list[tuple[int, int]]] = {}
    for index, entry in enumerate(costs):
        cost, bucket = entry if isinstance(entry, tuple) else (entry, 0)
        cost = int(cost)
        if cost > budget:
            msg = f"Group {index} costs {cost} image slots, which exceeds the budget of {budget}"
            raise ValueError(msg)
        buckets.setdefault(bucket, []).append((index, cost))

    packs: list[list[int]] = []
    for entries in buckets.values():
        bins: list[tuple[list[int], list[int]]] = []  # ([remaining slots], [group indices])
        # Decreasing cost, ties broken by original index so the result doesn't depend on sort stability
        for index, cost in sorted(entries, key=lambda entry: (-entry[1], entry[0])):
            for remaining, members in bins:
                if remaining[0] >= cost:
                    remaining[0] -= cost
                    members.append(index)
                    break
            else:
                bins.append(([budget - cost], [index]))
        packs.extend(sorted(members) for _remaining, members in bins)
    return packs


def build_packed_batch(
    groups: Sequence[ImageGroup],
    *,
    budget: int | None = None,
    pad: bool = False,
    width_bucket: int | None = None,
    padding_value: int = LABEL_PADDING_VALUE,
) -> PackedBatch:
    """Concatenate groups into a single :class:`PackedBatch`.

    Padding is 0, the upstream "no data in this pixel" sentinel, so padding rows read as an empty
    pileup rather than as out-of-distribution noise.

    Args:
        groups: Groups to pack, in the order they should appear in the batch.
        budget: Maximum number of image slots. Required if pad is True. Defaults to None.
        pad: Pad the batch out to budget rows. Defaults to False.
        width_bucket: Width to pad images out to. Defaults to the widest group.
        padding_value: Label value for padding rows and unlabeled groups. Defaults to -100.
    """
    assert len(groups) > 0, "Cannot build a packed batch from zero groups"

    channels, height = groups[0].query.shape[0], groups[0].query.shape[1]
    out_width = width_bucket or max(group.width for group in groups)

    support_lengths = [group.num_replicates * group.num_genotypes for group in groups]
    total_support = sum(support_lengths)
    num_images = total_support + len(groups)
    if budget is not None:
        assert num_images <= budget, f"Total of {num_images} images exceeds the budget of {budget}"
    assert not pad or budget is not None, "budget is required when pad is True"
    num_rows = budget if pad else num_images

    images = torch.zeros((num_rows, channels, height, out_width), dtype=groups[0].query.dtype)
    # Padding rows are genuine full-width (empty) images, so `widths` stays constant under a fixed geometry
    widths = torch.full((num_rows,), out_width, dtype=torch.int32)

    offsets = [0]  # Offset to the start of each new group in the support region of the batch
    for group, length in zip(groups, support_lengths, strict=True):
        next_offset = offsets[-1] + length
        images[offsets[-1] : next_offset, :, :, : group.width] = group.flat_support()
        widths[offsets[-1] : next_offset] = group.width
        offsets.append(next_offset)
    for offset, group in enumerate(groups, start=total_support):
        images[offset, :, :, : group.width] = group.query
        widths[offset] = group.width

    # Unpadded, labels cover only the support region; padded, they run the full height of the batch
    labels = torch.cat([group.flat_labels() for group in groups], dim=0).to(torch.long)
    if pad and num_rows > labels.shape[0]:
        labels = F.pad(labels, (0, num_rows - labels.shape[0]), value=padding_value)

    return PackedBatch(
        images=images,
        labels=labels,
        offsets=torch.tensor(offsets, dtype=torch.long),
        widths=widths,
        genotype_label=torch.tensor(
            [padding_value if group.genotype_label is None else group.genotype_label for group in groups],
            dtype=torch.long,
        ),
        num_genotypes=torch.tensor([group.num_genotypes for group in groups], dtype=torch.int32),
        meta=[group.meta for group in groups],
    )


def _pack_and_pad_images(
    data: Iterable[tuple],
    *,
    batch_size=256,
    image_transform=None,
    pad=False,
    padding_value=LABEL_PADDING_VALUE,
    width_bucket=None,
) -> Generator[PackedBatch, None, None]:
    """Packs and pads real and support images into batches of <= batch_size images.

    Greedy and sequential with no lookahead. See `select_packs` for the buffered alternative used by
    the energon loader.

    Args:
        data (Iterable[tuple]): Iterable of (query, support, label_rank, genotype_label, *meta) tuples,
            where support is replicates-major (R, G, C, H, W).
        batch_size (int, optional): Maximum number of images in the batch. Defaults to 256.
        image_transform (optional): Transform applied to the packed images, e.g. normalization. Applied to
            the whole batch at once, so padding rows are transformed as the empty images they are. Defaults to None.
        pad (bool, optional): Pad batches to maximum size. Images are zero padded, labels with padding_value.
        padding_value (int, optional): Value to use for padding labels. Defaults to -100.
        width_bucket (int | None, optional): Width to pad images out to. Defaults to the widest group in the batch.

    Yields:
        Generator[PackedBatch, None, None]: Packed batches. Offsets record the start of each group's support images.
    """

    def finish(groups: list[ImageGroup]) -> PackedBatch:
        batch = build_packed_batch(
            groups, budget=batch_size, pad=pad, width_bucket=width_bucket, padding_value=padding_value
        )
        if image_transform is not None:
            batch.images = image_transform(batch.images)
        return batch

    groups: list[ImageGroup] = []
    num_images = 0  # Number of images in the current batch (must be <= batch_size)
    for sample in data:
        query, support, label_rank, genotype_label, *addl = sample
        group = ImageGroup(
            query=query,
            support=support,
            label_rank=label_rank,
            genotype_label=genotype_label,
            meta=(addl[0] if len(addl) == 1 else tuple(addl)) if addl else None,
        )
        cost = group.cost(width_bucket=width_bucket)

        if groups and num_images + cost > batch_size:
            # If the current sample doesn't fit, yield the current batch and reset
            final = finish(groups)
            groups = []
            num_images = 0
            yield final

        groups.append(group)
        num_images += cost

    # If there are any remaining images, yield the final batch
    if groups:
        yield finish(groups)


pack_and_pad_images = wds.pipelinefilter(_pack_and_pad_images)


class PackedImageDataModule(BaseDataModule):
    """Data module for loading variants images as packed batches, e.g., [s_00, ..., s_0n, s_10, ..., s_1m, ..., q_0, ..., q_v]

    If pad is True, the number of images will be padded to the maximum batch size with zeros (the
    upstream "no data in this pixel" sentinel).

    Args:
        train_urls (str | None, optional): URLs for the training dataset, as webdataset '::' delimited brace URLs. Defaults to None.
        validate_urls (str | None, optional): URLs for the validation dataset, as webdataset '::' delimited brace URLs. Defaults to None.
        test_urls (str | None, optional): URLs for the test dataset, as webdataset '::' delimited brace URLs. Defaults to None.
        predict_urls (str | None, optional): URLs for the prediction dataset, as webdataset '::' delimited brace URLs. Defaults to None.
        batch_size (int, optional): Maximum number of images in the batch. Defaults to 256.
        num_workers (int, optional): Number of workers in torch.DataLoader. Defaults to 1.
        shuffle_size (int, optional): Size of the shuffle buffer for training loaders. Defaults to 1000.
        pad (bool, optional): Pad batches to maximum size. Defaults to False.
        legacy_axis_order (bool, optional): Read genotypes-major (G, R, H, W, C) shards written before the
            replicates-major convention, transposing them on read. Defaults to False.
        resampled (bool, optional): Resample to make dataset "infinite". Defaults to False.
        epoch_batches (int | None, optional): Number of batches per epoch. Required if resampled is True. Defaults to None.
    """

    def __init__(
        self,
        *,
        pad=False,
        resampled: bool = False,
        epoch_batches: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if resampled and epoch_batches is None:
            raise ValueError("epoch_batches must be specified if resampled is True")
        self.save_hyperparameters({"pad": pad, "resampled": resampled, "epoch_batches": epoch_batches})  # Save hyperparameters specific to the derived class

    def make_loader(self, urls, mode):
        dataset = self._make_base_dataset(urls, mode=mode, resampled=self.hparams.resampled).compose(
            pack_and_pad_images(
                batch_size=self.hparams.batch_size, image_transform=self.transforms, pad=self.hparams.pad
            )
        )

        # Since the data is pre-batched, we set batch_size to None and don't further shuffle (to reduce memory usage).
        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if self.hparams.epoch_batches is not None:
            # Set length of otherwise unknown/infinite dataset
            loader = loader.with_epoch(self.hparams.epoch_batches)
        return loader

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # PackedBatch.to only moves its tensor fields, leaving `meta` (region strings) on the host
        return batch.to(device)


def _mask_images(
    data: Iterable[tuple], *, image_transform=torch.nn.Identity, patch_size=(16, 16), mask_fraction=0.5
) -> Generator[tuple, None, None]:
    for sample in data:
        query, support, *_ = sample
        num_replicates, num_genotypes, *image_size = support.shape

        # Total number of patches
        num_patches = (image_size[1] // patch_size[0]) * (image_size[2] // patch_size[1])

        # Define indices to mask
        num_masked = int(mask_fraction * num_patches)
        selected_indices = torch.rand(1 + num_genotypes * num_replicates, num_patches).argsort(dim=1)[:, :num_masked]

        bool_mask = torch.zeros(1 + num_genotypes * num_replicates, num_patches, dtype=torch.bool)
        bool_mask[torch.arange(bool_mask.size(0)).unsqueeze(1), selected_indices] = True

        yield (
            torch.cat(
                [
                    image_transform(query).unsqueeze(0),
                    image_transform(support.reshape(-1, *image_size)),
                ]
            ),
            bool_mask,
        )


mask_images = wds.pipelinefilter(_mask_images)


class MaskedImageDataModule(BaseDataModule):
    """Data module for loading shuffled real and support images as batches with (b, num_patches[0] * num_patches[1]) boolean masks.

    Args:
        train_urls (str | None, optional): URLs for the training dataset, as webdataset '::' delimited brace URLs. Defaults to None.
        validate_urls (str | None, optional): URLs for the validation dataset, as webdataset '::' delimited brace URLs. Defaults to None.
        test_urls (str | None, optional): URLs for the test dataset, as webdataset '::' delimited brace URLs. Defaults to None.
        predict_urls (str | None, optional): URLs for the prediction dataset, as webdataset '::' delimited brace URLs. Defaults to None.
        batch_size (int, optional): Maximum number of images in the batch. Defaults to 256.
        num_workers (int, optional): Number of workers in torch.DataLoader. Defaults to 1.
        shuffle_size (int, optional): Size of the shuffle buffer for training loaders. Defaults to 1000.
        patch_size (tuple[int, int], optional): Size of the patches to mask. Defaults to (16, 16).
        mask_fraction (float, optional): Fraction of patches to mask. Defaults to 0.5.
        legacy_axis_order (bool, optional): Read genotypes-major (G, R, H, W, C) shards written before the
            replicates-major convention, transposing them on read. Defaults to False.
        resampled (bool, optional): Resample to make dataset "infinite". Defaults to False.
        epoch_batches (int | None, optional): Number of batches per epoch. Required if resampled is True. Defaults to None.
    """
    def __init__(
        self,
        *,
        patch_size: int|tuple[int, int]|Iterable[int] = (16, 16),
        mask_fraction: float = 0.5,
        resampled: bool = False,
        epoch_batches: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if resampled and epoch_batches is None:
            raise ValueError("epoch_batches must be specified if resampled is True")
        self.save_hyperparameters({"patch_size": ntuple(patch_size, 2), "mask_fraction": mask_fraction, "resampled": resampled, "epoch_batches": epoch_batches})

    def make_loader(self, urls, mode):
        # Adapted from: https://github.com/webdataset/webdataset/blob/main/examples/out/train-resnet50-multiray-wds.ipynb

        # Combine query and support images into a single batch
        dataset = self._make_base_dataset(urls, mode=mode, resampled=self.hparams.resampled).compose(
            mask_images(
                image_transform=self.transforms,
                patch_size=self.hparams.patch_size,
                mask_fraction=self.hparams.mask_fraction,
            )
        )

        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True, # With DDP we seem to have an issue where we stall at epoch end
        ).unbatched()
        if mode == "train":
            loader = loader.shuffle(self.hparams.shuffle_size)
        loader = loader.batched(self.hparams.batch_size, partial=mode != "train")
        if self.hparams.epoch_batches is not None:
            # Set length of otherwise unknown/infinite dataset
            loader = loader.with_epoch(self.hparams.epoch_batches)
        return loader
