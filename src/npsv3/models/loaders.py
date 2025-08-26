from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from itertools import repeat
from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from torch.utils import data
from torchvision.transforms import v2 as transforms


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


def _combine_values(values: tuple) -> tuple:
    if isinstance(values[0], int | float):
        return torch.tensor(values)
    if isinstance(values[0], torch.Tensor):
        return torch.stack(values)
    if isinstance(values[0], np.ndarray):
        return torch.from_numpy(np.stack(values))
    return list(values)


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
        def to_tuple(data):
            query_image = to_tensor(data["image.npy.gz"])
            support_images = to_tensor(data["sim.images.npy.gz"])
            # Synthesize ranked labels from the integer label if not present
            label = data["label.cls"]
            if (label_rank := data.get("label.rank.npy")) is None:
                label_rank = F.one_hot(torch.tensor(label, dtype=torch.long), support_images.size(0))
                if label > 0:
                    # The same "presence" for non-reference genotypes, i.e., non-reference concordant, are considered "third-rank" positives
                    label_rank[(label_rank == 0) & (torch.arange(support_images.size(0)) > 0)] = 3
            else:
                label_rank = torch.from_numpy(label_rank)
            return query_image, support_images, label_rank, data.get("region.txt", data["__key__"])

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


def _pack_image_batch(
    query_images,
    support_images,
    labels,
    batch_size,
    addl_fields=None,
    image_transform=torch.nn.Identity,
    padding_value=-100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform lists of images and labels into a single batch tensor, padding labels with padding_value if needed"""
    num_variants = len(query_images)
    assert num_variants == len(support_images) == len(labels), "Unexpected number of images or labels for variant"

    total_num_support = sum(img.shape[0] for img in support_images)
    assert total_num_support == sum(label.shape[0] for label in labels), "Unexpected number of labels for variant"
    assert total_num_support + num_variants <= batch_size, "Total number of images exceeds batch size"

    # Manually cat images while transforming and scaling. If batch_size is larger than the number of images,
    # the images will be padded with random values
    images_batch = torch.empty((batch_size, *query_images[0].shape), dtype=torch.float32)
    offsets = [0]  # Offset to the start of each new variant group in the batch
    for support in support_images:
        next_offset = offsets[-1] + support.shape[0]
        images_batch[offsets[-1] : next_offset, ...] = image_transform(support)
        offsets.append(next_offset)
    for offset, query in enumerate(query_images, start=offsets[-1]):
        images_batch[offset, ...] = image_transform(query)

    labels_batch = torch.cat(labels, dim=0)
    if batch_size > offsets[-1] + num_variants:
        labels_batch = F.pad(labels_batch, (0, batch_size - labels_batch.shape[0]), value=padding_value)

    # Transform list of tuples into a tuple of batched values (combining tensors, etc.)
    addl_fields = tuple(_combine_values(field) for field in zip(*addl_fields, strict=True)) if addl_fields else ()

    return (images_batch, labels_batch, torch.tensor(offsets, dtype=torch.long), *addl_fields)


def _pack_and_pad_images(
    data: Iterable[tuple], *, batch_size=256, image_transform=torch.nn.Identity, pad=False, padding_value=-100
) -> Generator[tuple, None, None]:
    """Packs and pads real and support images into a batch <= batch_size.

    Args:
        data (Iterable[tuple]): Iterable of (query, support, label) tuples.
        batch_size (int, optional): Maximum number of images in the batch. Defaults to 256.
        pad (bool, optional): Pad batches to maximum size. Images are randomly padded, labels are padded with padding_value.
        padding_values (int, optional): Value to use for padding labels. Defaults to -100.

    Yields:
        Generator[tuple, None, None]: (images, labels, offset) tuples. Offsets record start of each variants support images
    """
    query_images = []
    support_images = []
    labels = []
    addl_fields = []

    num_images = 0  # Number of images in the current batch (must be <= batch_size)
    for sample in data:
        query, support, label, *addl = sample
        num_genotypes, num_replicates, *image_size = support.shape

        remaining_space = batch_size - num_images
        if num_genotypes * num_replicates + 1 > remaining_space:
            # If the current sample doesn't fit, yield the current batch.
            final = _pack_image_batch(
                query_images,
                support_images,
                labels,
                batch_size if pad else num_images,
                addl_fields=addl_fields,
                image_transform=image_transform,
                padding_value=padding_value,
            )

            # Reset for the next batch
            query_images.clear()
            support_images.clear()
            labels.clear()
            addl_fields.clear()
            num_images = 0

            yield final

        # Append the "query" (real image) to the images list as a CHW tensor
        query_images.append(query)

        # Append the "support" (simulated) images to the images list as a (G*R)CHW tensor. Expand out the ranked labels
        # based on the number of replicates, if num_genotypes=3 and num_replicates=2, the ranked label is [0, 1, 3], then
        # the labels would be [0, 0, 1, 1, 3, 3].
        support_images.append(support.reshape(-1, *image_size))
        labels.append(torch.repeat_interleave(label, num_replicates))

        addl_fields.append(addl)

        num_images += num_genotypes * num_replicates + 1

    # If there are any remaining images, yield the final batch
    yield _pack_image_batch(
        query_images,
        support_images,
        labels,
        batch_size if pad else num_images,
        addl_fields=addl_fields,
        image_transform=image_transform,
        padding_value=padding_value,
    )


pack_and_pad_images = wds.pipelinefilter(_pack_and_pad_images)


class PackedImageDataModule(BaseDataModule):
    """Data module for loading variants images as packed batches, e.g., [s_00, ..., s_0n, s_10, ..., s_1m, ..., q_0, ..., q_v]

    If pad is True, the number of images will be padded to the maximum batch size with random data.

    Args:
        train_urls (str | None, optional): URLs for the training dataset, as webdataset '::' delimited brace URLs. Defaults to None.
        validate_urls (str | None, optional): URLs for the validation dataset, as webdataset '::' delimited brace URLs. Defaults to None.
        test_urls (str | None, optional): URLs for the test dataset, as webdataset '::' delimited brace URLs. Defaults to None.
        predict_urls (str | None, optional): URLs for the prediction dataset, as webdataset '::' delimited brace URLs. Defaults to None.
        batch_size (int, optional): Maximum number of images in the batch. Defaults to 256.
        num_workers (int, optional): Number of workers in torch.DataLoader. Defaults to 1.
        shuffle_size (int, optional): Size of the shuffle buffer for training loaders. Defaults to 1000.
        pad (bool, optional): Pad batches to maximum size. Defaults to False.
    """

    def __init__(
        self,
        *,
        pad=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters({"pad": pad})  # Save hyperparameters specific to the derived class

    def make_loader(self, urls, mode):
        dataset = self._make_base_dataset(urls, mode=mode).compose(
            pack_and_pad_images(
                batch_size=self.hparams.batch_size, image_transform=self.transforms, pad=self.hparams.pad
            )
        )

        # Since the data is pre-batched, we set batch_size to None and don't further shuffle (to reduce memory usage).
        return wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # We currently only need to transfer the images, labels, and offsets to the device, e.g., GPU
        images, labels, offsets, *addl_fields = batch
        return (images.to(device), labels.to(device), offsets.to(device), *addl_fields)


def _mask_images(
    data: Iterable[tuple], *, image_transform=torch.nn.Identity, patch_size=(16, 16), mask_fraction=0.5
) -> Generator[tuple, None, None]:
    for sample in data:
        query, support, *_ = sample
        num_genotypes, num_replicates, *image_size = support.shape

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
        dataset = self._make_base_dataset(urls, mode=mode, resampled=True).compose(
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
