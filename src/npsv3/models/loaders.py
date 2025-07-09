from collections.abc import Generator, Iterable

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from torch.utils import data
from torchvision.transforms import v2 as transforms


class ToTensor(torch.nn.Module):
    """Convert *HWC numpy array to a *CHW tensor"""

    def forward(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img).movedim(-1, -3).contiguous()


def transform_images(images: torch.Tensor) -> torch.Tensor:
    """Normalize uint8 *CHW image tensors"""
    return transforms.functional.to_dtype(images, torch.float32, scale=True)


def _pack_image_batch(
    images, labels, batch_size, transform_images=torch.nn.Identity, padding_value=-100
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform list of images and labels into a single batch tensor, padding labels with padding_value if needed"""
    assert len(images) // 2 == len(labels), "Unexpected number labels for number of images"
    assert sum(img.shape[0] for img in images) <= batch_size, "Total number of images exceeds batch size"

    # Manually cat images while transforming and scaling. If batch_size is larger than the number of images,
    # the images will be padded with random values
    images_batch = torch.empty((batch_size, *images[0].shape[1:]), dtype=torch.float32)
    offsets = [0]  # Offset to the start of each new variant group in the batch
    for query, support in zip(
        *[iter(images)] * 2, strict=True
    ):  # Get groups of length 2, i.e. implementation for `batched` (only available in Python>=3.12)
        images_batch[offsets[-1], ...] = transform_images(query)
        offsets.append(offsets[-1] + 1)
        images_batch[offsets[-1] : offsets[-1] + support.shape[0], ...] = transform_images(support)
        offsets[-1] += support.shape[0]

    labels_batch = torch.cat(labels, dim=0)
    if batch_size > offsets[-1]:
        labels_batch = F.pad(labels_batch, (0, batch_size - labels_batch.shape[0]), value=padding_value)

    return (images_batch, labels_batch, torch.tensor(offsets, dtype=torch.long))


def _pack_and_pad_images(
    data: Iterable[tuple], *, batch_size=256, transform_images=torch.nn.Identity, pad=False, padding_value=-100
) -> Generator[tuple, None, None]:
    """Packs and pads real and support images into a batch <= batch_size.

    Args:
        data (Iterable[tuple]): Iterable of (query, support, label) tuples.
        batch_size (int, optional): Maximum number of images in the batch. Defaults to 256.
        pad (bool, optional): Pad batches to maximum size. Images are randomly padded, labels are padded with padding_value.
        padding_values (int, optional): Value to use for padding labels. Defaults to -100.

    Yields:
        Generator[tuple, None, None]: (images, labels, offset) tuples. Offsets record start of each new variant in the batch.
    """
    images = []
    labels = []

    num_images = 0  # Number of images in the current batch (must be <= batch_size)
    for sample in data:
        query, support, label, *_ = sample
        num_genotypes, num_replicates, *image_size = support.shape

        remaining_space = batch_size - num_images
        if num_genotypes * num_replicates + 1 > remaining_space:
            # If the current sample doesn't fit, yield the current batch.
            final = _pack_image_batch(
                images,
                labels,
                batch_size if pad else num_images,
                transform_images=transform_images,
                padding_value=padding_value,
            )

            # Reset for the next batch
            images.clear()
            labels.clear()
            num_images = 0

            yield final

        # Append the "query" (real image) to the images list as a 1CHW tensor
        images.append(torch.unsqueeze(query, 0))

        # Append the "support" (simulated) images to the images list as a (G*R)CHW tensor. Generate one hot labels encoding correct
        # replicates, e.g., if num_genotypes=3 and num_replicates=2, and the label is 1, generate  [0, 0, 1, 1, 0, 0]
        images.append(support.reshape(-1, *image_size))
        labels.append(
            torch.repeat_interleave(F.one_hot(torch.tensor(label, dtype=torch.long), num_genotypes), num_replicates)
        )

        num_images += num_genotypes * num_replicates + 1

    # If there are any remaining images, yield the final batch
    yield _pack_image_batch(
        images,
        labels,
        batch_size if pad else num_images,
        transform_images=transform_images,
        padding_value=padding_value,
    )


pack_and_pad_images = wds.pipelinefilter(_pack_and_pad_images)


class EmptyDataset(data.Dataset):
    def __init__(self):
        super(EmptyDataset).__init__()

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


class PackedImageDataModule(L.LightningDataModule):
    """Data module for loading variants images as packed batches, e.g., [q_0, s_00, ..., s_0n, q_1, s_10, ..., s_1m, ...]

    If pad is True, the number of images will be padded to the maximum batch size with random data.

    Args:
        train_urls (str | None, optional): URLs for the training dataset, as webdataset '::' delimited brace URLs. Defaults to None.
        validate_urls (str | None, optional): URLs for the validation dataset, as webdataset '::' delimited brace URLs. Defaults to None.
        predict_urls (str | None, optional): URLs for the prediction dataset, as webdataset '::' delimited brace URLs. Defaults to None.
        test_urls (str | None, optional): URLs for the test dataset, as webdataset '::' delimited brace URLs. Defaults to None.
        batch_size (int, optional): Maximum number of images in the batch. Defaults to 256.
        pad (bool, optional): Pad batches to maximum size. Defaults to False.
        num_workers (int, optional): Number of workers in torch.DataLoader. Defaults to 1.
        shuffle_size (int, optional): Size of the shuffle buffer for training loaders. Defaults to 1000.
    """
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

    def make_loader(self, urls, mode="train"):
        # Adapted from: https://github.com/webdataset/webdataset/blob/main/examples/out/train-resnet50-multiray-wds.ipynb

        dataset = wds.WebDataset(urls, shardshuffle=100 if mode == "train" else False)
        if mode == "train":
            dataset = dataset.shuffle(self.hparams.shuffle_size)
        dataset = (
            dataset.decode()
            .to_tuple("image.npy.gz", "sim.images.npy.gz", "label.cls")
            .map_tuple(ToTensor(), ToTensor(), wds.utils.identity)
            .compose(
                pack_and_pad_images(
                    batch_size=self.hparams.batch_size, transform_images=self.transforms, pad=self.hparams.pad
                )
            )
        )

        # Since the data is pre-batched, we set batch_size to None. Since we have already batched, we don't further shuffle to minimize
        # memory usage.

        # def _seq_worker(_worker_id: int):
        #     # Set number of threads in dataset workers to prevent oversubscribing CPU cores
        #     torch.set_num_threads(1)

        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            #worker_init_fn=_seq_worker,
            pin_memory=torch.cuda.is_available(),
        )

        return loader

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
