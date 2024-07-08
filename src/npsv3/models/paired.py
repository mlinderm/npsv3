from collections.abc import Generator, Iterable

import numpy as np
import webdataset as wds
import lightning as L
import torch
from torchvision.transforms import v2 as transforms

def transform_images(images: np.ndarray) -> torch.Tensor:
    """Preprocess images from numpy array to torch tensor

    Args:
        images (np.ndarray): *HWC numpy array.

    Returns:
        torch.Tensor: Normalized *CHW torch tensor.
    """
    return transforms.functional.to_dtype(
        torch.from_numpy(images).movedim(-1, -3).contiguous(),
        torch.float32,
        scale=True,
    )


def _split_and_pad_support(data: Iterable[tuple], max_genotypes=6, padding_value=0) -> Generator[tuple, None, None]:
    """Split support images into groups of at most `max_genotypes` images, padding with `padding_value`.

    Transforms support images from GRCHW with variable size G to GCHW with fixed
    and padded G. A positive support image is guaranteed to be present in each group.

    Args:
        data (Iterable[tuple]): Iterable of (query, support, label) tuples.
        max_genotypes (int, optional): Maximum genotypes in output groups. Defaults to 6.
        padding_value (int, optional): Padding value. Defaults to 0.

    Yields:
        Generator[tuple, None, None]: (query, support, num_support, label) tuples.
    """
    for sample in data:
        query, support, label = sample
        genotypes, replicates, *image_size = support.shape
        assert label < genotypes and len(image_size) == 3, "Unexpected data shape"

        i = 0
        while i < genotypes:
            # Make sure there is a positive support image in each yielded example
            indices = list(range(i, min(i + max_genotypes, genotypes)))
            if i <= label < i + max_genotypes:
                # positive support image is already present in this group
                group_label = label - i
                i += len(indices)
            else:
                # Swap positive support image into group
                group_label = len(indices) - 1
                i, indices[group_label] = indices[group_label], label

            # yield a separate example for each replicate
            num_support = len(indices)
            padding = (0, 0) * len(image_size) + (0, max_genotypes - num_support)
            for j in range(replicates):
                yield query, torch.nn.functional.pad(
                    support[indices, j],
                    padding,
                    mode='constant',
                    value=padding_value,
                ), num_support, group_label


split_and_pad_support = wds.pipelinefilter(_split_and_pad_support)

class GroupedImageDataModule(L.LightningDataModule):
    def __init__(self, cfg, urls, batch_size=1, **kw):
        super().__init__()

        self.cfg = cfg
        self.training_urls = urls
        self.batch_size = batch_size

    def make_loader(self, urls, mode="train"):
        # Adapted from: https://github.com/webdataset/webdataset/blob/main/examples/out/train-resnet50-multiray-wds.ipynb

        if mode == "train":
            shuffle = 5000

        dataset = (
            wds.WebDataset(urls)
            .shuffle(shuffle)
            .decode()
            .to_tuple("image.npy.gz", "sim.images.npy.gz", "label.cls")
            .map_tuple(transform_images, transform_images, wds.utils.identity)
            .compose(split_and_pad_support(max_genotypes=6, padding_value=0))
            .batched(self.batch_size, partial=False)
        )

        # We unbatch, shuffle, and rebatch to mix samples from different workers as shown in webdataset examples
        loader = (
            wds.WebLoader(
                dataset,
                batch_size=None,
                num_workers=self.cfg.threads,
            )
            .unbatched()
            .shuffle(shuffle)
            .batched(self.batch_size)
        )

        return loader

    def train_dataloader(self):
        return self.make_loader(self.training_urls, mode="train")
