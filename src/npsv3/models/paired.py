from collections.abc import Generator, Iterable

import numpy as np
import webdataset as wds
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
from torchvision.transforms import v2 as transforms
import torchvision.models as models
import hydra


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
    def __init__(self, training_urls, batch_size=16, num_workers=1, max_group_size=6):
        super().__init__()

        self.training_urls = training_urls
        self.save_hyperparameters(ignore=["training_urls"])

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
            .compose(split_and_pad_support(max_genotypes=self.hparams.max_group_size, padding_value=0))
            .batched(self.hparams.batch_size, partial=False)
        )

        # We unbatch, shuffle, and rebatch to mix samples from different workers as shown in webdataset examples
        loader = (
            wds.WebLoader(
                dataset,
                batch_size=None,
                num_workers=self.hparams.num_workers,
            )
            .unbatched()
            .shuffle(shuffle)
            .batched(self.hparams.batch_size)
        )

        return loader

    def train_dataloader(self):
        return self.make_loader(self.training_urls, mode="train")


class InceptionEncoder(nn.Module):
    def __init__(self, num_channels=8, projection_size=512):
        super(InceptionEncoder, self).__init__()
        self.inception = models.inception_v3(weights=None, aux_logits=False)

        # Replace the first layer for our number of channels
        self.inception.Conv2d_1a_3x3.conv = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

        # Replace the final layer with our projection head
        self.inception.fc = nn.Linear(self.inception.fc.in_features, projection_size, bias=False)
        self.bn = nn.BatchNorm1d(projection_size)

    def forward(self, x):
        embeddings = self.inception(x)
        projection = self.bn(embeddings)
        return projection

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distances: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor):
        loss = y_true * torch.square(distances) + (1.0 - y_true) * torch.square(
            torch.clamp(self.margin - distances, min=0)
        )
        return torch.mean(loss[mask.to(torch.bool)])

class GroupedVariant(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        optimizer: torch.optim.Optimizer,
        max_group_size=6,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["encoder"])

        self.encoder = encoder
        self.batched_distance = torch.vmap(torch.cdist)
        self.loss = ContrastiveLoss()

        self.example_input_array = (torch.zeros(1, 8, 100, 300), torch.zeros(1, 6, 8, 100, 300))

    def forward(self, query, support):
        query_projections = self.encoder(query)

        # https://github.com/pytorch/pytorch/issues/1927#issuecomment-1245392571
        support = support.transpose(0, 1)
        support_projections = torch.stack([self.encoder(s) for s in support], dim=0)
        support_projections = support_projections.transpose(0, 1)

        # Compute pairwise distances between query and support projections for each group
        distances = torch.squeeze(
            self.batched_distance(torch.unsqueeze(query_projections, 1), support_projections), dim=1
        )

        return (distances, query_projections, support_projections)

    def training_step(self, batch, batch_idx):
        query, support, num_support, label = batch
        distances, query_projections, support_projections = self(query, support)

        # Create a mask for the valid support images in each group
        mask = torch.zeros(distances.shape, dtype=torch.long, device=distances.device)
        mask[(torch.arange(distances.shape[0]), num_support - 1)] = 1
        mask = 1 - (
            mask.cumsum(dim=-1) - mask
        )  # Fill ones out to the last valid support image (via "exclusive cumsum")

        # Compute the loss for each group
        label_pair = nn.functional.one_hot(label, self.hparams.max_group_size)

        # Contrastive loss
        loss = self.loss(distances, label_pair, mask)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return { "optimizer": optimizer }


def train(cfg, output_dir=None, **kw_args):
    dm = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    # Overwrite existing checkpoints, instead of creating new versions
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath=output_dir, enable_version_counter=False)
    trainer = L.Trainer(callbacks=[checkpoint_callback], **kw_args)
    
    # TODO: Check if we have reached the final, if not, continue training by setting ckpt_path
    # https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#resume-training-state
    trainer.fit(model=model, datamodule=dm)

    return checkpoint_callback.best_model_path
