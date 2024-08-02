from collections.abc import Generator, Iterable

import numpy as np
import webdataset as wds
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.transforms import v2 as transforms
import torchvision.models as models
from torchmetrics.classification.accuracy import Accuracy
import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf

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
            elif len(indices) < max_genotypes:
                # Space in current group to append positive
                i += len(indices)
                indices.append(label)
                group_label = len(indices) - 1
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

class EmptyDataset(data.Dataset):
    def __init__(self):
        super(EmptyDataset).__init__()
    
    def __iter__(self):
        return iter(())


class GroupedImageDataModule(L.LightningDataModule):
    def __init__(self, training_urls, validation_urls=None, batch_size=16, num_workers=1, max_group_size=6):
        super().__init__()

        self.training_urls = training_urls
        self.validation_urls = validation_urls
        self.save_hyperparameters(ignore=["training_urls", "validation_urls"])

    def make_loader(self, urls, mode="train"):
        # Adapted from: https://github.com/webdataset/webdataset/blob/main/examples/out/train-resnet50-multiray-wds.ipynb

        dataset = wds.WebDataset(urls, shardshuffle=100 if mode == "train" else False)
        if mode == "train":
            dataset = dataset.shuffle(5000)
        dataset = (
            dataset
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
                shuffle=False,
                num_workers=self.hparams.num_workers,
            )
            .unbatched()
        )
        if mode == "train":
            loader = loader.shuffle(5000)
        loader = loader.batched(self.hparams.batch_size)

        return loader

    def train_dataloader(self):
        return self.make_loader(self.training_urls, mode="train")

    def val_dataloader(self):
        # Make sure to return a valid dataloader, even if validation data is not available since
        # Lightning still calls this method with zero validation steps
        if self.validation_urls:
            return self.make_loader(self.validation_urls or [], mode="val")
        else:
            return data.DataLoader(EmptyDataset())


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

class EuclideanDistanceMetric(nn.Module):
    def __init__(self):
        super(EuclideanDistanceMetric, self).__init__()
        self.batched_distance = torch.vmap(torch.cdist)

    def forward(self, query_embeddings, support_embeddings):
        return torch.squeeze(
            self.batched_distance(
                torch.unsqueeze(torch.nn.functional.normalize(query_embeddings, p=2, dim=-1), 1),
                torch.nn.functional.normalize(support_embeddings, p=2, dim=-1),
            ),
            dim=1,
        )

class DotProductSimilarityMetric(nn.Module):
    def __init__(self):
        super(DotProductSimilarityMetric, self).__init__()
        self.batched_dot = torch.vmap(torch.mv)

    def forward(self, query_embeddings, support_embeddings):
        return self.batched_dot(support_embeddings, query_embeddings)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distances: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, query_embeddings: torch.Tensor, support_embeddings: torch.Tensor):
        label_pair = nn.functional.one_hot(label, distances.shape[1])
        loss = label_pair * torch.square(distances) + (1.0 - label_pair) * torch.square(
            torch.clamp(self.margin - distances, min=0)
        )
        return torch.mean(loss[mask])

class NPairsLoss(nn.Module):
    def __init__(self, l2_reg=0.002):
        super(NPairsLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, metric, label, mask, query_embeddings, support_embeddings):
        masked_metric = torch.where(mask, metric, metric.new_full([], -torch.inf))
        reg = 0.25*self.l2_reg*torch.mean(torch.torch.square(query_embeddings).sum(dim=1) + torch.square(support_embeddings).sum(dim=(1,2)))
        return nn.functional.cross_entropy(masked_metric, label, reduction="mean") + reg

class MinimizingPredictor(nn.Module):
    def __init__(self):
        super(MinimizingPredictor, self).__init__()

    def forward(self, metric, mask):
        masked_metric = torch.where(mask, metric, metric.new_full([], torch.inf))
        return torch.argmin(metric, dim=1)

class MaximizingPredictor(nn.Module):
    def __init__(self):
        super(MaximizingPredictor, self).__init__()

    def forward(self, metric, mask):
        masked_metric = torch.where(mask, metric, metric.new_full([], -torch.inf))
        return torch.argmax(metric, dim=1)

class GroupedVariant(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        metric: nn.Module,
        loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        predictor: nn.Module,
        max_group_size=6,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "metric"])

        self.encoder = encoder
        self.metric = metric
        self.loss = loss
        self.predictor = predictor

        self.train_acc = Accuracy(task="multiclass", num_classes=max_group_size)
        self.val_acc = Accuracy(task="multiclass", num_classes=max_group_size)

        self.example_input_array = (torch.zeros(1, 8, 100, 300), torch.zeros(1, max_group_size, 8, 100, 300))

    def on_train_start(self) -> None:
        # Reset validation metrics at the start of training to avoid effects of sanity batches
        self.val_acc.reset()

    def forward(self, query, support):
        query_embeddings = self.encoder(query)

        # https://github.com/pytorch/pytorch/issues/1927#issuecomment-1245392571
        support = support.transpose(0, 1)
        support_embeddings = torch.stack([self.encoder(s) for s in support], dim=0)
        support_embeddings = support_embeddings.transpose(0, 1)

        metric = self.metric(query_embeddings, support_embeddings)
        
        return (metric, query_embeddings, support_embeddings)

    def _model_step(self, batch, batch_idx):
        query, support, num_support, label = batch
        metric, query_embeddings, support_embeddings = self(query, support)

        # Create a mask for the valid support images in each group by filling ones out to the
        # last valid support image (via "exclusive cumsum")
        mask = torch.zeros(metric.shape, dtype=torch.long, device=metric.device)
        mask[(torch.arange(metric.shape[0]), num_support - 1)] = 1
        mask = (1 - (mask.cumsum(dim=-1) - mask)).to(torch.bool)

        loss = self.loss(metric, label, mask, query_embeddings, support_embeddings)
        preds = self.predictor(metric, mask)
        
        return (loss, preds, label)

    def training_step(self, batch, batch_idx):
        loss, preds, label = self._model_step(batch, batch_idx)
       
        self.log("train_loss", loss, prog_bar=True)
        self.train_acc(preds, label)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, label = self._model_step(batch, batch_idx)
       
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_acc(preds, label)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return { "optimizer": optimizer }


def train(cfg, output_dir=None, **kw_args):
    dm = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    # Overwrite existing checkpoints, instead of creating new versions
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath=output_dir, enable_version_counter=False)
    
    if cfg.data.validation_urls:
        limit_val_batches = OmegaConf.select(cfg, "data.limit_val_batches", default=1.0)
        num_sanity_val_steps = OmegaConf.select(cfg, "data.num_sanity_val_steps", default=2)
    else:
        # Skip validation if no validation data provided
        limit_val_batches = num_sanity_val_steps = 0

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[checkpoint_callback], limit_val_batches=limit_val_batches, num_sanity_val_steps=num_sanity_val_steps, **kw_args)
    
    # TODO: Check if we have reached the final, if not, continue training by setting ckpt_path
    # https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#resume-training-state
    trainer.fit(model=model, datamodule=dm)

    return checkpoint_callback.best_model_path
