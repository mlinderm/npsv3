import hydra
import lightning as L
import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F
from torchvision import models


class InceptionEncoder(nn.Module):
    def __init__(self, num_channels=8, projection_size=512):
        super().__init__()
        self.num_channels = num_channels
        self.projection_size = projection_size

        self.inception = models.inception_v3(weights=None, aux_logits=False)

        # Replace the first layer for our number of channels
        self.inception.Conv2d_1a_3x3.conv = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

        # Replace the final layer with our projection head
        self.inception.fc = nn.Linear(self.inception.fc.in_features, projection_size, bias=False)
        self.bn = nn.BatchNorm1d(projection_size)

    def forward(self, x):
        embeddings = self.inception(x)
        return self.bn(embeddings)


class EuclideanDistanceMetric(nn.Module):
    """
    Computes the L2 (Euclidean) distance between normalized query and support embeddings for a single variant.

    Embeddings are L2-normalized before distance computation to ensure scale invariance.

    Inputs:
        query_embedding (torch.Tensor): Tensor of shape (|support|, embedding_dim) representing query embeddings.
        support_embeddings (torch.Tensor): Tensor of shape (|support|, embedding_dim) representing support embeddings.

    Returns:
        Tensor of shape (|support|,): Euclidean distances between query and support pairs.
    """

    def __init__(self):
        super().__init__()

    def forward(self, query_embedding, support_embeddings):
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=-1)
        support_embeddings = torch.nn.functional.normalize(support_embeddings, p=2, dim=-1)

        # Compute distance between corresponding rows
        # return torch.cdist(query_embedding, support_embeddings)
        return F.pairwise_distance(query_embedding, support_embeddings)


class ContrastiveLoss(nn.Module):
    """
    Computes a contrastive loss based on distances between query and support embeddings

    Args:
        margin (float): The margin enforced between dissimilar pairs. Default is 1.0.

    Inputs:
        metric (torch.Tensor): A 2D nested tensor of shape(|variants|, j1=|support|) with distances between query and support embeddings
        target (torch.Tensor): A 2D nested tensor of shape(|variants|, j1=|support|) with 1 for support embeddings with the correct genotypes

    Returns:
        torch.Tensor: A scalar tensor representing the mean contrastive loss across all image pairs
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, metric: torch.Tensor, target: torch.Tensor):
        loss = target * torch.square(metric) + (1.0 - target) * torch.square(torch.clamp(self.margin - metric, min=0))
        return torch.mean(loss)


class InfoNCE(nn.Module):
    """
    Implements the InfoNCE loss for contrastive learning using precomputed similarity scores

    This formulation encourages positive pairs (query-support with label=1) to have higher
    similarity scores than negative pairs (label=0) within each variant group. It applies
    temperature scaling and uses log-ratio separation between positives and all negatives.

    Args:
        temperature (float, optional): Scaling factor applied to similarity scores before exponentiation. Default is 0.07.

    Inputs:
        metric (torch.Tensor): A 2D nested tensor of shape(|variants|, j1=|support|) with distances between query and support embeddings.
        target (torch.Tensor): A 2D nested tensor of shape(|variants|, j1=|support|) with 1 for support embeddings with the correct genotypes

    Returns:
        torch.Tensor: Scalar tensor representing the average InfoNCE loss across variant groups.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, metric: torch.Tensor, target: torch.Tensor):
        metric = metric / self.temperature  # Scale the similarity scores by the temperature

        loss = torch.tensor([0.0], dtype=metric.dtype, device=metric.device)
        for variant_metric, variant_target in zip(metric.unbind(), target.unbind(), strict=True):
            negative_mask = variant_target == 0
            negatives = variant_metric[negative_mask]
            positives = variant_metric[torch.logical_not(negative_mask)]

            # Compute log(softmax) using log-sum-exp trick (similar to log_softmax) to ensure numerical stability
            max_value = torch.max(variant_metric)
            numerator = positives - max_value
            # TODO: Normalize by the number of positives (i.e., mean vs. sum)? Some versions of infoNCE do, but not all
            loss += torch.sum(
                numerator - torch.log(torch.exp(numerator) + torch.sum(torch.exp(negatives - max_value)) + 1e-8)
            )

        return -loss / metric.size(0)  # Average loss across all variants


class MinimizingPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, metric):
        return torch.argmin(metric, dim=1)


class MaximizingPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, metric):
        return torch.argmax(metric, dim=1)


class GenotypingAccuracy(torchmetrics.Metric):
    """
    Compute genotyping accuracy(ies), allowing for multiple correct support images per variant.

    Predicted genotype labeled as 1 is considered correct, while incorrect predictions of 0 do not contribute to the calculated accuracy.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update accuracy measures with |variants|, dense integer prediction tensor and |variants|,|support| nested one-hot target tensor"""
        padded_target = torch.nested.to_padded_tensor(target, 0)
        onehot_preds = torch.nn.functional.one_hot(preds, num_classes=padded_target.shape[1])

        # TODO: Also collect non-reference concordance (need to know which are reference genotypes)
        self.correct += torch.sum(torch.any(padded_target & onehot_preds, dim=1))
        self.total += target.shape[0]

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total


class PackedVariant(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        metric: nn.Module,
        loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        predictor: nn.Module,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "metric", "loss", "predictor"])

        self.encoder = encoder
        self.metric = metric
        self.loss = loss
        self.predictor = predictor

        self.train_acc = GenotypingAccuracy()
        self.val_acc = GenotypingAccuracy()
        self.test_acc = GenotypingAccuracy()

    def on_train_start(self) -> None:
        # Reset validation metrics at the start of training to avoid effects of sanity batches
        self.val_acc.reset()

    def forward(self, batch: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        """Compute (metric, preds, labels, *metadata) as nested tensors from a batch of packed images, labels, and offsets."""
        images, labels, offsets, *metadata = batch
        image_embeddings = self.encoder(images)

        # TODO: Consider padding query and support embeddings if images are padded to enable consistent tensor sizes
        support_lengths = torch.diff(offsets)
        max_support = torch.max(support_lengths)
        # Extract query embeddings from after the supports and expand to compute the pairwise metric
        query_embeddings = torch.repeat_interleave(
            image_embeddings[offsets[-1]:offsets[-1]+support_lengths.size(0)],
            support_lengths,
            dim=0,
            output_size=offsets[-1],
        )
        support_embeddings = image_embeddings[: offsets[-1]]

        metrics = torch.nested.nested_tensor_from_jagged(
            self.metric(query_embeddings, support_embeddings), offsets=offsets, max_seqlen=max_support
        )

        preds = self.predictor(metrics)

        # Construct labels as nested tensor with offsets shared with metrics so the ragged dimension is recognized as matching.
        # Set the max_seqlen, since known, so that the nested tensor won't be padded more than needed.
        labels_nt = torch.nested.nested_tensor_from_jagged(labels, offsets=offsets, max_seqlen=max_support)

        return (metrics, preds, labels_nt, *metadata)

    # def forward(self, batch: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
    #     """Compute (metric, preds, labels) as nested tensors from a batch of packed images, labels, and offsets."""
    #     images, labels, offsets = batch
    #     images_embeddings = self.encoder(images)

    #     variant_lengths = torch.diff(offsets)
    #     query_embeddings = images_embeddings[offsets[:-1]].unsqueeze(1)
    #     support_embeddings_nt = torch.nested.nested_tensor_from_jagged(images_embeddings, offsets=offsets, max_seqlen=torch.max(variant_lengths))
    #     support_embeddings = support_embeddings_nt.to_padded_tensor(0.0)

    #     metrics = self.metric(query_embeddings, support_embeddings).squeeze(1)
    #     metrics_nt = torch.nested.as_nested_tensor([metrics[i,1:variant_lengths[i]] for i in range(len(offsets) - 1)], layout=torch.jagged)

    #     preds = self.predictor(metrics_nt)

    #     # Construct labels as nested tensor with offsets shared with metrics so the ragged dimension is recognized as matching.
    #     # Set the max_seqlen, since known, so that the nested tensor won't be padded more than needed.
    #     labels_nt = torch.nested.nested_tensor_from_jagged(
    #         labels, offsets=metrics_nt.offsets(), max_seqlen=torch.max(variant_lengths) - 1
    #     )

    #     return (metrics_nt, preds, labels_nt)

    def _model_step(self, batch, batch_idx):  # noqa: ARG002
        metric, preds, labels, *_ = self(batch)
        loss = self.loss(metric, labels)
        return (loss, preds, labels)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, label = self._model_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        self.train_acc(preds, label)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, label = self._model_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_acc(preds, label)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, label = self._model_step(batch, batch_idx)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_acc(preds, label)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return {"optimizer": optimizer}


def predict(cfg, **kw_args):
    dm = hydra.utils.instantiate(cfg.data)

    model_cls = hydra.utils.get_class(cfg.model._target_)
    model = model_cls.load_from_checkpoint(
        cfg.model.checkpoint,
        encoder=hydra.utils.instantiate(cfg.model.encoder),
        metric=hydra.utils.instantiate(cfg.model.metric),
        loss=hydra.utils.instantiate(cfg.model.loss),
        predictor=hydra.utils.instantiate(cfg.model.predictor),
    )
    trainer = L.Trainer(limit_predict_batches=2)
    for prediction in trainer.predict(model, dm):
        metrics, preds, labels_nt, *metadata = prediction
        print(metrics.offsets(), metrics.values(), preds, labels_nt.values(), *metadata, sep="\n")


def test(cfg, **kw_args):
    dm = hydra.utils.instantiate(cfg.data)

    model_cls = hydra.utils.get_class(cfg.model._target_)
    # We need to instantiate any of ignored components in the model
    model = model_cls.load_from_checkpoint(
        cfg.model.checkpoint,
        encoder=hydra.utils.instantiate(cfg.model.encoder),
        metric=hydra.utils.instantiate(cfg.model.metric),
    )

    trainer = L.Trainer(**kw_args)
    trainer.test(model, dm)
