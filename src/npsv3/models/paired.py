import hydra
import lightning as L
import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.transforms import v2 as transforms

from npsv3.models.metrics import (
    GenotypingConcordance,
    GenotypingNonRefConcordance,
    GenotypingNonRefF1,
    GenotypingNonRefPrecision,
    GenotypingNonRefRecall,
)
from npsv3.models.transformer import Classifier, ViTConfig, ViTModel


class InceptionEncoder(nn.Module):
    def __init__(self, num_channels=8, projection_size=512):
        super().__init__()
        self.num_channels = num_channels
        self.projection_size = projection_size

        self.inception = models.inception_v3(init_weights=True, aux_logits=False)

        # Replace the first layer for our number of channels
        self.inception.Conv2d_1a_3x3.conv = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

        # Replace the final layer with our projection head
        self.inception.fc = nn.Linear(self.inception.fc.in_features, projection_size, bias=False)
        self.bn = nn.BatchNorm1d(projection_size)

    def forward(self, x):
        embeddings = self.inception(x)
        return self.bn(embeddings)


class ViTEncoder(nn.Module):
    def __init__(self, num_channels=8, projection_size=512, pretrained_path=None):
        super(ViTEncoder, self).__init__()
        self.num_channels = num_channels
        # self.projection_size = projection_size

        config = ViTConfig(num_channels=self.num_channels, image_size = (96, 288), )

        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'), weights_only=False)
            self.model = ViTModel(config, add_pooling_layer=False)
            print(list(checkpoint["state_dict"].keys()))
            encoder_weights = {k.removeprefix("model.vit."): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.vit.")}
            self.model.load_state_dict(encoder_weights, strict=False)
        else:
            self.model = ViTModel(config, add_pooling_layer=False)

        # self.inception.Conv2d_1a_3x3.conv = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

        # self.inception.fc = nn.Linear(self.inception.fc.in_features, projection_size, bias=False)
        # self.bn = nn.BatchNorm1d(projection_size)

    def forward(self, x):
        outputs = self.model(x)
        sequence_output = outputs[0]
        return sequence_output[:, 0, :]

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
        tuple[torch.Tensor]: A scalar tensor representing the mean contrastive loss across all image pairs and the number of pairs in batch
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, metric: torch.Tensor, target: torch.Tensor):
        target = torch.where(torch.eq(target, 1) | torch.eq(target, 2), 1, 0) # Map "first-rank" and "second-rank" positives to 1
        loss = target * torch.square(metric) + (1.0 - target) * torch.square(torch.clamp(self.margin - metric, min=0))
        return torch.mean(loss), loss.size(0)  # Average loss across all pairs

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
        target (torch.Tensor): A 2D nested tensor of shape(|variants|, j1=|support|) with 1 for support embeddings with the correct genotypes.

    Returns:
        tuple[torch.Tensor]: Scalar tensor representing the average InfoNCE loss across variant groups and the number of variants in batch
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, metric: torch.Tensor, target: torch.Tensor):
        target = torch.where(torch.eq(target, 1) | torch.eq(target, 2), 1, 0) # Map "first-rank" and "second-rank" positives to 1
        metric = metric / self.temperature  # Scale the similarity scores by the temperature

        loss = torch.tensor([0.0], dtype=metric.dtype, device=metric.device)
        for variant_metric, variant_target in zip(metric.unbind(), target.unbind(), strict=True):
            negative_mask = variant_target == 0
            negatives = variant_metric[negative_mask]
            positives = variant_metric[torch.logical_not(negative_mask)]

            # Compute log(softmax) using log-sum-exp trick (similar to log_softmax) to ensure numerical stability
            max_value = torch.max(variant_metric)
            numerator = positives - max_value
            # Normalize by the number of positives "outside the log". This is the approach described in the SupCon paper.
            # Normalizing seems to improve learning and accuracy relative to just a "sum".
            variant_loss= torch.mean(
                numerator - torch.log(torch.exp(numerator) + torch.sum(torch.exp(negatives - max_value)) + 1e-8)
            )
            loss += variant_loss

        return -loss / metric.size(0), metric.size(0)  # Average loss across all variants

    
class RINCE(nn.Module):
    """
    Implements the Ranking Info Noise Contrastive Estimation (RINCE) loss for contrastive learning using precomputed similarity scores

    Args:
        temperature (float, optional): Scaling factor applied to similarity scores before exponentiation. Default is 0.07.

    Inputs:
        metric (torch.Tensor): A 2D nested tensor of shape(|variants|, j1=|support|) with distances between query and support embeddings.
        target (torch.Tensor): A 2D nested tensor of shape(|variants|, j1=|support|) with 1 for support embeddings with the correct genotypes
        

    Returns:
        torch.Tensor: Scalar tensor representing the average RINSE loss across variant groups.
    """

    def __init__(self, temperature= [0.1, 0.225 , 0.5], max_rank=3, ignore=0.0):
        super().__init__()
        self.temperature = torch.tensor(temperature, dtype=torch.float32)
        self.max_rank = max_rank
        self.ignore = ignore  # Value to ignore in the target tensor, e.g., for padding

    def forward(self, metric: torch.Tensor, target: torch.Tensor):
        loss = torch.tensor(0.0, dtype=metric.dtype, device=metric.device)

        for variant_metric, variant_target in zip(metric.unbind(), target.unbind(), strict=True):
            negative_mask = variant_target == 0
            hard_negatives = variant_metric[negative_mask]
            variant_loss = 0 
            for rank in range(1, self.max_rank + 1): 
                #Rank 1 positives
                
                R1mask = variant_target == rank
                positivesR1 = variant_metric[R1mask]

                if rank > 1:
                    positivesR1 = positivesR1[positivesR1 >= self.ignore] #Threshold for rank 1 positives

                #Skip empty ranks
                if positivesR1.numel() == 0:
                    continue

                R2mask = variant_target > rank
                positivesR2 = variant_metric[R2mask]
                positivesR2 = positivesR2[positivesR2 >= self.ignore] #Threshold for rank 2 positives

                numerator = torch.sum(torch.exp(positivesR1/self.temperature[rank-1]))
                posR2 = torch.sum(torch.exp(positivesR2/self.temperature[rank-1]))

                denominator = numerator + posR2 + torch.sum(torch.exp(hard_negatives/self.temperature[rank-1]))

                variant_loss += -torch.log(numerator/denominator + 1e-8)
                #ranks[rank-1] += -torch.log(numerator/denominator + 1e-8)
            loss += variant_loss
            #ranks = ranks/metric.size(0)
            #loss = ranks.mean()
        return loss / metric.size(0), metric.size(0) # Average loss across all variants


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


class PackedVariant(L.LightningModule):
    ignored_hyperparameters = ("encoder", "metric", "loss", "predictor")

    def __init__(
        self,
        encoder: nn.Module,
        metric: nn.Module,
        loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        predictor: nn.Module,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=type(self).ignored_hyperparameters)

        self.encoder = encoder
        self.metric = metric
        self.loss = loss
        self.predictor = predictor

        self.train_metrics = torchmetrics.MetricCollection({
            "concordance": GenotypingConcordance(),
            "nonrefconcordance": GenotypingNonRefConcordance(),
        }, prefix="train_")
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = torchmetrics.MetricCollection({
            "concordance": GenotypingConcordance(),
            "nonrefconcordance": GenotypingNonRefConcordance(),
            "nonrefprecision": GenotypingNonRefPrecision(),
            "nonrefrecall": GenotypingNonRefRecall(),
            "nonreff1": GenotypingNonRefF1(),
        }, prefix="test_")

    def on_train_start(self) -> None:
        # Reset validation metrics at the start of training to avoid effects of sanity batches
        self.val_metrics.reset()

    def forward(self, batch: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        """Compute (metric, preds, labels) as nested tensors from a batch of packed images, labels, and offsets."""
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
        # The different loss functions should return their relevant batch size (which may be pairs or number of variants)
        loss, batch_size = self.loss(metric, labels)
        return (loss, preds, labels, batch_size)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, label, *_ = self._model_step(batch, batch_idx)

        self.log("train_loss", loss, prog_bar=True)
        batch_value = self.train_metrics(preds, label)
        num_variants = label.size(0)
        self.log_dict(batch_value, on_step=False, on_epoch=True, prog_bar=True, batch_size=num_variants)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, label, batch_size = self._model_step(batch, batch_idx)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        batch_value = self.val_metrics(preds, label)
        num_variants = label.size(0)
        self.log_dict(batch_value, on_step=False, on_epoch=True, prog_bar=True, batch_size=num_variants)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, label, batch_size = self._model_step(batch, batch_idx)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        batch_value = self.test_metrics(preds, label)
        num_variants = label.size(0)
        self.log_dict(batch_value, on_step=False, on_epoch=True, prog_bar=True, batch_size=num_variants)

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
        print("\nOffsets", metrics.offsets(),"\nMetric",  metrics.values(),
              "\nPredictions", preds, "\nLabels",  labels_nt.values(),"\nMetadata", *metadata)
        padded_target = torch.nested.to_padded_tensor(labels_nt, 0)
        onehot_preds = torch.nn.functional.one_hot(preds, num_classes=padded_target.size(1))
 
        correct = torch.sum(torch.any(onehot_preds & (torch.eq(padded_target, 1) | torch.eq(padded_target, 2)), dim=1))
        total = labels_nt.shape[0]
        
        targ = padded_target > 0
        indices = torch.arange(targ.size(0))
        non_reference_concordance = torch.sum(targ[indices, preds])

        print("\nAccuracy", correct.float() / total, 
                "\nnon-reference concordance", non_reference_concordance.float() / total)

