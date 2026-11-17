import inspect
import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import hydra
import lightning as L
import pandas as pd
import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F
from torchvision import models

from npsv3.models.loaders import PackedBatch
from npsv3.models.metrics import (
    ExponentialMovingAverage,
    GenotypingConcordance,
    GenotypingNonRefConcordance,
    GenotypingNonRefF1,
    GenotypingNonRefPrecision,
    GenotypingNonRefRecall,
)
from npsv3.models.transformer import ViTConfig, ViTModel


class InceptionEncoder(nn.Module):
    def __init__(self, num_channels=8, projection_size=512):
        super().__init__()
        self.num_channels = num_channels
        self.projection_size = projection_size

        # TODO: How do we handle the variable batch size? Slice or replace BatchNorm2D with GroupNorm? Does the latter require
        # Weight Standardization?
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
    def __init__(self, num_channels=8, pretrained_path=None):
        super().__init__()
        self.num_channels = num_channels

        config = ViTConfig(num_channels=self.num_channels, image_size = (96, 288), )

        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'), weights_only=False)
            self.model = ViTModel(config, add_pooling_layer=False)
            encoder_weights = {k.removeprefix("model.vit."): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.vit.")}
            self.model.load_state_dict(encoder_weights, strict=False)
        else:
            self.model = ViTModel(config, add_pooling_layer=False)


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

    Each variant group has R replicates of its G genotypes' supports, laid out replicates-major (support r*G + g), with
    logits s_{r,g} = metric_{r,g} / temperature. Let P be the positive genotypes ("first-" and "second-rank" labels) and
    N the negatives (label 0). Every positive support is an anchor, averaged outside the log (SupCon's L_out), against
    the negatives in its denominator. With `negatives="all"`, those are the negatives of every replicate

        L = -1/(R |P|) sum_r sum_{g in P} log( e^{s_{r,g}} / (e^{s_{r,g}} + sum_{r'} sum_{h in N} e^{s_{r',h}}) )

    and with `negatives="replicate"`, only the negatives of the anchor's replicate, i.e., a softmax within each replicate

        L = -1/(R |P|) sum_r sum_{g in P} log( e^{s_{r,g}} / (e^{s_{r,g}} + sum_{h in N} e^{s_{r,h}}) )

    Every genotype of a replicate is simulated from the same reads, so the latter is invariant to a replicate-level
    shift in the similarity to the query. The batch loss is the mean of the groups' losses. With R = 1 they are the same.

    Args:
        temperature (float, optional): Scaling factor applied to similarity scores before exponentiation. Default is 0.07.
        negatives (str, optional): The negatives in each anchor's denominator, those of "all" replicates or the
            anchor's "replicate" only (requires `num_genotypes`). Default is "all".

    Inputs:
        metric (torch.Tensor): A 2D nested tensor of shape(|variants|, j1=|support|) with distances between query and support embeddings.
        target (torch.Tensor): A 2D nested tensor of shape(|variants|, j1=|support|) with 1 for support embeddings with the correct genotypes.
        num_genotypes (torch.Tensor | None): A (|variants|,) tensor with the number of genotypes G of each variant
            group. Only needed for `negatives="replicate"`.

    Returns:
        tuple[torch.Tensor]: Scalar tensor representing the average InfoNCE loss across variant groups and the number of variants in batch
    """

    NEGATIVES = ("all", "replicate")

    def __init__(self, temperature=0.07, negatives: Literal["all", "replicate"] = "all"):
        super().__init__()
        if negatives not in self.NEGATIVES:
            msg = f"Unknown negatives {negatives!r}, expected one of {self.NEGATIVES}"
            raise ValueError(msg)
        self.temperature = temperature
        self.negatives = negatives

    @staticmethod
    def _group_genotypes(metric: torch.Tensor, num_genotypes: torch.Tensor | None, *, needed: bool) -> list[int | None]:
        """The number of genotypes of each variant group, as Python ints, or None for each group if not `needed`"""
        if not needed:
            return [None] * metric.size(0)
        if num_genotypes is None:
            msg = "The replicate-aware loss needs the number of genotypes (`num_genotypes`) of each variant group"
            raise ValueError(msg)
        return [int(g) for g in num_genotypes.tolist()]

    def forward(self, metric: torch.Tensor, target: torch.Tensor, num_genotypes: torch.Tensor | None = None):
        target = torch.where(torch.eq(target, 1) | torch.eq(target, 2), 1, 0) # Map "first-rank" and "second-rank" positives to 1
        metric = metric / self.temperature  # Scale the similarity scores by the temperature
        group_genotypes = self._group_genotypes(metric, num_genotypes, needed=self.negatives == "replicate")

        loss = torch.tensor([0.0], dtype=metric.dtype, device=metric.device)
        for variant_metric, variant_target, variant_genotypes in zip(metric.unbind(), target.unbind(), group_genotypes, strict=True):
            # Each row is the supports contrasted with each other: every replicate (1, R*G), or each replicate (R, G)
            row_size = variant_genotypes if variant_genotypes is not None else variant_metric.size(0)
            rows = variant_metric.view(-1, row_size)
            negative_mask = variant_target.view(-1, row_size) == 0

            # Compute log(softmax) using log-sum-exp trick (similar to log_softmax) to ensure numerical stability
            max_value = torch.amax(rows, dim=1, keepdim=True)
            numerator = rows - max_value
            negatives = torch.sum(torch.where(negative_mask, torch.exp(numerator), 0.0), dim=1, keepdim=True)
            # Normalize by the number of positives "outside the log". This is the approach described in the SupCon paper.
            # Normalizing seems to improve learning and accuracy relative to just a "sum".
            variant_loss = torch.mean(
                (numerator - torch.log(torch.exp(numerator) + negatives + 1e-8))[torch.logical_not(negative_mask)]
            )
            loss += variant_loss

        return -loss / metric.size(0), metric.size(0)  # Average loss across all variants


class CopositiveInfoNCE(InfoNCE):
    """
    InfoNCE with co-positive aware targets: (partial) positives beyond the label and down-weighted (or masked) negatives
    (see `npsv3.models.copositive`).

    Each variant group has R replicates of its G genotypes' supports, laid out replicates-major (support r*G + g), with
    logits s_{r,g} = metric_{r,g} / temperature. Each genotype g has a numerator weight c_g in [0, 1], how much it is a
    positive, and a denominator weight a_g >= c_g, so it contributes a_g - c_g as a negative. Let y be the label (the
    "first-rank" genotype). The `formulation` determines how the replicates enter the loss:

    "support": Every support (r, g) with c_g > 0 is an anchor, against the negatives of every replicate, and the group's
    loss is the c-weighted mean of its anchors' losses ("all" over the positives and replicates)

        l_{r,g} = -log( e^{s_{r,g}} / (e^{s_{r,g}} + sum_{(r',h) != (r,g)} (a_h - c_h) e^{s_{r',h}}) )
        L = sum_{r,g} c_g l_{r,g} / (R sum_g c_g)

    With hard positives (c = a = 1) and negatives (c = 0, a = 1), this is `InfoNCE` (with `negatives="all"`): each
    positive against all the negatives, excluding the other positives, averaged over the positives. Down-weighting a
    negative's a_g interpolates between a full negative and being masked, and a partial positive (0 < c_g < 1, a_g = 1)
    is both an anchor (weighted by c_g) and a negative (weighted by 1 - c_g) for the other anchors. The label's share of
    the loss is only 1 / sum_g c_g, and every replicate of every positive must beat the negatives.

    "replicate" (option 3A/r): A softmax within each replicate, with the (c-weighted) positives pooled inside the log,
    averaged over the replicates ("any" over the positives, "all" over the replicates)

        L = -1/R sum_r log( sum_g c_g e^{s_{r,g}} / sum_g a_g e^{s_{r,g}} )

    "all_negatives" (option 3A): Each replicate is an anchor, with its (c-weighted) positives pooled inside the log as in
    "replicate", against the negatives of every replicate (not averaged over R) as in "support". The anchors don't
    depend on the positives, so unlike "support" the label isn't diluted by alternatives ("any" over the positives, "all"
    over the replicates)

        N_r = sum_g c_g e^{s_{r,g}}
        L = -1/R sum_r log( N_r / (N_r + sum_g sum_{r'} (a_g - c_g) e^{s_{r',g}}) )

    "pooled" (option 3B/pooled): Each label replicate is an anchor, with the other genotypes pooled by their mean over
    the replicates, e_g = 1/R sum_r e^{s_{r,g}}, consistent with `transforms.pooled_softmax`

        L = -1/R sum_r log( (c_y e^{s_{r,y}} + sum_{g != y} c_g e_g) / (c_y e^{s_{r,y}} + sum_{g != y} c_g e_g + sum_g (a_g - c_g) e_g) )

    With a single positive (c = 1[g = y]) and R > 1, "all_negatives" and "support" are `InfoNCE` with `negatives="all"`,
    while "replicate" is `InfoNCE` with `negatives="replicate"`, and "pooled" contrasts each label replicate with the
    replicate-mean of the negatives, so neither of the latter is `InfoNCE` with `negatives="all"`. With R = 1,
    "replicate", "pooled" and "all_negatives" are the same, as are all four with a single positive.

    Without `positive_weights` and `denominator_weights`, the weights are derived from `target`: c = 1 for the
    "first-" and "second-rank" positives and a = 1. The batch loss is the mean of the groups' losses.

    Args:
        temperature (float, optional): Scaling factor applied to similarity scores before exponentiation. Default is 0.07.
        formulation (str, optional): How the replicates enter the loss, "support", "replicate", "pooled" or
            "all_negatives" (all but "support" require `num_genotypes`). Default is "support".

    Inputs:
        metric (torch.Tensor): A 2D nested tensor of shape(|variants|, j1=|support|) with distances between query and support embeddings.
        target (torch.Tensor): A 2D nested tensor of shape(|variants|, j1=|support|) with 1 for support embeddings with the correct genotypes.
        positive_weights (torch.Tensor | None): A 2D nested tensor of shape(|variants|, j1=|support|) with the numerator
            weights c (tiled over the replicates). None (with `denominator_weights`) for the single-positive targets from
            `target`, as in `InfoNCE`.
        denominator_weights (torch.Tensor | None): A 2D nested tensor of shape(|variants|, j1=|support|) with the
            denominator weights a (tiled over the replicates).
        num_genotypes (torch.Tensor | None): A (|variants|,) tensor with the number of genotypes G of each variant
            group. Only needed for the "replicate", "pooled" and "all_negatives" formulations.

    Returns:
        tuple[torch.Tensor]: Scalar tensor representing the average loss across variant groups and the number of variants in batch
    """

    FORMULATIONS = ("support", "replicate", "pooled", "all_negatives")

    def __init__(
        self, temperature=0.07, formulation: Literal["support", "replicate", "pooled", "all_negatives"] = "support"
    ):
        super().__init__(temperature)
        if formulation not in self.FORMULATIONS:
            msg = f"Unknown co-positive formulation {formulation!r}, expected one of {self.FORMULATIONS}"
            raise ValueError(msg)
        self.formulation = formulation

    @staticmethod
    def _log_negatives(metric: torch.Tensor, positive: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """log sum (a_g - c_g) e^{metric} over all of the (..., G) logits, with (G,) weights"""
        # Select the negatives instead of relying on log(0) = -inf: with no negatives, as when every genotype is a positive,
        # the gradient of logsumexp over only -inf is NaN, while over nothing it is -inf and has no gradient
        negative = torch.clamp(denominator - positive, min=0.0)
        is_negative = negative > 0
        return torch.logsumexp((metric[..., is_negative] + torch.log(negative[is_negative])).ravel(), dim=0)

    @staticmethod
    def _support_loss(metric: torch.Tensor, positive: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """The "support" formulation's loss for one group's (R*G,) logits and weights"""
        anchors = torch.nonzero(positive > 0).squeeze(1)
        # The weight of each support in each anchor's denominator, as an (|anchors|, |support|) log-weight: the anchor
        # itself with weight 1, and every other support with its negative weight a - c (log(0) = -inf excludes it)
        negative = torch.clamp(denominator - positive, min=0.0)
        is_anchor = torch.eq(torch.arange(metric.size(0), device=anchors.device), anchors.unsqueeze(1))
        log_weight = torch.where(is_anchor, 0.0, torch.log(negative))
        log_denominator = torch.logsumexp(metric + log_weight, dim=1)
        anchor_weight = positive[anchors]
        return torch.sum(anchor_weight * (log_denominator - metric[anchors])) / torch.sum(anchor_weight)

    @staticmethod
    def _replicate_loss(metric: torch.Tensor, positive: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """The "replicate" formulation's loss for one group's (R, G) logits and (G,) weights"""
        log_numerator = torch.logsumexp(metric + torch.log(positive), dim=1)
        log_denominator = torch.logsumexp(metric + torch.log(denominator), dim=1)
        return torch.mean(log_denominator - log_numerator)

    @staticmethod
    def _pooled_loss(
        metric: torch.Tensor, positive: torch.Tensor, denominator: torch.Tensor, is_label: torch.Tensor
    ) -> torch.Tensor:
        """The "pooled" formulation's loss for one group's (R, G) logits and (G,) weights and label indicator"""
        # log e_g, the log-mean-exp of each genotype over the replicates
        log_pooled = torch.logsumexp(metric, dim=0) - math.log(metric.size(0))
        log_positive = torch.log(positive)
        # (R, G) numerator terms: the label in the anchor's replicate and the pooled alternatives (log(0) = -inf excludes)
        numerator = torch.where(is_label, metric + log_positive, log_pooled + log_positive)
        log_numerator = torch.logsumexp(numerator, dim=1)
        log_negatives = CopositiveInfoNCE._log_negatives(log_pooled, positive, denominator)
        return torch.mean(torch.logaddexp(log_numerator, log_negatives) - log_numerator)

    @staticmethod
    def _all_negatives_loss(metric: torch.Tensor, positive: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """The "all_negatives" formulation's loss for one group's (R, G) logits and (G,) weights"""
        log_numerator = torch.logsumexp(metric + torch.log(positive), dim=1)
        log_negatives = CopositiveInfoNCE._log_negatives(metric, positive, denominator)
        return torch.mean(torch.logaddexp(log_numerator, log_negatives) - log_numerator)

    def forward(
        self,
        metric: torch.Tensor,
        target: torch.Tensor,
        positive_weights: torch.Tensor | None = None,
        denominator_weights: torch.Tensor | None = None,
        num_genotypes: torch.Tensor | None = None,
    ):
        metric = metric / self.temperature  # Scale the similarity scores by the temperature
        if positive_weights is None or denominator_weights is None:
            # Map "first-rank" and "second-rank" positives to 1, with every support in the denominator
            positives = [(torch.eq(t, 1) | torch.eq(t, 2)).to(metric.dtype) for t in target.unbind()]
            weights = zip(positives, (torch.ones_like(p) for p in positives), strict=True)
        else:
            weights = zip(positive_weights.unbind(), denominator_weights.unbind(), strict=True)
        group_genotypes = self._group_genotypes(metric, num_genotypes, needed=self.formulation != "support")

        loss = torch.tensor(0.0, dtype=metric.dtype, device=metric.device)
        for variant_metric, variant_target, (positive, denominator), variant_genotypes in zip(
            metric.unbind(), target.unbind(), weights, group_genotypes, strict=True
        ):
            if self.formulation == "support":
                loss = loss + self._support_loss(variant_metric, positive, denominator)
                continue
            # The (R, G) logits, and the per-genotype weights and labels from the first replicate (they are tiled)
            logits = variant_metric.view(-1, variant_genotypes)
            genotype_positive, genotype_denominator = positive[:variant_genotypes], denominator[:variant_genotypes]
            if self.formulation == "replicate":
                loss = loss + self._replicate_loss(logits, genotype_positive, genotype_denominator)
            elif self.formulation == "all_negatives":
                loss = loss + self._all_negatives_loss(logits, genotype_positive, genotype_denominator)
            else:
                is_label = torch.eq(variant_target[:variant_genotypes], 1)
                loss = loss + self._pooled_loss(logits, genotype_positive, genotype_denominator, is_label)

        return loss / metric.size(0), metric.size(0)  # Average loss across all variants


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
        scheduler: Callable | None = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=type(self).ignored_hyperparameters)

        self.encoder = encoder
        self.metric = metric
        self.loss = loss
        self.predictor = predictor

        self.train_loss_ema = ExponentialMovingAverage()
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

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        """Set of parameters that should not use weight decay."""
        # This is used by timm's optimizers to determine which parameters should not use weight decay
        return { f"encoder.{name}" for name in self.encoder.no_weight_decay() }

    def on_train_start(self) -> None:
        # Reset validation metrics at the start of training to avoid effects of sanity batches
        self.val_metrics.reset()

    def forward(self, batch: PackedBatch) -> tuple[torch.Tensor]:
        """Compute (metric, preds, labels) as nested tensors from a batch of packed images, labels, and offsets."""
        images, offsets = batch.images, batch.offsets
        # A padded batch carries labels for the full batch height so the pack has a fixed shape, but
        # the nested tensor below indexes only the support region and requires an exact match.
        labels = batch.labels[: offsets[-1]]
        # Neither encoder masks its global pool, so a mixed-width pack would silently blend padding into
        # the embeddings. This is where a varlen (cu_seqlens) path would attach if widths ever vary.
        assert torch.all(batch.widths == batch.widths[0]), "Packed batch mixes image widths, which the encoders cannot mask"
        # Pass the canonical positive through so predict-time consumers can name the called genotype
        metadata = (batch.genotype_label, batch.num_genotypes, batch.meta)
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

    def _model_step(self, batch, batch_idx):  # noqa: ARG002
        metric, preds, labels, *_ = self(batch)
        # The different loss functions should return their relevant batch size (which may be pairs or number of variants)
        # The replicate-aware InfoNCE losses need each group's number of genotypes to recover its (R, G) supports
        loss_kwargs = {"num_genotypes": batch.num_genotypes} if isinstance(self.loss, InfoNCE) else {}
        if batch.positive_weights is None or batch.denominator_weights is None:
            loss, batch_size = self.loss(metric, labels, **loss_kwargs)
        else:
            if not isinstance(self.loss, CopositiveInfoNCE):
                msg = f"The batch has co-positive targets (`data.copositive`), which {type(self.loss).__name__} doesn't use; use CopositiveInfoNCE"
                raise TypeError(msg)
            # As for the labels, construct the targets with the offsets shared with metrics so the ragged dimension matches
            offsets = metric.offsets()
            max_support = torch.max(torch.diff(offsets))
            positive_weights, denominator_weights = (
                torch.nested.nested_tensor_from_jagged(weights, offsets=offsets, max_seqlen=max_support)
                for weights in (batch.positive_weights, batch.denominator_weights)
            )
            loss, batch_size = self.loss(metric, labels, positive_weights, denominator_weights, **loss_kwargs)
        return (loss, preds, labels, batch_size)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, label, *_ = self._model_step(batch, batch_idx)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_loss_ema", self.train_loss_ema(loss), on_step=True, on_epoch=False, prog_bar=True)
        self.train_metrics(preds, label)
        # Log the metric objects, not their per-batch values, so Lightning computes from counts summed across ranks
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=label.size(0))

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, label, batch_size = self._model_step(batch, batch_idx)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.val_metrics(preds, label)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=label.size(0))

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, label, batch_size = self._model_step(batch, batch_idx)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.test_metrics(preds, label)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=label.size(0))

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        # timm's `create_optimizer_v2` takes the model (to apply `no_weight_decay`), while torch optimizers take the
        # parameters
        optimizer_factory = self.hparams.optimizer
        if getattr(getattr(optimizer_factory, "func", optimizer_factory), "__module__", "").startswith("timm."):
            optimizer = optimizer_factory(self.trainer.model)
        else:
            optimizer = optimizer_factory(params=self.trainer.model.parameters())

        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}

        # Based on https://lightning.ai/docs/pytorch/stable/common/optimization.html#bring-your-own-custom-learning-rate-schedulers
        scheduler_factory = self.hparams.scheduler
        scheduler_kwargs = {}
        if "total_steps" in inspect.signature(scheduler_factory).parameters:
            # A step-based schedule (e.g., `schedulers.warmup_cosine`) needs the run's total optimizer steps, which
            # accounts for max_epochs/max_steps, the (virtual) epoch length set via `limit_train_batches`, gradient
            # accumulation and the number of DDP ranks
            total_steps = self.trainer.estimated_stepping_batches
            if not math.isfinite(total_steps):
                raise ValueError("A step-based LR schedule requires a finite trainer max_epochs or max_steps")
            scheduler_kwargs["total_steps"] = int(total_steps)
        scheduler = scheduler_factory(optimizer=optimizer, **scheduler_kwargs)

        if isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            # torch schedulers are stepped once per optimizer step
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        # timm schedulers are stepped once per epoch (see `lr_scheduler_step`)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, metric):
        if isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            super().lr_scheduler_step(scheduler, metric)
        else:
            # timm's schedulers appear to need the "next" epoch at each step
            # https://github.com/huggingface/pytorch-image-models/blob/954613a470652e4a113ff45b62dbd15c4e229218/train.py#L1067
            scheduler.step(epoch=self.current_epoch + 1)


class RecordLossCallback(L.pytorch.callbacks.Callback):
    """Record the loss, prediction, etc. of every region seen during prediction as a pickled DataFrame.

    Each packed batch holds many regions (groups), so the loss is recomputed per group rather than for whole batches.
    The DataFrame is intended for offline analysis, e.g., loss histograms to identify likely label errors. Under DDP
    each rank writes its own file with `.rank<N>` inserted before the suffix(es).

    Columns:
        region, loss, genotype (label), prediction, correct, num_genotypes, num_replicates,
        label_rank: (G,) per-genotype label ranks
        metric: (R, G) support metric
        genotype_metric: (G,) per-genotype metric averaged over the replicates
        label_metric, prediction_metric: `genotype_metric` of the labeled and predicted genotypes
    """

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.rows = []

    def on_predict_start(self, trainer, model):  # noqa: ARG002
        self.rows = []

    def on_predict_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):  # noqa: ARG002
        metric, preds, labels, genotype_label, num_genotypes, regions = outputs
        offsets = metric.offsets()
        metric_values, label_values = metric.values(), labels.values()
        cpu_offsets = offsets.tolist()
        cpu_metric, cpu_labels = metric_values.float().cpu().numpy(), label_values.cpu().numpy()

        for group, region in enumerate(regions):
            start, end = offsets[group], offsets[group + 1]
            group_offsets = torch.tensor([0, end - start], device=offsets.device)
            group_metric = torch.nested.nested_tensor_from_jagged(metric_values[start:end], offsets=group_offsets)
            group_labels = torch.nested.nested_tensor_from_jagged(label_values[start:end], offsets=group_offsets)
            loss_kwargs = {"num_genotypes": num_genotypes[group : group + 1]} if isinstance(model.loss, InfoNCE) else {}
            loss = model.loss(group_metric, group_labels, **loss_kwargs)[0].item()

            # Supports are replicate-major replicates of the G genotypes, i.e., support index = r*G + g, and the labels
            # are the per-genotype label ranks tiled R times
            num_genotype = int(num_genotypes[group])
            genotype = int(genotype_label[group])
            prediction = int(preds[group]) % num_genotype
            support_metric = cpu_metric[cpu_offsets[group] : cpu_offsets[group + 1]].reshape(-1, num_genotype)
            genotype_metric = support_metric.mean(axis=0)

            self.rows.append({
                "region": region,
                "loss": loss,
                "genotype": genotype,
                "prediction": prediction,
                "correct": genotype == prediction,
                "num_genotypes": num_genotype,
                "num_replicates": support_metric.shape[0],
                "label_rank": cpu_labels[cpu_offsets[group] : cpu_offsets[group] + num_genotype].copy(),
                "metric": support_metric.copy(),
                "genotype_metric": genotype_metric,
                "label_metric": genotype_metric[genotype].item() if 0 <= genotype < num_genotype else float("nan"),
                "prediction_metric": genotype_metric[prediction].item(),
            })

    def on_predict_end(self, trainer, model):  # noqa: ARG002
        path = Path(self.path)
        if trainer.world_size > 1:
            suffixes = "".join(path.suffixes)
            path = path.with_name(f"{path.name.removesuffix(suffixes)}.rank{trainer.global_rank}{suffixes}")
        path.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(self.rows).to_pickle(path)
        logging.info("Wrote losses for %d regions to %s", len(self.rows), path)


# def predict(cfg, **kw_args):
#     dm = hydra.utils.instantiate(cfg.data)

#     model_cls = hydra.utils.get_class(cfg.model._target_)
#     model = model_cls.load_from_checkpoint(
#         cfg.model.checkpoint,
#         encoder=hydra.utils.instantiate(cfg.model.encoder),
#         metric=hydra.utils.instantiate(cfg.model.metric),
#         loss=hydra.utils.instantiate(cfg.model.loss),
#         predictor=hydra.utils.instantiate(cfg.model.predictor),
#     )
#     trainer = L.Trainer(limit_predict_batches=2)
#     for prediction in trainer.predict(model, dm):
#         metrics, preds, labels_nt, *metadata = prediction
#         print("\nOffsets", metrics.offsets(),"\nMetric",  metrics.values(),
#               "\nPredictions", preds, "\nLabels",  labels_nt.values(),"\nMetadata", *metadata)
#         padded_target = torch.nested.to_padded_tensor(labels_nt, 0)
#         onehot_preds = torch.nn.functional.one_hot(preds, num_classes=padded_target.size(1))

#         correct = torch.sum(torch.any(onehot_preds & (torch.eq(padded_target, 1) | torch.eq(padded_target, 2)), dim=1))
#         total = labels_nt.shape[0]

#         targ = padded_target > 0
#         indices = torch.arange(targ.size(0))
#         non_reference_concordance = torch.sum(targ[indices, preds])

#         print("\nAccuracy", correct.float() / total,
#                 "\nnon-reference concordance", non_reference_concordance.float() / total)

