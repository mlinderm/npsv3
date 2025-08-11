import torch
import torchmetrics
from torchmetrics.utilities.compute import _safe_divide


class GenotypingMetrics(torchmetrics.Metric):
    """
    Compute genotyping metrics, allowing for multiple correct support images per variant.

    This class currently computes metrics on a per-example basis, as opposed to per-variant
    basis, i.e., it does not take into account multiple variants merged into a single example.
    """

    is_differentiable: bool = False
    higher_is_better: bool | None = True
    full_state_update: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        # Store non-reference concordance as full binary contingency table
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update accuracy measures with |examples|, dense integer prediction tensor and |examples|,|support| nested one-hot target tensor"""
        padded_target = torch.nested.to_padded_tensor(target, 0)
        onehot_preds = torch.nn.functional.one_hot(preds, num_classes=padded_target.size(1))

        # Treat "first-rank" and "second-rank" positives (same genotype, different phase) as correct
        self.correct += torch.sum(
            torch.any(onehot_preds & (torch.eq(padded_target, 1) | torch.eq(padded_target, 2)), dim=1)
        )
        self.total += target.size(0)

        # Collect non-reference concordance as the contingency table
        matching_presence = padded_target[torch.arange(padded_target.size(0)), preds] != 0
        nonref_target = padded_target[:, 0] == 0
        self.tp += (matching_presence & nonref_target).sum()
        self.fn += ((~matching_presence) & nonref_target).sum()
        self.fp += ((~matching_presence) & (~nonref_target)).sum()
        self.tn += (matching_presence & (~nonref_target)).sum()

    def compute(self) -> torch.Tensor:
        return _safe_divide(self.correct, self.total)


class GenotypingConcordance(GenotypingMetrics):
    def compute(self) -> torch.Tensor:
        return _safe_divide(self.correct, self.total)


class GenotypingNonRefConcordance(GenotypingMetrics):
    def compute(self) -> torch.Tensor:
        assert torch.equal(self.tp + self.fp + self.tn + self.fn, self.total), (
            "Inconsistent total and contingency table"
        )
        return _safe_divide(self.tp + self.tn, self.tp + self.tn + self.fp + self.fn)


class GenotypingNonRefPrecision(GenotypingMetrics):
    def compute(self) -> torch.Tensor:
        return _safe_divide(self.tp, self.tp + self.fp)


class GenotypingNonRefRecall(GenotypingMetrics):
    def compute(self) -> torch.Tensor:
        return _safe_divide(self.tp, self.tp + self.fn)


class GenotypingNonRefF1(GenotypingMetrics):
    def compute(self) -> torch.Tensor:
        return _safe_divide(2.0 * self.tp, 2.0 * self.tp + self.fp + self.fn)
