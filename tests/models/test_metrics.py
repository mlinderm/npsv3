import pytest
import torch
import torchmetrics

from npsv3.models.metrics import (
    ExponentialMovingAverage,
    GenotypingConcordance,
    GenotypingNonRefConcordance,
    GenotypingNonRefF1,
    GenotypingNonRefPrecision,
    GenotypingNonRefRecall,
)


class TestGenotypingMetrics:
    def _create_metrics(self):
        return torchmetrics.MetricCollection({
            "concordance": GenotypingConcordance(),
            "nonrefconcordance": GenotypingNonRefConcordance(),
            "nonrefprecision": GenotypingNonRefPrecision(),
            "nonrefrecall": GenotypingNonRefRecall(),
            "nonreff1": GenotypingNonRefF1(),
        })

    def test_perfect_concordance(self):
        preds = torch.tensor([0, 1, 1, 3])
        target = torch.nested.nested_tensor(
            # Matching genotype but incorrect phase (example index 2) is treated as correct
            [torch.tensor(t) for t in [[1, 0, 0, 0], [0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 3, 1]]],
            layout=torch.jagged
        )

        metrics = self._create_metrics()
        metrics.update(preds, target)

        results = metrics.compute()
        for name, result in results.items():
            assert result == pytest.approx(1.0), f"Should be perfect metric for {name}"


    def test_false_positive(self):
        preds = torch.tensor([0, 1, 1, 3])
        target = torch.nested.nested_tensor(
            # Matching genotype but incorrect phase (example index 2) is treated as correct
            [torch.tensor(t) for t in [[1, 0, 0, 0], [1, 0, 0, 0], [0, 2, 1, 3], [0, 3, 3, 1]]],
            layout=torch.jagged
        )

        metrics = self._create_metrics()
        metrics.update(preds, target)

        results = metrics.compute()
        assert {
            "concordance": pytest.approx(0.75),
            "nonrefconcordance": pytest.approx(0.75),
            "nonrefprecision": pytest.approx(2 / 3),
            "nonrefrecall": pytest.approx(1.0),
            "nonreff1": pytest.approx(4/(4 + 1 + 0)),
        }.items() <= results.items(), "Should have 1 false positive"


class TestExponentialMovingAverage:
    def test_first_update_is_bias_corrected_to_raw_value(self):
        metric = ExponentialMovingAverage(decay=0.9)
        assert metric(torch.tensor(2.0)) == pytest.approx(2.0)

    def test_converges_to_constant_series(self):
        metric = ExponentialMovingAverage(decay=0.9)
        for _ in range(100):
            result = metric(torch.tensor(3.0))
        assert result == pytest.approx(3.0)

    def test_smooths_noisy_series_between_its_extremes(self):
        metric = ExponentialMovingAverage(decay=0.9)
        values = [1.0, 5.0, 1.0, 5.0, 1.0, 5.0]
        for value in values:
            result = metric(torch.tensor(value))
        assert 1.0 < result < 5.0

    def test_rejects_out_of_range_decay(self):
        with pytest.raises(ValueError, match="decay"):
            ExponentialMovingAverage(decay=1.0)
