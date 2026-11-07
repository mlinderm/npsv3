import json
import os
import subprocess
import sys
import textwrap

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


def _scripted_step(rank: int, batch_idx: int, seed: int):
    """(loss, preds, target, examples) for one batch, with sizes and accuracy that differ by batch and rank, so that the
    pooled result is distinct from any one rank's result and from a mean of per-batch ratios."""
    generator = torch.Generator().manual_seed(1000 * seed + 10 * rank + batch_idx)
    examples = [8, 64, 16][batch_idx % 3] * (1 + rank)
    nonref = torch.rand(examples, generator=generator) < [0.2, 0.7, 0.5][batch_idx % 3]
    target = torch.nested.nested_tensor(
        [torch.tensor([0, 1, 0, 0] if is_nonref else [1, 0, 0, 0]) for is_nonref in nonref],
        layout=torch.jagged,
    )
    correct = torch.rand(examples, generator=generator) < (0.9 if rank == 0 else 0.4)
    preds = torch.where(correct, nonref.long(), 1 - nonref.long())
    return 1.0 + rank + 0.1 * batch_idx, preds, target, examples


def _pooled_results(seed: int, num_batches: int, prefix: str, names: tuple[str, ...]) -> dict[str, float]:
    """The result of pooling the counts of both ranks"""
    metrics = torchmetrics.MetricCollection({
        "concordance": GenotypingConcordance(),
        "nonrefconcordance": GenotypingNonRefConcordance(),
        "nonrefprecision": GenotypingNonRefPrecision(),
        "nonrefrecall": GenotypingNonRefRecall(),
        "nonreff1": GenotypingNonRefF1(),
    })
    weighted_loss, total_examples = 0.0, 0
    for rank in (0, 1):
        for batch_idx in range(num_batches):
            loss, preds, target, examples = _scripted_step(rank, batch_idx, seed)
            metrics.update(preds, target)
            weighted_loss += loss * examples
            total_examples += examples
    computed = metrics.compute()
    results = {f"{prefix}_{name}": computed[name].item() for name in names}
    if prefix != "train":
        results[f"{prefix}_loss"] = weighted_loss / total_examples
    return results


class TestDistributedLogging:
    def test_logged_metrics_pool_counts_across_ranks(self, tmp_path):
        """Both ranks must report what pooling all counts would give, not their own value or a mean of batch ratios."""
        num_batches, num_epochs = 6, 2
        script = textwrap.dedent(
            f"""
            import json, sys, torch, lightning as L
            sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})
            from models.test_metrics import _scripted_step
            from npsv3.models.paired import PackedVariant

            class ScriptedVariant(PackedVariant):
                def _model_step(self, batch, batch_idx):
                    if self.trainer.sanity_checking:
                        seed = 100
                    elif self.trainer.testing:
                        seed = 200
                    else:
                        seed = self.current_epoch
                    loss, preds, target, examples = _scripted_step(self.global_rank, batch_idx, seed)
                    return (self.encoder.weight.sum() * 0.0 + loss, preds, target, examples)

                def configure_optimizers(self):
                    return torch.optim.SGD(self.parameters(), lr=0.0)

            class DataModule(L.LightningDataModule):
                def _loader(self):
                    return torch.utils.data.DataLoader(list(range({num_batches})), batch_size=None)
                train_dataloader = val_dataloader = test_dataloader = _loader

            model = ScriptedVariant(
                encoder=torch.nn.Linear(1, 1, bias=False),
                metric=torch.nn.Identity(),
                loss=torch.nn.Identity(),
                optimizer=None,
                predictor=torch.nn.Identity(),
            )
            trainer = L.Trainer(
                accelerator="cpu", devices=2, strategy="ddp", max_epochs={num_epochs},
                num_sanity_val_steps=2, use_distributed_sampler=False,
                enable_checkpointing=False, logger=False, enable_progress_bar=False,
            )
            trainer.fit(model, datamodule=DataModule())
            fit_results = dict(trainer.callback_metrics)
            trainer.test(model, datamodule=DataModule(), verbose=False)
            results = {{name: value.item() for name, value in {{**fit_results, **trainer.callback_metrics}}.items()}}
            with open(f"{tmp_path}/rank{{trainer.global_rank}}.json", "w") as results_file:
                json.dump(results, results_file)
            """
        )
        script_path = tmp_path / "ddp_metrics.py"
        script_path.write_text(script)

        # Lightning would otherwise defer to the SLURM job's (single-task) allocation
        env = {name: value for name, value in os.environ.items() if not name.startswith("SLURM_")}
        subprocess.run([sys.executable, str(script_path)], check=True, capture_output=True, text=True, timeout=600, env=env)

        fit_names = ("concordance", "nonrefconcordance")
        test_names = (*fit_names, "nonrefprecision", "nonrefrecall", "nonreff1")
        expected = {
            # The last epoch, so that these also fail if counts accumulate across epochs
            **_pooled_results(num_epochs - 1, num_batches, "train", fit_names),
            **_pooled_results(num_epochs - 1, num_batches, "val", fit_names),
            **_pooled_results(200, num_batches, "test", test_names),
        }
        for rank in (0, 1):
            with open(tmp_path / f"rank{rank}.json") as results_file:
                results = json.load(results_file)
            for name, value in expected.items():
                assert results[name] == pytest.approx(value), f"{name} on rank {rank}"
