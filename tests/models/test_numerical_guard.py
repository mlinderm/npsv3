import logging
import os
import subprocess
import sys
import textwrap

import lightning as L
import pytest
import torch
from lightning.pytorch.plugins.precision import MixedPrecision
from torch import nn

from npsv3.models.numerical_guard import NumericalGuard, NumericalInstabilityError, checkpoint_problem, first_nonfinite


class TinyModule(L.LightningModule):
    """Linear + BatchNorm regression whose loss is replaced by `bad_value` (NaN, inf, ...) at `bad_steps`.

    With `constant_bad=True` the bad value is *added* to the loss as a constant, so the loss is non-finite but the
    gradients stay finite (only the loss check can catch it). `bad_ranks` limits the bad steps to those DDP ranks.
    """

    def __init__(self, bad_steps=(), bad_value=float("nan"), constant_bad=False, bad_ranks=None, lr=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 1))
        self.bad_steps = set(bad_steps)
        self.bad_value = bad_value
        self.constant_bad = constant_bad
        self.bad_ranks = bad_ranks
        self.lr = lr
        self.snapshots = {}

    def training_step(self, batch, batch_idx):  # noqa: ARG002
        # Record the state before this step's forward pass (and so before this step's BN update)
        self.snapshots[self.global_step] = {k: v.detach().clone() for k, v in self.state_dict().items()}
        x, y = batch
        loss = nn.functional.mse_loss(self.net(x), y)
        if self.global_step in self.bad_steps and (self.bad_ranks is None or self.global_rank in self.bad_ranks):
            bad = torch.tensor(self.bad_value)
            loss = loss + bad if self.constant_bad else loss * bad
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)

    def train_dataloader(self):
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(64, 4, generator=generator)
        y = x.sum(dim=1, keepdim=True)
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=8)


def _trainer(guard=None, max_steps=6, devices=1, **kwargs):
    callbacks = [guard] if guard is not None else []
    return L.Trainer(
        accelerator="cpu",
        devices=devices,
        max_steps=max_steps,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        callbacks=callbacks,
        **kwargs,
    )


def _state_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def _cpu_fp16_scaler_plugin(init_scale):
    # Lightning doesn't create a GradScaler for 16-mixed on CPU, so provide one, to exercise the GradScaler code paths
    return MixedPrecision("16-mixed", "cpu", scaler=torch.amp.GradScaler("cpu", init_scale=init_scale))


class TestFirstNonfinite:
    def test_finds_nested_key(self):
        state = {"a": torch.zeros(2), "b": [{"c": torch.tensor([1.0, float("inf")])}], "d": torch.tensor([1, 2])}
        assert first_nonfinite(state) == "b.0.c"

    def test_all_finite(self):
        assert first_nonfinite({"a": torch.zeros(2), "n": 3, "s": "x"}) is None


class TestCheckpointProblem:
    def test_zero_scale(self):
        assert "scale" in checkpoint_problem({"state_dict": {}, "MixedPrecision": {"scale": 0.0}})

    def test_nonfinite_state_dict(self):
        assert "w" in checkpoint_problem({"state_dict": {"w": torch.tensor([float("nan")])}})

    def test_nonfinite_optimizer_state(self):
        checkpoint = {"state_dict": {}, "optimizer_states": [{"state": {0: {"exp_avg": torch.tensor([float("nan")])}}}]}
        assert "optimizer" in checkpoint_problem(checkpoint)

    def test_usable(self):
        checkpoint = {"state_dict": {"w": torch.ones(2)}, "MixedPrecision": {"scale": 65536.0}}
        assert checkpoint_problem(checkpoint) is None


class TestNumericalGuard:
    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
    def test_skips_nonfinite_step_and_continues(self, bad_value, caplog):
        model = TinyModule(bad_steps={2}, bad_value=bad_value)
        guard = NumericalGuard()
        with caplog.at_level(logging.WARNING):
            _trainer(guard).fit(model)

        # The weights, optimizer moments *and* BN running stats after the bad step are what they were before it
        assert _state_equal(model.snapshots[2], model.snapshots[3])
        assert not _state_equal(model.snapshots[1], model.snapshots[2]), "Good steps must update the model"
        assert not _state_equal(model.snapshots[3], model.snapshots[4]), "Training must continue after a skipped step"
        assert all(torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point())
        assert guard.skipped_steps == 1
        assert guard.consecutive_nonfinite == 0
        assert "Skipping step 2" in caplog.text

    def test_skips_nonfinite_loss_with_finite_gradients(self):
        """A non-finite loss whose gradients are finite would otherwise be applied, so the loss itself is checked"""
        model = TinyModule(bad_steps={1}, bad_value=float("inf"), constant_bad=True)
        guard = NumericalGuard()
        _trainer(guard, max_steps=3).fit(model)
        assert _state_equal(model.snapshots[1], model.snapshots[2])
        assert guard.skipped_steps == 1

    def test_without_guard_nan_corrupts_weights(self):
        """Control for the above: with bf16/fp32 there is no GradScaler to skip a non-finite step"""
        model = TinyModule(bad_steps={2})
        _trainer(max_steps=4).fit(model)
        assert not torch.isfinite(model.net[0].weight).all()

    def test_stops_after_consecutive_nonfinite_steps(self):
        model = TinyModule(bad_steps=set(range(2, 100)))
        guard = NumericalGuard(max_consecutive_nonfinite=3)
        with pytest.raises(NumericalInstabilityError, match="3 consecutive .* through step 4"):
            _trainer(guard, max_steps=20).fit(model)
        # Every bad step before the limit was skipped, so the model is still finite
        assert torch.isfinite(model.net[0].weight).all()

    def test_consecutive_count_resets_on_good_step(self):
        model = TinyModule(bad_steps={1, 2, 4, 5})
        guard = NumericalGuard(max_consecutive_nonfinite=3)
        _trainer(guard, max_steps=7).fit(model)
        assert guard.skipped_steps == 4

    def test_check_every_n_steps(self):
        model = TinyModule(bad_steps={1, 2})
        guard = NumericalGuard(check_every_n_steps=2)
        _trainer(guard, max_steps=4).fit(model)
        # Step 2 is checked (and skipped), step 1 isn't and so is applied, corrupting the model
        assert guard.skipped_steps == 1
        assert not torch.isfinite(model.net[0].weight).all()

    def test_logs_diagnostics(self):
        model = TinyModule()
        trainer = _trainer(NumericalGuard(), max_steps=2)
        trainer.fit(model)
        metrics = trainer.progress_bar_metrics
        assert metrics["lr"] == pytest.approx(0.1)
        assert metrics["grad_norm"] > 0
        assert "grad_scale" not in metrics, "There is no GradScaler without 16-mixed"

    def test_logs_grad_scale_under_fp16(self):
        trainer = _trainer(NumericalGuard(), max_steps=2, plugins=[_cpu_fp16_scaler_plugin(2.0**10)])
        trainer.fit(TinyModule())
        assert trainer.progress_bar_metrics["grad_scale"] == pytest.approx(2.0**10)

    def test_stops_on_collapsed_grad_scale(self):
        trainer = _trainer(NumericalGuard(), plugins=[_cpu_fp16_scaler_plugin(2.0**-9)])
        with pytest.raises(NumericalInstabilityError, match=r"scale collapsed to 0.00195312 .* at step 0"):
            trainer.fit(TinyModule())

    def test_zero_grad_scale_doesnt_apply_nan_gradients(self):
        """The motivating failure: at scale 0, unscaling turns zero gradients into NaN without the GradScaler flagging
        them, so the optimizer applies them. With the scale check disabled, the gradient check must still skip them."""
        model = TinyModule()
        trainer = _trainer(NumericalGuard(min_grad_scale=0.0), max_steps=3, plugins=[_cpu_fp16_scaler_plugin(0.0)])
        trainer.fit(model)
        assert torch.isfinite(model.net[0].weight).all()
        assert _state_equal(model.snapshots[0], model.snapshots[2])

    def test_refuses_to_save_nonfinite_checkpoint(self, tmp_path):
        model = TinyModule()
        trainer = _trainer(NumericalGuard(), max_steps=1)
        trainer.fit(model)
        trainer.save_checkpoint(tmp_path / "good.ckpt")
        assert (tmp_path / "good.ckpt").exists()

        with torch.no_grad():
            model.net[0].weight[0, 0] = float("nan")
        with pytest.raises(NumericalInstabilityError, match=r"Refusing to save .* net\.0\.weight"):
            trainer.save_checkpoint(tmp_path / "bad.ckpt")
        assert not (tmp_path / "bad.ckpt").exists()


def test_ddp_ranks_agree_to_skip(tmp_path):
    """A non-finite loss on one rank only, with finite gradients (so DDP's gradient all-reduce doesn't spread it), must
    make every rank skip the step; otherwise the ranks' weights diverge, and a rank that stops would hang the others."""
    script = textwrap.dedent(
        f"""
        import json, os, sys, torch
        sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
        from test_numerical_guard import TinyModule, _trainer
        from npsv3.models.numerical_guard import NumericalGuard

        model = TinyModule(bad_steps={{1}}, bad_value=float("inf"), constant_bad=True, bad_ranks={{1}})
        guard = NumericalGuard()
        _trainer(guard, max_steps=3, devices=2, strategy="ddp").fit(model)
        skipped_step_1 = all(torch.equal(model.snapshots[1][k], model.snapshots[2][k]) for k in model.snapshots[1])
        with open(os.path.join({str(tmp_path)!r}, f"rank{{model.global_rank}}.json"), "w") as f:
            json.dump({{
                "skipped": guard.skipped_steps,
                "skipped_step_1": skipped_step_1,
                "weight": model.net[0].weight.flatten().tolist(),
            }}, f)
        """
    )
    script_path = tmp_path / "ddp_guard.py"
    script_path.write_text(script)
    result = subprocess.run(
        [sys.executable, str(script_path)], cwd=tmp_path, capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, result.stderr[-4000:]

    import json

    ranks = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert all(rank["skipped"] == 1 and rank["skipped_step_1"] for rank in ranks)
    assert ranks[0]["weight"] == ranks[1]["weight"], "Ranks must stay in sync"
