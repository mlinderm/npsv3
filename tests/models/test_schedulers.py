import math

import pytest
import torch

from npsv3.models.schedulers import warmup_cosine

PEAK = 1e-3


def _lrs(num_steps, **kwargs):
    """The learning rate used at each of `num_steps` optimizer steps"""
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=PEAK)
    scheduler = warmup_cosine(optimizer, **kwargs)
    lrs = []
    for _ in range(num_steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return lrs


class TestWarmupCosine:
    def test_is_lambda_lr(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=PEAK)
        assert isinstance(warmup_cosine(optimizer, total_steps=10), torch.optim.lr_scheduler.LambdaLR)

    def test_closed_form(self):
        warmup, total, min_ratio, start_ratio = 4, 14, 0.01, 0.1
        lrs = _lrs(20, total_steps=total, warmup_steps=warmup, min_lr_ratio=min_ratio, warmup_start_ratio=start_ratio)
        for step, lr in enumerate(lrs):
            if step < warmup:
                expected = start_ratio + (1 - start_ratio) * step / warmup
            else:
                progress = min(1.0, (step - warmup) / (total - warmup))
                expected = min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
            assert lr == pytest.approx(PEAK * expected), f"step {step}"

    def test_shape(self):
        lrs = _lrs(120, total_steps=100, warmup_steps=10, min_lr_ratio=0.01)
        assert lrs[0] == 0.0
        assert all(a < b for a, b in zip(lrs[:10], lrs[1:11], strict=True)), "Warmup increases linearly"
        assert lrs[10] == pytest.approx(PEAK), "Peak at the end of warmup"
        assert all(a >= b for a, b in zip(lrs[10:100], lrs[11:101], strict=True)), "Cosine decays monotonically"
        assert lrs[55] == pytest.approx(PEAK * (0.01 + 0.99 * 0.5)), "Halfway through decay"
        assert lrs[100:] == pytest.approx([PEAK * 0.01] * 20), "Held at the minimum after total_steps"

    def test_warmup_fraction(self):
        assert _lrs(200, total_steps=200, warmup_fraction=0.05) == pytest.approx(
            _lrs(200, total_steps=200, warmup_steps=10)
        )

    def test_no_warmup(self):
        lrs = _lrs(3, total_steps=10)
        assert lrs[0] == pytest.approx(PEAK)
        assert lrs[1] < PEAK

    def test_state_dict_round_trip(self):
        """Resuming restores the step, while the lambda is rebuilt from the configuration"""
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=PEAK)
        scheduler = warmup_cosine(optimizer, total_steps=50, warmup_steps=5)
        for _ in range(17):
            optimizer.step()
            scheduler.step()
        state = scheduler.state_dict()

        resumed_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=PEAK)
        resumed = warmup_cosine(resumed_optimizer, total_steps=50, warmup_steps=5)
        resumed.load_state_dict(state)
        optimizer.step()
        scheduler.step()
        resumed_optimizer.step()
        resumed.step()
        assert resumed.get_last_lr() == pytest.approx(scheduler.get_last_lr())

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"total_steps": 0},
            {"total_steps": float("inf")},
            {"total_steps": 10, "warmup_steps": 10},
            {"total_steps": 10, "warmup_steps": 1, "warmup_fraction": 0.1},
            {"total_steps": 10, "warmup_fraction": 1.0},
            {"total_steps": 10, "min_lr_ratio": 1.5},
            {"total_steps": 10, "warmup_start_ratio": -0.1},
        ],
    )
    def test_rejects_invalid_arguments(self, kwargs):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=PEAK)
        with pytest.raises(ValueError):
            warmup_cosine(optimizer, **kwargs)
