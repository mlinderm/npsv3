"""Learning rate schedules built only from `torch.optim.lr_scheduler`.

These are a `LambdaLR` with a closed-form multiplier, rather than a `ChainedScheduler`/`SequentialLR` composition, which
doesn't pass the step through to the underlying schedulers (see the note in `models/transformer.py`), and is stepped
once per optimizer step (Lightning `interval: "step"`).
"""

import math

import torch


def warmup_cosine(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int | None = None,
    warmup_fraction: float | None = None,
    min_lr_ratio: float = 0.0,
    warmup_start_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup to the optimizer's (peak) learning rate, then cosine decay to `min_lr_ratio` times the peak, then
    held there.

    The multiplier at step `s` (the number of prior optimizer steps) for `W` warmup and `T` total steps is

    * `s < W`: `warmup_start_ratio + (1 - warmup_start_ratio) * s / W`
    * `W <= s`: `min_lr_ratio + (1 - min_lr_ratio) * (1 + cos(pi * min(1, (s - W) / (T - W)))) / 2`

    Args:
        optimizer: The optimizer whose learning rate(s) are the peak.
        total_steps: Total number of optimizer steps in the run (e.g., Lightning's `trainer.estimated_stepping_batches`),
            including the warmup.
        warmup_steps: Number of warmup steps. Specify at most one of this or `warmup_fraction` (neither means no warmup).
        warmup_fraction: Warmup steps as a fraction of `total_steps` in [0, 1).
        min_lr_ratio: Final learning rate as a fraction of the peak, in [0, 1].
        warmup_start_ratio: Learning rate at the first warmup step as a fraction of the peak, in [0, 1].
    """
    if not (isinstance(total_steps, int) or (isinstance(total_steps, float) and math.isfinite(total_steps))):
        raise ValueError(f"total_steps must be a finite number of steps, got {total_steps}")
    total_steps = int(total_steps)
    if total_steps < 1:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if warmup_steps is not None and warmup_fraction is not None:
        raise ValueError("Specify at most one of warmup_steps or warmup_fraction")
    if warmup_fraction is not None:
        if not 0.0 <= warmup_fraction < 1.0:
            raise ValueError(f"warmup_fraction must be in [0, 1), got {warmup_fraction}")
        warmup_steps = round(warmup_fraction * total_steps)
    warmup_steps = int(warmup_steps or 0)
    if not 0 <= warmup_steps < total_steps:
        raise ValueError(f"warmup_steps must be in [0, total_steps={total_steps}), got {warmup_steps}")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")
    if not 0.0 <= warmup_start_ratio <= 1.0:
        raise ValueError(f"warmup_start_ratio must be in [0, 1], got {warmup_start_ratio}")

    decay_steps = total_steps - warmup_steps

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return warmup_start_ratio + (1.0 - warmup_start_ratio) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
