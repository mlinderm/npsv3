"""Fail fast on numerical collapse during training, rather than silently corrupting the run and its checkpoints.

Motivating failure: an fp16 (`16-mixed`) run had ~166 consecutive non-finite steps, each of which halved the GradScaler
scale until it reached 0. At scale 0, a finite loss gives all-zero scaled gradients, which unscaling (multiplying by
inv_scale = inf) turns into NaN without flagging them as non-finite, so the optimizer applied them, and every weight,
BN running statistic and optimizer moment became NaN. The NaN model was then checkpointed and resumed from.
"""

import logging
import math

import lightning as L
import torch


class NumericalInstabilityError(RuntimeError):
    """Training has numerically collapsed (non-finite losses/gradients/weights or a collapsed GradScaler scale)."""


def _tensors(obj):
    """Yield every tensor nested in dicts/lists/tuples of `obj`, e.g., a state dict or optimizer state."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _tensors(value)


def first_nonfinite(obj, prefix: str = "") -> str | None:
    """Return the key path of the first floating-point tensor in `obj` with a non-finite value, or None if all finite."""
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point() and not bool(torch.isfinite(obj).all()):
            return prefix or "<tensor>"
        return None
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = enumerate(obj)
    else:
        return None
    for key, value in items:
        if (found := first_nonfinite(value, f"{prefix}.{key}" if prefix else str(key))) is not None:
            return found
    return None


def checkpoint_problem(checkpoint: dict) -> str | None:
    """Describe why a (Lightning) checkpoint dict is numerically unusable, or return None if it is usable.

    Checks the GradScaler scale saved by the `MixedPrecision` plugin (16-mixed only), the model `state_dict` and the
    optimizer states (e.g., AdamW moments).
    """
    scaler_state = checkpoint.get("MixedPrecision")
    if isinstance(scaler_state, dict) and "scale" in scaler_state:
        scale = float(scaler_state["scale"])
        if not (math.isfinite(scale) and scale > 0.0):
            return f"its GradScaler scale is {scale}"
    if (key := first_nonfinite(checkpoint.get("state_dict", {}))) is not None:
        return f"state_dict tensor {key} is non-finite"
    if (key := first_nonfinite(checkpoint.get("optimizer_states", []))) is not None:
        return f"optimizer state {key} is non-finite"
    return None


class NumericalGuard(L.pytorch.callbacks.Callback):
    """Skip non-finite optimizer steps, stop training on numerical collapse, refuse to save non-finite checkpoints, and
    log diagnostics (`lr`, pre-clip `grad_norm` and, under 16-mixed, `grad_scale`).

    Every `check_every_n_steps` optimizer steps (all of them by default), on every DDP rank:

    * If the training loss (of any rank or accumulated micro-batch) or the (unscaled, pre-clip) gradients are
      non-finite, the step is skipped (the gradients are set to None, so the optimizer leaves the weights and its moments
      unchanged, and any buffers, e.g., BN running statistics, updated by the forward pass are restored)
      and a warning is logged. Under 16-mixed the GradScaler skips such steps itself, but not the NaN gradients produced
      once its scale reaches 0 (see the module docstring); under bf16-mixed/32-bit there is no scaler and so nothing
      else would stop a non-finite update.
    * After `max_consecutive_nonfinite` consecutive skipped (checked) steps, training stops with a
      `NumericalInstabilityError`.
    * Under 16-mixed, training stops if the GradScaler scale falls below `min_grad_scale`.

    The ranks all-reduce a single flag (any rank non-finite) so that they agree on whether to skip or stop, since a rank
    that diverged would hang at the next collective.

    Overhead: a check costs a gradient-norm reduction, one scalar all-reduce (under DDP), a device copy of the (small)
    module buffers, and one host-device sync to read the flag (16-mixed's GradScaler already syncs every step to read
    its own inf flag). That is small relative to a forward/backward pass, so the default checks every step.
    `check_every_n_steps > 1` amortizes it, at the cost that a non-finite step between checks is applied unchecked. Such
    a step leaves NaN weights that make every later step non-finite, which the next checks still catch and stop on
    (and the checkpoint check refuses to save), but the run can no longer recover by skipping.
    """

    def __init__(
        self,
        max_consecutive_nonfinite: int = 50,
        min_grad_scale: float = 2**-8,
        check_every_n_steps: int = 1,
        check_checkpoints: bool = True,
        log_diagnostics: bool = True,
    ):
        super().__init__()
        if max_consecutive_nonfinite < 1:
            raise ValueError(f"max_consecutive_nonfinite must be positive, got {max_consecutive_nonfinite}")
        if check_every_n_steps < 1:
            raise ValueError(f"check_every_n_steps must be positive, got {check_every_n_steps}")
        self.max_consecutive_nonfinite = max_consecutive_nonfinite
        self.min_grad_scale = min_grad_scale
        self.check_every_n_steps = check_every_n_steps
        self.check_checkpoints = check_checkpoints
        self.log_diagnostics = log_diagnostics

        self.consecutive_nonfinite = 0
        self.skipped_steps = 0
        self._batch_idx = 0
        self._nonfinite_loss = None  # Device flag accumulated over the micro-batches of the current optimizer step
        self._buffers = None  # Snapshot of the buffers before the current optimizer step's forward(s)

    def _is_check_step(self, trainer) -> bool:
        # `global_step` is the index of the upcoming optimizer step until that step completes, and is the same on every
        # rank, so all ranks check (and so all-reduce) at the same steps
        return trainer.global_step % self.check_every_n_steps == 0

    def _is_log_step(self, trainer) -> bool:
        # Matches `TextLoggingCallback`, so the diagnostics are current when it logs
        return (self._batch_idx + 1) % trainer.log_every_n_steps == 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):  # noqa: ARG002
        self._batch_idx = batch_idx
        if self._buffers is None and self._is_check_step(trainer):
            # Snapshot at the first micro-batch of the step, since the forward pass updates BN running stats
            # before we can decide whether to skip the step
            self._buffers = [(buffer, buffer.detach().clone()) for buffer in pl_module.buffers()]

    def on_before_backward(self, trainer, pl_module, loss):  # noqa: ARG002
        # Accumulate on device (no sync). Under 16-mixed this is the scaled loss, which is finite if the loss is (the
        # scale is finite), and the NaN gradients when the scale is 0 are caught by the gradient check
        if self._is_check_step(trainer):
            nonfinite = ~torch.isfinite(loss.detach()).all()
            self._nonfinite_loss = nonfinite if self._nonfinite_loss is None else self._nonfinite_loss | nonfinite

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Lightning calls this after the GradScaler unscales the gradients and before gradient clipping, at every
        # optimizer step in automatic optimization (even if `training_step` returned None), and so always consumes
        # the per-step state set by the hooks above
        is_check_step = self._is_check_step(trainer)
        is_log_step = self.log_diagnostics and self._is_log_step(trainer)

        nonfinite_loss, self._nonfinite_loss = self._nonfinite_loss, None
        buffers, self._buffers = self._buffers, None

        if self.log_diagnostics:
            pl_module.log("lr", optimizer.param_groups[0]["lr"], on_step=True, on_epoch=False, prog_bar=True)

        if not (is_check_step or is_log_step):
            return

        grads = [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        grad_norm = torch.nn.utils.get_total_norm(grads) if grads else None

        scaler = getattr(trainer.precision_plugin, "scaler", None)
        scale = scaler.get_scale() if scaler is not None and scaler.is_enabled() else None

        if is_log_step:
            if grad_norm is not None:
                pl_module.log("grad_norm", grad_norm.float(), on_step=True, on_epoch=False, prog_bar=True)
            if scale is not None:
                pl_module.log("grad_scale", scale, on_step=True, on_epoch=False, prog_bar=True)

        if not is_check_step:
            return

        step = trainer.global_step
        if scale is not None and scale < self.min_grad_scale:
            raise NumericalInstabilityError(
                f"GradScaler scale collapsed to {scale:g} (< min_grad_scale={self.min_grad_scale:g}) at step {step}, "
                "i.e., many consecutive steps had non-finite gradients"
            )

        device = pl_module.device
        nonfinite = torch.zeros((), dtype=torch.float32, device=device)
        if nonfinite_loss is not None:
            nonfinite = nonfinite + nonfinite_loss.to(device=device, dtype=torch.float32)
        if grad_norm is not None:
            nonfinite = nonfinite + (~torch.isfinite(grad_norm)).to(device=device, dtype=torch.float32)
        # Agree across ranks (a no-op for single-device strategies)
        nonfinite = trainer.strategy.reduce(nonfinite, reduce_op="sum")
        if float(nonfinite) == 0.0:
            self.consecutive_nonfinite = 0
            return

        self.consecutive_nonfinite += 1
        self.skipped_steps += 1
        if self.consecutive_nonfinite >= self.max_consecutive_nonfinite:
            raise NumericalInstabilityError(
                f"{self.consecutive_nonfinite} consecutive (checked) steps had a non-finite loss or gradients, through "
                f"step {step}"
            )

        # Skip the step: optimizers (and the GradScaler) skip parameters whose gradient is None, so neither the
        # weights, their weight decay, nor the optimizer moments change, and gradient clipping ignores them
        optimizer.zero_grad(set_to_none=True)
        if buffers is not None:
            with torch.no_grad():
                for buffer, saved in buffers:
                    buffer.copy_(saved)
        if trainer.is_global_zero:
            logging.warning(
                "Skipping step %d with a non-finite loss or gradients (%d consecutive, %d total)",
                step,
                self.consecutive_nonfinite,
                self.skipped_steps,
            )

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):  # noqa: ARG002
        if self.check_checkpoints and (problem := checkpoint_problem(checkpoint)) is not None:
            raise NumericalInstabilityError(f"Refusing to save a checkpoint at step {trainer.global_step}: {problem}")

    def state_dict(self):
        return {"consecutive_nonfinite": self.consecutive_nonfinite, "skipped_steps": self.skipped_steps}

    def load_state_dict(self, state_dict):
        self.consecutive_nonfinite = state_dict.get("consecutive_nonfinite", 0)
        self.skipped_steps = state_dict.get("skipped_steps", 0)
