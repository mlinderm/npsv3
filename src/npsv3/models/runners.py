import glob
import logging
import os

import hydra
import lightning as L
import torch
from omegaconf import ListConfig, OmegaConf

from npsv3.models.energon import available_splits, estimate_epoch_batches
from npsv3.models.numerical_guard import NumericalGuard, checkpoint_problem


class TextLoggingCallback(L.pytorch.callbacks.Callback):
    """Log current metrics as one plain-text line every `trainer.log_every_n_steps` steps.

    `TQDMProgressBar` rewrites its line in place using `\\r`, which a terminal renders as a single
    updating line but when written to a file (the common case) produces multiple lines.
    """

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # noqa: ARG002
        if not trainer.is_global_zero or (batch_idx + 1) % trainer.log_every_n_steps != 0:
            return
        metrics = ", ".join(f"{name}={value:.4g}" for name, value in trainer.progress_bar_metrics.items())
        logging.info(
            "Epoch %d, step %d/%d: %s", trainer.current_epoch, batch_idx + 1, trainer.num_training_batches, metrics
        )

    def on_validation_epoch_end(self, trainer, pl_module):  # noqa: ARG002
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        metrics = ", ".join(
            f"{name}={value:.4g}" for name, value in trainer.callback_metrics.items() if name.startswith("val_")
        )
        logging.info("Epoch %d validation: %s", trainer.current_epoch, metrics)

    def on_test_epoch_end(self, trainer, pl_module):  # noqa: ARG002
        if not trainer.is_global_zero:
            return
        metrics = ", ".join(
            f"{name}={value:.4g}" for name, value in trainer.callback_metrics.items() if name.startswith("test_")
        )
        logging.info("Test: %s", metrics)


def _resolve_world_size(cfg) -> int:
    """Return a best-effort DDP world size from `cfg.trainer`, for scaling a `limit_train_batches` estimate.

    Not authoritative, since Lightning only resolves the real world size once the `Trainer`/strategy are built, but good
    enough to scale an *estimate*. `devices` left unset (Lightning's default is `"auto"`) or `"auto"` falls back to the
    visible GPU count, matching Lightning's own "auto" resolution for the GPU accelerator.
    """
    devices = OmegaConf.select(cfg, "trainer.devices", default="auto")
    num_nodes = OmegaConf.select(cfg, "trainer.num_nodes", default=1)
    if devices is None or devices == "auto":
        devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    elif isinstance(devices, (list, ListConfig)):
        devices = len(devices)
    return max(1, int(devices) * int(num_nodes))


def load_model_from_checkpoint(cfg, *, strict=True, weights_only=None):
    # Load the model from the checkpoint, instantiating any "child" objects that were not saved as part of the checkpoint
    model_cls = hydra.utils.get_class(cfg.model._target_)

    model_args = {}
    for key, value in cfg.model.items():
        if key in model_cls.ignored_hyperparameters and "_target_" in value:
            model_args[key] = hydra.utils.instantiate(value)

    return model_cls.load_from_checkpoint(
        cfg.model.checkpoint,
        strict=strict,
        **model_args,
        weights_only=weights_only,
    )

def _resume_checkpoint_problem(path) -> str | None:
    """Describe why the checkpoint file at `path` can't be resumed from (e.g., it is unreadable, its GradScaler scale is
    0 or it has non-finite weights), or return None if it can."""
    try:
        try:
            # Memory-map to avoid reading all of a large checkpoint up front (requires the default zipfile format)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except RuntimeError:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        return f"it could not be loaded ({type(e).__name__}: {e})"
    if not isinstance(checkpoint, dict):
        return "it is not a Lightning checkpoint"
    return checkpoint_problem(checkpoint)


def _resolve_resume_checkpoint(resume_from, output_dir) -> str | None:
    """Return the checkpoint to resume training from (as `ckpt_path`) for `resume_from`.

    `None` defers to Lightning, which resumes from the newest `hpc_ckpt_N.ckpt` in `default_root_dir` when running under
    SLURM. Otherwise resume from the most recently modified usable checkpoint among `resume_from` (unless it is "last")
    and the full checkpoints in `output_dir`, i.e., `ModelCheckpoint` and requeue (`hpc_ckpt_N.ckpt`) checkpoints.
    Including `output_dir` means a requeued job, which re-runs the same command, continues from where it was interrupted
    rather than from `resume_from` again. Returns `None` (train from scratch) if "last" finds no checkpoints.

    Candidates are loaded newest first, and skipped (with a warning) if unusable, e.g., unreadable or numerically
    collapsed (GradScaler scale of 0 or non-finite weights/optimizer state), until the first usable one. Raises if there
    are candidates but none are usable, rather than silently training from scratch.
    """
    if resume_from is None:
        return None
    if resume_from != "last" and not os.path.exists(resume_from):
        raise FileNotFoundError(f"resume_from checkpoint {resume_from} does not exist")

    # Skip symlinks, e.g., `model.ckpt`, which point to (and so would duplicate) another checkpoint
    candidates = [path for path in glob.glob(os.path.join(output_dir, "*.ckpt")) if not os.path.islink(path)]
    if resume_from != "last":
        candidates.append(resume_from)
    if not candidates:
        logging.warning("No checkpoints found in %s to resume from, training from scratch", output_dir)
        return None
    for path in sorted(candidates, key=os.path.getmtime, reverse=True):
        if (problem := _resume_checkpoint_problem(path)) is None:
            return path
        logging.warning("Skipping checkpoint %s, which can't be resumed from since %s", path, problem)
    raise RuntimeError(f"None of the checkpoints to resume from are usable: {', '.join(sorted(candidates))}")


def train(cfg, output_dir=None, **kw_args):
    """Train a Lightning model from scratch or resume from a checkpoint 

    Args:
        cfg: The configuration for the training.
        output_dir (_type_, optional): The directory to save the training output. Defaults to None.

    Returns:
        The path to the saved checkpoint.
    """
    # Reduce precision to enable use of GPU tensor cores
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # `model.checkpoint` (fine-tuning) loads only the weights, while `resume_from` restores the full training state,
    # including the weights, and so would overwrite them
    resume_from = OmegaConf.select(cfg, "resume_from", default=None)
    if resume_from is not None and OmegaConf.select(cfg, "model.checkpoint", default=None) is not None:
        raise ValueError("Specify only one of model.checkpoint (fine-tuning) or resume_from (resume training)")
    if resume_from is not None and output_dir is None:
        raise ValueError("resume_from requires an output directory")
    if resume_from is not None and resume_from != "last":
        resume_from = hydra.utils.to_absolute_path(resume_from)

    dm = hydra.utils.instantiate(cfg.data)

    if OmegaConf.select(cfg, "model.checkpoint", default=None) is None:
        model = hydra.utils.instantiate(cfg.model)
    else:
        model = load_model_from_checkpoint(cfg, strict=False)

    # Compile the model (if requested) to attempt to speed up training
    if cfg.torch_compile:
        model = torch.compile(model)

    # Create callback for saving checkpoints in the specified directory
    checkpoint_callback = hydra.utils.instantiate(
        cfg.checkpoint_callback,
        dirpath=output_dir,
    )

    # The energon data config is addressed by metadataset(s), either a single pre-combined `metadataset_path`
    # with its own real train/val/test splits, or per-split `*_metadataset_paths` lists.
    combined_metadataset_path = OmegaConf.select(cfg, "data.metadataset_path", default=None)
    train_metadataset_paths = OmegaConf.select(cfg, "data.train_metadataset_paths", default=None)

    if combined_metadataset_path is not None:
        splits = available_splits(combined_metadataset_path)
        has_val_metadataset = "val" in splits
        has_test_metadataset = "test" in splits
    elif train_metadataset_paths is not None:
        has_val_metadataset = OmegaConf.select(cfg, "data.val_metadataset_paths", default=None) is not None
        has_test_metadataset = OmegaConf.select(cfg, "data.test_metadataset_paths", default=None) is not None
    else:
        raise ValueError("Either `data.metadataset_path` or `data.train_metadataset_paths` must be specified in the config.")

    if has_val_metadataset:
        limit_val_batches = OmegaConf.select(cfg, "data.limit_val_batches", default=1.0)
        num_sanity_val_steps = OmegaConf.select(cfg, "data.num_sanity_val_steps", default=2)
    else:
        # Skip validation if no validation data provided
        limit_val_batches = num_sanity_val_steps = 0

    # Skip testing if no testing data provided
    limit_test_batches = (
        OmegaConf.select(cfg, "data.limit_test_batches", default=1.0) if has_test_metadataset else 0
    )

    # Energon train datasets repeat indefinitely so a virtual epoch length is mandatory. `data.limit_train_batches`
    # is used if set, otherwise estimate one true pass from the dataset's metadata jsonl shards (excluding any
    # filtered examples), scaled by the (best-effort) DDP world size so every rank gets the same count and none reaches
    # the all-reduce an unequal number of times.
    limit_train_batches = OmegaConf.select(cfg, "data.limit_train_batches", default=None)
    if limit_train_batches is None:
        limit_train_batches = estimate_epoch_batches(
            combined_metadataset_path if combined_metadataset_path is not None else train_metadataset_paths,
            budget=cfg.data.batch_size,
            w_bucket=OmegaConf.select(cfg, "data.w_bucket", default=None),
            w_max=OmegaConf.select(cfg, "data.w_max", default=None),
            g_max=OmegaConf.select(cfg, "data.g_max", default=None),
            r_max=OmegaConf.select(cfg, "data.r_max", default=None),
            world_size=_resolve_world_size(cfg),
            filters=getattr(dm, "train_filters", ()),
            genotype_subset=getattr(dm, "train_genotype_subset", None),
            occupancy=OmegaConf.select(cfg, "data.estimated_occupancy", default=0.95),
        )

    # Lightning saves `hpc_ckpt_N.ckpt` checkpoints on the SLURM requeue signal to, and resumes from them in,
    # `default_root_dir`, which defaults to the current directory. Use the stable output directory instead, since the
    # current directory can differ between runs (e.g., timestamped Hydra directories).
    if output_dir is not None:
        kw_args = {"default_root_dir": output_dir, **kw_args}

    # Text logging in place of a progress bar, which redraws via "\r" and so leaves duplicate lines
    # once stdout is redirected to a file (see `TextLoggingCallback`)
    callbacks = [checkpoint_callback, TextLoggingCallback()]
    # Skip non-finite steps, stop on numerical collapse and refuse to save non-finite checkpoints (enabled unless
    # `numerical_guard.enabled=false`)
    guard_cfg = OmegaConf.select(cfg, "numerical_guard", default=None)
    guard_cfg = {} if guard_cfg is None else OmegaConf.to_container(guard_cfg, resolve=True)
    if guard_cfg.pop("enabled", True):
        callbacks.append(NumericalGuard(**guard_cfg))

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        enable_progress_bar=False,
        callbacks=callbacks,
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=num_sanity_val_steps,
        limit_test_batches=limit_test_batches,
        limit_train_batches=limit_train_batches,
        **kw_args,
    )

    ckpt_path = _resolve_resume_checkpoint(resume_from, output_dir)
    if ckpt_path is not None:
        logging.info("Resuming training from %s", ckpt_path)
    # Our own checkpoints include non-tensor state (e.g., the energon loader position) that `weights_only` rejects. This
    # also applies to Lightning's automatic resume from `hpc_ckpt_N.ckpt` when `ckpt_path` is None.
    trainer.fit(model=model, datamodule=dm, ckpt_path=ckpt_path, weights_only=False)

    return checkpoint_callback.best_model_path

def test(cfg, **kw_args):
    # Reduce precision to enable use of GPU tensor cores
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    dm = hydra.utils.instantiate(cfg.data)
    # Our own checkpoints include non-tensor state (e.g., the energon loader position) that `weights_only` rejects
    model = load_model_from_checkpoint(cfg, weights_only=False)

    trainer_args = {
        "enable_progress_bar": False,
        "callbacks": [TextLoggingCallback()],
        **kw_args,
    }

    trainer = hydra.utils.instantiate(cfg.trainer, **trainer_args)
    return trainer.test(model=model, datamodule=dm)

def predict(cfg, return_predictions=None, **kw_args):
    # Reduce precision to enable use of GPU tensor cores
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    dm = hydra.utils.instantiate(cfg.data)
    model = load_model_from_checkpoint(cfg, weights_only=False)

    trainer_args = {
        "enable_progress_bar": False,
        **kw_args,
    }

    trainer = hydra.utils.instantiate(cfg.trainer, **trainer_args)
    return trainer.predict(model=model, datamodule=dm, return_predictions=return_predictions)
