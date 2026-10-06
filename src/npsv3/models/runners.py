import logging

import hydra
import lightning as L
import torch
from omegaconf import ListConfig, OmegaConf

from npsv3.models.energon import available_splits, estimate_epoch_batches


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
    enough to scale an *estimate*. `devices` left unset or `"auto"` falls back to the visible GPU count, matching
    Lightning's own "auto" resolution for the GPU accelerator.
    """
    devices = OmegaConf.select(cfg, "trainer.devices", default=1)
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
    # is used if set, otherwise estimate one true pass from the dataset's metadata jsonl shards, scaled by the
    # (best-effort) DDP world size so every rank gets the same count and none reaches the all-reduce an unequal number
    # of times.
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
        )

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        # Text logging in place of a progress bar, which redraws via "\r" and so leaves duplicate lines
        # once stdout is redirected to a file (see `TextLoggingCallback`)
        enable_progress_bar=False,
        callbacks=[checkpoint_callback, TextLoggingCallback()],
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=num_sanity_val_steps,
        limit_test_batches=limit_test_batches,
        limit_train_batches=limit_train_batches,
        **kw_args,
    )

    # TODO: Check if we have reached the final, if not, continue training by setting ckpt_path
    # https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#resume-training-state
    trainer.fit(model=model, datamodule=dm)

    return checkpoint_callback.best_model_path

def test(cfg, **kw_args):
    # Reduce precision to enable use of GPU tensor cores
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    dm = hydra.utils.instantiate(cfg.data)
    model = load_model_from_checkpoint(cfg)

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
