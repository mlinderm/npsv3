import hydra
import lightning as L
from omegaconf import OmegaConf


def train(cfg, output_dir=None, **kw_args):
    dm = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)

    # Overwrite existing checkpoints, instead of creating new versions
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath=output_dir, enable_version_counter=False)

    if cfg.data.validate_urls:
        limit_val_batches = OmegaConf.select(cfg, "data.limit_val_batches", default=1.0)
        num_sanity_val_steps = OmegaConf.select(cfg, "data.num_sanity_val_steps", default=2)
    else:
        # Skip validation if no validation data provided
        limit_val_batches = num_sanity_val_steps = 0

    if cfg.data.test_urls:
        limit_test_batches = OmegaConf.select(cfg, "data.limit_test_batches", default=1.0)
    else:
        # Skip testing if no testing data provided
        limit_test_batches = 0

    profiler = "simple"#L.pytorch.profilers.PyTorchProfiler(profile_memory=True)

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=[checkpoint_callback, L.pytorch.callbacks.TQDMProgressBar(refresh_rate=50)],
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=num_sanity_val_steps,
        limit_test_batches=limit_test_batches,
        profiler=profiler,
        **kw_args,
    )

    # TODO: Check if we have reached the final, if not, continue training by setting ckpt_path
    # https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#resume-training-state
    trainer.fit(model=model, datamodule=dm)

    return checkpoint_callback.best_model_path
