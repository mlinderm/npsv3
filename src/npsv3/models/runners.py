import hydra
import lightning as L
import torch
from omegaconf import OmegaConf


def train(cfg, output_dir=None, **kw_args):
    # Reduce precision to enable use of GPU tensor cores
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    dm = hydra.utils.instantiate(cfg.data)

    if cfg.model.checkpoint is None:
        model = hydra.utils.instantiate(cfg.model)
    else:
        # Load the model from the checkpoint, instantiating any "child" objects that were not saved as part of the checkpoint
        model_cls = hydra.utils.get_class(cfg.model._target_)

        model_args = {}
        for key, value in cfg.model.items():
            if key in model_cls.ignored_hyperparameters and "_target_" in value:
                model_args[key] = hydra.utils.instantiate(value)

        model = model_cls.load_from_checkpoint(
            cfg.model.checkpoint,
            strict=False,
            **model_args,
        )

    # Make model compilation optional
    #model = torch.compile(model)

    # Create callback for saving checkpoints in the specified directory
    checkpoint_callback = hydra.utils.instantiate(
        cfg.checkpoint_callback,
        dirpath=output_dir,
    )

    if cfg.data.validate_urls:
        limit_val_batches = OmegaConf.select(cfg, "data.limit_val_batches", default=1.0)
        num_sanity_val_steps = OmegaConf.select(cfg, "data.num_sanity_val_steps", default=2)
    else:
        # Skip validation if no validation data provided
        limit_val_batches = num_sanity_val_steps = 0

    # Skip testing if no testing data provided
    limit_test_batches = OmegaConf.select(cfg, "data.limit_test_batches", default=1.0) if cfg.data.test_urls else 0

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        # Decrease TQDM progress refresh rate to mitigate performance issues
        callbacks=[checkpoint_callback, L.pytorch.callbacks.TQDMProgressBar(refresh_rate=50)],
        limit_val_batches=limit_val_batches,
        num_sanity_val_steps=num_sanity_val_steps,
        limit_test_batches=limit_test_batches,
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

    # Load the model from the checkpoint, instantiating any "child" objects that were not saved as part of the checkpoint
    model_cls = hydra.utils.get_class(cfg.model._target_)

    model_args = {}
    for key, value in cfg.model.items():
        if key in model_cls.ignored_hyperparameters and "_target_" in value:
            model_args[key] = hydra.utils.instantiate(value)

    model = model_cls.load_from_checkpoint(
        cfg.model.checkpoint,
        callbacks=[L.pytorch.callbacks.TQDMProgressBar(refresh_rate=50)],
        **model_args,
    )

    trainer = hydra.utils.instantiate(cfg.trainer, **kw_args)
    return trainer.test(model=model, datamodule=dm)
