import hydra
import lightning as L
import torch
from omegaconf import OmegaConf
from lightning.pytorch.callbacks import TQDMProgressBar
from npsv3.models.transformer import Classifier, ModelAssessmentCallback


def train(cfg, output_dir=None, **kw_args):
    # print("\nmasking scheme:",cfg.model.masking_scheme)
    dm = hydra.utils.instantiate(cfg.data)

    # OmegaConf.update(cfg, "model.patch_size", cfg.data.patch_size, merge=False)

    if cfg.pretrained.path:
        # print(f"\npretrained loaded from {cfg.pretrained.path}")
        model = Classifier.load_from_checkpoint(cfg.pretrained.path, strict=False)
    else: 
        model = hydra.utils.instantiate(cfg.model)
        model = torch.compile(model)

    # Overwrite existing checkpoints, instead of creating new versions
    # print("\ncheckpoint name:",cfg.checkpoint.name)
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath=output_dir, filename=cfg.checkpoint.name, enable_version_counter=False)

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

    profiler = "advanced" if torch.cuda.is_available() else None
    trainer = hydra.utils.instantiate(cfg.trainer, profiler=profiler, callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=50)], limit_val_batches=limit_val_batches, num_sanity_val_steps=num_sanity_val_steps, limit_test_batches=limit_test_batches, **kw_args)

    # TODO: Check if we have reached the final, if not, continue training by setting ckpt_path
    # https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#resume-training-state
    trainer.fit(model=model, datamodule=dm)

    return checkpoint_callback.best_model_path

def assess_accuracy(cfg, ckpt_path, **kw_args):
    dm = hydra.utils.instantiate(cfg.data)
    print("\n",ckpt_path)
    model = Classifier.load_from_checkpoint(ckpt_path, strict=False)
    
    trainer = L.Trainer(
        # I believe "callbacks" means that something runs after each iteration of the trainer
        callbacks=[ModelAssessmentCallback(), TQDMProgressBar(refresh_rate=50)], **kw_args
    )
    trainer.predict(model, dm)