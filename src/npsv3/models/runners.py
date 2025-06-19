import hydra
import lightning as L
import torch
from omegaconf import OmegaConf
from lightning.pytorch.callbacks import TQDMProgressBar, DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from npsv3.models.transformer import Classifier, LabelsToWebDatasetCallback


def train(cfg, output_dir=None, **kw_args):
    dm = hydra.utils.instantiate(cfg.data)

    if cfg.pretrained.path:
        # print("pretrained loaded")
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

    full_profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
        schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=20)
    )
    profiler = "advanced" if torch.cuda.is_available() else None
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=50), DeviceStatsMonitor()], limit_val_batches=limit_val_batches, num_sanity_val_steps=num_sanity_val_steps, limit_test_batches=limit_test_batches, profiler=profiler, **kw_args)

    # TODO: Check if we have reached the final, if not, continue training by setting ckpt_path
    # https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#resume-training-state
    trainer.fit(model=model, datamodule=dm)

    return checkpoint_callback.best_model_path

def assess_accuracy(cfg, ckpt_path, **kw_args):
    dm = hydra.utils.instantiate(cfg.data)
    data_path = cfg.data.predict_urls
    model = Classifier.load_from_checkpoint(ckpt_path, strict=False)
    
    trainer = L.Trainer(
        # I believe "callbacks" means that something runs after each iteration of the trainer
        callbacks=[LabelsToWebDatasetCallback(data_path), TQDMProgressBar(refresh_rate=50)], **kw_args
    )
    trainer.predict(model, dm)