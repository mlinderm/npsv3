import os
import sys

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf

from npsv3.simulation import bwa_index_loaded


def _configure_gpu():
    """
    Configure GPU options (seems to be required for RTX GPUs)
    """
    try:
        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("Memory growth error:", e, file=sys.stderr)
        pass


def _check_shared_reference(cfg: DictConfig):
    """Check if BWA shared index is loaded, loading it if specified configuration"""
    if cfg.simulation.replicates > 0:
        cfg.shared_reference = bwa_index_loaded(hydra.utils.to_absolute_path(cfg.reference), load=cfg.load_reference)
        if not cfg.shared_reference:
            logging.warning(
                "Consider loading BWA indices into shared memory before generating examples with 'bwa shm %s'",
                cfg.reference,
            )


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.command == "images":
        from npsv3.image import vcf_to_tfrecords
        from npsv3.util.sample import Sample

        _check_shared_reference(cfg)

        sample = Sample.from_json(hydra.utils.to_absolute_path(cfg.stats_path))

        # If no output directory is specified, use the Hydra output directory (the current working directory)
        if OmegaConf.is_missing(cfg, "output"):
            output = os.getcwd()
        else:
            output = hydra.utils.to_absolute_path(cfg.output)

        vcf_to_tfrecords(
            cfg,
            hydra.utils.to_absolute_path(cfg.background),
            hydra.utils.to_absolute_path(cfg.input),
            hydra.utils.to_absolute_path(cfg.reads),
            sample,
            output,
            progress_bar=True,
        )
    else:
        msg = f"Command {cfg.command} not implemented"
        raise NotImplementedError(msg)


if __name__ == "__main__":
    main()
