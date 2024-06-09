import logging
import os
import sys

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf

from npsv3.simulation import bwa_index_loaded


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
        from npsv3.images.example import vcf_to_region_examples
        from npsv3.util.sample import Sample

        _check_shared_reference(cfg)

        sample = Sample.from_json(hydra.utils.to_absolute_path(cfg.stats_path))

        # If no output directory is specified, use the Hydra output directory (the current working directory)
        if OmegaConf.is_missing(cfg, "output"):
            output = os.getcwd()
        else:
            output = hydra.utils.to_absolute_path(cfg.output)

        vcf_to_region_examples(
            cfg,
            hydra.utils.to_absolute_path(cfg.reads),
            sample,
            hydra.utils.to_absolute_path(cfg.input),
            output,
            background_vcf=hydra.utils.to_absolute_path(cfg.background),
            progress_bar=True,
        )
    elif cfg.command == "train_vae":
        import torch
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(i).name)

    else:
        msg = f"Command {cfg.command} not implemented"
        raise NotImplementedError(msg)


if __name__ == "__main__":
    main()
