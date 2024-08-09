import json
import logging
import os
import sys
import typing

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


def _make_paths_absolute(cfg: DictConfig, keys: typing.Iterable[str]):
    """Make list of hydra configuration keys, e.g. 'pileup.snv_vcf_input' absolute paths"""
    for key in keys:
        if not OmegaConf.is_missing(cfg, key) and OmegaConf.select(cfg, key) is not None:
            OmegaConf.update(cfg, key, hydra.utils.to_absolute_path(OmegaConf.select(cfg, key)))


def _to_webdataset_urls(urls: typing.Union[str, typing.Iterable[str]]) -> str:
    # Join multiple URLs with "::" separator expected by WebDataset
    list_of_urls = urls.split("::") if isinstance(urls, str) else urls
    list_of_urls = [hydra.utils.to_absolute_path(url) for url in list_of_urls]
    return "::".join(list_of_urls)
    

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.command == "preprocess":
        from npsv3.util.sample import compute_read_stats

        # If no output file is specified, create a fixed file in the Hydra output directory
        if OmegaConf.is_missing(cfg, "output"):
            output = "stats.json"
        else:
            output = hydra.utils.to_absolute_path(cfg.output)

        _make_paths_absolute(cfg, ["reference"])

        stats = compute_read_stats(cfg, hydra.utils.to_absolute_path(cfg.reads))
        with open(output, "w") as file:
            json.dump(stats, file)

    elif cfg.command == "images":
        from npsv3.images.example import vcf_to_region_examples
        from npsv3.util.sample import Sample

        _check_shared_reference(cfg)

        sample = Sample.from_json(hydra.utils.to_absolute_path(cfg.stats_path))

        # If no output directory is specified, use the Hydra output directory (the current working directory)
        if OmegaConf.is_missing(cfg, "output"):
            output = os.getcwd()
        else:
            output = hydra.utils.to_absolute_path(cfg.output)

        vcf_to_examples = hydra.utils.get_method(cfg.pileup.example_fn)
        vcf_to_examples(
            cfg,
            hydra.utils.to_absolute_path(cfg.reads),
            sample,
            hydra.utils.to_absolute_path(cfg.input),
            output,
            background_vcf=hydra.utils.to_absolute_path(cfg.background),
            progress_bar=True,
        )
    elif cfg.command == "train":
        import torch
        from npsv3.models.paired import train

        torch.set_num_threads(cfg.threads)

        # If no output directory is specified, use the Hydra output directory (the current working directory)
        if OmegaConf.is_missing(cfg, "output"):
            output = os.getcwd()
        else:
            output = hydra.utils.to_absolute_path(cfg.output)

        OmegaConf.update(cfg, "data.training_urls", _to_webdataset_urls(cfg.data.training_urls), merge=False)
        if not OmegaConf.is_missing(cfg, "data.validation_urls"):
             OmegaConf.update(cfg, "data.validation_urls", _to_webdataset_urls(cfg.data.validation_urls), merge=False)

        train(cfg, output_dir=output)
        # TODO: Create link to the best model to serve as the final model
    else:
        msg = f"Command {cfg.command} not implemented"
        raise NotImplementedError(msg)


if __name__ == "__main__":
    main()
