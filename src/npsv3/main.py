import json
import logging
import os
import typing

import hydra
from omegaconf import DictConfig, OmegaConf

from npsv3.simulation import bwa_index_loaded
from npsv3.util.config import setup_resolvers


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


def _to_webdataset_urls(urls: str | typing.Iterable[str]) -> str:
    # Join multiple URLs with "::" separator expected by WebDataset
    list_of_urls = urls.split("::") if isinstance(urls, str) else urls
    list_of_urls = [hydra.utils.to_absolute_path(url) for url in list_of_urls]
    return "::".join(list_of_urls)

# Register resolvers for OmegaConf
setup_resolvers()

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.command == "preprocess":
        from npsv3.util.sample import compute_read_stats

        # If no output file is specified, create a fixed file in the Hydra output directory
        output = "stats.json" if OmegaConf.is_missing(cfg, "output") else hydra.utils.to_absolute_path(cfg.output)

        _make_paths_absolute(cfg, ["reference"])

        stats = compute_read_stats(cfg, hydra.utils.to_absolute_path(cfg.reads))
        with open(output, "w") as file:
            json.dump(stats, file)

    elif cfg.command == "genotype":
        from npsv3.genotype import genotype
        from npsv3.util.sample import Sample

        _make_paths_absolute(cfg, ["reference", "stats_path", "model.checkpoint"])
        _check_shared_reference(cfg)

        sample = Sample.from_json(hydra.utils.to_absolute_path(cfg.stats_path))

        # If no output file is specified, create a fixed file in the Hydra output directory
        output = "genotypes.vcf.gz" if OmegaConf.is_missing(cfg, "output") else hydra.utils.to_absolute_path(cfg.output)

        genotype(
            cfg,
            hydra.utils.to_absolute_path(cfg.reads),
            sample,
            hydra.utils.to_absolute_path(cfg.input),
            output,
            background_vcf=hydra.utils.to_absolute_path(cfg.background),
        )
    elif cfg.command == "images":
        from npsv3.util.sample import Sample

        _make_paths_absolute(cfg, ["reference", "stats_path"])
        _check_shared_reference(cfg)

        sample = Sample.from_json(hydra.utils.to_absolute_path(cfg.stats_path))

        # If no output directory is specified, use the Hydra output directory (the current working directory)
        output = os.getcwd() if OmegaConf.is_missing(cfg, "output") else hydra.utils.to_absolute_path(cfg.output)

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

        from npsv3.models.runners import train

        torch.set_num_threads(cfg.threads)

        # If no output directory is specified, use the Hydra output directory (the current working directory)
        output = os.getcwd() if OmegaConf.is_missing(cfg, "output") else hydra.utils.to_absolute_path(cfg.output)

        _make_paths_absolute(cfg, ["model.encoder.checkpoint_path"])

        OmegaConf.update(cfg, "data.train_urls", _to_webdataset_urls(cfg.data.train_urls), merge=False)
        if not OmegaConf.is_missing(cfg, "data.validate_urls") and OmegaConf.select(cfg, "data.validate_urls") is not None:
            OmegaConf.update(cfg, "data.validate_urls", _to_webdataset_urls(cfg.data.validate_urls), merge=False)

        checkpoint = train(cfg, output_dir=output)
        if not OmegaConf.select(cfg, "trainer.fast_dev_run", default=False):
            # If checkpoints are active, create a symlink to the returned checkpoint, removing existing symlink if it exists
            final_checkpoint = os.path.join(output, "model.ckpt")
            if os.path.exists(final_checkpoint):
                os.unlink(final_checkpoint)
            os.symlink(checkpoint, final_checkpoint)

    elif cfg.command == "test":
        import torch

        from npsv3.models.runners import test

        torch.set_num_threads(cfg.threads)

        # If no output directory is specified, use the Hydra output directory (the current working directory)
        output = os.getcwd() if OmegaConf.is_missing(cfg, "output") else hydra.utils.to_absolute_path(cfg.output)

        _make_paths_absolute(cfg, ["model.checkpoint"])
        OmegaConf.update(cfg, "data.test_urls", _to_webdataset_urls(cfg.data.test_urls), merge=False)

        metrics = test(cfg)
        with open(os.path.join(output, "metrics.json"), "w") as file:
            json.dump(metrics, file, indent=2)

    elif cfg.command == "predict":
        import torch

        from npsv3.models.runners import predict

        torch.set_num_threads(cfg.threads)

        _make_paths_absolute(cfg, ["model.checkpoint"])
        OmegaConf.update(cfg, "data.predict_urls", _to_webdataset_urls(cfg.data.predict_urls), merge=False)

        predict(cfg, return_predictions=False)

    elif cfg.command == "encode":
        import torch

        from npsv3.models.dvae import encode

        torch.set_num_threads(cfg.threads)

        # If no output directory is specified, use the Hydra output directory (the current working directory)
        output = os.getcwd() if OmegaConf.is_missing(cfg, "output") else hydra.utils.to_absolute_path(cfg.output)

        _make_paths_absolute(cfg, ["model.checkpoint"])
        OmegaConf.update(cfg, "data.predict_urls", _to_webdataset_urls(cfg.data.predict_urls), merge=False)

        encode(cfg, output_dir=output)

    elif cfg.command == "split_and_filter":
        from npsv3.images.example import split_and_filter_vcf

        # If no output directory is specified, use the Hydra output directory (the current working directory)
        output = os.getcwd() if OmegaConf.is_missing(cfg, "output") else hydra.utils.to_absolute_path(cfg.output)

        split_and_filter_vcf(
            cfg,
            hydra.utils.to_absolute_path(cfg.input),
            output,
        )
    elif cfg.command == "update_filter":
        from npsv3.images.population import update_filter

        # If no output file is specified, write to stdout (defaulting to vcf)
        if OmegaConf.is_missing(cfg, "output"):
            output = "-" # HTSLib interprets "-" as stdout
            if OmegaConf.select(cfg, "output_format") is None:
                OmegaConf.update(cfg, "output_format", "vcf")
        else:
            output = hydra.utils.to_absolute_path(cfg.output)

        update_filter(
            cfg,
            hydra.utils.to_absolute_path(cfg.input),
            output,
        )
    else:
        msg = f"Command {cfg.command} not implemented"
        raise NotImplementedError(msg)


if __name__ == "__main__":
    main()
