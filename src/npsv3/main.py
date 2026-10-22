import glob
import itertools
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


def _make_path_or_list_absolute(cfg: DictConfig, key: str):
    """Like `_make_paths_absolute`, but `key`'s value may be a single path or a list of paths, e.g.
    `data.train_metadataset_paths` (see `EnergonPackedImageDataModule`)."""
    if OmegaConf.is_missing(cfg, key):
        return
    value = OmegaConf.select(cfg, key)
    if value is None:
        return
    if isinstance(value, str):
        OmegaConf.update(cfg, key, hydra.utils.to_absolute_path(value))
    else:
        OmegaConf.update(cfg, key, [hydra.utils.to_absolute_path(path) for path in value])


def _graph_shards(cfg: DictConfig) -> list[str] | None:
    """Pre-generated graph shards if specified (`graph_shards`), otherwise None to generate them on-the-fly (slower)"""
    if OmegaConf.is_missing(cfg, "graph_shards") or OmegaConf.select(cfg, "graph_shards") is None:
        return None
    # Pre-expand shard patterns using the '::' separator also used by WebDataset, and glob each pattern to find matching files
    graph_shards = list(
        itertools.chain.from_iterable(glob.glob(hydra.utils.to_absolute_path(shard)) for shard in cfg.graph_shards.split("::"))
    )
    graph_shards.sort()  # Sort for deterministic order, since glob order is arbitrary
    return graph_shards


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

        _make_paths_absolute(cfg, ["reference", "reads", "kmc_prefix"])

        stats = compute_read_stats(cfg, cfg.reads, kmc_prefix=OmegaConf.select(cfg, "kmc_prefix"))
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

        _make_paths_absolute(cfg, ["reference", "reads", "input", "stats_path", "unique_kmer_path", "filtered_kmer_path"])
        _check_shared_reference(cfg)

        sample = Sample.from_json(hydra.utils.to_absolute_path(cfg.stats_path))

        # If no output directory is specified, use the Hydra output directory (the current working directory)
        output = os.getcwd() if OmegaConf.is_missing(cfg, "output") else hydra.utils.to_absolute_path(cfg.output)

        # Use precomputed haplotypes (from the "haplotypes" command) if specified, otherwise generate them on-the-fly
        addl_args = {}
        if (haplotypes_dir := OmegaConf.select(cfg, "haplotypes_dir")) is not None:
            addl_args["haplotypes_dir"] = hydra.utils.to_absolute_path(haplotypes_dir)

        vcf_to_examples = hydra.utils.get_method(cfg.pileup.example_fn)
        vcf_to_examples(
            cfg,
            cfg.reads,
            sample,
            cfg.input,
            output,
            progress_bar=True,
            graph_shards=_graph_shards(cfg),
            unique_kmer_path=OmegaConf.select(cfg, "unique_kmer_path"),
            filtered_kmer_path=OmegaConf.select(cfg, "filtered_kmer_path"),
            **addl_args,
        )

    elif cfg.command == "haplotypes":
        # Pre-generate a sample's haplotypes (and comparisons) for subsequent image generation
        from npsv3.images.example import vcf_to_haplotypes
        from npsv3.util.sample import Sample

        _make_paths_absolute(cfg, ["reference", "input", "stats_path", "unique_kmer_path", "filtered_kmer_path"])
        sample = Sample.from_json(cfg.stats_path)

        # If no output directory is specified, use the Hydra output directory (the current working directory)
        output = os.getcwd() if OmegaConf.is_missing(cfg, "output") else hydra.utils.to_absolute_path(cfg.output)

        vcf_to_haplotypes(
            cfg,
            sample,
            cfg.input,
            output,
            progress_bar=True,
            graph_shards=_graph_shards(cfg),
            unique_kmer_path=OmegaConf.select(cfg, "unique_kmer_path"),
            filtered_kmer_path=OmegaConf.select(cfg, "filtered_kmer_path"),
        )

    elif cfg.command == "train":
        import torch
        from npsv3.models.runners import train

        torch.set_num_threads(cfg.threads)

        # If no output directory is specified, use the Hydra output directory (the current working directory)
        output = os.getcwd() if OmegaConf.is_missing(cfg, "output") else hydra.utils.to_absolute_path(cfg.output)

        _make_paths_absolute(cfg, ["model.encoder.checkpoint_path", "data.cache_dir"])

        # The energon data config is addressed by metadataset(s) rather than by WebDataset URLs,
        # either a single pre-combined `metadataset_path` or per-split `*_metadataset_paths`.
        for key in (
            "data.metadataset_path",
            "data.train_metadataset_paths",
            "data.val_metadataset_paths",
            "data.test_metadataset_paths",
        ):
            _make_path_or_list_absolute(cfg, key)
       
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

    elif cfg.command == "loss":
        import torch

        from npsv3.models.runners import predict
        from npsv3.models.paired import RecordLossCallback

        torch.set_num_threads(cfg.threads)

        _make_paths_absolute(cfg, ["model.checkpoint", "data.cache_dir"])
        
        # The energon data config is addressed by metadataset(s) rather than by WebDataset URLs,
        # either a single pre-combined `metadataset_path` or per-split `*_metadataset_paths`.
        for key in (
            "data.metadataset_path",
            "data.train_metadataset_paths",
            "data.val_metadataset_paths",
            "data.test_metadataset_paths",
        ):
            _make_path_or_list_absolute(cfg, key)

        # If no output file is specified, create a fixed file in the Hydra output directory
        output = "losses.pkl.gz" if OmegaConf.is_missing(cfg, "output") else hydra.utils.to_absolute_path(cfg.output)

        predict(cfg, return_predictions=False, callbacks=[RecordLossCallback(output)])

    elif cfg.command == "copositive":
        import torch

        from npsv3.models.transforms import copositive_transform

        torch.set_num_threads(cfg.threads)

        # input is a per-split metadataset path
        _make_paths_absolute(cfg, ["input", "model.checkpoint", "data.cache_dir"])

        # If no output directory is specified, use the Hydra output directory (the current working directory)
        output = os.getcwd() if OmegaConf.is_missing(cfg, "output") else hydra.utils.to_absolute_path(cfg.output)

        copositive_transform(cfg, cfg.input, output)

    elif cfg.command == "compare":
        from npsv3.images.sequence_compare import add_comparisons

        _make_paths_absolute(cfg, ["input"])
        add_comparisons(cfg, cfg.input)

    elif cfg.command == "encode":
        import torch

        from npsv3.models.dvae import encode

        torch.set_num_threads(cfg.threads)

        # If no output directory is specified, use the Hydra output directory (the current working directory)
        output = os.getcwd() if OmegaConf.is_missing(cfg, "output") else hydra.utils.to_absolute_path(cfg.output)

        _make_paths_absolute(cfg, ["model.checkpoint"])
        OmegaConf.update(cfg, "data.predict_urls", _to_webdataset_urls(cfg.data.predict_urls), merge=False)

        encode(cfg, output_dir=output)

    elif cfg.command == "update_filter":
        from npsv3.util.variant import update_filter

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
    elif cfg.command == "graphs":
        # Pre-generate graphs for a given VCF (since it doesn't depend on any particular sample)
        from npsv3.graphs.haplotype import serialize_graph_and_unique_kmers

        _make_paths_absolute(cfg, ["reference", "input", "graph.ref_kmer_counts_kmc_prefix"])
         # If no output directory is specified, use the Hydra output directory (the current working directory)
        output = os.getcwd() if OmegaConf.is_missing(cfg, "output") else hydra.utils.to_absolute_path(cfg.output)

        serialize_graph_and_unique_kmers(
            cfg,
            cfg.input,
            output,
            pool_kmers=True,
            ref_kmer_counts_path=OmegaConf.select(cfg, "graph.ref_kmer_counts_kmc_prefix"),
            progress_bar=True
       )

    elif cfg.command == "diplotype_topk":
        from npsv3.graphs.haplotype import diplotypes_in_topk
        from npsv3.util.sample import Sample

        _make_paths_absolute(cfg, ["reference", "stats_path", "input", "unique_kmer_path", "filtered_kmer_path"])
        # If no output file is specified, create a fixed file in the Hydra output directory
        output = "topk_statistics.pkl.gz" if OmegaConf.is_missing(cfg, "output") else hydra.utils.to_absolute_path(cfg.output)

        sample = Sample.from_json(cfg.stats_path)
        statistics = diplotypes_in_topk(
            cfg,
            cfg.input,
            sample,
            progress_bar=True,
            graph_shards=_graph_shards(cfg),
            unique_kmer_path=OmegaConf.select(cfg, "unique_kmer_path"),
            filtered_kmer_path=OmegaConf.select(cfg, "filtered_kmer_path"),
        )
        statistics.to_pickle(output)

    else:
        msg = f"Command {cfg.command} not implemented"
        raise NotImplementedError(msg)


if __name__ == "__main__":
    main()
