"""Build the graphs and generate the images for genotyping a sample (stages 2 and 3 of the genotyper).

Genotyping reuses training-data generation (`serialize_graph_and_unique_kmers` and `vcf_to_graph_examples`) so the images
can't drift from those the model was trained with, with a few differences ("genotyping mode", see `genotyping_config`):

- The graphs are built for exactly the regions to genotype (`npsv3.genotyping.inputs.genotyping_regions`).
- The sample is not in the VCF (see `npsv3.genotyping.inputs`), so every example is unlabeled, and settings that only make
  sense for training (skipping unlabeled examples, downsampling, genotype subsets, population top-up, ...) are disabled.
- Every example's haplotype record includes the allele of *every* VCF record in the region for each candidate haplotype
  (`variant_ids`/`variant_alleles`), so the VCF writer never needs to reload the graphs.
- Every region without an example is recorded in the `failures-*.jsonl` sidecar files (`npsv3.images.example.FailureWriter`),
  so every region is accounted for in the output VCF. Regions with only a single haplotype record its haplotype record.
- The settings that determine the images are recorded in `generation.yaml`, to check against the model's configuration.
"""

import contextlib
import copy
import dataclasses
import logging
import os
from collections.abc import Sequence

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from npsv3.genotyping.inputs import GenotypingInputs
from npsv3.graphs.haplotype import serialize_graph_and_unique_kmers
from npsv3.images.example import _init_ray, read_failures, vcf_to_graph_examples
from npsv3.images.haplotype_tree import FILTERED_KMERS
from npsv3.images.haplotypes import SampledHaplotypeStrategy
from npsv3.types import PathType
from npsv3.util.range import Range
from npsv3.util.sample import Sample, kmc_filter

GRAPHS_DIR = "graphs_ds"
"""Graph shards for the genotyped regions, in the working directory"""

UNIQUE_KMERS = "unique_kmers"
"""The pooled unique k-mers of the graphs, in the working directory"""

IMAGES_DIR = "images"
"""The energon image dataset (and failures sidecar files), in the working directory"""

GENERATION_FILE = "generation.yaml"
"""The settings the images were generated with, in the image dataset's directory"""

FORCED_SETTINGS = {
    "skip_unlabeled": False,
    "hom_ref_downsample.enabled": False,
    "genotype_subset.enabled": False,
    "compare.enabled": False,
    "haplotype.population_haplotype_topup": False,
    "haplotype.add_true_haplotypes": False,
}
"""Settings that only make sense for training, and so are forced for genotyping (see `genotyping_config`)"""


def genotyping_config(cfg: DictConfig, sample_name: str) -> DictConfig:
    """Return a copy of `cfg` with the image generation settings forced for genotyping `sample_name`.

    - Unlabeled examples are kept (every example is unlabeled), and hom. ref. downsampling, genotype subsets and
      comparisons with the label are disabled.
    - Population top-up is disabled, since it adds the population's haplotypes rather than ones supported by the sample.
    - The sample's own ("true") haplotypes are never added to its candidate haplotypes, since its genotypes may still be
      in the genotyping VCF (see `npsv3.genotyping.inputs.prepare_genotyping_vcf`).
    - The sample is added to `graph.population_excluded_samples` (if not already), so its genotypes, if still in the
      genotyping VCF, don't inform the population prior.

    Raises:
        ValueError: If `haplotype.strategy` isn't a `SampledHaplotypeStrategy`, e.g., `TrueHaplotypeStrategy` requires the
            sample's own ("true") haplotypes
    """
    strategy = hydra.utils.get_class(cfg.haplotype.strategy._target_)
    if not issubclass(strategy, SampledHaplotypeStrategy):
        msg = f"Genotyping requires a SampledHaplotypeStrategy (haplotype.strategy), not {cfg.haplotype.strategy._target_}"
        raise ValueError(msg)

    cfg = copy.deepcopy(cfg)
    for key, value in FORCED_SETTINGS.items():
        if (current := OmegaConf.select(cfg, key, default=None)) not in (None, value):
            logging.info("Setting %s=%s (was %s) for genotyping", key, value, current)
        OmegaConf.update(cfg, key, value, force_add=True)

    excluded = list(OmegaConf.select(cfg, "graph.population_excluded_samples", default=None) or [])
    if sample_name not in excluded:
        OmegaConf.update(cfg, "graph.population_excluded_samples", [*excluded, sample_name], force_add=True)
    return cfg


def _container(cfg: DictConfig, key: str):
    node = OmegaConf.select(cfg, key, default=None)
    return OmegaConf.to_container(node, resolve=True) if isinstance(node, DictConfig) else node


def generation_settings(cfg: DictConfig, *, ploidy: int, min_variant_size: int) -> dict:
    """The settings that determine the images (and their haplotypes), as plain YAML-serializable data.

    `image` summarizes the geometry the model must agree with: `channels` (the model's `encoder.num_channels`), `height`,
    `width` and `replicates` (the model's `r`).
    """
    settings = {
        "ploidy": ploidy,
        "min_variant_size": min_variant_size,
        "image": {
            "channels": len(cfg.pileup.image_channels),
            "height": cfg.pileup.image_height,
            "width": cfg.pileup.image_width,
            "replicates": cfg.simulation.replicates,
        },
        "pileup": _container(cfg, "pileup"),
        "haplotype": _container(cfg, "haplotype"),
        "simulation": _container(cfg, "simulation"),
        "graph": _container(cfg, "graph"),
    }
    # Round-trip so settings are compared exactly as they would be after being recorded
    return yaml.safe_load(yaml.safe_dump(settings))


def write_generation(directory: PathType, settings: dict) -> None:
    with open(os.path.join(directory, GENERATION_FILE), "w") as file:
        yaml.safe_dump(settings, file, sort_keys=True)


def read_generation(directory: PathType) -> dict:
    with open(os.path.join(directory, GENERATION_FILE)) as file:
        return yaml.safe_load(file)


@dataclasses.dataclass
class GenotypingImages:
    """The outputs of image generation for genotyping"""

    dataset_path: str | None
    """The image dataset's `dataset.yaml`, None if there are no examples (e.g., no regions, or every region failed)"""
    images_dir: str
    """The image dataset's directory, with the `failures-*.jsonl` sidecar files and `generation.yaml`"""
    graph_shards: list[str]
    """The graph shards (built or provided)"""
    unique_kmer_path: str | None
    """The graphs' pooled unique k-mers, if built (None if the graphs or filtered k-mers were provided)"""
    filtered_kmer_path: str | None
    """The sample's k-mers filtered to the unique k-mers (built or provided)"""
    counts: dict[str, int]
    """The counts of regions by outcome (see `GraphWriter.counts`)"""
    failures: list[dict]
    """The regions without examples (see `npsv3.images.example.FailureWriter`)"""


def generate_genotyping_images(  # noqa: PLR0913
    cfg: DictConfig,
    read_path: PathType,
    sample: Sample,
    inputs: GenotypingInputs,
    regions: Sequence[Range],
    work_dir: PathType,
    *,
    graph_shards: Sequence[PathType] | None = None,
    unique_kmer_path: PathType | None = None,
    filtered_kmer_path: PathType | None = None,
    images_dir: PathType | None = None,
    ploidy: int = 2,
    min_variant_size: int = 50,
    progress_bar: bool = False,
) -> GenotypingImages:
    """Build the graphs for `regions` (unless `graph_shards` are provided) and generate the images for genotyping `sample`.

    Ray is initialized (if not already) only for the duration of this function, and so is shut down before inference
    (so the two don't compete for memory).

    Args:
        cfg: Hydra config, the genotyping-mode settings are applied (see `genotyping_config`)
        read_path: The sample's (preprocessed) reads
        sample: The sample, with its KMC k-mer database (unless `filtered_kmer_path` is provided)
        inputs: The prepared genotyping inputs (see `npsv3.genotyping.inputs.prepare_genotyping_vcf`)
        regions: The regions to genotype (see `npsv3.genotyping.inputs.genotyping_regions`)
        work_dir: The working directory, for the graphs (`GRAPHS_DIR`), unique and filtered k-mers and (by default) images
        graph_shards: Previously built graph shards for `regions` (built from `inputs.vcf_path`)
        unique_kmer_path: The unique k-mers for `graph_shards` (to filter the sample's k-mers)
        filtered_kmer_path: The sample's k-mers filtered to the unique k-mers of `graph_shards`
        images_dir: The image dataset's directory, `work_dir/IMAGES_DIR` by default
        ploidy: The sample's ploidy
        min_variant_size: Minimum size of the analysis variants
        progress_bar: Show progress bars

    Returns:
        GenotypingImages: The image dataset and associated outputs
    """
    cfg = genotyping_config(cfg, sample.name)
    work_dir = os.fspath(work_dir)
    images_dir = os.path.join(work_dir, IMAGES_DIR) if images_dir is None else os.fspath(images_dir)
    os.makedirs(images_dir, exist_ok=True)
    settings = generation_settings(cfg, ploidy=ploidy, min_variant_size=min_variant_size)

    with contextlib.ExitStack() as stack:
        _init_ray(cfg, stack)

        # Stage 2: Build the graphs for exactly the regions to genotype, pooling their unique k-mers to filter the sample's
        # k-mers (unless already available)
        if graph_shards is None:
            graphs_dir = os.path.join(work_dir, GRAPHS_DIR)
            os.makedirs(graphs_dir, exist_ok=True)
            logging.info("Building graphs for %d regions in %s", len(regions), graphs_dir)
            graph_shards, unique_kmer_path, _region_count = serialize_graph_and_unique_kmers(
                cfg,
                inputs.vcf_path,
                graphs_dir,
                min_variant_size=min_variant_size,
                pool_kmers=filtered_kmer_path is None and unique_kmer_path is None,
                ref_kmer_counts_path=cfg.graph.ref_kmer_counts_kmc_prefix,
                regions=regions,
                progress_bar=progress_bar,
                unique_kmer_dir=work_dir,
            )
        graph_shards = [os.fspath(shard) for shard in graph_shards]

        if not graph_shards:
            logging.warning("No regions to genotype for sample %s", sample.name)
            write_generation(images_dir, settings)
            return GenotypingImages(
                dataset_path=None, images_dir=images_dir, graph_shards=[], unique_kmer_path=None,
                filtered_kmer_path=None if filtered_kmer_path is None else os.fspath(filtered_kmer_path), counts={},
                failures=[],
            )

        if filtered_kmer_path is None:
            if unique_kmer_path is None:
                msg = "The unique k-mers (or filtered k-mers) are required with previously built graph shards"
                raise ValueError(msg)
            if sample.kmc_prefix is None:
                msg = f"Sample {sample.name} has no KMC database, so its k-mers can't be filtered"
                raise ValueError(msg)
            filtered_kmer_path = os.path.join(work_dir, FILTERED_KMERS)
            kmc_filter(sample.kmc_prefix, unique_kmer_path, filtered_kmer_path, threads=cfg.threads)

        # Stage 3: Generate the (unlabeled) images, recording every region's variants' alleles and any failures
        dataset_path, counts = vcf_to_graph_examples(
            cfg,
            os.fspath(read_path),
            sample,
            inputs.vcf_path,
            images_dir,
            progress_bar=progress_bar,
            ploidy=ploidy,
            graph_shards=graph_shards,
            filtered_kmer_path=filtered_kmer_path,
            min_variant_size=min_variant_size,
            variant_alleles=True,
            record_failures=True,
            allow_empty=True,
        )
        write_generation(images_dir, settings)

    return GenotypingImages(
        dataset_path=dataset_path,
        images_dir=images_dir,
        graph_shards=graph_shards,
        unique_kmer_path=None if unique_kmer_path is None else os.fspath(unique_kmer_path),
        filtered_kmer_path=os.fspath(filtered_kmer_path),
        counts=counts,
        failures=read_failures(images_dir),
    )
