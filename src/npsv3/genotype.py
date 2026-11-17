"""Genotype a sample with a trained model (`command=genotype`).

The stages reuse the same components as training-data generation, so genotyping can't drift from how the model was trained:

1. Prepare the VCF the graphs are built from, without the sample and with any inference VCF's sites
   (`npsv3.genotyping.inputs`), and determine the regions to genotype.
2. Build the graphs for those regions, and 3. generate their images in parallel (`npsv3.genotyping.images`).
4. Score the images with the model on the CPU or GPU(s) (`npsv3.genotyping.predict`).
5. Write the genotyped VCF (`npsv3.genotyping.vcf`).

See `npsv3/genotyping/GENOTYPING_PLAN.md` for the design.
"""

import contextlib
import logging
import os
import tempfile
from collections import Counter
from collections.abc import Sequence

from omegaconf import DictConfig, ListConfig, OmegaConf

from npsv3.genotyping.images import (
    GENERATION_FILE,
    GRAPHS_DIR,
    IMAGES_DIR,
    generate_genotyping_images,
    genotyping_config,
)
from npsv3.genotyping.inputs import (
    GENOTYPING_VCF,
    INFERENCE_SITES_VCF,
    GenotypingInputs,
    check_provenance,
    file_provenance,
    genotyping_regions,
    prepare_genotyping_vcf,
    write_provenance,
)
from npsv3.genotyping.predict import PREDICTIONS_DIR, predict_genotypes, read_predictions, resolve_devices
from npsv3.genotyping.vcf import scoring_options, write_genotypes_vcf
from npsv3.images.example import read_failures
from npsv3.types import PathType
from npsv3.util.sample import Sample

PLOIDY = 2
MIN_VARIANT_SIZE = 50
"""Minimum size of the analysis variants, as for training"""

INPUTS_DIR = "inputs"
"""The prepared inputs, in the working directory (or a specified images directory, so they can be reused together)"""

DATASET_FILE = "dataset.yaml"

REGION_SELECTIONS = ("inference", "all")
"""With an inference VCF, genotype the regions with an analysis variant with an inference allele, or every region"""

GRAPH_PROVENANCE_KEYS = (
    "population", "inference", "reference", "sample", "ploidy", "min_variant_size", "variant_padding", "region_selection", "graph",
)
"""The inputs and settings that determine the graphs (i.e., not the sample's reads)"""


def _region_selection(cfg: DictConfig) -> str:
    selection = OmegaConf.select(cfg, "region_selection", default="inference")
    if selection not in REGION_SELECTIONS:
        msg = f"region_selection must be one of {', '.join(REGION_SELECTIONS)}, not {selection!r}"
        raise ValueError(msg)
    return selection


def _provenance(cfg: DictConfig, reads: str, sample: Sample, population_vcf: str, inference_vcf: str | None) -> dict:
    """The inputs and settings that determine the prepared VCF, graphs and images. The image settings the model depends on
    are checked separately (`npsv3.genotyping.predict.check_generation`)."""
    graph = genotyping_config(cfg, sample.name).graph
    return {
        "population": file_provenance(population_vcf),
        "inference": file_provenance(inference_vcf),
        "reads": file_provenance(reads),
        "reference": file_provenance(cfg.reference),
        "sample": sample.name,
        "ploidy": PLOIDY,
        "min_variant_size": MIN_VARIANT_SIZE,
        "variant_padding": cfg.pileup.variant_padding,
        "region_selection": _region_selection(cfg),
        "graph": OmegaConf.to_container(graph, resolve=True),
    }


def _resolved_devices(cfg: DictConfig):
    devices = OmegaConf.select(cfg, "devices", default="cpu")
    return OmegaConf.to_object(devices) if isinstance(devices, ListConfig) else devices


def _check_empty(images_dir: str) -> None:
    """Images can't be added to an earlier run's, even one that didn't finish, since the two would be mixed"""
    if os.path.isdir(images_dir) and os.listdir(images_dir):
        msg = (
            f"{images_dir} already has files. Remove it to regenerate the images or, if it has complete images, specify it "
            "with images_dir to reuse them."
        )
        raise ValueError(msg)


def genotype(  # noqa: PLR0913
    cfg: DictConfig,
    reads: PathType,
    sample: Sample,
    population_vcf: PathType,
    output: PathType,
    *,
    inference_vcf: PathType | None = None,
    work_dir: PathType | None = None,
    images_dir: PathType | None = None,
    graph_shards: Sequence[PathType] | None = None,
    unique_kmer_path: PathType | None = None,
    filtered_kmer_path: PathType | None = None,
) -> Counter:
    """Genotype `sample` and write a phased VCF to `output`.

    Args:
        cfg: The genotyping configuration
        reads: The sample's (preprocessed) reads
        sample: The sample, with its KMC k-mer database
        population_vcf: Phased population VCF (bgzipped and indexed). The sample is excluded if present.
        output: The output VCF
        inference_vcf: VCF (bgzipped and indexed) of the variants to genotype, which may not be in the population VCF. If
            None, all of the population's analysis variants are genotyped. Otherwise, `region_selection` determines the
            regions genotyped, and every inference record is reported.
        work_dir: Directory for the intermediate files, which is always kept. If None, a temporary directory.
        images_dir: Directory for the images (`work_dir`/images by default). If it already has complete images, the input
            preparation, graphs and images are skipped, after checking they were created from the same inputs.
        graph_shards: Graphs from an earlier run with the same inputs to skip graph building
        unique_kmer_path: The unique k-mers for `graph_shards`
        filtered_kmer_path: The sample's k-mers filtered to the unique k-mers of `graph_shards`

    Returns:
        The number of records written by outcome

    Raises:
        ValueError: If the configuration is invalid, or the intermediate files to reuse don't match the inputs
    """
    reads, population_vcf, output = os.path.abspath(reads), os.path.abspath(population_vcf), os.path.abspath(output)
    inference_vcf = None if inference_vcf is None else os.path.abspath(inference_vcf)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    # Check everything we can before any of the expensive stages
    if cfg.simulation.replicates < 1:
        msg = "At least one replicate (simulation.replicates) is required for genotyping"
        raise ValueError(msg)
    if graph_shards is not None and len(graph_shards) == 0:
        # E.g., a pattern that didn't match, which otherwise looks like there is nothing to genotype
        msg = "graph_shards was specified, but has no graph shards"
        raise ValueError(msg)

    # Ensure valid devices specification before getting too far
    devices = _resolved_devices(cfg)
    resolve_devices(devices)
    
    options = scoring_options(cfg)
    if not os.path.isfile(cfg.model.checkpoint):
        msg = f"Model checkpoint {cfg.model.checkpoint} doesn't exist"
        raise ValueError(msg)
    provenance = _provenance(cfg, reads, sample, population_vcf, inference_vcf)

    if work_dir is None:
        work_context = tempfile.TemporaryDirectory(prefix="npsv3-genotype-", ignore_cleanup_errors=True)
    else:
        os.makedirs(work_dir, exist_ok=True)
        work_context = contextlib.nullcontext(os.path.abspath(work_dir))

    with work_context as work_path:
        provided_images = None if images_dir is None else os.path.abspath(images_dir)
        resolved_images = os.path.join(work_path, IMAGES_DIR) if provided_images is None else provided_images
        # Prepared inputs live with images that were specified, so those images can be reused without the working directory
        inputs_dir = os.path.join(work_path if provided_images is None else provided_images, INPUTS_DIR)
        reuse_images = provided_images is not None and os.path.exists(os.path.join(provided_images, GENERATION_FILE))

        if reuse_images:
            logging.info("Using the existing images in %s", resolved_images)
            check_provenance(inputs_dir, provenance)
            merged_vcf, sites_vcf = os.path.join(inputs_dir, GENOTYPING_VCF), os.path.join(inputs_dir, INFERENCE_SITES_VCF)
            vcf_path = merged_vcf if os.path.exists(merged_vcf) else population_vcf
            inference_sites = sites_vcf if inference_vcf is not None else None
            failures = read_failures(resolved_images)
        else:
            _check_empty(resolved_images)
            if graph_shards:
                check_provenance(os.path.dirname(os.path.abspath(graph_shards[0])), provenance, keys=GRAPH_PROVENANCE_KEYS)
            os.makedirs(inputs_dir, exist_ok=True)
            inputs: GenotypingInputs = prepare_genotyping_vcf(
                population_vcf, inputs_dir, sample_name=sample.name, inference_vcf=inference_vcf, threads=cfg.threads,
            )
            write_provenance(inputs_dir, provenance)
            vcf_path, inference_sites = inputs.vcf_path, inputs.inference_sites
            failures = []

        regions = genotyping_regions(
            vcf_path,
            flank=cfg.pileup.variant_padding,
            min_variant_size=MIN_VARIANT_SIZE,
            inference_sites=inference_sites if _region_selection(cfg) == "inference" else None,
        )
        logging.info("Genotyping %d regions in sample %s", len(regions), sample.name)

        dataset_path = None
        if reuse_images:
            candidate = os.path.join(resolved_images, DATASET_FILE)
            dataset_path = candidate if os.path.exists(candidate) else None
        elif regions:
            images = generate_genotyping_images(
                cfg,
                reads,
                sample,
                inputs,
                regions,
                work_path,
                graph_shards=graph_shards,
                unique_kmer_path=unique_kmer_path,
                filtered_kmer_path=filtered_kmer_path,
                images_dir=resolved_images,
                ploidy=PLOIDY,
                min_variant_size=MIN_VARIANT_SIZE,
                progress_bar=True,
            )
            dataset_path, failures = images.dataset_path, images.failures
            if not graph_shards:
                # So these graphs can be reused, but only for these same inputs
                write_provenance(os.path.join(work_path, GRAPHS_DIR), provenance)

        predictions, rejected = None, []
        if dataset_path is not None:
            result = predict_genotypes(cfg, dataset_path, os.path.join(work_path, PREDICTIONS_DIR), devices=devices)
            predictions, rejected = read_predictions(result.predictions_dir), result.rejected

        counts = write_genotypes_vcf(
            vcf_path,
            output,
            sample_name=sample.name,
            regions=regions,
            images_dir=resolved_images,
            predictions=predictions,
            rejected=rejected,
            failures=failures,
            options=options,
            inference_sites=inference_sites,
            checkpoint=cfg.model.checkpoint,
            ploidy=PLOIDY,
        )
    return counts
