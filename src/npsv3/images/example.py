from dataclasses import dataclass
import itertools
import logging
import math
import os
import random
import shutil
import sys
import tempfile
import typing

import hydra
import numpy as np
import pysam
import ray
import webdataset as wds
from omegaconf import OmegaConf, DictConfig
from PIL import Image
from tqdm import tqdm

from npsv3.graphs.graph import Graph
from npsv3.images.generator import ImageGenerator
from npsv3.realigner import FragmentRealigner
from npsv3.simulation import augment_sample, simulate_variant_sequencing
from npsv3.util.config import setup_resolvers
from npsv3.util.range import Range
from npsv3.util.reads import downsample_reads, haplotag_reads
from npsv3.util.sample import Sample
from npsv3.util.timeout import Timeout
from npsv3.util.vcf import index_variant_file, pysam_write_mode
from npsv3.variant import Variant, overlapping_records


def _reference_sequence(reference_fasta: str, region: Range) -> str:
    with pysam.FastaFile(reference_fasta) as ref_fasta:
        # Make sure reference sequence is all upper case
        return ref_fasta.fetch(reference=region.contig, start=region.start, end=region.end).upper()


def example_to_image(
    cfg, example, out_path: str | None=None, with_simulations=False, margin=10, max_replicates=1, **kwargs
):
    generator = hydra.utils.instantiate(cfg.generator, cfg, _recursive_=False)

    image_tensor = example["image"]
    real_image = generator.render(image_tensor, **kwargs)

    genotypes, replicates, *_ = example["sim.images"].shape if with_simulations and "sim.images" in example else (3, 0)
    if replicates > 0:
        width, height = real_image.size
        replicates = min(replicates, max_replicates)

        image = Image.new(real_image.mode, (width + (genotypes - 1) * (width + margin), height + replicates * (height + margin)))
        # Paste the real image overlapping the correct genotype
        label = example.get("label", 0)
        image.paste(real_image, (label * width + max(label - 1, 0) * margin, 0))

        synth_tensor = example["sim.images"]
        for gt in range(genotypes):
            for repl in range(replicates):
                synth_image_tensor = synth_tensor[gt, repl]
                synth_image = generator.render(synth_image_tensor, **kwargs)

                coord = (gt * (width + margin), (repl + 1) * (height + margin))
                image.paste(synth_image, coord)
    else:
        image = real_image

    if out_path:
        image.save(out_path)
    return image


def make_example_from_region(
    cfg,
    region: Range,
    read_path: str,
    sample: Sample,
    background_vcf: str | None = None,
    generator: ImageGenerator = None,
    addl_features: dict | None = None,
    **kwargs,
):
    # Create generator if not provided
    generator = generator or hydra.utils.instantiate(cfg.generator, cfg=cfg, _recursive_=False)

    # Construct image for "real" data
    with tempfile.TemporaryDirectory() as tempdir:
        local_read_path = read_path
        if cfg.pileup.downsample < 1.0:
            # Downsample reads if specified
            local_read_path = downsample_reads(
                local_read_path,
                region.expand(cfg.pileup.fetch_flank),
                tempdir,
                downsample=cfg.pileup.downsample,
            )

        if cfg.pileup.haplotag_reads and background_vcf is not None:
            # Haplotag reads on the fly using whatshap
            local_read_path = haplotag_reads(
                cfg.reference, sample, local_read_path, background_vcf, region.expand(cfg.pileup.fetch_flank), tempdir
            )

        image_tensor = generator.generate(
            local_read_path,
            sample,
            region,
            ref_seq=_reference_sequence(cfg.reference, region),
        )

    example = {"region": str(region), "image": image_tensor}
    # Add any additional (extension) features
    if addl_features:
        example.update(addl_features)

    return example

def make_graph_example_from_region(
    cfg,
    region: Range,
    read_path: str,
    sample: Sample,
    background_vcf: str,
    inference_vcf: str,
    generator: ImageGenerator = None,
    addl_features: dict | None = None,
    ploidy: int = 2,
    **kwargs,
):
    # Create generator if not provided
    generator = generator or hydra.utils.instantiate(cfg.generator, cfg=cfg, _recursive_=False)

    # Generate graph from the VCF(s)
    graph = Graph.from_vcf(
        cfg.reference, background_vcf, region.expand(cfg.pileup.graph_flank), inference_vcf=inference_vcf
    )
    assert graph.is_bubble_path(region.contig), f"Graph must form bubble for reference background for region {region}"

    # Set up image flanks to minimize compression
    example_region = generator.image_region(region) if cfg.pileup.compress else generator.image_region_variable(region)

    with tempfile.TemporaryDirectory() as tempdir:
        # Generate haplotypes for re-alignment, i.e., with reference as the background (as opposed to a specific haplotype)
        realign_haplotypes = graph.all_haplotypes(inference_vcf, region.contig, region.expand(cfg.pileup.variant_padding))
        assert len(realign_haplotypes) >= 2, f"Fewer than 2 haplotypes in region {region}"  # noqa: PLR2004
        assert realign_haplotypes[0].nodes == graph.nodes_on_path(
            region.contig
        ), f"First haplotype must be the reference for region {region}"

        realign_fasta_path = os.path.join(tempdir, "realign.fasta")
        with open(realign_fasta_path, "w") as fasta:
            for i, haplotype in enumerate(realign_haplotypes):
                fasta.write(f">seq{i}\n")
                sequence = haplotype.sequence()
                assert sequence.isupper(), f"Sequence for haplotype {i} is not upper case in region {region}"
                fasta.write(haplotype.sequence() + "\n")

        # Construct realigner once for all images for this variant,
        addl_args = {"num_alts": len(realign_haplotypes) - 1}  # This is needed to prevent C++ errors
        realigner = FragmentRealigner(realign_fasta_path, sample.mean_insert_size, sample.std_insert_size, **addl_args)

        # Extract the reference sequence from the first haplotype
        ref_seq = realign_haplotypes[0].sequence()[
            example_region.start - graph.region.start : example_region.end - graph.region.end
        ]
        assert len(ref_seq) == example_region.length, f"Reference sequence length does not match the region size for region {region}"

    # Construct image for "real" data
    with tempfile.TemporaryDirectory() as tempdir:
        local_read_path = read_path
        if cfg.pileup.downsample < 1.0:
            # Downsample reads if specified
            local_read_path = downsample_reads(
                local_read_path,
                example_region.expand(cfg.pileup.fetch_flank),
                tempdir,
                downsample=cfg.pileup.downsample,
            )

        if cfg.pileup.haplotag_reads:
            # Haplotag reads on the fly using whatshap
            local_read_path = haplotag_reads(
                cfg.reference, sample, local_read_path, background_vcf, graph.region, tempdir
            )

        # What if adding padding to the image causes it to be compressed? Should we then not pad the image to preserve 1 pixel to 1 base? For example if a variant is 500 bp in length, padding it will increase it to 692, thus compression will be required for a max image width of 512.
        do_compress = cfg.pileup.compress
        if region.length > cfg.pileup.max_image_width:
            do_compress = True

        image_tensor = generator.generate(
            local_read_path,
            sample,
            example_region,
            realigner=realigner,
            ref_seq=ref_seq,
            compress=do_compress,
        )

    example = {"region": str(example_region), "image": image_tensor}
    # Add any additional (extension) features
    if addl_features:
        example.update(addl_features)

    # Generate the possible haplotypes for this region on the possible backgrounds
    # TODO: Check if the backgrounds are identical, if so, we can generate the haplotypes once
    backgrounds = [
        graph.all_haplotypes(inference_vcf, f"{sample.name}#{i}#{region.contig}", region.expand(cfg.pileup.variant_padding))
        for i in range(ploidy)
    ]
    total_genotypes = math.prod(len(haplotypes) for haplotypes in backgrounds)

    # For fully labeled data, one of the haplotypes should be the true haplotype
    labels = []
    for allele, haplotypes in enumerate(backgrounds):
        assert len(haplotypes) > 1, f"Fewer than 2 haplotypes for allele {allele} in region {region}"
        base_path_nodes = graph.shortest_path(f"{sample.name}#{allele}#{region.contig}")
        for allele_index, haplotype in enumerate(haplotypes):
            if haplotype.nodes == base_path_nodes:
                labels.append(allele_index)
                break
        else:
            raise ValueError(f"True haplotype not found in possible haplotypes for region {region}")
    assert len(labels) == ploidy, f"Expected {ploidy} labels, got {len(labels)} for region {region}"
    gt_label = np.ravel_multi_index(tuple([i] for i in labels), tuple(len(b) for b in backgrounds)).item()
    example["label"] = gt_label

    # Determine "ranked" positives based on shared inference paths, presence, etc.
    ranked_positives = np.zeros(total_genotypes, dtype=np.long)
    ranked_positives[gt_label] = 1 # True (or "first-rank") positive

    true_inference_paths = set.union(*(backgrounds[i][allele_index].paths for i, allele_index in enumerate(labels)))
    for g, allele_indices in enumerate(itertools.product(*(range(len(haplotypes)) for haplotypes in backgrounds))):
        if g == gt_label:
            continue  # Skip the true positive
        # Diplotypes that have the same inference paths as the true haplotype, but with a different phase, are considered "second-rank" positives
        inf_paths = set.union(*(backgrounds[i][allele_index].paths for i, allele_index in enumerate(allele_indices)))
        if inf_paths == true_inference_paths:
            ranked_positives[g] = 2

    # The same "presence" for non-reference genotypes, i.e., non-reference concordant, are considered "third-rank" positives
    # TODO: Should only variants with the true allele be considered "third-rank" positives? And any presence be "fourth-rank"?
    if gt_label > 0:
        ranked_positives[(ranked_positives == 0) & (np.arange(total_genotypes) > 0)] = 3

    example["label.rank"] = ranked_positives

    # Create listing of alleles for each genotype
    genotype_alleles = []
    for allele_indices in itertools.product(*(range(len(haplotypes)) for haplotypes in backgrounds)):
        genotype_alleles.append(tuple(tuple(backgrounds[i][allele_index].paths) for i, allele_index in enumerate(allele_indices)))  # noqa: PERF401
    example["label.alleles"] = genotype_alleles

    if cfg.simulation.replicates == 0:
        # No more work to be done if there are not simulations
        return example

    # If we are augmenting the simulated data, use the provided statistics for the first example, so it
    # will hopefully be most similar to the real data and then augment the remaining replicates
    if cfg.simulation.augment:
        repl_samples = augment_sample(sample, cfg.simulation.replicates, keep_original=True)
    else:
        repl_samples = [sample] * cfg.simulation.replicates

    # Generate the relevant sequences once, which are then combined to create the fasta for simulation.
    # TODO: Do we need pad out the shorter sequences?
    sequences = [[haplotype.sequence() for haplotype in background] for background in backgrounds]

    # TODO: Do we want to only have one of 0/1, 1/0?
    alleles_encoded_images = []
    for allele_indices in itertools.product(*(range(len(haplotypes)) for haplotypes in backgrounds)):
        # Get the sequences for this haplotype combination
        gt_sequences = [sequences[i][allele_index] for i, allele_index in enumerate(allele_indices)]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tempdir:
            # Write the fasta file for this haplotype combination
            fasta_path = os.path.join(tempdir, "haplotypes.fasta")
            with open(fasta_path, "w") as fasta:
                for i, sequence in enumerate(gt_sequences):
                    fasta.write(f">{allele_indices[i]}#{i}#{graph.region.contig}#0\n")
                    fasta.write(sequence + "\n")

            repl_encoded_images = []
            for repl, repl_sample in enumerate(repl_samples):
                # Simulate reads from this haplotype combination
                try:
                    sample_coverage = (
                        repl_sample.chrom_mean_coverage(graph.region.contig)
                        if cfg.simulation.chrom_norm_covg
                        else repl_sample.mean_coverage
                    )
                    replicate_bam_path = simulate_variant_sequencing(
                        fasta_path,
                        (sample_coverage * cfg.pileup.downsample) / ploidy,
                        repl_sample,
                        reference=cfg.reference,
                        shared_reference=cfg.shared_reference,
                        dir=tempdir,
                        stats_path=cfg.stats_path if cfg.simulation.gc_norm_covg else None,
                        region=example_region.expand(cfg.pileup.realigner_flank),
                        phase_vcf_path=background_vcf if cfg.pileup.haplotag_sim else None,
                        aligner=cfg.pileup.aligner,
                    )
                except ValueError:
                    logging.exception(
                        "Failed to synthesize data for alleles (%d,%d) for region %s",
                        *allele_indices,
                        str(graph.region),
                    )
                    raise

                synth_image_tensor = generator.generate(
                    replicate_bam_path,
                    repl_sample,
                    example_region,
                    realigner=realigner,
                    ref_seq=ref_seq,
                    # Unsure if this actually needs to be changed to the config or can remain as "True"
                    compress=do_compress,
                )
                repl_encoded_images.append(synth_image_tensor)

                if not OmegaConf.is_missing(cfg.simulation, "save_sim_bam_dir"):
                    sim_bam_path = os.path.join(cfg.simulation.save_sim_bam_dir, f"{'_'.join(map(str, allele_indices))}_{repl}.bam")
                    shutil.copy(replicate_bam_path, sim_bam_path)
                    shutil.copy(f"{replicate_bam_path}.bai", f"{sim_bam_path}.bai")

            # Stack all of the image replicates into a tensor
            alleles_encoded_images.append(np.stack(repl_encoded_images))

        # Stack the replicated images for each genotype into a tensor
        sim_image_tensor = np.stack(alleles_encoded_images)
        example["sim.images"] = sim_image_tensor

    return example

class ExampleActor:
    def __init__(self, index: int, output_dir: str, cfg, *args, **kwargs):
        self.output_path = os.path.join(output_dir, f"images-{index:04d}.tar")
        self.cfg = cfg
        self.args = args
        self.kwargs = kwargs

        self._writer = wds.TarWriter(self.output_path)

    # TODO: Convert this to a finalizer? https://docs.python.org/3/library/weakref.html#comparing-finalizers-with-del-methods
    def cleanup(self):
        self._writer.close()

@ray.remote
class RegionWriter(ExampleActor):
    def from_region(self, region: Range):
        try:
            # Attempt to timeout long running regions.
            with Timeout(self.cfg.timeout):
                example = make_example_from_region(self.cfg, region, *self.args, **self.kwargs)
            sample = {
                "__key__": region.slug,
                "image.npy.gz": example["image"],
            }
            self._writer.write(sample)
        except TimeoutError:
            logging.exception("Timeout error for region %s", region)


@ray.remote
class VariantWriter(ExampleActor):
    def from_region(self, region: Range):
        # Loop through inference VCF in region, creating a new VCF for each variant
        # Then call make_graph_example_from_region for that variant alone
        inference_vcf = self.kwargs.get("inference_vcf")
        assert inference_vcf, "Inference VCF is not provided"

        with pysam.VariantFile(inference_vcf) as src_vcf_file:
            src_header = src_vcf_file.header
            for record in src_vcf_file.fetch(**region.pysam_fetch):
                variant = Variant.from_pysam(record)
                with tempfile.TemporaryDirectory() as dst_dir:
                    dst_vcf = os.path.join(dst_dir, "variant.vcf.gz")
                    with pysam.VariantFile(dst_vcf, mode="wz", header=src_header) as dst_vcf_file:
                        dst_vcf_file.write(record)
                    index_variant_file(dst_vcf)

                    kwargs = self.kwargs.copy()
                    kwargs["inference_vcf"] = dst_vcf
                    example = make_graph_example_from_region(self.cfg, variant.reference_region, *self.args, **kwargs)

                    sample = {
                        "__key__": variant.vg_variant_id,
                        "region.txt": example["region"],
                        "image.npy.gz": example["image"],
                    }
                    if "label" in example:
                        sample["label.cls"] = example["label"]
                    if "label.rank" in example:
                        sample["label.rank.npy"] = example["label.rank"]
                    if "sim.images" in example:
                        sample["sim.images.npy.gz"] = example["sim.images"]
                    self._writer.write(sample)


@ray.remote
class GraphWriter(ExampleActor):
    def from_region(self, region: Range):
        try:
            # Attempt to gracefully timeout long running regions.
            with Timeout(self.cfg.timeout):
                example = make_graph_example_from_region(self.cfg, region, *self.args, **self.kwargs)
            sample = {
                "__key__": region.slug,
                # TODO: Write region.txt as well
                "image.npy.gz": example["image"],
            }
            if "label" in example:
                sample["label.cls"] = example["label"]
            if "label.rank" in example:
                sample["label.rank.npy"] = example["label.rank"]
            if "sim.images" in example:
                sample["sim.images.npy.gz"] = example["sim.images"]
            self._writer.write(sample)
        except TimeoutError:
            logging.exception("Timeout error for region %s", region)


def vcf_to_region_examples(
    cfg,
    read_path: str,
    sample: Sample,
    inference_vcf: str,
    output_dir: str,
    background_vcf: str | None = None,
    progress_bar: bool = False,
):

    regions = []
    with pysam.VariantFile(inference_vcf) as vcf_file:
        for record in vcf_file:
            variant = Variant.from_pysam(record)
            regions.append(variant.reference_region.expand(cfg.pileup.variant_padding))

    os.makedirs(output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as ray_dir:
        # We currently just use ray for the CPU-side work, specifically simulating the SVs. We use a private temporary directory
        # to avoid conflicts between clusters running on the same node.
        # To ensure Ray worker processes know about our custom resolvers, we set up the runtime environment with a worker process hook.
        ray.init(num_cpus=cfg.threads, num_gpus=0, _temp_dir=ray_dir, ignore_reinit_error=True, include_dashboard=False, runtime_env=ray.runtime_env.RuntimeEnv(worker_process_setup_hook=setup_resolvers))

        actors = [
            RegionWriter.remote(i, output_dir, cfg, read_path, sample, background_vcf=background_vcf, inference_vcf=inference_vcf)
            for i in range(cfg.threads)
        ]
        pool = ray.util.ActorPool(actors)

        gen = pool.map_unordered(lambda actor, region: actor.from_region.remote(region), regions)
        for _ in tqdm(gen, total=len(regions), disable=not progress_bar):
            pass

        # Make sure all the writers are cleaned up
        ray.wait([actor.cleanup.remote() for actor in actors], num_returns=len(actors))


def vcf_to_variant_examples(
    cfg,
    read_path: str,
    sample: Sample,
    inference_vcf: str,
    output_dir: str,
    background_vcf: str | None = None,
    progress_bar: bool = False,
    ploidy: int = 2,
):
    group_padding = cfg.pileup.variant_padding
    regions = [region for region, *_ in overlapping_records(inference_vcf, flank=group_padding)]

    os.makedirs(output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as ray_dir:
        # We currently just use ray for the CPU-side work, specifically simulating the SVs. We use a private temporary directory
        # to avoid conflicts between clusters running on the same node.
        ray.init(num_cpus=cfg.threads, num_gpus=0, _temp_dir=ray_dir, ignore_reinit_error=True, include_dashboard=False, runtime_env=ray.runtime_env.RuntimeEnv(worker_process_setup_hook=setup_resolvers))

        actors = [
            VariantWriter.remote(i, output_dir, cfg, read_path, sample, background_vcf=background_vcf, inference_vcf=inference_vcf, ploidy=ploidy)
            for i in range(cfg.threads)
        ]
        pool = ray.util.ActorPool(actors)

        gen = pool.map_unordered(lambda actor, region: actor.from_region.remote(region), regions)

        for _ in tqdm(gen, total=len(regions), disable=not progress_bar):
            pass

        # Make sure all the writers are cleaned up
        ray.wait([actor.cleanup.remote() for actor in actors], num_returns=len(actors))


def vcf_to_graph_examples(
    cfg,
    read_path: str,
    sample: Sample,
    inference_vcf: str,
    output_dir: str,
    background_vcf: str | None = None,
    progress_bar: bool = False,
    ploidy: int = 2,
):
    # Identify regions in the inference with overlapping SVs (i.e., identify bubbles)
    regions, search_regions = [], []
    running_total = running_max = 0
    group_padding = cfg.pileup.variant_padding // 2
    for region, records in overlapping_records(inference_vcf, flank=group_padding):
        count = len(records)

        running_total += count
        running_max = max(running_max, count)

        # We can't exhaustively generate examples for regions with too many records, so we skip any more complex regions
        (regions if count <= cfg.pileup.max_exhaustive_records else search_regions).append(region.expand(-group_padding))

    logging.info(
        "Identified %d regions with mean %f and max %d records",
        running_total,
        running_total / (len(regions) + len(search_regions)),
        running_max,
    )
    logging.info(
        "Generating exhaustive images for %d regions (across %d threads)", len(regions), cfg.threads
    )

    os.makedirs(output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as ray_dir:
        # We currently just use ray for the CPU-side work, specifically simulating the SVs. We use a private temporary directory
        # to avoid conflicts between clusters running on the same node.
        ray.init(num_cpus=cfg.threads, num_gpus=0, _temp_dir=ray_dir, ignore_reinit_error=True, include_dashboard=False, runtime_env=ray.runtime_env.RuntimeEnv(worker_process_setup_hook=setup_resolvers))

        actors = [
            GraphWriter.remote(i, output_dir, cfg, read_path, sample, background_vcf=background_vcf, inference_vcf=inference_vcf, ploidy=ploidy)
            for i in range(cfg.threads)
        ]
        pool = ray.util.ActorPool(actors)

        gen = pool.map_unordered(lambda actor, region: actor.from_region.remote(region), regions)
        for _ in tqdm(gen, total=len(regions), disable=not progress_bar):
            pass

        # Make sure all the writers are cleaned up
        ray.wait([actor.cleanup.remote() for actor in actors], num_returns=len(actors))


VCF_HEADER_TYPES_TO_COPY = frozenset(["GENERIC", "STRUCTURED", "INFO", "FILTER", "CONTIG", "FORMAT"])


def _complete_genotype(record: pysam.VariantRecord, sample: str) -> bool:
    """Return True if the sample's genotype is completely defined (no missing alleles)"""
    return all(allele is not None for allele in record.samples[sample].allele_indices)

def _non_ref_genotype(record: pysam.VariantRecord, sample: str) -> bool:
    """Return True if the sample's genotype is defined and has non-reference allele"""
    non_ref = False
    for allele in record.samples[sample].allele_indices:
        if allele is None:
            return False
        non_ref = non_ref or allele > 0
    return non_ref

def _write_subset_record(
    vcf_file: pysam.VariantFile,
    record: pysam.VariantRecord,
    sample: str,
    info_handlers: typing.Optional[dict] = None,
) -> pysam.VariantRecord:
    """Write record with just sample's genotype to vcf_file.

    Args:
        vcf_file (pysam.VariantFile): The VCF file to write to.
        record (pysam.VariantRecord): The VCF record to write.
        sample (str): The sample name to include.
        info_handlers (dict): Handlers for cleaning up INFO fields.

    Returns:
        pysam.VariantRecord: The written VCF record.
    """
    # Create a new record with only the relevant sample
    src_sample = record.samples[sample]
    try:
        dst_record = vcf_file.new_record(
            contig=record.contig,
            start=record.start,
            stop=record.stop,
            id=record.id,
            alleles=record.alleles,
            qual=record.qual,
            filter=record.filter,
            # Fix up INFO fields as needed to produce a valid VCF from ill-formed inputs
            info={ key: (handler(value) if (handler := info_handlers and info_handlers.get(key)) else value) for key, value in record.info.items() },
            samples=[src_sample]
        )
        dst_record.samples[0].phased = src_sample.phased # Reapply the phasing information (which is otherwise lost)
    except TypeError:
        print(record)
        raise
    vcf_file.write(dst_record)
    return dst_record

@dataclass
class _SplitAndFilterStats:
    nonref_records: int = 0
    matching_ref_records: int = 0
    ref_records: int = 0
    dropped_regions: int = 0


def split_and_filter_vcf(
    cfg: DictConfig,
    inference_vcf: str,
    output_dir: str,
):
    """Split a multi-sample VCF into a single-samples VCFs suitable for paired-model training.

    For each region with nearby or overlapping variants, we write out non-reference variants to the respective single
    sample VCFs (if there are fewer than cfg.pileup.max_exhaustive_records non-reference variants in the region). And attempt
    to find corresponding variants in reference-only regions in other samples.

    Args:
        cfg (DictConfig): Global configuration
        inference_vcf (str): Path to multi-sample VCF to be split
        output_dir (str): Directory to write the split VCFs as {sample}.vcf.gz
    """
    logging.info("Splitting and filtering VCF into %s", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with pysam.VariantFile(inference_vcf) as src_vcf_file:
        src_header = src_vcf_file.header

        # Create headers for each of the output VCFs (one per sample)
        headers = {}
        info_handlers = {}
        for src_sample in src_header.samples:
            dst_header = pysam.VariantHeader()
            for record in src_header.records:
                if record.type in VCF_HEADER_TYPES_TO_COPY:
                    dst_header.add_record(record)
                    # Certain INFO fields can be ill-formatted upstream (and PySAM is strict), so we record handlers here for cleanup
                    if record.type == "INFO" and record["Number"] == "0":
                        info_handlers[record["ID"]] = bool  # Force FLAG fields to be boolean
            dst_header.add_sample(src_sample)
            headers[src_sample] = dst_header

        dst_vcf_files = {}
        for sample, dst_header in headers.items():
            output_vcf = os.path.join(output_dir, f"{sample}.vcf.gz")
            dst_vcf_files[sample] = pysam.VariantFile(output_vcf, mode="wz", header=dst_header)

        stats = {sample: _SplitAndFilterStats() for sample in dst_vcf_files}

        for _region_count, (_region, records) in enumerate(overlapping_records(src_vcf_file, flank=cfg.pileup.variant_padding), start=1):
            # Partition samples into reference-only and non-reference genotypes
            non_ref_samples = {}
            for sample in dst_vcf_files:
                non_ref = [record for record in records if _non_ref_genotype(record, sample)]
                if len(non_ref) > 0:
                    non_ref_samples[sample] = non_ref

            ref_samples = list(set(dst_vcf_files).difference(non_ref_samples))
            random.shuffle(ref_samples)
            for sample, non_ref_records in non_ref_samples.items():
                if len(non_ref_records) > cfg.pileup.max_exhaustive_records:
                    stats[sample].dropped_regions += 1
                    continue  # Skip regions with too many non-reference variants

                # We found a tractable number of non-ref variants. Write them to the sample's VCF file and find
                # corresponding reference-only examples in other samples
                for record in non_ref_records:
                    _write_subset_record(dst_vcf_files[sample], record, sample, info_handlers)
                stats[sample].nonref_records += len(non_ref_records)

                # Select another random sample without replacement and write out the reference only records
                while len(ref_samples) > 0:
                    ref_sample = ref_samples.pop()
                    ref_records = [record for record in records if record in non_ref_records and _complete_genotype(record, ref_sample)]
                    if len(records) != len(ref_records):
                        continue # This samples does not have fully genotyped records for all of the needed variants
                    for record in ref_records:
                        _write_subset_record(dst_vcf_files[ref_sample], record, ref_sample, info_handlers)
                    stats[ref_sample].ref_records += len(ref_records)
                    stats[sample].matching_ref_records += len(ref_records)
                    break  # We successfully wrote a reference sample

        for dst_vcf_file in dst_vcf_files.values():
            dst_vcf_file.close()
        for sample, sample_stats in stats.items():
            logging.info(
                "Sample %s: Wrote %d non-ref variants (with %d matching ref variants) and %d ref variants (%d/%d regions dropped due to too many non-ref variants)",
                sample,
                sample_stats.nonref_records,
                sample_stats.matching_ref_records,
                sample_stats.ref_records,
                sample_stats.dropped_regions,
                _region_count,
            )

