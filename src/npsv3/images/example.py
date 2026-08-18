import contextlib
import itertools
import logging
import math
import os
import random
import shutil
import sys
import tempfile
import typing
from collections.abc import Sequence
from dataclasses import dataclass

import hydra
import numpy as np
import pysam
import ray
import webdataset as wds
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from npsv3.graphs.graph import Graph
from npsv3.graphs.haplotype import (
    HaplotypeSamplerOverlay,
    KmerClassify,
    UniqueKmersOverlay,
    _filter_variants,
    prepare_genotyping_haplotypes,
    serialize_graph_and_unique_kmers,
)
from npsv3.images.generator import ImageGenerator
from npsv3.realigner import FragmentRealigner
from npsv3.simulation import augment_sample, simulate_variant_sequencing, split_bam_by_tag
from npsv3.types import PathType
from npsv3.util.config import setup_resolvers
from npsv3.util.range import Range
from npsv3.util.reads import downsample_reads, haplotag_reads
from npsv3.util.sample import Sample, _kmc_db_kmer_size, kmc_filter
from npsv3.util.timeout import Timeout
from npsv3.util.variant import Variant, VariantFileReader, overlapping_records
from npsv3.util.vcf import index_variant_file, pysam_write_mode


def _reference_sequence(reference_fasta: str, region: Range) -> str:
    with pysam.FastaFile(reference_fasta) as ref_fasta:
        # Make sure reference sequence is all upper case
        return ref_fasta.fetch(reference=region.contig, start=region.start, end=region.end).upper()


def example_to_image(
    cfg, example, out_path: PathType | None=None, *, with_simulations=False, margin=10, max_replicates=1, **kwargs
):
    generator = hydra.utils.instantiate(cfg.generator, cfg, _recursive_=False)

    image_tensor = example["image"]
    real_image = generator.render(image_tensor, **kwargs)

    replicates, genotypes, *_ = example["sim.images"].shape if with_simulations and "sim.images" in example else (0, 3)
    if replicates > 0:
        width, height = real_image.size
        replicates = min(replicates, max_replicates)

        image = Image.new(real_image.mode, (width + (genotypes - 1) * (width + margin), height + replicates * (height + margin)))
        # Paste the real image overlapping the correct genotype
        label = example.get("label", 0)
        image.paste(real_image, (label * (width + margin), 0))

        synth_tensor = example["sim.images"]
        for repl in range(replicates):
            for gt in range(genotypes):
                synth_image_tensor = synth_tensor[repl, gt]
                synth_image = generator.render(synth_image_tensor, **kwargs)

                coord = (gt * (width + margin), (repl + 1) * (height + margin))
                image.paste(synth_image, coord)
    else:
        image = real_image

    if out_path:
        image.save(out_path)
    return image


def make_graph_example(
    cfg,
    graph: Graph,
    sampler: HaplotypeSamplerOverlay,
    region: Range,
    vcf_path: str,
    sample: Sample,
    read_path: PathType,
    filtered_kmer_path: str,
    analysis_variants: Sequence[Variant],
    generator: ImageGenerator | None = None,
    addl_features: dict | None = None,
    ploidy: int = 2,
    **kwargs,
):
    with contextlib.ExitStack() as stack:
        tmp_dir = stack.enter_context(tempfile.TemporaryDirectory(ignore_cleanup_errors=True))
        example: dict = { "region": str(region) }

        # Sample haplotypes using sample-specific k-mer counts and filtered k-mers (i.e., only those k-mers that are
        # unique to the graph and present in the sample)
        counts = KmerClassify(filtered_kmer_path, sample.kmer_coverage)
        sampler.initialize_scores(counts)
        haplotypes = sampler.sample_haplotypes(n=cfg.kmer.max_haplotypes)
        assert len(haplotypes) > 0, f"No haplotypes sampled for region {region} in {sample.name}"

        # Prepare haplotypes for genotyping by ensuring the reference haplotype, and any "true" haplotypes, if present
        # in the graph, are included with the reference haplotype guaranteed to be at index 0.
        haplotypes, alleles, true_haplotype_idxs = prepare_genotyping_haplotypes(
            graph, sampler, haplotypes, analysis_variants, region.contig, sample.name, ploidy=ploidy
        )
        assert len(haplotypes) >= 2, f"Fewer than 2 haplotypes for region {region} in {sample.name}"  # noqa: PLR2004
        assert len(true_haplotype_idxs) == ploidy, f"Expected {ploidy} true haplotypes, got {len(true_haplotype_idxs)}"

        # Allow for all possible genotypes, or TODO: apply sampling here to reduce the potential number considered
        possible_genotypes = list(itertools.combinations_with_replacement(range(len(haplotypes)), ploidy))

        # Generate labels if the true haplotype combination is fully defined (i.e., no missing alleles)
        if None not in true_haplotype_idxs:
            norm_true_haplotype_idxs = tuple(sorted(true_haplotype_idxs)) # type: ignore
            genotype_idx = next((i for i, gt in enumerate(possible_genotypes) if gt == norm_true_haplotype_idxs), -1)
            assert genotype_idx >= 0, f"True haplotype combination {norm_true_haplotype_idxs} not found in possible genotypes {possible_genotypes}"

            example["label"] = genotype_idx

            # Determined "ranked" positives based on shared inference paths, presence, etc.
            # TODO: Incorporate shared alleles, allele similarity, etc. into the ranking of positives
            ranked_positives = np.zeros(len(possible_genotypes), dtype=np.long)
            ranked_positives[genotype_idx] = 1 # True (or "first-rank") positive

            # The same "presence" for non-reference genotypes, i.e., non-reference concordant, are considered "third-rank" positives
            # TODO: Should only variants with the true allele be considered "third-rank" positives? And any presence be "fourth-rank"?
            if genotype_idx > 0:
                ranked_positives[(ranked_positives == 0) & (np.arange(len(possible_genotypes)) > 0)] = 3

            example["label.rank"] = ranked_positives

        # The graph (and thus each haplotype's sequence) does not extend beyond `region`, but realignment and simulation need
        # sequences comfortably longer than the fragment/insert size. Pad with reference sequence out to realigner_flank.
        sim_region = region.expand(cfg.pileup.realigner_flank)
        flank_seq = _reference_sequence(cfg.reference, sim_region)
        left_flank = flank_seq[: region.start - sim_region.start]
        right_flank = flank_seq[len(flank_seq) - (sim_region.end - region.end) :]

        # Create realigner once for this region. The (flanked) sequences are also reused, unmodified, for the
        # simulation fasta below.
        realign_fasta_path = os.path.join(tmp_dir, "realign.fasta")
        with open(realign_fasta_path, "w") as fasta:
            # Extract the reference sequence from the first haplotype
            ref_seq = graph.path_sequence(haplotypes[0])
            assert ref_seq.isupper(), f"Sequence for haplotype 0 is not upper case in region {region}"
            fasta.write(">seq-0\n")
            fasta.write(left_flank + ref_seq + right_flank + "\n")

            for i, haplotype in enumerate(itertools.islice(haplotypes, 1, None), start=1):
                sequence = graph.path_sequence(haplotype)
                assert sequence.isupper(), f"Sequence for haplotype {i} is not upper case in region {region}"
                fasta.write(f">seq-{i}\n")
                fasta.write(left_flank + sequence + right_flank + "\n")

        addl_args = { "num_alts": len(haplotypes) - 1 }  # This is needed to prevent C++ errors
        realigner = FragmentRealigner(realign_fasta_path, sample.mean_insert_size, sample.std_insert_size, **addl_args)

        # Because the reference sequence may be sampled, it is not guaranteed to be the same length as the region
        # TODO: Do we need to pad out the shorter sequences so everything has identical length?

        # Create generator if not provided
        generator: ImageGenerator = generator or hydra.utils.instantiate(cfg.generator, cfg=cfg, _recursive_=False)

        # Construct image for "real" data, possibly downsampling the reads if specified
        with tempfile.TemporaryDirectory(dir=tmp_dir) as local_read_tmp_dir:
            local_read_path = read_path
            if cfg.pileup.downsample < 1.0:
                # Downsample reads if specified
                local_read_path = downsample_reads(
                    local_read_path,
                    region.expand(cfg.pileup.fetch_flank),
                    local_read_tmp_dir,
                    downsample=cfg.pileup.downsample,
                )

            image_tensor = generator.generate(
                local_read_path,
                sample,
                region,
                realigner=realigner,
                ref_seq=ref_seq,
                compress=True,
            )

        example["image"] = image_tensor
        if addl_features:
            # Add any additional (extension) features
            example.update(addl_features)

        if cfg.simulation.replicates == 0:
            # No more work to be done if there are no simulations specified
            return example
        assert cfg.simulation.replicates == 1, "Multiple replicates not yet supported"


        # Simulate reads for `ploidy`` copies of all haplotypes as a single step, then construct genotype images from
        # subsets of the simulated reads. Copy the (flanked) sequences already written to the realigner fasta above,
        # just relabeling the headers for downstream splitting by "SQ" tag.
        simulation_fasta_path = os.path.join(tmp_dir, "simulation.fasta")
        with pysam.FastxFile(realign_fasta_path) as realign_fasta, open(simulation_fasta_path, "w") as simulation_fasta_file:
            for i, entry in enumerate(realign_fasta):
                for p in range(ploidy):
                    simulation_fasta_file.write(f">seq-{i}-{p}\n")
                    simulation_fasta_file.write(f"{entry.sequence}\n")

        # TODO: This would be become a loop over replicates with different coverage, etc. For example:
        # If we are augmenting the simulated data, use the provided statistics for the first example, so it
        # will hopefully be most similar to the real data and then augment the remaining replicates
        # if cfg.simulation.augment:
        #     repl_samples = augment_sample(sample, cfg.simulation.replicates, keep_original=True)
        # else:
        #     repl_samples = [sample] * cfg.simulation.replicates

        repl_sample = sample
        try:
            sample_coverage = (
                sample.chrom_mean_coverage(graph.region.contig)
                if cfg.simulation.chrom_norm_covg
                else sample.mean_coverage
            )
            replicate_bam_path = simulate_variant_sequencing(
                simulation_fasta_path,
                (sample_coverage * cfg.pileup.downsample) / ploidy, # Haplotype coverage
                sample,
                reference=cfg.reference,
                shared_reference=cfg.shared_reference,
                dir=tmp_dir,
                stats_path=cfg.stats_path if cfg.simulation.gc_norm_covg else None,
                region=sim_region,
                #phase_vcf_path=background_vcf if cfg.pileup.haplotag_sim else None,
                aligner=cfg.pileup.aligner,
            )
        except ValueError as e:
            e.add_note(f"Failed to synthesize data for region {region} in sample {sample.name}")
            raise

        # Split the single replicate BAM into separate BAMs for each haplotype, which are then used to construct
        # the images for each genotype
        replicate_bam_splits = split_bam_by_tag(replicate_bam_path, tmp_dir, tag="SQ")

        # TODO: Save BAMs for debugging
        # if not OmegaConf.is_missing(cfg.simulation, "save_sim_bam_dir"):
        #   sim_bam_path = os.path.join(cfg.simulation.save_sim_bam_dir, f"{'_'.join(map(str, allele_indices))}_{repl}.bam")
        #   shutil.copy(replicate_bam_path, sim_bam_path)
        #   shutil.copy(f"{replicate_bam_path}.bai", f"{sim_bam_path}.bai")

        # Use the split BAM files to construct images for all genotypes of interest
        replicate_images = []
        for possible_genotype in possible_genotypes:
            # For each possible genotype, use the relevant split BAM file to construct the image
            replicate_bams = [replicate_bam_splits[f"{h}_{i}"] for i, h in enumerate(possible_genotype)]
            synth_image_tensor = generator.generate(
                replicate_bams,
                repl_sample,
                region,
                realigner=realigner,
                ref_seq=ref_seq,
                compress=True,
            )
            replicate_images.append(synth_image_tensor)

        # Stack all the genotypes into a single tensor (adding singleton replicate dimension)
        sim_image_tensor = np.stack(replicate_images)
        example["sim.images"] =  np.expand_dims(sim_image_tensor, axis=0)

    return example



class ExampleActor:
    def __init__(self, index: int, output_dir: str, cfg):
        self.output_path = os.path.join(output_dir, f"images-{index:04d}.tar.gz")
        self.cfg = cfg

        self._writer = wds.TarWriter(self.output_path)

    def close(self):
        # `__ray_shutdown__` would eventually close the writer on actor termination, but that runs
        # asynchronously at some point in the future. To ensure files are closed, invoke and wait on this
        # remote method before using the webdataset files.
        self._writer.close()

@ray.remote
class GraphWriter(ExampleActor):
    def __init__(
        self,
        index: int,
        output_dir: str,
        cfg,
        vcf_path: str,
        read_path: PathType,
        sample: Sample,
        filtered_kmer_path: str,
        *,
        ploidy: int = 2,
        min_variant_size: int = 50,
    ):
        super().__init__(index, output_dir, cfg)
        self.vcf_path = vcf_path
        self.read_path = read_path
        self.sample = sample
        self.filtered_kmer_path = filtered_kmer_path
        self.ploidy = ploidy
        self.min_variant_size = min_variant_size
        self.generator = hydra.utils.instantiate(cfg.generator, cfg=cfg, _recursive_=False)

    def from_graph(
        self,
        region: Range,
        graph: Graph,
        sampler: HaplotypeSamplerOverlay,
        analysis_variants: Sequence[Variant],
    ) -> dict:
        return make_graph_example(
            self.cfg,
            graph,
            sampler,
            region,
            self.vcf_path,
            self.sample,
            self.read_path,
            self.filtered_kmer_path,
            analysis_variants,
            generator=self.generator,
            ploidy=self.ploidy,
        )

    def from_shard(self, shard_path: PathType) -> int:
        num_regions = 0
        with VariantFileReader.open(self.vcf_path) as vcf_file:
            for record in wds.WebDataset([shard_path], shardshuffle=False):
                region = Range(record["region.txt"].decode())
                graph = Graph.load_bytes(record["graph.bytes"])
                unique_kmers = UniqueKmersOverlay(graph, record["unique_kmer_overlay.bytes"])
                sampler = HaplotypeSamplerOverlay(graph, unique_kmers, self.vcf_path, region, self.min_variant_size)
                analysis_variants = _filter_variants(list(vcf_file.fetch(region)), self.min_variant_size)

                try:
                    # Attempt to gracefully timeout long running regions.
                    with Timeout(self.cfg.timeout):
                        example = self.from_graph(region, graph, sampler, analysis_variants)
                except TimeoutError:
                    logging.exception("Timeout error for region %s", region)
                    continue

                sample = {
                    "__key__": region.slug,
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
                num_regions += 1
        return num_regions

def vcf_to_graph_examples(
    cfg,
    read_path: str,
    sample: Sample,
    vcf_path: str,
    output_dir: str,
    *,
    progress_bar: bool = False,
    ploidy: int = 2,
    graph_shards: Sequence[PathType]| None =None,
    unique_kmer_path: PathType | None = None,
    filtered_kmer_path: PathType | None = None,
    region: Range|None=None,
    min_variant_size: int = 50,
):
    with contextlib.ExitStack() as stack:
        # We currently just use ray for the CPU-side work, specifically simulating the SVs. We use a private temporary directory
        # to avoid conflicts between clusters running on the same node. We seem to observe race condition on Ray cleanup of temporary
        # directories, so we ignore cleanup errors here.
        tmp_dir = stack.enter_context(tempfile.TemporaryDirectory(ignore_cleanup_errors=True))
        if not ray.is_initialized():
            ray.init(
                num_cpus=cfg.threads,
                num_gpus=0,
                _temp_dir=tmp_dir,
                include_dashboard=False,
                runtime_env=ray.runtime_env.RuntimeEnv(worker_process_setup_hook=setup_resolvers), # type: ignore
            )
            stack.callback(ray.shutdown)

        # Phase 1: Create and serialize graphs for subsequent analysis along with filtered k-mers
        if graph_shards is None:
            graph_shards, unique_kmer_path, region_count = serialize_graph_and_unique_kmers(
                cfg,
                vcf_path,
                ref_kmer_counts_path=cfg.kmer.ref_kmer_counts_kmc_prefix,
                output_dir=tmp_dir,
                min_variant_size=min_variant_size,
                pool_kmers=(filtered_kmer_path is None and unique_kmer_path is None),
                progress_bar=progress_bar,
                region=region,
            )
        else:
            region_count = None

        if filtered_kmer_path is None:
            assert unique_kmer_path is not None, "Unique k-mer path must be defined if filtered kmer path is not provided"
            assert sample.kmc_prefix is not None, "Sample must have a KMC database when filtered_kmer_path is not provided"

            filtered_kmer_path = os.path.join(tmp_dir, "filtered_kmers")
            kmc_filter(sample.kmc_prefix, unique_kmer_path, filtered_kmer_path, threads=cfg.threads)

        assert _kmc_db_kmer_size(filtered_kmer_path) == cfg.kmer.kmer_size, "Filtered k-mer database has unexpected k"

        if region_count is not None:
            logging.info("Generating exhaustive images for %d regions (across %d threads)", region_count, cfg.threads)
        else:
            logging.info("Generating exhaustive images across %d threads", cfg.threads)

        os.makedirs(output_dir, exist_ok=True)

        # Phase 2: Process each shard as a Ray task in parallel to generate images
        actors = [
            GraphWriter.remote(
                i,
                output_dir,
                cfg,
                str(vcf_path),
                str(read_path),
                sample,
                str(filtered_kmer_path),
                ploidy=ploidy,
                min_variant_size=min_variant_size,
            )
            for i in range(cfg.threads)
        ]
        pool = ray.util.ActorPool(actors)

        gen = pool.map_unordered(lambda actor, shard: actor.from_shard.remote(shard), graph_shards) # type: ignore
        with tqdm(disable=not progress_bar) as progress:
            for num_regions in gen:
                progress.update(num_regions) # type: ignore

        # Explicitly close (and wait for) each actor's webdataset writer before deleting the actor handles
        # and reading the files actors produced.
        ray.get([actor.close.remote() for actor in actors]) # type: ignore
        del pool, actors




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
