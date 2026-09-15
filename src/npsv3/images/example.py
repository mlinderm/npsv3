import contextlib
import gzip
import itertools
import json
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
    HaplotypePriorOverlay,
    HaplotypeSamplerOverlay,
    KmerClassify,
    UniqueKmersOverlay,
    _filter_variants,
    add_population_haplotypes,
    build_kmer_evidence_index,
    prepare_genotyping_haplotypes,
    serialize_graph_and_unique_kmers,
)
from npsv3.images.generator import ImageGenerator
from npsv3.models.loaders import REAL_SUFFIX, SIMS_SUFFIX
from npsv3.realigner import FragmentRealigner
from npsv3.simulation import augment_sample, simulate_variant_sequencing, split_bam_by_tag
from npsv3.types import PathType
from npsv3.util.config import setup_resolvers
from npsv3.util.range import Range
from npsv3.util.reads import downsample_reads, haplotag_reads
from npsv3.util.sample import Sample, _kmc_db_kmer_size, kmc_filter
from npsv3.util.timeout import Timeout
from npsv3.util.variant import Variant, VariantFileReader
from npsv3.util.vcf import index_variant_file, pysam_write_mode


class InsufficientHaplotypesError(RuntimeError):
    """Raised when make_graph_example cannot assemble >=2 distinct haplotypes for a region, even after
    attempting to recover a population-panel individual's embedded haplotype for any still-missing
    analysis-variant ALT allele. Caught by GraphWriter.from_shard, which skips (not crashes) the region."""


def _reference_sequence(reference_fasta: str, region: Range) -> str:
    with pysam.FastaFile(reference_fasta) as ref_fasta:
        # Make sure reference sequence is all upper case
        return ref_fasta.fetch(reference=region.contig, start=region.start, end=region.end).upper()


def example_to_image(
    cfg, example, out_path: PathType | None=None, *, with_simulations=False, margin=10, max_replicates=1, **kwargs
):
    generator = hydra.utils.instantiate(cfg.pileup.generator, cfg, _recursive_=False)

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
    unique_kmers: UniqueKmersOverlay,
    region: Range,
    vcf_path: str,
    sample: Sample,
    read_path: PathType,
    filtered_kmer_path: str,
    analysis_variants: Sequence[Variant],
    generator: ImageGenerator | None = None,
    addl_features: dict | None = None,
    ploidy: int = 2,
    agreement_sink=None,
    **kwargs,
):
    with contextlib.ExitStack() as stack:
        tmp_dir = stack.enter_context(tempfile.TemporaryDirectory(ignore_cleanup_errors=True))
        example: dict = { "region": str(region) }

        # Sample haplotypes using sample-specific k-mer counts and filtered k-mers (i.e., only those k-mers that are
        # unique to the graph and present in the sample)
        counts = KmerClassify(filtered_kmer_path, sample.kmer_coverage)
        sampler.initialize_scores(counts)
        haplotypes = sampler.sample_haplotypes(n=cfg.graph.max_haplotypes)
        assert len(haplotypes) > 0, f"No haplotypes sampled for region {region} in {sample.name}"

        # Prepare haplotypes for genotyping by ensuring the reference haplotype, and any "true" haplotypes, if present
        # in the graph, are included with the reference haplotype guaranteed to be at index 0.
        haplotypes, alleles, true_haplotype_idxs = prepare_genotyping_haplotypes(
            graph, sampler, haplotypes, analysis_variants, region.contig, sample.name, ploidy=ploidy
        )
        if len(haplotypes) < 2 and cfg.graph.population_haplotype_topup:  # noqa: PLR2004
            # Only attempt "top-up" from the population if the sampler did not produce any haplotype diversity (and if
            # explicitly requested).
            excluded_samples = { sample.name, *cfg.graph.population_excluded_samples }
            haplotypes, alleles = add_population_haplotypes(
                graph,
                sampler,
                haplotypes,
                alleles,
                analysis_variants,
                region.contig,
                excluded_samples,
                cfg.graph.max_haplotypes,
                ploidy=ploidy,
            )
        if len(haplotypes) < 2:  # noqa: PLR2004
            msg = (
                f"Fewer than 2 haplotypes for region {region} in {sample.name} even after population-panel "
                f"top-up (have {len(haplotypes)})"
            )
            raise InsufficientHaplotypesError(msg)
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
        # sequences comfortably longer than the fragment/insert size. Pad with reference sequence out to `realigner_flank`.
        # TODO: If this large memory consumer, refactor to use memoryview to avoid copying the reference sequence.
        sim_region = region.expand(cfg.pileup.realigner_flank)
        flank_seq = _reference_sequence(cfg.reference, sim_region)
        left_flank = flank_seq[: region.start - sim_region.start]
        ref_seq = flank_seq[region.start - sim_region.start : len(flank_seq) - (sim_region.end - region.end)]
        right_flank = flank_seq[len(flank_seq) - (sim_region.end - region.end) :]
        assert len(ref_seq) == region.length, f"Reference sequence length {len(ref_seq)} does not match region length {region.length} for {region}"

        # Create realigner once for this region. The (flanked) sequences are also reused, unmodified, for the
        # simulation fasta below.
        realign_fasta_path = os.path.join(tmp_dir, "realign.fasta")
        with open(realign_fasta_path, "w") as fasta:
            for i, haplotype in enumerate(haplotypes):
                sequence = graph.path_sequence(haplotype)
                assert sequence.isupper(), f"Sequence for haplotype {i} is not upper case in region {region}"
                fasta.write(f">seq-{i}\n")
                fasta.write(left_flank)
                fasta.write(sequence)
                fasta.write(right_flank)
                fasta.write("\n")

        # The BWA realigner (channels 6/7) and the k-mer evidence index (channel 8, Stage 1 of the
        # "Cheaper allele evidence" plan) are independently gated by `pileup.allele_evidence` so that the
        # `kmer`-only production path can skip the realigner's per-(fragment, allele) BWA cost entirely --
        # the whole point of the k-mer path. `both` builds both, for the dual-run agreement harness.
        realigner = None
        if cfg.pileup.allele_evidence in ("bwa", "both"):
            addl_args = { "num_alts": len(haplotypes) - 1 }  # This is needed to prevent C++ errors
            realigner = FragmentRealigner(realign_fasta_path, sample.mean_insert_size, sample.std_insert_size, **addl_args)

        kmer_index = None
        if cfg.pileup.allele_evidence in ("kmer", "both"):
            kmer_index = build_kmer_evidence_index(cfg, graph, sampler, unique_kmers, haplotypes, left_flank, right_flank)

        # Shared across the real image and every simulated-genotype image below: the same haplotype/ploidy-copy
        # split BAM is reused, unmodified, across every genotype combination that includes that haplotype, so
        # this avoids re-realigning identical fragments once per genotype.
        realign_cache = {}

        # TODO: Do we need to pad out the shorter sequences so everything has identical length?

        # Create generator if not provided
        generator: ImageGenerator = generator or hydra.utils.instantiate(cfg.pileup.generator, cfg=cfg, _recursive_=False)

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

            if agreement_sink is not None:
                # Dual-run agreement harness (see images/agreement.py): tag the sink with this region's
                # context and capture the pre-resize/pre-channel-select tensor too, so channel 6/7 vs. 8
                # pixel correlation is possible -- restricted to the real image (not the simulated-genotype
                # loop below), which is the most representative comparison and keeps harness cost bounded.
                agreement_sink.region_label = str(region)
                agreement_sink.image_index = 0
                agreement_sink.analysis_variants = analysis_variants
                agreement_sink.num_alleles = len(haplotypes)
                image_tensor, raw_image = generator.generate(
                    local_read_path,
                    sample,
                    region,
                    realigner=realigner,
                    kmer_index=kmer_index,
                    ref_seq=ref_seq,
                    realign_cache=realign_cache,
                    compress=True,
                    agreement_sink=agreement_sink,
                    return_raw=True,
                )
                agreement_sink.raw_image = raw_image
            else:
                image_tensor = generator.generate(
                    local_read_path,
                    sample,
                    region,
                    realigner=realigner,
                    kmer_index=kmer_index,
                    ref_seq=ref_seq,
                    realign_cache=realign_cache,
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
                kmer_index=kmer_index,
                ref_seq=ref_seq,
                realign_cache=realign_cache,
                compress=True,
            )
            replicate_images.append(synth_image_tensor)

        # Stack all the genotypes into a single tensor (adding singleton replicate dimension)
        sim_image_tensor = np.stack(replicate_images)
        example["sim.images"] =  np.expand_dims(sim_image_tensor, axis=0)

    return example


PIXEL_COMPRESS_LEVEL = 1
"""zlib level for per-member pixel compression. Pileup pixels are ~38% exact zeros, so level 1
already reaches ~7.6x; higher levels cost generation time for very little.
"""

def energon_key(sample_name: str, region: Range) -> str:
    """Return globally unique, dot-free key for a region's example (incorporating sample name)"""
    # region.slug alone collides when per-sample trees are merged. WebDataset splits the key from the extension at the
    # **first** dot, so a dot anywhere in the key (hg38 alt/decoy contigs like `GL000207.1` have them) would silently
    # truncate it. Dots become underscores rather than being deleted, so distinct contigs stay distinct.
    key = f"{sample_name}_{region.slug}".replace(".", "_")
    assert "." not in key, f"Energon key {key!r} must not contain a dot"
    return key


class EnergonExampleWriter:
    """Writes generated examples to one shard stream owned by a single actor.

    Polylithic energon trees: one flat JSONL metadata file per actor beside a sharded,
    gzipped-per-member pixel tar.

    Splitting a region's metadata from its pixels is what lets the packing buffer decide a pack from
    metadata alone -- `g`, `r` and the geometry are all the cost function needs -- and only then fetch
    the pixels for the groups that were actually selected. `label_rank` rides along in the JSON (a few
    dozen ints) so `pack_selected_samples` does not need a third fetch.

    The metadata file does not rotate with the pixel tar's shards: `prepare_energon_dataset` merges
    every actor's fragment, in `imgs_ds`'s own shard order, into the single `meta_ds/train.jsonl` that
    `dataset.yaml` references, so there is no need for this writer to keep the two in lockstep, or for
    its output to line up 1:1 with `imgs_ds`'s tar shard boundaries.

    The metadata JSONL doubles as the offline manifest that used to be a separate `manifest-*.csv.gz`
    file: since it is already one small flat record per example, `estimate_epoch_batches` can stream
    it directly with plain `json.loads`, no separate artifact needed.

    The pixel tar stays plain and uncompressed with per-member gzip: gzipping the tar itself would
    destroy the random access within a shard that the aux lookup depends on.
    """

    def __init__(self, index: int, output_dir: str, *, maxcount: int = 100_000, maxsize: float = 3e9):
        meta_dir = os.path.join(output_dir, "meta_ds")
        imgs_dir = os.path.join(output_dir, "imgs_ds")
        os.makedirs(meta_dir, exist_ok=True)
        os.makedirs(imgs_dir, exist_ok=True)

        pixel_pattern = os.path.join(imgs_dir, f"imgs-{index:04d}-%06d.tar")
        self._writer = wds.ShardWriter(
            pixel_pattern,
            maxcount=maxcount,
            maxsize=maxsize,
            mtime=0,  # Deterministic shards, so a regenerated shard compares byte-for-byte
            encoder=False,  # Members below are already-encoded, pre-gzipped bytes
            verbose=0,
        )
        meta_path = os.path.join(meta_dir, f"meta-{index:04d}.jsonl")
        self._meta_file: typing.TextIO = open(meta_path, "w")  # noqa: SIM115

    @staticmethod
    def _compress(array: np.ndarray) -> bytes:
        assert array.dtype == np.uint8, f"Expected uint8 pixels, got {array.dtype}"
        return gzip.compress(np.ascontiguousarray(array).tobytes(), compresslevel=PIXEL_COMPRESS_LEVEL, mtime=0)

    def write(self, example: dict, *, region: Range, sample_name: str) -> None:
        key = energon_key(sample_name, region)

        real = np.asarray(example["image"])
        height, width, channels = real.shape

        sims = example.get("sim.images")
        if sims is None:
            num_replicates = num_genotypes = 0
        else:
            sims = np.asarray(sims)
            # Replicates-major (R, G, H, W, C), the single convention in the codebase
            num_replicates, num_genotypes = sims.shape[:2]
            assert sims.shape[2:] == real.shape, f"Simulated images {sims.shape} disagree with the real image {real.shape}"

        pixel_sample = {"__key__": key, REAL_SUFFIX: self._compress(real)}
        if sims is not None:
            pixel_sample[SIMS_SUFFIX] = self._compress(sims)
        self._writer.write(pixel_sample)

        # `label` is the canonical positive; `label.rank` is its derived view, recorded rather than
        # recomputed so the ranking rule can evolve without invalidating shards. Both are null for
        # unlabeled examples.
        label = example.get("label")
        label_rank = example.get("label.rank")
        self._meta_file.write(
            json.dumps(
                {
                    "key": key,
                    "r": num_replicates,
                    "g": num_genotypes,
                    "h": height,
                    "w": width,
                    "c": channels,
                    "label": None if label is None else int(label),
                    "label_rank": (
                        None if label_rank is None else [int(rank) for rank in label_rank]
                    ),
                    "region": example["region"],
                    "sample": sample_name,
                }
            )
        )
        self._meta_file.write("\n")

    def close(self) -> None:
        self._writer.close()
        self._meta_file.close()


def make_example_writer(cfg, index: int, output_dir: str) -> EnergonExampleWriter:
    """Create the per-actor energon example writer."""
    maxcount = OmegaConf.select(cfg, "pileup.example_shard_maxcount", default=100_000)
    maxsize = OmegaConf.select(cfg, "pileup.example_shard_maxsize", default=3e9)
    return EnergonExampleWriter(index, output_dir, maxcount=maxcount, maxsize=maxsize)


class ExampleActor:
    def __init__(self, index: int, output_dir: str, cfg):
        self.cfg = cfg

        # If NPSV3_REALIGN_PROFILE profiling is requested, give this actor's process its own aux log file
        # alongside its webdataset shard, rather than every actor in the pool interleaving REALIGN_PROFILE
        # lines onto a shared stderr. Native code (realigner.cpp) picks this up via getenv on first use, so
        # it must be set before any realignment happens in this process -- true here since __init__ always
        # runs before from_shard/from_graph. No-op (and no file created) unless profiling is enabled.
        if os.environ.get("NPSV3_REALIGN_PROFILE"):
            os.environ["NPSV3_REALIGN_PROFILE_PATH"] = os.path.join(output_dir, f"images-{index:04d}.realign_profile.log")

        self._writer = make_example_writer(cfg, index, output_dir)

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
        self.generator = hydra.utils.instantiate(cfg.pileup.generator, cfg=cfg, _recursive_=False)
        self.sampler_params = hydra.utils.instantiate(cfg.graph.haplotype_sampler_params)

    def from_graph(
        self,
        region: Range,
        graph: Graph,
        sampler: HaplotypeSamplerOverlay,
        unique_kmers: UniqueKmersOverlay,
        analysis_variants: Sequence[Variant],
    ) -> dict:
        return make_graph_example(
            self.cfg,
            graph,
            sampler,
            unique_kmers,
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
                region_string = record["region.txt"].decode()
                region = Range(region_string)
                analysis_variants = _filter_variants(list(vcf_file.fetch(region)), self.min_variant_size)

                graph = Graph.load_bytes(record["graph.bytes"])
                unique_kmers = UniqueKmersOverlay(graph, record["unique_kmer_overlay.bytes"])

                # The population-prior overlay is only present in shards built with population_prior=True. If None, it
                # will be ignored by the HaplotypeSamplerOverlay, which will fall back to a uniform prior.
                prior_overlay = (
                    HaplotypePriorOverlay(graph, prior_bytes)
                    if (prior_bytes := record.get("haplotype_prior_overlay.bytes"))
                    else None
                )

                sampler = HaplotypeSamplerOverlay(
                    graph,
                    unique_kmers,
                    self.vcf_path,
                    region,
                    self.min_variant_size,
                    prior=prior_overlay,
                    params=self.sampler_params,
                )

                try:
                    # Attempt to gracefully timeout long running regions.
                    with Timeout(self.cfg.timeout):
                        example = self.from_graph(region, graph, sampler, unique_kmers, analysis_variants)
                except TimeoutError:
                    logging.exception("Timeout error for region %s for sample %s", region, self.sample.name)
                    continue
                except InsufficientHaplotypesError as e:
                    logging.warning("Skipping region %s for sample %s due to too few haplotypes sampled: %s", region, self.sample.name, e)
                    continue
                except Exception as e:
                    e.add_note(
                        f"Error generating images for region {region_string} for sample {self.sample.name} from shard {shard_path}"
                    )
                    raise

                self._writer.write(example, region=region, sample_name=self.sample.name)
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
                ref_kmer_counts_path=cfg.graph.ref_kmer_counts_kmc_prefix,
                output_dir=tmp_dir,
                min_variant_size=min_variant_size,
                pool_kmers=(filtered_kmer_path is None and unique_kmer_path is None),
                progress_bar=progress_bar,
                region=region,
            )
        else:
            logging.info("Using %d pre-generated graph shards", len(graph_shards))
            region_count = None

        if filtered_kmer_path is None:
            assert unique_kmer_path is not None, "Unique k-mer path must be defined if filtered kmer path is not provided"
            assert sample.kmc_prefix is not None, "Sample must have a KMC database when filtered_kmer_path is not provided"

            filtered_kmer_path = os.path.join(tmp_dir, "filtered_kmers")
            kmc_filter(sample.kmc_prefix, unique_kmer_path, filtered_kmer_path, threads=cfg.threads)

        assert _kmc_db_kmer_size(filtered_kmer_path) == cfg.graph.kmer_size, "Filtered k-mer database has unexpected k"

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

        # Programmatic preparation of the energon dataset
        from npsv3.models.energon import prepare_energon_dataset
        prepare_energon_dataset(output_dir, num_workers=cfg.threads)

