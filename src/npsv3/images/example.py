import collections
import contextlib
import dataclasses
import gzip
import itertools
import json
import logging
import os
import pathlib
import tempfile
import typing
from collections.abc import Iterator, Sequence

import hydra
import numpy as np
import pysam
import ray
import webdataset as wds
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

try:
    from compression import zstd  # Python 3.14+
except ImportError:
    from backports import zstd

from npsv3.constants import (
    CHW_LAYOUT,
    COMPARISONS_AUX,
    COMPARISONS_DIR,
    HAPLOTYPES_SUFFIX,
    HWC_LAYOUT,
    LEGACY_REAL_SUFFIX,
    LEGACY_SIMS_SUFFIX,
    REAL_SUFFIX,
    SIMS_SUFFIX,
)
from npsv3.graphs.graph import Graph
from npsv3.graphs.haplotype import (
    HaplotypePriorOverlay,
    HaplotypeSamplerOverlay,
    UniqueKmersOverlay,
    _filter_variants,
    build_kmer_evidence_index,
    serialize_graph_and_unique_kmers,
)
from npsv3.graphs.sharding import (
    GRAPH_MEMBER,
    HAPLOTYPE_PRIOR_MEMBER,
    REGION_MEMBER,
    UNIQUE_KMERS_MEMBER,
    decode_graph_record,
    graph_aux_directory,
    graph_digest,
    iter_graph_records,
    split_graph_shards,
)
from npsv3.images.sequence_compare import ComparisonOptions, make_comparison_writer, prepare_comparison_tree
from npsv3.images.generator import ImageGenerator
from npsv3.images.haplotype_tree import (
    FILTERED_KMERS,
    HaplotypeTree,
    HaplotypeTreeWriter,
    haplotype_options,
    prepare_haplotype_tree,
    source_digest,
)
from npsv3.images.haplotypes import (
    FilteredKmers,
    HaplotypeStrategy,
    InsufficientHaplotypesError,
    as_kmer_classify,
    generate_haplotypes,
)
from npsv3.pileup import close_alignment_file_cache, evict_alignment_file
from npsv3.realigner import FragmentRealigner
from npsv3.simulation import simulate_variant_sequencing, split_bam_by_tag
from npsv3.types import PathType
from npsv3.util.config import setup_resolvers
from npsv3.util.range import Range
from npsv3.util.reads import downsample_reads
from npsv3.util.sample import Sample, _kmc_db_kmer_size, kmc_filter
from npsv3.util.timeout import Timeout
from npsv3.util.variant import Variant, VariantFileReader


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
    sample_idx: int,
    sample: Sample,
    read_path: PathType,
    filtered_kmers: FilteredKmers | None,
    analysis_variants: Sequence[Variant],
    generator: ImageGenerator | None = None,
    haplotype_strategy: HaplotypeStrategy | None = None,
    addl_features: dict | None = None,
    ploidy: int = 2,
    exact_reference: bool = False,
    agreement_sink=None,
    haplotypes: dict | None = None,
    **kwargs,
):
    """Generate the example for `region`. Its candidate haplotypes are the precomputed `haplotypes` record (see
    `images.haplotype_tree`), if provided, otherwise they are generated with `haplotype_strategy` (from `filtered_kmers`)."""
    with contextlib.ExitStack() as stack:
        tmp_dir = stack.enter_context(tempfile.TemporaryDirectory(ignore_cleanup_errors=True))
        example: dict = { "region": str(region) }

        # Generate the haplotypes to genotype with the configured strategy (the reference haplotype is at index 0)
        if haplotypes is None:
            assert filtered_kmers is not None, "Filtered k-mers are required to generate haplotypes"
            haplotype_strategy = haplotype_strategy or hydra.utils.instantiate(cfg.haplotype.strategy, cfg=cfg, _recursive_=False)
            haplotypes = generate_haplotypes(
                cfg, graph, sampler, region, sample, sample_idx, filtered_kmers, analysis_variants,
                strategy=haplotype_strategy, ploidy=ploidy, exact_reference=exact_reference,
            )
        paths = haplotypes["paths"]
        true_haplotype_idxs = haplotypes["true"]
        assert len(true_haplotype_idxs) == ploidy, f"Expected {ploidy} true haplotypes, got {len(true_haplotype_idxs)}"

        # Allow for all possible genotypes, or TODO: apply sampling here to reduce the potential number considered
        possible_genotypes = list(itertools.combinations_with_replacement(range(len(paths)), ploidy))
        # Recorded even if no label is defined or no simulated images are generated below, since packing cost
        # and downstream consumers may want the genotype count independent of `label`/`sim.images`.
        example["num_genotypes"] = len(possible_genotypes)

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

        # Link genotype indices back to the graph, e.g., to compare candidates with the sample's assembled paths
        example["haplotypes"] = haplotypes

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
            for i, path in enumerate(paths):
                sequence = graph.path_sequence(path)
                assert sequence.isupper(), f"Sequence for haplotype {i} is not upper case in region {region}"
                fasta.write(f">seq-{i}\n")
                fasta.write(left_flank)
                fasta.write(sequence)
                fasta.write(right_flank)
                fasta.write("\n")

        realigner = None
        if cfg.pileup.allele_evidence in ("bwa", "both"):
            addl_args = { "num_alts": len(paths) - 1 }  # This is needed to prevent C++ errors
            realigner = FragmentRealigner(realign_fasta_path, sample.mean_insert_size, sample.std_insert_size, **addl_args)

        kmer_index = None
        if cfg.pileup.allele_evidence in ("kmer", "both"):
            kmer_index = build_kmer_evidence_index(cfg, graph, sampler, unique_kmers, paths, left_flank, right_flank)

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
                agreement_sink.num_alleles = len(paths)
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

        # Simulate reads for `ploidy`` copies of all haplotypes as a single step, then construct genotype images from
        # subsets of the simulated reads. Copy the (flanked) sequences already written to the realigner fasta above,
        # just relabeling the headers for downstream splitting by "SQ" tag.
        simulation_fasta_path = os.path.join(tmp_dir, "simulation.fasta")
        with pysam.FastxFile(realign_fasta_path) as realign_fasta, open(simulation_fasta_path, "w") as simulation_fasta_file:
            for i, entry in enumerate(realign_fasta):
                for p in range(ploidy):
                    simulation_fasta_file.write(f">seq-{i}-{p}\n")
                    simulation_fasta_file.write(f"{entry.sequence}\n")

        # TODO: Vary the coverage, etc. across replicates. For example, if we are augmenting the simulated data, use
        # the provided statistics for the first replicate, so it will hopefully be most similar to the real data, and
        # then augment the remaining replicates:
        # if cfg.simulation.augment:
        #     repl_samples = augment_sample(sample, cfg.simulation.replicates, keep_original=True)
        # For now, every replicate is simulated with the original sample's parameters.
        repl_samples = [sample] * cfg.simulation.replicates

        sim_image_tensors = []
        for repl, repl_sample in enumerate(repl_samples):
            sample_coverage = (
                repl_sample.chrom_mean_coverage(graph.region.contig) if cfg.simulation.chrom_norm_covg else repl_sample.mean_coverage
            )

            try:
                replicate_bam_path = simulate_variant_sequencing(
                    simulation_fasta_path,
                    (sample_coverage * cfg.pileup.downsample) / ploidy,  # Haplotype coverage
                    repl_sample,
                    reference=cfg.reference,
                    shared_reference=cfg.shared_reference,
                    dir=tmp_dir,
                    stats_path=cfg.stats_path if cfg.simulation.gc_norm_covg else None,
                    region=sim_region,
                    aligner=cfg.pileup.aligner,
                )
            except ValueError as e:
                e.add_note(f"Failed to synthesize data for region {region} in sample {sample.name} (replicate {repl})")
                raise

            # Split the replicate BAM into separate BAMs for each haplotype, which are then used to construct
            # the images for each genotype
            replicate_bam_splits = split_bam_by_tag(replicate_bam_path, tmp_dir, tag="SQ")

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

            # Stack all the genotypes for this replicate into a single tensor
            sim_image_tensors.append(np.stack(replicate_images))

            # The simulated reads are fully consumed, so remove the replicate BAM and its splits (and indices) now
            # rather than at the end of the region, to bound peak disk usage across replicates.
            for split_path in replicate_bam_splits.values():
                evict_alignment_file(split_path)
            for path in pathlib.Path(tmp_dir).glob(f"{pathlib.Path(replicate_bam_path).stem}*"):
                path.unlink(missing_ok=True)

        # Replicate-major (R, G, ...) simulated images
        example["sim.images"] = np.stack(sim_image_tensors)

    return example


PIXEL_COMPRESS_LEVEL = 10
"""zstd level for per-member (channel-first) pixel compression. Many channels are mostly constant within an image, so
channel-first pixels compress to ~57% of the size of channel-last pixels with gzip level 1 (the `HWC_LAYOUT`) and
decompress faster. Higher levels cost generation time for little further reduction.
"""

LEGACY_PIXEL_COMPRESS_LEVEL = 1
"""gzip level for per-member channel-last pixel compression (the `HWC_LAYOUT`)"""

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
    compressed-per-member pixel tar.

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

    The pixel tar stays plain and uncompressed with per-member compression: compressing the tar itself would
    destroy the random access within a shard that the aux lookup depends on. Pixels are written channel-first and
    zstd-compressed (`CHW_LAYOUT`), so the loader can use the decompressed bytes as-is, or channel-last and
    gzip-compressed (`HWC_LAYOUT`), as for datasets generated before `layout` was recorded.
    """

    def __init__(
        self,
        index: int,
        output_dir: str,
        *,
        maxcount: int = 100_000,
        maxsize: float = 3e9,
        layout: str = CHW_LAYOUT,
    ):
        assert layout in (CHW_LAYOUT, HWC_LAYOUT), f"Unknown pixel layout {layout!r}"
        self._layout = layout
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
            encoder=False,  # Members below are already-encoded, pre-compressed bytes
            verbose=0,
        )
        meta_path = os.path.join(meta_dir, f"meta-{index:04d}.jsonl")
        self._meta_file: typing.TextIO = open(meta_path, "w")  # noqa: SIM115

    def _compress(self, array: np.ndarray) -> bytes:
        """Compress channel-last `(..., H, W, C)` pixels in the writer's layout"""
        assert array.dtype == np.uint8, f"Expected uint8 pixels, got {array.dtype}"
        if self._layout == HWC_LAYOUT:
            return gzip.compress(np.ascontiguousarray(array).tobytes(), compresslevel=LEGACY_PIXEL_COMPRESS_LEVEL, mtime=0)
        return zstd.compress(np.ascontiguousarray(np.moveaxis(array, -1, -3)).tobytes(), level=PIXEL_COMPRESS_LEVEL)

    def write(self, example: dict, *, region: Range, sample_name: str) -> None:
        key = energon_key(sample_name, region)

        real = np.asarray(example["image"])
        height, width, channels = real.shape

        # `num_genotypes` (`g`) reflects the number of genotypes considered during example generation, even
        # if there are no simulated images.
        num_genotypes = example.get("num_genotypes", 0)
        sims = example.get("sim.images")
        if sims is None:
            num_replicates = 0
        else:
            sims = np.asarray(sims)
            # Replicates-major (R, G, H, W, C), the single convention in the codebase
            num_replicates, sims_genotypes = sims.shape[:2]
            assert sims_genotypes == num_genotypes, f"Simulated genotypes {sims_genotypes} disagree with genotypes considered {num_genotypes}"
            assert sims.shape[2:] == real.shape, f"Simulated images {sims.shape} disagree with the real image {real.shape}"

        if self._layout == CHW_LAYOUT:
            real_suffix, sims_suffix = REAL_SUFFIX, SIMS_SUFFIX
        else:
            real_suffix, sims_suffix = LEGACY_REAL_SUFFIX, LEGACY_SIMS_SUFFIX
        pixel_sample = {"__key__": key, real_suffix: self._compress(real)}
        if sims is not None:
            pixel_sample[sims_suffix] = self._compress(sims)
        if (haplotypes := example.get("haplotypes")) is not None:
            # Fetched lazily by key (like the pixels), so the metadata the packing buffer reads stays small
            pixel_sample[HAPLOTYPES_SUFFIX] = gzip.compress(json.dumps(haplotypes).encode("utf-8"), mtime=0)
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
                    "layout": self._layout,
                    "label": None if label is None else int(label),
                    "label_rank": (
                        None if label_rank is None else [int(rank) for rank in label_rank]
                    ),
                    "region": example["region"],
                    "sample": sample_name,
                    # The region's members in the `graphs` aux tree, and that graph's identity (see `npsv3.graphs.sharding`)
                    **{field: example[field] for field in ("graph_key", "graph_digest") if field in example},
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


@dataclasses.dataclass
class GraphRegion:
    """A region's graph, and the objects derived from it, read from a graph shard record"""

    graph_key: str
    region: Range
    sample_idx: int
    analysis_variants: list[Variant]
    graph: Graph
    unique_kmers: UniqueKmersOverlay
    sampler: HaplotypeSamplerOverlay
    graph_digest: str
    source_digest: str
    """Digest of all the graph members the region's haplotypes depend on (see `images.haplotype_tree.source_digest`)"""


class RegionActor:
    """Base for actors that process every region in (strided splits of) graph shards for a sample"""

    def __init__(
        self,
        cfg,
        vcf_path: str,
        sample: Sample,
        filtered_kmer_path: str | None,
        *,
        ploidy: int = 2,
        min_variant_size: int = 50,
        exact_reference: bool = False,
    ):
        self.cfg = cfg
        self.vcf_path = vcf_path
        self.sample = sample
        self.ploidy = ploidy
        self.min_variant_size = min_variant_size
        self.exact_reference = exact_reference
        self.haplotype_strategy = hydra.utils.instantiate(cfg.haplotype.strategy, cfg=cfg, _recursive_=False)
        self.sampler_params = hydra.utils.instantiate(cfg.haplotype.sampler_params)
        self._filtered_kmer_path = filtered_kmer_path
        self._filtered_kmers = None

    @property
    def filtered_kmers(self):
        """The filtered k-mer database, opened once per actor (instead of per region) so the random-access prefix LUT is
        loaded once and reused for every region the actor processes, and only when first needed"""
        if self._filtered_kmers is None:
            assert self._filtered_kmer_path is not None, "Filtered k-mers are required to generate haplotypes"
            self._filtered_kmers = as_kmer_classify(self._filtered_kmer_path, self.sample)
        return self._filtered_kmers

    def generate_haplotypes(self, item: GraphRegion) -> dict:
        return generate_haplotypes(
            self.cfg,
            item.graph,
            item.sampler,
            item.region,
            self.sample,
            item.sample_idx,
            self.filtered_kmers,
            item.analysis_variants,
            strategy=self.haplotype_strategy,
            ploidy=self.ploidy,
            exact_reference=self.exact_reference,
        )

    def iter_regions(self, shard_path: PathType, split: int = 0, num_splits: int = 1) -> Iterator[GraphRegion]:
        """Yield the `split`-th of `num_splits` strided subsets of the regions in `shard_path`"""
        with VariantFileReader.open(self.vcf_path) as vcf_file:
            sample_idx = vcf_file.samples().index(self.sample.name)
            for record in iter_graph_records(shard_path, split, num_splits):
                region = Range(record[REGION_MEMBER].decode())
                analysis_variants = _filter_variants(list(vcf_file.fetch(region)), self.min_variant_size)

                members = decode_graph_record(record)
                graph = Graph.load_bytes(members[GRAPH_MEMBER])
                unique_kmers = UniqueKmersOverlay(graph, members[UNIQUE_KMERS_MEMBER])

                # The population-prior overlay is only present in shards built with population_prior=True. If None, it
                # will be ignored by the HaplotypeSamplerOverlay, which will fall back to a uniform prior.
                prior_overlay = (
                    HaplotypePriorOverlay(graph, prior_bytes)
                    if (prior_bytes := members.get(HAPLOTYPE_PRIOR_MEMBER))
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
                yield GraphRegion(
                    graph_key=record["__key__"],
                    region=region,
                    sample_idx=sample_idx,
                    analysis_variants=analysis_variants,
                    graph=graph,
                    unique_kmers=unique_kmers,
                    sampler=sampler,
                    graph_digest=graph_digest(members[GRAPH_MEMBER]),
                    source_digest=source_digest(members),
                )


@ray.remote
class GraphWriter(RegionActor):
    """Generate each region's example (images), using its precomputed haplotypes (and comparison) from `haplotypes_dir`, if
    available, or otherwise generating them inline"""

    def __init__(
        self,
        index: int,
        output_dir: str,
        cfg,
        vcf_path: str,
        read_path: PathType,
        sample: Sample,
        filtered_kmer_path: str | None,
        *,
        ploidy: int = 2,
        min_variant_size: int = 50,
        exact_reference: bool = False,
        haplotypes_dir: str | None = None,
        reuse_comparisons: bool = False,
    ):
        super().__init__(
            cfg, vcf_path, sample, filtered_kmer_path,
            ploidy=ploidy, min_variant_size=min_variant_size, exact_reference=exact_reference,
        )

        # If NPSV3_REALIGN_PROFILE profiling is requested, give this actor's process its own aux log file
        # alongside its webdataset shard, rather than every actor in the pool interleaving REALIGN_PROFILE
        # lines onto a shared stderr. Native code (realigner.cpp) picks this up via getenv on first use, so
        # it must be set before any realignment happens in this process -- true here since __init__ always
        # runs before from_shard/from_graph. No-op (and no file created) unless profiling is enabled.
        if os.environ.get("NPSV3_REALIGN_PROFILE"):
            os.environ["NPSV3_REALIGN_PROFILE_PATH"] = os.path.join(output_dir, f"images-{index:04d}.realign_profile.log")

        self._writer = make_example_writer(cfg, index, output_dir)
        # Compare each example's genotypes with its label for the optional `comparisons` aux tree (if `compare.enabled`)
        self._comparisons = make_comparison_writer(cfg, index, output_dir)

        self.read_path = read_path
        self.generator = hydra.utils.instantiate(cfg.pileup.generator, cfg=cfg, _recursive_=False)
        # Precomputed haplotypes (already checked against the current settings by `vcf_to_graph_examples`)
        self._haplotypes = HaplotypeTree(haplotypes_dir) if haplotypes_dir is not None else None
        self._reuse_comparisons = reuse_comparisons
        # Unlabeled examples are rejected during training (`models.energon._group_accepted`), so needn't be generated
        self._skip_unlabeled = OmegaConf.select(cfg, "skip_unlabeled", default=False)
        self._counts: collections.Counter[str] = collections.Counter()

    def from_graph(
        self,
        region: Range,
        graph: Graph,
        sampler: HaplotypeSamplerOverlay,
        unique_kmers: UniqueKmersOverlay,
        analysis_variants: Sequence[Variant],
        sample_idx: int,
        haplotypes: dict | None = None,
    ) -> dict:
        return make_graph_example(
            self.cfg,
            graph,
            sampler,
            unique_kmers,
            region,
            sample_idx,
            self.sample,
            self.read_path,
            self.filtered_kmers if haplotypes is None else None,
            analysis_variants,
            generator=self.generator,
            haplotype_strategy=self.haplotype_strategy,
            ploidy=self.ploidy,
            exact_reference=self.exact_reference,
            haplotypes=haplotypes,
        )

    def from_shard(self, shard_path: PathType, split: int = 0, num_splits: int = 1) -> int:
        """Generate examples for the `split`-th of `num_splits` strided subsets of the regions in `shard_path`"""
        num_regions = 0
        for item in self.iter_regions(shard_path, split, num_splits):
            region = item.region
            key = energon_key(self.sample.name, region)

            haplotypes = self._haplotypes.lookup(key, item.source_digest) if self._haplotypes is not None else None
            if haplotypes is not None and (reason := haplotypes.get("skipped")) is not None:
                logging.warning("Skipping region %s for sample %s as for its precomputed haplotypes: %s", region, self.sample.name, reason)
                self._counts["skipped"] += 1
                continue
            precomputed = haplotypes is not None

            try:
                # Attempt to gracefully timeout long running regions.
                with Timeout(self.cfg.timeout):
                    if self._skip_unlabeled:
                        # Determine the label (from the haplotypes) before generating any images
                        if haplotypes is None:
                            haplotypes = self.generate_haplotypes(item)
                        if None in haplotypes["true"]:
                            self._counts["unlabeled"] += 1
                            continue
                    example = self.from_graph(
                        region, item.graph, item.sampler, item.unique_kmers, item.analysis_variants, item.sample_idx,
                        haplotypes=haplotypes,
                    )
            except TimeoutError:
                logging.exception("Timeout error for region %s for sample %s", region, self.sample.name)
                continue
            except InsufficientHaplotypesError as e:
                logging.warning("Skipping region %s for sample %s due to too few haplotypes sampled: %s", region, self.sample.name, e)
                continue
            except Exception as e:
                e.add_note(f"Error generating images for region {region} for sample {self.sample.name} from shard {shard_path}")
                raise
            self._counts["precomputed" if precomputed else "generated"] += 1

            example["graph_key"] = item.graph_key
            example["graph_digest"] = item.graph_digest
            self._writer.write(example, region=region, sample_name=self.sample.name)
            if self._comparisons is not None:
                # Precomputed comparisons are only valid for the precomputed haplotypes they were made from
                if precomputed and self._reuse_comparisons and (payload := self._haplotypes.comparison(key)) is not None:  # type: ignore[union-attr]
                    self._comparisons.write_member(key, payload)
                else:
                    self._comparisons.write(key, example["haplotypes"])
            num_regions += 1
        return num_regions

    def counts(self) -> dict[str, int]:
        """The number of regions whose haplotypes were "precomputed", "generated" inline or "skipped" as precomputed, and
        the number of regions skipped as "unlabeled" (with `skip_unlabeled`)"""
        return dict(self._counts)

    def close(self):
        # `__ray_shutdown__` would eventually close the writer on actor termination, but that runs
        # asynchronously at some point in the future. To ensure files are closed, invoke and wait on this
        # remote method before using the webdataset files.
        self._writer.close()
        if self._comparisons is not None:
            self._comparisons.close()
        close_alignment_file_cache()


@ray.remote
class HaplotypeWriter(RegionActor):
    """Generate each region's haplotypes (and comparison) ahead of image generation (see `images.haplotype_tree`)"""

    def __init__(
        self,
        index: int,
        output_dir: str,
        cfg,
        vcf_path: str,
        sample: Sample,
        filtered_kmer_path: str,
        *,
        ploidy: int = 2,
        min_variant_size: int = 50,
        exact_reference: bool = False,
    ):
        super().__init__(
            cfg, vcf_path, sample, filtered_kmer_path,
            ploidy=ploidy, min_variant_size=min_variant_size, exact_reference=exact_reference,
        )
        self._writer = HaplotypeTreeWriter(index, output_dir)
        self._comparisons = make_comparison_writer(cfg, index, output_dir)

    def from_shard(self, shard_path: PathType, split: int = 0, num_splits: int = 1) -> int:
        """Generate haplotypes for the `split`-th of `num_splits` strided subsets of the regions in `shard_path`"""
        num_regions = 0
        for item in self.iter_regions(shard_path, split, num_splits):
            region = item.region
            key = energon_key(self.sample.name, region)
            try:
                with Timeout(self.cfg.timeout):
                    haplotypes = self.generate_haplotypes(item)
            except TimeoutError:
                # Without a record, image generation will attempt the region again
                logging.exception("Timeout error for region %s for sample %s", region, self.sample.name)
                continue
            except InsufficientHaplotypesError as e:
                logging.warning("Skipping region %s for sample %s due to too few haplotypes sampled: %s", region, self.sample.name, e)
                self._writer.write_skipped(key, str(e), digest=item.source_digest)
                num_regions += 1
                continue
            except Exception as e:
                e.add_note(f"Error generating haplotypes for region {region} for sample {self.sample.name} from shard {shard_path}")
                raise

            self._writer.write(key, haplotypes, digest=item.source_digest)
            if self._comparisons is not None:
                self._comparisons.write(key, haplotypes)
            num_regions += 1
        return num_regions

    def close(self):
        self._writer.close()
        if self._comparisons is not None:
            self._comparisons.close()


def _init_ray(cfg, stack: contextlib.ExitStack) -> str:
    """Initialize ray (if not already) for the duration of `stack`, returning a private temporary directory"""
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
    return tmp_dir


def _prepare_graphs_and_kmers(
    cfg,
    sample: Sample,
    vcf_path: str,
    output_dir: str,
    tmp_dir: str,
    *,
    graph_shards: Sequence[PathType] | None,
    unique_kmer_path: PathType | None,
    filtered_kmer_path: PathType | None,
    region: Range | None,
    min_variant_size: int,
    progress_bar: bool,
    filtered_kmer_dir: str,
    require_filtered_kmers: bool = True,
) -> tuple[Sequence[PathType], str | None, int | None, str | None]:
    """Create and serialize graphs (if not provided), kept in `output_dir`'s `graphs_ds`, and the sample's filtered k-mers
    (if not provided), written to `filtered_kmer_dir`.

    Returns:
        Graph shards, filtered k-mer path (None if not available and not `require_filtered_kmers`), the number of regions
        (if known) and the prepared graph tree containing the shards (if any)
    """
    if graph_shards is None:
        graphs_output = os.path.join(output_dir, "graphs_ds")
        os.makedirs(graphs_output, exist_ok=True)
        graph_shards, unique_kmer_path, region_count = serialize_graph_and_unique_kmers(
            cfg,
            vcf_path,
            ref_kmer_counts_path=cfg.graph.ref_kmer_counts_kmc_prefix,
            output_dir=graphs_output,
            unique_kmer_dir=tmp_dir,
            min_variant_size=min_variant_size,
            pool_kmers=(filtered_kmer_path is None and unique_kmer_path is None),
            progress_bar=progress_bar,
            region=region,
        )
    else:
        logging.info("Using %d pre-generated graph shards", len(graph_shards))
        region_count = None
    graphs_dir = graph_aux_directory(graph_shards) if graph_shards else None

    if filtered_kmer_path is None:
        if require_filtered_kmers or (unique_kmer_path is not None and sample.kmc_prefix is not None):
            assert unique_kmer_path is not None, "Unique k-mer path must be defined if filtered kmer path is not provided"
            assert sample.kmc_prefix is not None, "Sample must have a KMC database when filtered_kmer_path is not provided"

            filtered_kmer_path = os.path.join(filtered_kmer_dir, FILTERED_KMERS)
            kmc_filter(sample.kmc_prefix, unique_kmer_path, filtered_kmer_path, threads=cfg.threads)
        else:
            logging.warning("No filtered k-mers are available, so only regions with precomputed haplotypes can be generated")

    if filtered_kmer_path is not None:
        actual_filtered_kmer_size = _kmc_db_kmer_size(filtered_kmer_path)
        assert actual_filtered_kmer_size is not None, f"Filtered k-mer database {filtered_kmer_path} does not exist or could not be read"
        assert actual_filtered_kmer_size == cfg.graph.kmer_size, f"Filtered k-mer database {filtered_kmer_path} has unexpected k"

    return graph_shards, None if filtered_kmer_path is None else str(filtered_kmer_path), region_count, graphs_dir


def _process_shards(actors: list, tasks: list, *, total: int | None, progress_bar: bool) -> None:
    """Process each `(shard, split, num_splits)` task with the next available actor's `from_shard`, then close (and wait
    for) each actor's writers so their files can be used"""
    pool = ray.util.ActorPool(actors)
    gen = pool.map_unordered(lambda actor, task: actor.from_shard.remote(*task), tasks) # type: ignore
    with tqdm(disable=not progress_bar, total=total) as progress:
        for num_regions in gen:
            progress.update(num_regions) # type: ignore

    # Explicitly close (and wait for) each actor's webdataset writer before reading the files actors produced.
    ray.get([actor.close.remote() for actor in actors]) # type: ignore


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
    exact_reference: bool = False,
    haplotypes_dir: PathType | None = None,
):
    """Generate the image dataset for `sample` in `output_dir`.

    Args:
        haplotypes_dir: A `command=haplotypes` output directory (see `vcf_to_haplotypes`) for the sample, generated with
            the current settings, whose haplotypes (and comparisons, if `compare.enabled` with the same options) are used
            for each region they were generated for with the same graph. Haplotypes are generated inline for the rest.
    """
    with contextlib.ExitStack() as stack:
        tmp_dir = _init_ray(cfg, stack)

        tree, reuse_comparisons = None, False
        if haplotypes_dir is not None:
            haplotypes_dir = os.path.abspath(haplotypes_dir)
            tree = HaplotypeTree(haplotypes_dir)
            tree.check(
                haplotype_options(cfg, sample, ploidy=ploidy, exact_reference=exact_reference, min_variant_size=min_variant_size)
            )
            logging.info("Using precomputed haplotypes from %s for sample %s", haplotypes_dir, sample.name)
            if filtered_kmer_path is None and unique_kmer_path is None:
                filtered_kmer_path = tree.filtered_kmer_path
            if OmegaConf.select(cfg, "compare.enabled", default=False):
                options = tree.comparison_options()
                reuse_comparisons = options == ComparisonOptions.from_config(cfg.compare)
                if reuse_comparisons:
                    logging.info("Reusing precomputed comparisons from %s for sample %s", haplotypes_dir, sample.name)
                else:
                    logging.warning(
                        "Comparing every example, since %s has %s", haplotypes_dir,
                        "no comparisons" if options is None else "comparisons made with different options",
                    )

        # Phase 1: Create and serialize graphs for subsequent analysis along with filtered k-mers. Graphs generated here
        # are kept in the output (`graphs_ds`) as the dataset's `graphs` aux tree. With precomputed haplotypes, filtered
        # k-mers are only needed for regions without them.
        graph_shards, filtered_kmer_path, region_count, graphs_dir = _prepare_graphs_and_kmers(
            cfg,
            sample,
            vcf_path,
            output_dir,
            tmp_dir,
            graph_shards=graph_shards,
            unique_kmer_path=unique_kmer_path,
            filtered_kmer_path=filtered_kmer_path,
            region=region,
            min_variant_size=min_variant_size,
            progress_bar=progress_bar,
            filtered_kmer_dir=tmp_dir,
            require_filtered_kmers=tree is None,
        )

        if region_count is not None:
            logging.info("Generating images into %s for %d regions (across %d threads)", output_dir, region_count, cfg.threads)
        else:
            logging.info("Generating images into %s across %d threads", output_dir, cfg.threads)

        os.makedirs(output_dir, exist_ok=True)

        # Phase 2: Process (strided splits of) each shard as a Ray task in parallel to generate images. Shards are split
        # when there are fewer shards than threads so parallelism isn't limited by the number of graph shards. Every actor
        # creates its own output shard, so don't create more actors than there are tasks to process.
        tasks = split_graph_shards(graph_shards, cfg.threads)
        num_actors = min(cfg.threads, len(tasks))
        actors = [
            GraphWriter.remote(
                i,
                output_dir,
                cfg,
                str(vcf_path),
                str(read_path),
                sample,
                filtered_kmer_path,
                ploidy=ploidy,
                min_variant_size=min_variant_size,
                exact_reference=exact_reference,
                haplotypes_dir=None if haplotypes_dir is None else str(haplotypes_dir),
                reuse_comparisons=reuse_comparisons,
            )
            for i in range(num_actors)
        ]
        _process_shards(actors, tasks, total=region_count, progress_bar=progress_bar)
        counts = collections.Counter()
        for actor_counts in ray.get([actor.counts.remote() for actor in actors]): # type: ignore
            counts.update(actor_counts)
        if counts["unlabeled"]:
            logging.info("Skipped %d unlabeled regions for sample %s", counts["unlabeled"], sample.name)
        if tree is not None:
            logging.info(
                "Used precomputed haplotypes from %s for %d regions (and skipped %d), generated them for %d",
                haplotypes_dir, counts["precomputed"], counts["skipped"], counts["generated"],
            )
        del actors

        # Programmatic preparation of the energon dataset
        from npsv3.models.energon import prepare_energon_dataset, set_aux_tree
        dataset_path = prepare_energon_dataset(
            output_dir, num_workers=cfg.threads, aux={"graphs": graphs_dir} if graphs_dir is not None else None
        )
        if OmegaConf.select(cfg, "compare.enabled", default=False) and prepare_comparison_tree(
            output_dir, ComparisonOptions.from_config(cfg.compare), num_workers=cfg.threads
        ):
            set_aux_tree(dataset_path, COMPARISONS_AUX, f"./{COMPARISONS_DIR}")


def vcf_to_haplotypes(
    cfg,
    sample: Sample,
    vcf_path: str,
    output_dir: str,
    *,
    progress_bar: bool = False,
    ploidy: int = 2,
    graph_shards: Sequence[PathType] | None = None,
    unique_kmer_path: PathType | None = None,
    filtered_kmer_path: PathType | None = None,
    region: Range | None = None,
    min_variant_size: int = 50,
    exact_reference: bool = False,
):
    """Generate `sample`'s haplotypes (and comparisons, if `compare.enabled`) for every region into `output_dir`, to be
    used by subsequent image generation for that sample (`vcf_to_graph_examples`'s `haplotypes_dir`). See
    `images.haplotype_tree` for the layout. Image generation must use the same graphs and settings."""
    with contextlib.ExitStack() as stack:
        tmp_dir = _init_ray(cfg, stack)
        os.makedirs(output_dir, exist_ok=True)

        # Filtered k-mers computed here are kept in the output, so image generation can generate the haplotypes for any
        # regions it needs to without recomputing them.
        graph_shards, filtered_kmer_path, region_count, _graphs_dir = _prepare_graphs_and_kmers(
            cfg,
            sample,
            vcf_path,
            output_dir,
            tmp_dir,
            graph_shards=graph_shards,
            unique_kmer_path=unique_kmer_path,
            filtered_kmer_path=filtered_kmer_path,
            region=region,
            min_variant_size=min_variant_size,
            progress_bar=progress_bar,
            filtered_kmer_dir=output_dir,
        )
        assert filtered_kmer_path is not None

        logging.info("Generating haplotypes into %s across %d threads", output_dir, cfg.threads)
        tasks = split_graph_shards(graph_shards, cfg.threads)
        actors = [
            HaplotypeWriter.remote(
                i,
                output_dir,
                cfg,
                str(vcf_path),
                sample,
                filtered_kmer_path,
                ploidy=ploidy,
                min_variant_size=min_variant_size,
                exact_reference=exact_reference,
            )
            for i in range(min(cfg.threads, len(tasks)))
        ]
        _process_shards(actors, tasks, total=region_count, progress_bar=progress_bar)
        del actors

        prepare_haplotype_tree(
            output_dir,
            haplotype_options(cfg, sample, ploidy=ploidy, exact_reference=exact_reference, min_variant_size=min_variant_size),
            filtered_kmer_path=filtered_kmer_path,
            num_workers=cfg.threads,
        )
        if OmegaConf.select(cfg, "compare.enabled", default=False):
            prepare_comparison_tree(output_dir, ComparisonOptions.from_config(cfg.compare), num_workers=cfg.threads)
