import contextlib
import glob
import itertools
import logging
import os
import tempfile
from collections.abc import Sequence

import numpy as np
import pandas as pd
import ray
import webdataset as wds
from tqdm import tqdm

from npsv3 import PathType
from npsv3._native_graph import Diplotype
from npsv3._native_graph import HaplotypeSamplerOverlay as HaplotypeSamplerOverlay
from npsv3._native_graph import KmerClassify as KmerClassify
from npsv3._native_graph import KmerCounts as KmerCounts
from npsv3._native_graph import UniqueKmersOverlay as UniqueKmersOverlay
from npsv3.graphs.graph import Graph
from npsv3.util.config import setup_resolvers
from npsv3.util.range import Range
from npsv3.util.sample import Sample, _kmc_db_kmer_size, filter_kmers_by_unique_kmers, kmc_build_from_fasta, kmc_filter
from npsv3.util.variant import Variant, VariantFileReader


def _create_graph_and_sampler(
    reference: PathType,
    vcf_path: PathType,
    region: Range,
    *,
    k: int,
    min_variant_size=50,
    max_edges=5,
    exclude_universal=True,
    canonicalize=False,
    ref_kmer_counts: KmerCounts | None = None,
) -> tuple[Graph, UniqueKmersOverlay, HaplotypeSamplerOverlay]:
    graph = Graph(str(reference), str(vcf_path), region)
    unique_kmers = UniqueKmersOverlay(
        graph, k, max_edges=max_edges, exclude_universal=exclude_universal, canonicalize=canonicalize, ref_kmer_counts=ref_kmer_counts
    )
    sampler = HaplotypeSamplerOverlay(
        graph, unique_kmers, str(vcf_path), region, min_variant_size
    )
    return graph, unique_kmers, sampler


def _sample_diplotypes_from_counts(
    sampler: HaplotypeSamplerOverlay,
    unique_kmers: UniqueKmersOverlay,
    kmer_path: str | os.PathLike,
    *,
    k: int,
    kmer_coverage: float,
    max_haplotypes=6,
    max_diplotypes=32,
    filter_kmers=False,
):
    with contextlib.ExitStack() as stack:
        if filter_kmers:
            tmp_dir = stack.enter_context(tempfile.TemporaryDirectory())
            filtered_kmer_path = os.path.join(tmp_dir, "kmers")
            filter_kmers_by_unique_kmers(
                kmer_path, unique_kmers, k, filtered_kmer_path, tmp_dir=tmp_dir
            )
        else:
            filtered_kmer_path = kmer_path

        counts = KmerClassify(str(filtered_kmer_path), kmer_coverage)
        sampler.initialize_scores(counts)

        haplotypes = sampler.sample_haplotypes(n=max_haplotypes)
        diplotypes = sampler.sample_diplotypes(haplotypes, n=max_diplotypes)
        return haplotypes, diplotypes


def sample_diplotypes(
    reference: str | os.PathLike,
    vcf_path: str | os.PathLike,
    region: Range,
    kmer_path: str | os.PathLike,
    *,
    k: int,
    kmer_coverage,
    min_variant_size=50,
    max_haplotypes=6,
    max_diplotypes=32,
    filter_kmers=False,
    max_edges=5,
    exclude_universal=True,
    canonicalize=False,
):
    graph, unique_kmers, sampler = _create_graph_and_sampler(
        reference,
        vcf_path,
        region,
        k=k,
        min_variant_size=min_variant_size,
        max_edges=max_edges,
        exclude_universal=exclude_universal,
        canonicalize=canonicalize,
    )
    haplotypes, diplotypes = _sample_diplotypes_from_counts(
        sampler,
        unique_kmers,
        kmer_path,
        k=k,
        kmer_coverage=kmer_coverage,
        max_haplotypes=max_haplotypes,
        max_diplotypes=max_diplotypes,
        filter_kmers=filter_kmers,
    )
    return graph, haplotypes, diplotypes, sampler, unique_kmers


def _filter_variants(variants, min_variant_size):
    """Filter variants to those with one or more alternate alleles having length change >= min_variant_size"""
    result = []
    for variant in variants:
        if any(abs(variant.allele_length_change(i) or 0) >= min_variant_size for i in range(1, variant.num_alleles)):
            result.append(variant)  # noqa: PERF401
    return result

def overlapping_variants(vcf_file: VariantFileReader|PathType, region: Range|None = None, flank=0, min_variant_size=0):
    """Yield separated (non-overlapping) regions and corresponding variants

    Args:
        vcf_file (VariantFileReader | str): VCF file
        region (Range | None, optional): Region to fetch variants from. Defaults to None.
        flank (int, optional): Required separation between variants to define region. Defaults to 0.
        min_variant_size (int, optional): Minimum size of variant to include. Defaults to 0.

    Yields:
        tuple[Region, list[Variant]]: Region and a list of overlapping variants
    """
    # We assume VCF is in sorted order
    current_range = None
    current_variants = []

    if not isinstance(vcf_file, VariantFileReader):
        vcf_file = VariantFileReader.open(str(vcf_file))
        # TODO: Automatically close this file when done
    assert isinstance(vcf_file, VariantFileReader)

    for variant in vcf_file.fetch(region=region):
        if min_variant_size > 0 and not any(abs(variant.allele_length_change(i) or 0) >= min_variant_size for i in range(1, variant.num_alleles)):
            continue  # Skip variants that don't meet the minimum size requirement

        variant_range = variant.reference_region().expand(flank)
        if current_range is None:
            current_range = variant_range
            current_variants = [variant]
        elif current_range.overlaps(variant_range):
            current_range.union_with(variant_range)
            current_variants.append(variant)
        else:
            # Next variant doesn't overlap, so yield current variants and then reset
            yield current_range, current_variants
            current_range = variant_range
            current_variants = [variant]

    # yield any remaining records
    if current_variants:
        yield current_range, current_variants

@ray.remote
class _SerializeGraphAndUniqueKmers:
    """Ray actor to construct a Graph and UniqueKmersOverlay for a region and return as serialized payload."""
    def __init__(
        self,
        reference: str,
        vcf_path: str,
        kmer_size: int,
        graph_shard: str,
        *,
        max_edges=5,
        exclude_universal=True,
        canonicalize=False,
        ref_kmer_counts_path: str | None = None,
        filter_kmer_fasta_path: str | None = None,
        max_size_shard=200*1024*1024, # 200 MB
    ):
        self.reference = reference
        self.vcf_path = vcf_path
        self.kmer_size = kmer_size
        self._graph_writer = wds.ShardWriter(graph_shard, maxsize=max_size_shard, verbose=False)
        self.max_edges = max_edges
        self.exclude_universal = exclude_universal
        self.canonicalize = canonicalize
        if ref_kmer_counts_path is not None:
            self.ref_kmer_counts = KmerCounts(ref_kmer_counts_path)
        else:
            self.ref_kmer_counts = None
        if filter_kmer_fasta_path is not None:
            self.kmer_fasta = open(filter_kmer_fasta_path, "w")
        else:
            self.kmer_fasta = None

    def close(self):
        # `__ray_shutdown__` would eventually close the fasta file on actor termination, but that runs
        # asynchronously at some point in the future. To ensure files are closed, invoke and wait on this
        # remote method before using the fasta files.
        self._graph_writer.close()
        if self.kmer_fasta and not self.kmer_fasta.closed:
            self.kmer_fasta.close()

    def construct_from_region(self, region_str: str):
        """Return a serialized Graph and UniqueKmersOverlay for region_str"""
        region = Range(region_str)
        try:
            graph = Graph(self.reference, self.vcf_path, region)
            unique_kmers = UniqueKmersOverlay(
                graph,
                self.kmer_size,
                max_edges=self.max_edges,
                exclude_universal=self.exclude_universal,
                canonicalize=self.canonicalize,
                ref_kmer_counts=self.ref_kmer_counts,
            )
        except Exception as e:
            e.add_note(f"Error constructing graph and unique kmers for region {region_str}")
            raise

        slug = region.slug
        if self.kmer_fasta is not None:
            for i, seq in enumerate(unique_kmers.sequences):
                self.kmer_fasta.write(f">{slug}_{i}\n{seq}\n")
        self._graph_writer.write({
            "__key__": slug,
            "region.txt": region_str,
            "graph.bytes": graph.save_bytes(),
            "unique_kmer_overlay.bytes": unique_kmers.save_bytes(),
        })


def serialize_graph_and_unique_kmers(
    cfg,
    vcf_path: PathType,
    output_dir: PathType,
    *,
    min_variant_size=50,
    pool_kmers=False,
    ref_kmer_counts_path: PathType | None = None,
    region: Range|None = None,
    max_size_shard=200*1024*1024, # 200 MB
    progress_bar=False,
) -> tuple[list[str], PathType|None, int]:
    """Serialize all graphs and unique_kmer overlays for regions in vcf_path

    Args:
        cfg: Hydra config
        vcf_path (PathType): _description_
        sample (Sample): _description_
        output_dir (PathType): _description_
        min_variant_size (int, optional): _description_. Defaults to 50.
        pool_kmers (bool, optional): _description_. Defaults to False.
        region (Range | None, optional): Specific region to process. Defaults to None.
        max_size_shard (int, optional): _description_. Defaults to 200MB.
        progress_bar (bool, optional): _description_. Defaults to False.
    Returns:
        tuple[list[str], str]: List of shard paths and path to (filtered) k-mer database
    """

    with contextlib.ExitStack() as stack:
        tmp_dir = stack.enter_context(tempfile.TemporaryDirectory(ignore_cleanup_errors=True))
        vcf_file = stack.enter_context(VariantFileReader.open(str(vcf_path)))

        # Create thread-specific graph shards and k-mer fasta files to avoid contention on a single file handle across threads.
        graph_shards = [os.path.join(output_dir, f"graphs-{i:05d}-%05d.tar.gz") for i in range(cfg.threads)]
        kmer_fasta_paths = [os.path.join(tmp_dir, f"combined_kmers.{i}.fa") if pool_kmers else None for i in range(cfg.threads)]
        actors = [
            # Convert all arguments to easily serializable types, e.g, paths to str
            _SerializeGraphAndUniqueKmers.remote(
                str(cfg.reference),
                str(vcf_path),
                cfg.kmer.kmer_size,
                graph_shard,
                max_edges=cfg.kmer.max_edges,
                canonicalize=cfg.kmer.canonicalize,
                ref_kmer_counts_path=str(ref_kmer_counts_path) if ref_kmer_counts_path is not None else None,
                filter_kmer_fasta_path=kmer_fasta_path,
                max_size_shard=max_size_shard,
            ) for graph_shard, kmer_fasta_path in zip(graph_shards, kmer_fasta_paths, strict=True)
        ]
        pool = ray.util.ActorPool(actors)

        logging.info("Generating all graphs and unique k-mers to graphs-%%05d-%%05d.tar.gz")
        region_count = 0
        for variants_region, _ in tqdm(
            overlapping_variants(vcf_file, flank=cfg.pileup.variant_padding, region=region, min_variant_size=min_variant_size),
            disable=not progress_bar,
            desc="Pre-generating graphs and associated k-mers",
            mininterval=1.0,
        ):
            # Utilize _pending_submits to implement back pressure on the number of regions in-flight
            # to avoid excessive memory usage (ActorPool doesn't provide a public API to implement back pressure)
            if len(pool._pending_submits) >= cfg.threads and pool.has_next(): # noqa: SLF001
                pool.get_next_unordered()

            pool.submit(lambda a, v: a.construct_from_region.remote(str(v)), variants_region)
            region_count += 1

        while pool.has_next():
            pool.get_next_unordered()

        # Explicitly close (and wait for) each actor's webdataset writer before deleting the actor handles and
        # reading the files actors produced.
        ray.get([actor.close.remote() for actor in actors]) # type: ignore
        del pool, actors
        logging.info("Created and saved graphs for %d region(s)", region_count)

        if region_count > 0 and pool_kmers:
            assert all(kmer_fasta_paths), "All kmer_fasta_paths must be defined when filter_kmers is True"
            # Combine all k-mers into a single KMC database for subsequent filtering
            unique_kmer_path = os.path.join(output_dir, "unique_kmers")
            kmc_build_from_fasta(
                kmer_fasta_paths, # type: ignore
                cfg.kmer.kmer_size,
                unique_kmer_path,
                tmp_dir,
                canonicalize=cfg.kmer.canonicalize,
                threads=cfg.threads,
            )
        else:
            unique_kmer_path = None

    return glob.glob(os.path.join(output_dir, "graphs-*.tar.gz")), unique_kmer_path, region_count



def _star_alleles(analysis_variants: Sequence[Variant]) -> dict[tuple[str, int], int]:
    """Return map of (variant_id, allele) -> num_alleles for every star ('*') allele among analysis_variants"""
    star_alleles = {}
    for variant in analysis_variants:
        variant_id = variant.variant_id
        for a in range(1, variant.num_alleles):
            if variant.allele_length_change(a) is None:
                star_alleles[(variant_id, a)] = variant.num_alleles
    return star_alleles


def haplotype_alleles(
    sampler: HaplotypeSamplerOverlay, haplotypes: Sequence[Sequence[int]], variants: Sequence[Variant]
) -> list[set[tuple[str, int]]]:
    """Return the set of compatible variant alleles, as (variant_id, allele) tuples, for each haplotype.

    Since a '*' allele has no path of its own, we consider a haplotype compatible with a '*' allele whenever
    none of the other alleles for that variant are in the haplotype.

    Args:
        sampler (HaplotypeSamplerOverlay): Haplotype sampler
        haplotypes (Sequence[Sequence[int]]): List of haplotypes, each as a sequences of node IDs
        variants (Sequence[Variant]): List of variants to consider for compatibility

    Returns:
        list[set[tuple[str, int]]]: List of sets of (variant_id, allele) tuples for each haplotype
    """
    star_alleles = _star_alleles(variants)

    haplotype_paths = []
    for haplotype in haplotypes:
        paths = set(sampler.decode_haplotype(haplotype))
        for (variant_id, star_idx), num_alleles in star_alleles.items():
            if not any((variant_id, other) in paths for other in range(1, num_alleles) if other != star_idx):
                paths.add((variant_id, star_idx))
        haplotype_paths.append(paths)
    return haplotype_paths


def prepare_genotyping_haplotypes(
    graph: Graph,
    sampler: HaplotypeSamplerOverlay,
    haplotypes: Sequence[Sequence[int]],
    analysis_variants: Sequence[Variant],
    contig: str,
    sample_name: str,
    ploidy: int = 2,
) -> tuple[list[Sequence[int]], list[set[tuple[str, int]]], list[int | None]]:
    """Extend and reorder sampled haplotypes so they are suitable for genotyping.

    The returned haplotypes are guaranteed to include the reference haplotype (at index 0) and, for
    any "true" haplotype path present in the graph for `sample_name`, that haplotype as well. Haplotypes
    already present among `haplotypes` are reused (and reordered as needed); the same true haplotype
    repeated across ploidy indices (e.g. a homozygous genotype) is only added once.

    Args:
        graph (Graph): Graph for the region, used to obtain the reference and "true" haplotype paths.
        sampler (HaplotypeSamplerOverlay): Haplotype sampler used to translate haplotypes into sets of
            compatible variant alleles.
        haplotypes (Sequence[Sequence[int]]): Previously sampled haplotypes, each as a sequence of node IDs.
            Not modified in place.
        analysis_variants (Sequence[Variant]): Variants to consider when computing haplotype-allele
            compatibility.
        contig (str): The contig for the reference path.
        sample_name (str): Sample name used to look up "true" haplotype paths in the graph.
        ploidy (int, optional): Number of "true" haplotype paths to look for. Defaults to 2.

    Returns:
        tuple[list[Sequence[int]], list[set[tuple[str, int]]], list[int | None]]: A tuple of:
            - haplotypes: Extended list of haplotypes, with the reference haplotype at index 0.
            - alleles: List of sets of (variant_id, allele) tuples for each haplotype, parallel to `haplotypes`.
            - true_haplotype_idxs: The sample's genotype as indices into `haplotypes`/`alleles`, one per
              ploidy index (in ploidy order), with `None` for any ploidy index whose path isn't present in
              the graph. Empty if none of the sample's "true" haplotype paths are present in the graph.
    """
    haplotypes = list(haplotypes) # Make copy of haplotypes so we can modify it without affecting the caller
    alleles = haplotype_alleles(sampler, haplotypes, analysis_variants)

    # Ensure the reference haplotype is present and at index 0
    ref_alleles = {(variant.variant_id, 0) for variant in analysis_variants}
    ref_haplotype_idx = next(
        (i for i, haplotype_alleles_ in enumerate(alleles) if haplotype_alleles_ == ref_alleles), None
    )
    if ref_haplotype_idx is None:
        # Add the reference haplotype to the sampled haplotypes at index 0
        haplotypes.insert(0, graph.path_nodes(contig))
        alleles.insert(0, ref_alleles)
    elif ref_haplotype_idx != 0:
        # Move the reference haplotype to index 0
        haplotypes.insert(0, haplotypes.pop(ref_haplotype_idx))
        alleles.insert(0, alleles.pop(ref_haplotype_idx))

    # For each ploidy index with a fully resolved "true" path in the graph, ensure that haplotype is
    # present in the haplotype list and report its index (i.e., the sample's genotype).
    # TODO: Generate a "best" path for ploidy indices without a fully resolved path in the graph.
    true_hap_names = [f"{sample_name}#{i}#{contig}#0" for i in range(ploidy)]
    true_hap_allele_idxs, true_hap_paths = tuple(zip(
        *((i, graph.path_nodes(name)) for i, name in enumerate(true_hap_names) if graph.has_path(name)), strict=True
    )) or ((), ())
    if not true_hap_allele_idxs:
        return haplotypes, alleles, []

    true_hap_alleles = haplotype_alleles(sampler, true_hap_paths, analysis_variants)

    true_hap_idxs: list[int | None] = [None] * ploidy
    for allele_idx, hap_path, hap_alleles in zip(true_hap_allele_idxs, true_hap_paths, true_hap_alleles, strict=True):
        hap_idx = next((i for i, haplotype_alleles_ in enumerate(alleles) if haplotype_alleles_ == hap_alleles), None)
        if hap_idx is None:
            # Add the true haplotype to the sampled haplotypes
            haplotypes.append(hap_path)
            alleles.append(hap_alleles)
            hap_idx = len(haplotypes) - 1
        true_hap_idxs[allele_idx] = hap_idx

    return haplotypes, alleles, true_hap_idxs

def _find_matching_diplotype(diplotypes: Sequence[Diplotype], matching_haplotypes: np.ndarray) -> int:
    """Return the index of the first diplotype that matches the genotype in `matching_halotypes` or -1

    Assumes `diplotypes` are not globally phased, but `matching_haplotypes` might be. All permutations of haplotype
    indices in diplotypes are considered when checking for a match.

    Args:
        diplotypes (Sequence[Diplotype]): Sequence of genotypes as ploidy-length tuple of haplotype indices
        matching_haplotypes (np.ndarray): ploidy x haplotype boolean array indicating compatible haplotypes for each allele

    Returns:
        int: Matching index or -1 if no matching diplotype is found
    """
    for diplotype_idx, diplotype in enumerate(diplotypes):
        for haplotype in itertools.permutations(diplotype.haplotypes):
            if all(matching_haplotypes[j][h] for j, h in enumerate(haplotype)):
                return diplotype_idx
    return -1

@ray.remote # type: ignore
def _diplotypes_in_topk_shard(
    shard_path: str,
    vcf_path: str,
    sample_name: str,
    filtered_kmer_path: str,
    *,
    kmer_coverage: float,
    min_variant_size: int,
    max_haplotypes: int,
    max_diplotypes: int,
) -> list[dict]:
    """Sample diplotypes and compute genotype ranks for every region in a single WebDataset shard."""
    result_rows = []
    with VariantFileReader.open(vcf_path) as vcf_file:
        sample_idx = vcf_file.samples().index(sample_name)
        counts = KmerClassify(filtered_kmer_path, kmer_coverage)

        for record in wds.WebDataset([shard_path], shardshuffle=False):
            region_string = record["region.txt"].decode()
            region = Range(region_string)
            analysis_variants = _filter_variants(list(vcf_file.fetch(region)), min_variant_size)

            graph = Graph.load_bytes(record["graph.bytes"])
            unique_kmers = UniqueKmersOverlay(graph, record["unique_kmer_overlay.bytes"])

            # TODO: Initialize HaplotypeSamplerOverlay from precomputed list of variants to avoid redundant VCF parsing
            sampler = HaplotypeSamplerOverlay(graph, unique_kmers, vcf_path, region, min_variant_size)
            sampler.initialize_scores(counts)
            haplotypes = sampler.sample_haplotypes(n=max_haplotypes)
            diplotypes = sampler.sample_diplotypes(haplotypes, n=max_diplotypes)
            assert len(haplotypes) > 0, f"No haplotypes sampled for region {region_string} in shard {shard_path}"

            # Translate haplotypes to sets of (variant_id, allele) pairs they are compatible with
            haplotype_paths = haplotype_alleles(sampler, haplotypes, analysis_variants)

            record_rows = []
            all_matching_haplotypes = []
            for variant in analysis_variants:
                # Get genotype as allele indices, e.g. (0,1), for this variant and sample
                alleles = variant.genotype(sample_idx)
                if any(allele < 0 for allele in alleles):
                    continue  # Skip missing genotypes

                # alleles x haplotypes boolean array indicating indicating compatible haplotypes for each allele
                matching_haplotypes = np.array([
                    [(variant.variant_id, allele) in paths for paths in haplotype_paths]
                    for allele in alleles
                ], dtype=bool)
                all_matching_haplotypes.append(matching_haplotypes)

                # Compute the rank (0-indexed) of the true haplotypes, or -1 if no sampled haplotype match for this variant
                haplotype_idxs = np.where(np.any(matching_haplotypes, axis=1), np.argmax(matching_haplotypes, axis=1), -1)

                # Find the first diplotype that matches both haplotypes, which may not the be the same as the diplotype
                # with the first matching haplotypes (if they don't co-occur). The diplotypes are sampled in sorted order,
                # i.e., (0,1) can be observed, but (1,0) will not be observed. Thus we are reporting the first matching diplotype
                # independent of phase, i.e., for any permutation of haplotype indices in a given diplotype.
                diplotype_idx = _find_matching_diplotype(diplotypes, matching_haplotypes) if -1 not in haplotype_idxs else -1

                record_rows.append({
                    "region": region_string,
                    "variant": variant.variant_id,
                    "sample": sample_name,
                    "haplotypes": len(haplotypes),
                    "haplotype_idxs": tuple(haplotype_idxs),
                    "diplotypes": len(diplotypes),
                    "diplotype_idx": diplotype_idx,
                })

            if len(record_rows) == 0:
                logging.info("Missing genotypes for analysis variants for sample %s in region %s", sample_name, region_string)
                continue

            # Find the first diplotype that matches both haplotypes across all variants
            all_matching_haplotypes = np.all(np.stack(all_matching_haplotypes), axis=0)
            all_haplotype_idxs = tuple(np.where(np.any(all_matching_haplotypes, axis=1), np.argmax(all_matching_haplotypes, axis=1), -1))
            all_diplotype_idx = _find_matching_diplotype(diplotypes, all_matching_haplotypes) if -1 not in all_haplotype_idxs else -1

            # Determine if we match the the complete haplotype, including all variants.
            # TODO: We assume fully phased haplotypes here, but that may not be the case.
            true_haplotypes = [graph.path_nodes(f"{sample_name}#{i}#{region.contig}#0") for i in range(2)]
            true_haplotypes_idxs = tuple(
                next((i for i, haplotype in enumerate(haplotypes) if haplotype == true_haplotype), -1)
                for true_haplotype in true_haplotypes
            )
            if -1 not in true_haplotypes_idxs:
                # Find the first diplotype that matches the true haplotypes
                norm_true_haplotypes_idxs = sorted(true_haplotypes_idxs)
                true_diplotype_idx = next((
                    i for i, diplotype in enumerate(diplotypes)
                    if sorted(diplotype.haplotypes) == norm_true_haplotypes_idxs
                ), -1)
            else:
                true_diplotype_idx = -1

            for record_row in record_rows:
                record_row["all_haplotype_idxs"] = all_haplotype_idxs
                record_row["all_diplotype_idx"] = all_diplotype_idx
                record_row["true_haplotype_idxs"] = true_haplotypes_idxs
                record_row["true_diplotype_idx"] = true_diplotype_idx

            result_rows.extend(record_rows)
    return result_rows


def diplotypes_in_topk(
    cfg,
    vcf_path: PathType,
    sample: Sample,
    *,
    min_variant_size=50,
    progress_bar=False,
    graph_shards: Sequence[PathType]|None =None,
    unique_kmer_path: PathType | None = None,
    filtered_kmer_path: PathType | None = None,
    region: Range | None = None,
) -> pd.DataFrame:
    """For each variant region, score sampled diplotypes against the true diplotype and
    accumulate rank statistics.

    Args:
        cfg: Hydra config
        vcf_path (str | os.PathLike): Path to the input VCF (must be indexed).
        sample (Sample): Sample object with name, kmc_prefix and kmer_coverage attributes.
        min_variant_size (int, optional): _description_. Defaults to 50.
        filter_kmers (bool, optional): _description_. Defaults to False.
        ref_kmer_counts (KmerClassify | None, optional): Reference k-mer counts. Defaults to None.
        graph_shards (list, optional): List of graph shard paths. Defaults to None.
        unique_kmer_path (str, optional): Path to the unique k-mer database. Defaults to None.
        filtered_kmer_path (str, optional): Path to the filtered k-mer database. Defaults to None.
        region (Range | None, optional): Region to analyze. Defaults to None.
    """
    result_table = []

    with contextlib.ExitStack() as stack:
        # We seem to observe race condition on Ray cleanup of temporary directories, so we ignore cleanup errors here
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
            graph_shards, unique_kmer_path, *_ = serialize_graph_and_unique_kmers(
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
            logging.info("Using %d pre-generated graph shards", len(graph_shards))
        if filtered_kmer_path is None:
            assert unique_kmer_path is not None, "Unique k-mer path must be defined if filtered k-mer path is not provided"
            assert sample.kmc_prefix is not None, "sample must have a KMC database when filtered_kmer_path is not provided"

            filtered_kmer_path = os.path.join(tmp_dir, "filtered_kmers")
            kmc_filter(sample.kmc_prefix, unique_kmer_path, filtered_kmer_path, threads=cfg.threads)
        else:
            logging.info("Using pre-generated filtered k-mer database at %s", filtered_kmer_path)

        assert _kmc_db_kmer_size(filtered_kmer_path) == cfg.kmer.kmer_size, "Filtered k-mer database has unexpected k"

        # Phase 2: Process each shard as a Ray task in parallel to sample diplotypes and compute genotype ranks
        pending = [
            _diplotypes_in_topk_shard.remote( # type: ignore
                shard_path,
                str(vcf_path),
                sample.name,
                str(filtered_kmer_path),
                kmer_coverage=sample.kmer_coverage,
                min_variant_size=min_variant_size,
                max_haplotypes=cfg.kmer.max_haplotypes,
                max_diplotypes=cfg.kmer.max_diplotypes,
            )
            for shard_path in graph_shards
        ]

        with tqdm(
            disable=not progress_bar,
            desc="Computing top-k genotypes in graph regions",
        ) as progress:
            while pending:
                done, pending = ray.wait(pending, num_returns=1)
                for ref in done:
                    shard_results = ray.get(ref)
                    result_table.extend(shard_results)
                    progress.update(len(shard_results))

    return pd.DataFrame(result_table)
