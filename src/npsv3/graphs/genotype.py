import contextlib
import glob
import logging
import os
import sys
import tempfile
from collections.abc import Sequence

import numpy as np
import pandas as pd
import ray
import webdataset as wds
from tqdm import tqdm

from npsv3 import PathType
from npsv3._native_graph import (
    Graph,
    HaplotypeSamplerOverlay,
    KmerClassify,
    KmerCounts,
    Range,
    UniqueKmersOverlay,
    VariantFileReader,
)
from npsv3.images.population import overlapping_variants
from npsv3.util.config import setup_resolvers
from npsv3.util.sample import Sample, _kmc_db_kmer_size, filter_kmc_database, filter_kmc_database_from_fasta


def _create_graph_and_sampler(
    reference: str | os.PathLike,
    vcf_path: str | os.PathLike,
    region: Range,
    *,
    k: int,
    min_variant_size=50,
    max_edges=5,
    exclude_universal=True,
    canonicalize=False,
    ref_kmer_counts: KmerClassify | None = None,
) -> tuple[Graph, UniqueKmersOverlay, HaplotypeSamplerOverlay]:
    graph = Graph(reference, vcf_path, region)
    unique_kmers = UniqueKmersOverlay(
        graph, k, max_edges=max_edges, exclude_universal=exclude_universal, canonicalize=canonicalize, ref_kmer_counts=ref_kmer_counts
    )
    sampler = HaplotypeSamplerOverlay(
        graph, unique_kmers, vcf_path, region, min_variant_size
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
            filter_kmc_database(
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

@ray.remote
class _SerializeGraphAndUniqueKmers:
    """Ray actor to construct a Graph and UniqueKmersOverlay for a region and return as serialized payload."""
    def __init__(
        self,
        reference: str,
        vcf_path: str,
        *,
        kmer_size: int,
        max_edges=5,
        exclude_universal=True,
        canonicalize=False,
        ref_kmer_counts_path: str | None = None,
        filter_kmer_fasta_path: str | None = None,
    ):
        self.reference = reference
        self.vcf_path = vcf_path
        self.kmer_size = kmer_size
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

    def __ray_shutdown__(self):
        if self.kmer_fasta and not self.kmer_fasta.closed:
            self.kmer_fasta.close()

    def construct_from_region(self, region_str: str):
        """Return a serialized Graph and UniqueKmersOverlay for region_str"""
        region = Range(region_str)
        graph = Graph(self.reference, self.vcf_path, region)
        unique_kmers = UniqueKmersOverlay(
            graph,
            self.kmer_size,
            max_edges=self.max_edges,
            exclude_universal=self.exclude_universal,
            canonicalize=self.canonicalize,
            ref_kmer_counts=self.ref_kmer_counts,
        )

        slug = region.slug
        if self.kmer_fasta is not None:
            for i, seq in enumerate(unique_kmers.sequences):
                self.kmer_fasta.write(f">{slug}_{i}\n{seq}\n")

        return {
            "slug": slug,
            "region_str": region_str,
            # Wrap the already-produced bytes as uint8 arrays (a zero-copy view, not a
            # copy) so Ray returns them to the driver as zero-copy views into the object
            # store on `ray.get()`, instead of allocating and copying a fresh `bytes`
            # object as it would for a plain `bytes`/`str` return value.
            "graph_bytes": np.frombuffer(graph.save_bytes(), dtype=np.uint8),
            "unique_kmer_bytes": np.frombuffer(unique_kmers.save_bytes(), dtype=np.uint8),
        }


def _serialize_graph_and_unique_kmers(
    cfg,
    vcf_path: PathType,
    sample: Sample,
    output_dir: PathType,
    *,
    min_variant_size=50,
    filter_kmers=False,
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
        filter_kmers (bool, optional): _description_. Defaults to False.
        region (Range | None, optional): Specific region to process. Defaults to None.
        max_size_shard (int, optional): _description_. Defaults to 200MB.
        progress_bar (bool, optional): _description_. Defaults to False.
    Returns:
        tuple[list[str], str]: List of shard paths and path to (filtered) k-mer database
    """

    with contextlib.ExitStack() as stack:
        tmp_dir = stack.enter_context(tempfile.TemporaryDirectory())

        vcf_file = stack.enter_context(VariantFileReader.open(vcf_path))

        shard_pattern = os.path.join(output_dir, "graphs-%05d.tar.gz")
        sink = stack.enter_context(wds.ShardWriter(shard_pattern, maxsize=max_size_shard, verbose=False))

        kmer_fasta_paths = [os.path.join(tmp_dir, f"combined_kmers.{i}.fa") if filter_kmers else None for i in range(cfg.threads)]
        actors = [
            # Convert all arguments to easily serializable types, e.g, paths to str
            _SerializeGraphAndUniqueKmers.remote(
                str(cfg.reference),
                str(vcf_path),
                kmer_size=cfg.kmer.kmer_size,
                max_edges=cfg.kmer.max_edges,
                canonicalize=cfg.kmer.canonicalize,
                ref_kmer_counts_path=str(ref_kmer_counts_path) if ref_kmer_counts_path is not None else None,
                filter_kmer_fasta_path=kmer_fasta_path,
            ) for kmer_fasta_path in kmer_fasta_paths
        ]
        pool = ray.util.ActorPool(actors)

        def _consume_region():
            result = pool.get_next_unordered()  # noqa: F821
            sink.write({
                "__key__": result["slug"],
                "region.txt": result["region_str"],
                # `.tobytes()` copies, but the FASTA/tar path already needs an owned buffer
                # here (and the shard is gzip-compressed downstream, touching every byte
                # regardless), so there's nothing left to gain by avoiding this last copy.
                "graph.bytes": result["graph_bytes"].tobytes(),
                "unique_kmer_overlay.bytes": result["unique_kmer_bytes"].tobytes(),
            })

        logging.info("Pre-generating graphs and unique k-mers to %s", shard_pattern)
        region_count = 0
        for variants_region, variants in tqdm(
            overlapping_variants(vcf_file, flank=cfg.pileup.variant_padding, region=region),
            disable=not progress_bar,
            desc="Pre-generating graphs and associated k-mers",
            mininterval=1.0,
        ):
            analysis_variants = _filter_variants(variants, min_variant_size)
            if not analysis_variants:
                continue

            # Utilize _pending_submits to implement back pressure on the number of regions in-flight
            # to avoid excessive memory usage (ActorPool doesn't provide a public API to implement back pressure)
            if len(pool._pending_submits) >= cfg.threads and pool.has_next(): # noqa: SLF001
                _consume_region()

            pool.submit(lambda a, v: a.construct_from_region.remote(str(v)), variants_region)
            region_count += 1

        while pool.has_next():
            _consume_region()
        del pool, actors # Clean up actors (i.e., ensure files are closed)
        logging.info("Created and saved graphs for %d region(s)", region_count)

        # Single k-mer filtering step across all regions
        if region_count > 0 and filter_kmers:
            assert sample.kmc_prefix is not None, "sample must have a KMC database when filter_kmers is True"
            assert all(kmer_fasta_paths), "All kmer_fasta_paths must be non-None when filter_kmers is True"

            filtered_kmer_path = os.path.join(output_dir, "filtered_kmers")
            filter_kmc_database_from_fasta(
                sample.kmc_prefix,
                kmer_fasta_paths, # type: ignore
                cfg.kmer.kmer_size,
                filtered_kmer_path,
                canonicalize=cfg.kmer.canonicalize,
                tmp_dir=tmp_dir,
                threads=cfg.threads,
            )
        else:
            filtered_kmer_path = sample.kmc_prefix

    return glob.glob(os.path.join(output_dir, "graphs-*.tar.gz")), filtered_kmer_path, region_count


def _star_alleles(analysis_variants) -> dict[tuple[str, int], int]:
    """Return map of (variant_id, allele) -> num_alleles for every star ('*') allele among analysis_variants"""
    star_alleles = {}
    for variant in analysis_variants:
        variant_id = variant.variant_id
        for a in range(1, variant.num_alleles):
            if variant.allele_length_change(a) is None:
                star_alleles[(variant_id, a)] = variant.num_alleles
    return star_alleles


def _haplotype_paths(sampler: HaplotypeSamplerOverlay, haplotypes: Sequence[int], star_alleles: dict[tuple[str, int], int]):
    """For each haplotype, determine the set of (variant_id, allele) pairs it is compatible with

    `sampler.decode_haplotype` reports the alleles a haplotype actually traverses distinguishing nodes for
    (using the same trimmed inference masks that drove sampling, so overlapping variants' shared boundary
    nodes never make a haplotype ambiguous about its own alleles). A star allele has no graph path of its
    own -- it reuses whatever nodes its variant's REF allele would -- so it can never be decoded directly.
    Instead, a star allele is compatible with a haplotype whenever none of that variant's other *real*
    alleles were decoded for it; REF is excluded from that check since REF and a star share the same nodes
    and so cannot be told apart from decoding alone.
    """
    haplotype_paths = []
    for haplotype in haplotypes:
        paths = set(sampler.decode_haplotype(haplotype))
        for (variant_id, star_idx), num_alleles in star_alleles.items():
            if not any((variant_id, other) in paths for other in range(1, num_alleles) if other != star_idx):
                paths.add((variant_id, star_idx))
        haplotype_paths.append(paths)
    return haplotype_paths


@ray.remote # type: ignore
def _genotypes_in_topk_shard(
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
            haplotype_paths = _haplotype_paths(sampler, haplotypes, _star_alleles(analysis_variants))

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

                if -1 not in haplotype_idxs:
                    # Find the first diplotype that matches both haplotypes, which may not the be the same as the diplotype
                    # with the first matching haplotypes.
                    diplotype_idx = next((
                        i for i, diplotype in enumerate(diplotypes)
                        if all(matching_haplotypes[j][h] for j, h in enumerate(diplotype.haplotypes))
                    ), -1)
                else:
                    diplotype_idx = -1

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
            all_diplotype_idx = next((
                i for i, diplotype in enumerate(diplotypes)
                if all(all_matching_haplotypes[j][h] for j, h in enumerate(diplotype.haplotypes))
            ), -1)

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


def genotypes_in_topk(
    cfg,
    vcf_path: PathType,
    sample: Sample,
    *,
    min_variant_size=50,
    filter_kmers=False,
    progress_bar=False,
    graph_shards=None,
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
            graph_shards, filtered_kmer_path, *_ = _serialize_graph_and_unique_kmers(
                cfg,
                vcf_path,
                sample,
                ref_kmer_counts_path=cfg.kmer.ref_kmer_counts_kmc_prefix,
                output_dir=tmp_dir,
                min_variant_size=min_variant_size,
                filter_kmers=filter_kmers,
                progress_bar=progress_bar,
                region=region,
            )
        if filtered_kmer_path is None:
            filtered_kmer_path = sample.kmc_prefix
        if not graph_shards or not filtered_kmer_path:
            msg = "No graph shards or filtered k-mer path available for genotype analysis"
            raise ValueError(msg)
        if (filter_kmer_k := _kmc_db_kmer_size(filtered_kmer_path)) != cfg.kmer.kmer_size:
            msg = f"Filtered k-mer database has k={filter_kmer_k} but expected k={cfg.kmer.kmer_size}"
            raise ValueError(msg)

        # Phase 2: Process each shard as a Ray task in parallel to sample diplotypes and compute genotype ranks
        pending = [
            _genotypes_in_topk_shard.remote( # type: ignore
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
