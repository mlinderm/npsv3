import contextlib
import glob
import logging
import os
import tempfile

import pandas as pd
import webdataset as wds
from tqdm import tqdm

from npsv3._native_graph import Graph, HaplotypeSamplerOverlay, KmerClassify, Range, UniqueKmersOverlay, VariantFileReader
from npsv3.images.population import overlapping_variants
from npsv3.util.sample import Sample, filter_kmc_database, filter_kmc_database_from_fasta


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
            result.append(variant)
    return result

def _unique_node_sets(graph: Graph, variants) -> dict[tuple[str, int], set[int]]:
    """Construct dictionary of uniquely identifying nodes for variant alleles"""
    unique_node_sets = {}
    for variant in variants:
        variant_id = variant.variant_id
        ref_nodes = set(graph.path_nodes(f"_alt_{variant_id}_0"))

        all_alt_nodes = set()
        for a in range(1, variant.num_alleles):
            if variant.allele_length_change(a) is None:
                continue  # Skip star alleles
            alt_nodes = set(graph.path_nodes(f"_alt_{variant_id}_{a}"))
            alt_nodes.difference_update(ref_nodes)
            all_alt_nodes.update(alt_nodes)
            unique_node_sets[(variant_id, a)] = alt_nodes

        unique_node_sets[(variant_id, 0)] = ref_nodes.difference(all_alt_nodes)

    return unique_node_sets

def serialize_graph_and_unique_kmers(cfg, vcf_path: str | os.PathLike, sample: Sample, output_dir: str | os.PathLike, *, min_variant_size=50, filter_kmers=False, max_count_shard=1000) -> tuple[list[str], str, int]:
    """Serialize all graphs and unique_kmer overlays for regions in vcf_path

    Args:
        cfg: Hydra config
        vcf_path (str | os.PathLike): _description_
        sample (Sample): _description_
        output_dir (str | os.PathLike): _description_
        min_variant_size (int, optional): _description_. Defaults to 50.
        filter_kmers (bool, optional): _description_. Defaults to False.
        max_count_shard (int, optional): _description_. Defaults to 1000.

    Returns:
        tuple[list[str], str]: List of shard paths and path to (filtered) k-mer database
    """

    # Build graphs and unique k-mers one region at a time, serializing each
    # directly into a WebDataset shard so only one region's native objects are in memory
    # at once. K-mer sequences are written incrementally to a single shared FASTA for the
    # combined filtering step.

    with contextlib.ExitStack() as stack:
        tmp_dir =stack.enter_context(tempfile.TemporaryDirectory())  

        combined_fasta_path = os.path.join(tmp_dir, "combined_kmers.fa")
        shard_pattern = os.path.join(output_dir, "graphs-%05d.tar.gz")
        region_idx = None

        vcf_file = stack.enter_context(VariantFileReader.open(vcf_path))
        fasta = stack.enter_context(open(combined_fasta_path, "w")) if filter_kmers else FileNotFoundError
        sink = stack.enter_context(wds.ShardWriter(shard_pattern, maxcount=max_count_shard, verbose=False))

        for region_idx, (region, variants) in enumerate(overlapping_variants(vcf_file, flank=cfg.pileup.variant_padding)):
            analysis_variants = _filter_variants(variants, min_variant_size)
            if not analysis_variants:
                continue

            # TODO: For the unique k-mer analysis to work correctly, we may need to expand
            # the region to include nearby repetitive sequence. For example, if the variant
            # is a DEL of a single copy of a longer repeat, a small flank may not include
            # other copies of the repeat and thus overestimate the number of unique k-mers.

            graph = Graph(str(cfg.reference), str(vcf_path), region)
            unique_kmers = UniqueKmersOverlay(
                graph,
                cfg.kmer.kmer_size,
                max_edges=cfg.kmer.max_edges,
                exclude_universal=True,
                canonicalize=cfg.kmer.canonicalize,
            )
            if filter_kmers:
                for kmer_idx, seq in enumerate(unique_kmers.sequences):
                    fasta.write(f">r{region_idx}_{kmer_idx}\n{seq}\n")

            sink.write({
                "__key__": f"{region.slug}",
                "region.txt": str(region),
                "graph.bytes": graph.save_bytes(),
                "unique_kmer_overlay.bytes": unique_kmers.save_bytes(),
            })

        region_count = region_idx + 1 if region_idx is not None else 0
        logging.info("Created and saved graphs for %d region(s)", region_count)

        # Single k-mer filtering step across all regions
        if region_count > 0 and filter_kmers:
            filtered_kmer_path = os.path.join(output_dir, "filtered_kmers")
            filter_kmc_database_from_fasta(
                sample.kmc_prefix,
                combined_fasta_path,
                cfg.kmer.kmer_size,
                filtered_kmer_path,
                canonicalize=cfg.kmer.canonicalize,
                tmp_dir=tmp_dir,
                threads=cfg.threads,
            )
        else:
            filtered_kmer_path = sample.kmc_prefix

    return glob.glob(os.path.join(output_dir, "graphs-*.tar.gz")), filtered_kmer_path, region_count


def genotypes_in_topk(cfg, vcf_path: str | os.PathLike, sample: Sample, *, min_variant_size=50, filter_kmers=False, progress_bar=False, graph_shards=None, filtered_kmer_path=None) -> pd.DataFrame:
    """For each variant region, score sampled diplotypes against the true diplotype and
    accumulate rank statistics.

    Args:
        cfg: Hydra config
        vcf_path (str | os.PathLike): Path to the input VCF (must be indexed).
        sample (Sample): Sample object with name, kmc_prefix and kmer_coverage attributes.
        min_variant_size (int, optional): _description_. Defaults to 50.
        filter_kmers (bool, optional): _description_. Defaults to False.
        graph_shards (list, optional): List of graph shard paths. Defaults to None.
        filtered_kmer_path (str, optional): Path to the filtered k-mer database. Defaults to None.
    """
    result_table = []

    with contextlib.ExitStack() as stack:
        tmp_dir = stack.enter_context(tempfile.TemporaryDirectory())

        # Phase 1: Create and serialize graphs for subsequent analysis along with filtered k-mers
        region_count = None
        if graph_shards is None:
            graph_shards, filtered_kmer_path, region_count = serialize_graph_and_unique_kmers(
                cfg, vcf_path, sample, output_dir=tmp_dir, min_variant_size=min_variant_size, filter_kmers=filter_kmers
            )
        if filtered_kmer_path is None:
            filtered_kmer_path = sample.kmc_prefix
        if not graph_shards or not filtered_kmer_path:
            msg = "No graph shards or filtered k-mer path available for genotype analysis"
            raise ValueError(msg)

        # Phase 2: Process each region, potentially in parallel across shards, to sample diplotypes and compute genotype ranks
        vcf_file = stack.enter_context(VariantFileReader.open(vcf_path))
        sample_idx = vcf_file.samples().index(sample.name)
        for record_idx, record in enumerate(tqdm(wds.WebDataset(graph_shards, shardshuffle=False), total=region_count, disable=not progress_bar, desc="Processing graph regions")):
            if record_idx == 5:
                break
            region_string = record["region.txt"].decode()

            region = Range(region_string)
            analysis_variants = _filter_variants(vcf_file.fetch(region), min_variant_size)

            graph = Graph.load_bytes(record["graph.bytes"])
            #graph.dump()
            unique_kmers = UniqueKmersOverlay(graph, record["unique_kmer_overlay.bytes"])
            # TODO: Enable initializing HaplotypeSamplerOverlay from precomputed list of variants to avoid redundant parsing
            # of the VCF
            sampler = HaplotypeSamplerOverlay(graph, unique_kmers, str(vcf_path), region, min_variant_size)
            unique_node_sets = _unique_node_sets(graph, analysis_variants)

            counts = KmerClassify(str(filtered_kmer_path), sample.kmer_coverage)
            sampler.initialize_scores(counts)
            haplotypes = sampler.sample_haplotypes(n=cfg.kmer.max_haplotypes)
            diplotypes = sampler.sample_diplotypes(haplotypes, n=cfg.kmer.max_diplotypes)
            print(haplotypes, [diplotype.haplotypes for diplotype in diplotypes])

            # Translate haplotypes to sets of unique paths
            haplotype_paths = []
            for haplotype in haplotypes:
                nodes = set(haplotype)
                paths = set()
                for path, path_nodes in unique_node_sets.items():
                    if nodes.intersection(path_nodes):
                        paths.add(path)
                haplotype_paths.append(paths)

            for variant in analysis_variants:
                # Get genotype as allele indices, e.g. (0,1), for this variant and sample
                alleles = variant.genotype(sample_idx)
                if any(allele < 0 for allele in alleles):
                    continue  # Skip missing genotypes

                # TODO: How to handle * star alleles here? There isn't a true haplotype path, so much as excluded paths.

                matching_haplotypes = tuple(
                    [(variant.variant_id, allele) in paths for paths in haplotype_paths]
                    for allele in alleles
                )

                # Compute the rank (0-indexed) of the true haplotypes
                haplotype_idxs = [
                    next((i for i, match in enumerate(allele_matches) if match), -1)
                    for allele_matches in matching_haplotypes
                ]
                if -1 not in haplotype_idxs:
                    # Normalize diplotype (remove phasing) and compute rank (0-indexed)
                    haplotype_idxs = sorted(haplotype_idxs)
                    diplotype_idx = next(
                        (i for i, diplotype in enumerate(diplotypes)
                         if sorted(diplotype.haplotypes) == haplotype_idxs),
                        -1,
                    )
                else:
                    # If either haplotype is not found, there can't be a correct diplotype match
                    diplotype_idx = -1

                result_table.append({
                    "region": region_string,
                    "variant": variant.variant_id,
                    "sample": sample.name,
                    "haplotypes": len(haplotypes),
                    "haplotype_idxs": tuple(haplotype_idxs),
                    "diplotypes": len(diplotypes),
                    "diplotype_idx": diplotype_idx,
                })

    return pd.DataFrame(result_table)
