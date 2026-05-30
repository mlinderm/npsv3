from collections import Counter
import contextlib
import json
import logging
import os
from shlex import quote
import subprocess
import tempfile

import pandas as pd

from npsv3._native_graph import Range, VariantFileReader, Graph, UniqueKmersOverlay, HaplotypeSamplerOverlay, KmerCounts 
from npsv3.images.population import overlapping_variants
from npsv3.util.sample import filter_kmc_database


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
) -> tuple[Graph, UniqueKmersOverlay, HaplotypeSamplerOverlay]:
    graph = Graph(reference, vcf_path, region)
    unique_kmers = UniqueKmersOverlay(
        graph, k, max_edges=max_edges, exclude_universal=exclude_universal, canonicalize=canonicalize
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

        counts = KmerCounts(str(filtered_kmer_path), kmer_coverage)
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
            alt_nodes = set(graph.path_nodes(f"_alt_{variant_id}_{a}"))
            alt_nodes.difference_update(ref_nodes)
            all_alt_nodes.update(alt_nodes)
            unique_node_sets[(variant_id, a)] = alt_nodes
        
        unique_node_sets[(variant_id, 0)] = ref_nodes.difference(all_alt_nodes)
    
    return unique_node_sets


def genotypes_in_topk(cfg, vcf_path, sample, min_variant_size=50, filter_kmers=False):
    """For each variant region, score sampled diplotypes against the true diplotype and
    accumulate rank statistics.

    Args:
        cfg: Hydra config with kmer_size, max_edge, n_haplotypes, max_diplotypes,
             min_sv_size, coverage, and reference fields.
        vcf_path: Path to the input VCF (must be indexed).
        sample_kmc_map: dict mapping sample name -> KMC db path (without extension).
    """
    result_table = []

    vcf_file = VariantFileReader.open(vcf_path)
    sample_idx = vcf_file.samples().index(sample.name)
    for region, variants in overlapping_variants(vcf_file, flank=cfg.pileup.variant_padding):
        analysis_variants = _filter_variants(variants, min_variant_size)
        if not analysis_variants:
            continue  # Skip regions with no variants meeting the size threshold

        # TODO: For the unique k-mer analysis to work correctly, we may need to expand the region to include nearby
        # repetitive sequence. For example, if the variant is a DEL of a single copy of a longer repeat, e.g., 1kb,m
        # a small flank may not include other copies of the repeat, and thus overestimate the number of unique k-mers. 

        # Per region operations (shared across samples)
        graph, unique_kmers, sampler = _create_graph_and_sampler(
            cfg.reference,
            vcf_path,
            region,
            k=cfg.kmer.kmer_size,
            min_variant_size=min_variant_size,
            max_edges=cfg.kmer.max_edges,
            exclude_universal=True,
            canonicalize=cfg.kmer.canonicalize,
        )

        # Prepare "unique node sets" for each allele of each variant
        unique_node_sets = _unique_node_sets(graph, analysis_variants)

        # Per sample processing
        haplotypes, diplotypes = _sample_diplotypes_from_counts(
            sampler,
            unique_kmers,
            sample.kmc_prefix,
            k=cfg.kmer.kmer_size,
            kmer_coverage=sample.kmer_coverage,
            max_haplotypes=cfg.kmer.max_haplotypes,
            max_diplotypes=cfg.kmer.max_diplotypes,
            filter_kmers=filter_kmers,
        )    

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

            matching_haplotypes = tuple([(variant.variant_id, allele) in paths for paths in haplotype_paths] for allele in alleles)

            # Compute the rank (0-indexed) of the true haplotypes (a True in matching_haplotypes)
            haplotype_idxs = [
                next((i for i, match in enumerate(allele_matches) if match), -1)
                for allele_matches in matching_haplotypes
            ]
            if -1 not in haplotype_idxs:
                # Normalize diplotype (compare genotypes without phasing) and compute rank (0-indexed) of the true diplotype
                haplotype_idxs = sorted(haplotype_idxs)
                diplotype_idx = next((i for i, diplotype in enumerate(diplotypes) if sorted(diplotype.haplotypes) == haplotype_idxs), -1)
            else:
                # If either haplotype is not found, there can't be a correct diplotype match
                diplotype_idx = -1

            result_table.append({
                "region": str(region),
                "variant": variant.variant_id,
                "sample": sample.name,
                "haplotypes": len(haplotypes),
                "haplotype_idxs": tuple(haplotype_idxs),
                "diplotypes": len(diplotypes),
                "diplotype_idx": diplotype_idx,
            })

    return pd.DataFrame(result_table)
