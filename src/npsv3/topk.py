import json
import logging

from npsv3._native_graph import Graph, HaplotypeSamplerOverlay, KmerCounts, VariantFileReader
from npsv3.images.population import overlapping_variants


def _find_rank(true_alleles, diplotypes, sampler):
    """Return 1-based rank of the true diplotype in the scored list, or None if not found.

    Comparison is unordered: (hap0, hap1) matches (hap1, hap0).
    """
    true_0 = tuple(true_alleles[0])
    true_1 = tuple(true_alleles[1])
    true_sorted = tuple(sorted([true_0, true_1]))

    for rank, (dip0, dip1) in enumerate(diplotypes, 1):
        cov0 = tuple(sampler.covered_alleles(dip0))
        cov1 = tuple(sampler.covered_alleles(dip1))
        if tuple(sorted([cov0, cov1])) == true_sorted:
            return rank
    return None


def top_k_genotype(cfg, vcf_path, sample_kmc_map, output_path):
    """For each variant region, score sampled diplotypes against the true diplotype and
    accumulate rank statistics.

    Args:
        cfg: Hydra config with kmer_size, max_edge, n_haplotypes, max_diplotypes,
             min_sv_size, coverage, and reference fields.
        vcf_path: Path to the input VCF (must be indexed).
        sample_kmc_map: dict mapping sample name -> KMC db path (without extension).
        output_path: Path to write JSON statistics.
    """
    reference = cfg.reference

    total_variant_regions = 0
    total_sample_regions = 0
    rank_counts = {}  # str(rank) -> int, plus "not_found"

    vcf_file = VariantFileReader.open(vcf_path)
    try:
        for region, _variants in overlapping_variants(vcf_file):
            total_variant_regions += 1
            try:
                graph = Graph(reference, vcf_path, region)
            except Exception:
                logging.warning("Failed to build graph for region %s, skipping", region)
                continue

            for sample, kmc_path in sample_kmc_map.items():
                hap0_nodes = graph.haplotype_paths(f"{sample}#0#{region.contig}")
                hap1_nodes = graph.haplotype_paths(f"{sample}#1#{region.contig}")
                if not hap0_nodes and not hap1_nodes:
                    logging.debug("No haplotype paths for sample %s in region %s", sample, region)
                    continue

                try:
                    counts = KmerCounts(kmc_path, cfg.coverage)
                    sampler = HaplotypeSamplerOverlay(
                        graph, cfg.kmer_size, cfg.max_edge,
                        counts, vcf_path, region, cfg.min_sv_size,
                    )
                except Exception:
                    logging.warning(
                        "Failed to build sampler for sample %s in region %s, skipping",
                        sample, region,
                    )
                    continue

                candidates = sampler.sample_haplotypes(cfg.n_haplotypes)
                if not candidates:
                    logging.debug("No candidates for sample %s in region %s", sample, region)
                    continue

                diplotypes = sampler.sample_diplotypes(candidates, cfg.max_diplotypes)

                true_alleles = (
                    sampler.covered_alleles(hap0_nodes),
                    sampler.covered_alleles(hap1_nodes),
                )
                rank = _find_rank(true_alleles, diplotypes, sampler)

                total_sample_regions += 1
                key = str(rank) if rank is not None else "not_found"
                rank_counts[key] = rank_counts.get(key, 0) + 1
    finally:
        vcf_file.close()

    # Compute recall@k for selected k values
    recall_at_k = {}
    for k in [1, 5, 10, cfg.max_diplotypes]:
        found = sum(v for rk, v in rank_counts.items()
                    if rk != "not_found" and int(rk) <= k)
        recall_at_k[str(k)] = round(found / total_sample_regions, 4) if total_sample_regions else 0.0

    stats = {
        "total_variant_regions": total_variant_regions,
        "total_sample_regions": total_sample_regions,
        "rank_counts": rank_counts,
        "recall_at_k": recall_at_k,
    }

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    logging.info(
        "top_k_genotype: %d regions, %d sample-regions, rank_counts=%s",
        total_variant_regions, total_sample_regions, rank_counts,
    )
    return stats
