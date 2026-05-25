from npsv3._native_graph import Graph, Range, VariantFileReader, KmerCounts
from npsv3.images.population import overlapping_variants


def sample_from_regions(cfg, samples, vcf_path):

    kmer_counters = {name: KmerCounts(sample.kmc_path, sample.kmer_coverage) for name, sample in samples.items()}
    # total_variant_regions = 0
    # total_sample_regions = 0
    # rank_counts = {}  # str(rank) -> int, plus "not_found"

    with VariantFileReader.open(vcf_path) as vcf_file:
        for region, _variants in overlapping_variants(vcf_file):
            graph = Graph(cfg.reference, vcf_path, region)


        # for sample, kmc_path in sample_kmc_map.items():
        #     hap0_nodes = graph.haplotype_paths(f"{sample}#0#{region.contig}")
        #     hap1_nodes = graph.haplotype_paths(f"{sample}#1#{region.contig}")
        #     if not hap0_nodes and not hap1_nodes:
        #         logging.debug("No haplotype paths for sample %s in region %s", sample, region)
        #         continue

        #     counts = KmerCounts(kmc_path, cfg.coverage)
        #     sampler = HaplotypeSamplerOverlay(
        #         graph, cfg.kmer_size, cfg.max_edge,
        #         counts, vcf_path, region, cfg.min_sv_size,
        #     )
        

        #     candidates = sampler.sample_haplotypes(cfg.n_haplotypes)
        #     if not candidates:
        #         logging.debug("No candidates for sample %s in region %s", sample, region)
        #         continue

        #     diplotypes = sampler.sample_diplotypes(candidates, cfg.max_diplotypes)

        #     true_alleles = (
        #         sampler.covered_alleles(hap0_nodes),
        #         sampler.covered_alleles(hap1_nodes),
        #     )
        #     rank = _find_rank(true_alleles, diplotypes, sampler)

        #     total_sample_regions += 1
        #     key = str(rank) if rank is not None else "not_found"
        #     rank_counts[key] = rank_counts.get(key, 0) + 1


    # # Compute recall@k for selected k values
    # recall_at_k = {}
    # for k in [1, 5, 10, cfg.max_diplotypes]:
    #     found = sum(v for rk, v in rank_counts.items()
    #                 if rk != "not_found" and int(rk) <= k)
    #     recall_at_k[str(k)] = round(found / total_sample_regions, 4) if total_sample_regions else 0.0

    # stats = {
    #     "total_variant_regions": total_variant_regions,
    #     "total_sample_regions": total_sample_regions,
    #     "rank_counts": rank_counts,
    #     "recall_at_k": recall_at_k,
    # }

    # with open(output_path, "w") as f:
    #     json.dump(stats, f, indent=2)

    # logging.info(
    #     "top_k_genotype: %d regions, %d sample-regions, rank_counts=%s",
    #     total_variant_regions, total_sample_regions, rank_counts,
    # )
    # return stats