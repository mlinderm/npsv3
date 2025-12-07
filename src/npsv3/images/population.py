from omegaconf import DictConfig

from npsv3._native_graph import Graph, VariantFileReader, VariantFileWriter

def update_filter(cfg: DictConfig, vcf_path: str, output_path: str):
    """Set variant FILTER to PASS if any defined genotype is not filtered

    Used with variant combination workflow that sets genotype filters from upstream variant-level filters. This function
    is significantly faster than the equivalent bcftools command.

    Args:
        cfg (DictConfig): Hydra configuration
        vcf_path (str): Input VCF file
        output_path (str): Output VCF file
    """
    src_vcf_file = VariantFileReader.open(vcf_path)
    dst_vcf_file = VariantFileWriter.open(output_path, src_vcf_file.header(), cfg.get("output_format"))
    for variant in src_vcf_file.fetch():
        if variant.has_passing_genotype():
            # Set FILTER to PASS is there are any passing genotypes
            variant.set_filter_pass()
        dst_vcf_file.write(variant)


def overlapping_variants(vcf_file: VariantFileReader|str, flank=0):
    # We assume VCF is in sorted order
    current_range = None
    current_variants = []
    
    if isinstance(vcf_file, str):
        vcf_file = VariantFileReader.open(vcf_file)

    for variant in vcf_file.fetch():
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

def split_and_filter_vcf(
    cfg: DictConfig,
    inference_vcf: str,
    output_dir: str,
):
    reader = VariantFileReader.open(inference_vcf)
    possible_samples = set(reader.samples())
    for _region_count, (region, variants) in enumerate(overlapping_variants(reader, flank=cfg.pileup.variant_padding), start=1):
        print(f"Region {region} has {len(variants)} records")
        graph = Graph(cfg.reference, inference_vcf, region)
        exclude_nodes = set()
        for variant in variants:
            sv_alleles = {
                i
                for i in range(1, variant.num_alleles)
                if abs(variant.allele_length_change(i) or 0) >= 50
            }
            if sv_alleles:
                variant_id = variant.variant_id
                ref_nodes = set(graph.path_nodes(f"_alt_{variant_id}_0"))
                exclude_nodes.update(*(graph.path_nodes(f"_alt_{variant_id}_{a}") for a in sv_alleles))
                exclude_nodes.difference_update(ref_nodes)  # Remove nodes shared with references paths to get nodes that distinguish ALT alleles
        
        ref_samples = possible_samples - set(graph.samples_including(list(exclude_nodes)))
        print(ref_samples)
        break
