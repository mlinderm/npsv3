from collections import defaultdict
import os
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
    try:
        for variant in src_vcf_file.fetch():
            if variant.has_passing_genotype():
                # Set FILTER to PASS is there are any passing genotypes
                variant.set_filter_pass()
            dst_vcf_file.write(variant)
    finally:
        src_vcf_file.close()
        dst_vcf_file.close()


def overlapping_variants(vcf_file: VariantFileReader|str, flank=0):
    """Yield separated (non-overlapping) regions and corresponding variants

    Args:
        vcf_file (VariantFileReader | str): VCF file
        flank (int, optional): Required separation between variants to define region. Defaults to 0.

    Yields:
        tuple[Region, list[Variant]]: Region and a list of overlapping variants
    """
    # We assume VCF is in sorted order
    current_range = None
    current_variants = []
    
    if isinstance(vcf_file, str):
        vcf_file = VariantFileReader.open(vcf_file)
    assert isinstance(vcf_file, VariantFileReader)

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
    file_template="{sample}.vcf.gz",
):
    """Split multi-sample VCF into individual VCFs to create training images

    Args:
        cfg (DictConfig): Application configuration
        inference_vcf (str): Input multi-sample VCF
        output_dir (str): Directory to create single-sample VCFs
        file_template (str, optional): Template for output VCF filenames, with {sample} placeholder. Defaults to "{sample}.vcf.gz".
    """
    try:
        reader = VariantFileReader.open(inference_vcf)
        possible_samples = { sample: [i] for i, sample in enumerate(reader.samples()) }
        header = reader.header()

        # Lazily create per-sample writers
        class WriterDict(defaultdict):
            def __missing__(self, key):
                sample_path = os.path.join(output_dir, file_template.format(sample=key))
                sample_header = header.subset([key])  # Create a subset header with only this sample
                writer = self[key] = VariantFileWriter.open(sample_path, sample_header)
                return writer
        writers = WriterDict()

        for _region_count, (region, variants) in enumerate(overlapping_variants(reader, flank=cfg.pileup.variant_padding), start=1):
            passing_variants = [v for v in variants if not v.is_filtered()]
            if len(passing_variants) == 0:
                continue  # Skip regions with no unfiltered variants

            # We want the positive examples to (only) have unfiltered variants in this region and negative samples to not contain any SV
            # alleles, regardless of FILTER status.

            # Build the graph for this region and classify SV alt-allele nodes by filter status
            graph = Graph(cfg.reference, inference_vcf, region)
            all_alt_nodes = set()
            filtered_alt_nodes = set()

            for variant in variants:
                sv_alleles = {
                    i
                    for i in range(1, variant.num_alleles)
                    if abs(variant.allele_length_change(i) or 0) >= 50
                }
                if sv_alleles:
                    variant_id = variant.variant_id
                    ref_nodes = set(graph.path_nodes(f"_alt_{variant_id}_0"))
                    alt_nodes = set()
                    for a in sv_alleles:
                        alt_nodes.update(graph.path_nodes(f"_alt_{variant_id}_{a}"))
                    # Keep only nodes that distinguish ALT alleles from REF
                    alt_nodes.difference_update(ref_nodes)

                    all_alt_nodes.update(alt_nodes)
                    if variant.is_filtered():
                        filtered_alt_nodes.update(alt_nodes)

            # Negative samples: No SV alleles at all (filtered or unfiltered)
            samples_with_any_sv = set(graph.samples_including(list(all_alt_nodes)))
            negative_samples = possible_samples.keys() - samples_with_any_sv

            # Positive samples: Have passing SVs but no filtered SVs
            samples_with_filtered_sv = set(graph.samples_including(list(filtered_alt_nodes)))
            positive_samples = samples_with_any_sv - samples_with_filtered_sv

            # Write passing variants for all qualifying samples
            for sample in positive_samples | negative_samples:
                writer = writers[sample]
                sample_idxs = possible_samples[sample]
                for variant in passing_variants:
                    writer.write(variant.subset_samples(sample_idxs))
    finally:
        # Close all writers and reader
        reader.close()
        for writer in writers.values():
            writer.close()
