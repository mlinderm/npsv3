import hashlib
import typing

from omegaconf import DictConfig
import pysam

from npsv3._native_graph import Variant as Variant  # noqa: PLC0414
from npsv3._native_graph import VariantFileReader as VariantFileReader  # noqa: PLC0414
from npsv3._native_graph import VariantFileWriter as VariantFileWriter
from npsv3.util.range import Range
from npsv3.types import PathType  # noqa: PLC0414

# Length of SHA1 hex digest used for variants IDs by vg
VARIANT_ID_LENGTH = 40

def vg_variant_id(record: pysam.VariantRecord) -> str:
    # https://github.com/vgteam/vg/blob/da34f4e54b0e64d1b741da102217c97d5333fabc/src/utility.cpp#L505
    assert record.ref is not None and record.alts is not None  # noqa: PT018
    variant_string = f"{record.contig}\n{record.pos}\n{record.ref.upper()}\n"
    for alt in record.alts:
        if alt != "*": # Ignore "*" alleles since they are not handled by VG
            variant_string += f"{alt.upper()}\n"
    return hashlib.sha1(bytes(variant_string, "ascii")).hexdigest()  # noqa: S324


# Adapted from nucleus:
# https://github.com/google/nucleus/blob/3bd27ac076a6f3f93e49a27ed60661858e727dda/nucleus/util/variant_utils.py#L718
def generate_allele_indices(num_alleles: int, ploidy: int) -> typing.Generator[tuple[int, ...], None, None]:
    """Generate VCF allele indices (genotype) in order of the VCF genotype likelihood (or other 'G') field

    Args:
        num_alt (int): Number of alternate alleles
        ploidy (int, optional): Ploidy. Defaults to 2.

    Raises:
        NotImplementedError: Specified ploidy is not supported

    Yields:
        Tuple[int,...]: Tuple of genotypes for each index in genotypes field, e.g. (0,0), (0,1)...
    """
    if ploidy == 1:
        for i in range(num_alleles):
            yield (i,)
    elif ploidy == 2:  # noqa: PLR2004
        for j in range(num_alleles):
            for i in range(j + 1):
                yield (i, j)
    else:
        msg = "Only ploidy <= 2 is currently supported"
        raise NotImplementedError(msg)


# Adapted from nucleus:
# https://github.com/google/nucleus/blob/3bd27ac076a6f3f93e49a27ed60661858e727dda/nucleus/util/variant_utils.py#L793
def genotype_field_index(allele_indices: typing.Sequence[int]) -> int:
    """Determine index in VCF genotype likelihood (or other 'G') field for genotype

    Args:
        allele_indices (Sequence[int]): Genotype, e.g. (0,1)

    Raises:
        NotImplementedError: Specified ploidy is not supported

    Returns:
        int: Index in genotype field
    """
    if len(allele_indices) == 1:
        return allele_indices[0]
    if len(allele_indices) == 2:  # noqa: PLR2004
        a1, a2 = sorted(allele_indices)
        return a1 + (a2 * (a2 + 1) // 2)
    msg = "Only ploidy <= 2 is currently supported"
    raise NotImplementedError(msg)


def genotype_count(num_alleles: int, ploidy: int):
    if ploidy <= 2:  # noqa: PLR2004
        return (num_alleles * (num_alleles + 1)) // ploidy
    msg = "Only ploidy <= 2 is currently supported"
    raise NotImplementedError(msg)


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