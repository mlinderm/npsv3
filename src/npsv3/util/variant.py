import hashlib
import typing

import pysam

# Variant is now the native class directly (like npsv3.util.range.Range): all the property/derived-value
# logic that used to live in a Python wrapper class has been pushed into npsv3::Variant and its bindings
# (src/npsv3/native/variant.{hpp,cpp}, src/npsv3/native/graph_bindings.cpp). A native Variant can only be
# constructed by reading through VariantFileReader -- there is no bridge from an already-parsed
# pysam.VariantRecord, and symbolic alleles are not supported (VariantFileReader raises when it hits one).
#
# Functionality that did NOT make the move, and why:
#  - Variant.from_pysam(record): no replacement. Construct via `Variant(next(reader.fetch()))` from a
#    VariantFileReader instead.
#  - variant.num_alt / variant.allele_indices / variant.alt_allele_indices: removed as pure Python sugar
#    over num_alleles/num_alts with no real logic. Use `v.num_alts`, `range(v.num_alleles)`,
#    `range(1, v.num_alleles)` directly.
#  - Renames: callers should use the native binding names directly rather than the old wrapper names:
#    alt_reference_region -> allele_reference_region, alt_length -> allele_length, alt_seq -> allele_sequence.
#  - variant_id caching: vg_variant_id used to be a functools.cached_property; the native `variant_id`
#    property recomputes the SHA1 digest on every access. Deferred -- cache on the C++ side if this shows
#    up as a hot path.
#  - length_change() SVLEN parsing: also recomputed on every call (native `info_int("SVLEN")` re-reads and
#    re-parses the INFO field each time). Deferred alongside variant_id caching, for the same reason.
from npsv3._native_graph import Variant as Variant  # noqa: PLC0414
from npsv3._native_graph import VariantFileReader as VariantFileReader  # noqa: PLC0414

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


def overlapping_records(vcf_path: str | VariantFileReader, flank=0):
    # We assume the file is in sorted order
    current_range = None
    current_variants = []

    reader = vcf_path if isinstance(vcf_path, VariantFileReader) else VariantFileReader.open(vcf_path)

    with reader:
        for variant in reader.fetch():
            variant_range = variant.reference_region().expand(flank)
            if current_range is None:
                current_range = variant_range
                current_variants = [variant]
            elif current_range.overlaps(variant_range):
                current_range = current_range.union(variant_range)
                current_variants.append(variant)
            else:
                # Next variant doesn't overlap, so yield current variants and then reset
                yield current_range, current_variants
                current_range = variant_range
                current_variants = [variant]

        # yield any remaining variants
        if current_variants:
            yield current_range, current_variants
