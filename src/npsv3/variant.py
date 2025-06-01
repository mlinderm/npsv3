import hashlib
import logging
import os
import re
import typing
from functools import cached_property

import pysam

from npsv3.util.range import Range

_VALID_BASES_RE = re.compile(r"[ACGTN]+", re.IGNORECASE)


def vg_variant_id(record: pysam.VariantRecord) -> str:
    # https://github.com/vgteam/vg/blob/da34f4e54b0e64d1b741da102217c97d5333fabc/src/utility.cpp#L505
    assert record.ref is not None and record.alts is not None
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
    elif ploidy == 2:
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
    if len(allele_indices) == 2:
        a1, a2 = sorted(allele_indices)
        return a1 + (a2 * (a2 + 1) // 2)
    msg = "Only ploidy <= 2 is currently supported"
    raise NotImplementedError(msg)


def genotype_count(num_alleles: int, ploidy: int):
    if ploidy <= 2:
        return (num_alleles * (num_alleles + 1)) // ploidy
    msg = "Only ploidy <= 2 is currently supported"
    raise NotImplementedError(msg)


def _has_symbolic_allele(record):
    for alt in record.alts:
        if alt.startswith("<") or alt.endswith(">"):
            return True
    return False


class Variant:
    def __init__(self, record: pysam.VariantRecord):
        """Initialize Variant object from pysam.VariantRecord

        Args:
            record (pysam.VariantRecord): Underlying VariantRecord
        """
        self._record = record

        self._sequence_resolved = not _has_symbolic_allele(record)
        if self._sequence_resolved:
            self._padding = len(os.path.commonprefix([a for a in record.alleles if a != "*"]))
            self._right_padding = [
                len(os.path.commonprefix([record.ref[self._padding :][::-1], a[self._padding :][::-1]]))
                for a in record.alts if a != "*"
            ]
        else:
            assert len(record.alts) == 1, "Multiallelic symbolic variants not currently supported"
            self._padding = 1
            self._right_padding = [0] * len(record.alts)

        if self._padding > 1:
            logging.info("Variant has more than expected number of padding bases, is the VCF normalized?")

    @property
    def contig(self):
        return self._record.contig

    @property
    def start(self):
        return self._record.start

    @property
    def end(self):
        return self._record.stop

    @property
    def num_alt(self):
        # TODO: Exclude alleles?
        return len(self._record.alts)

    @property
    def allele_indices(self):
        return range(1 + self.num_alt)

    @property
    def alt_allele_indices(self):
        return range(1, 1 + self.num_alt)

    @property
    def reference_region(self) -> Range:
        """Returns changed region of the reference genome, excluding any padding bases"""
        return Range(self.contig, self.start + self._padding, self.end)

    def alt_reference_region(self, allele: int) -> Range:
        raise NotImplementedError

    @property
    def ref_length(self):
        """Length of reference allele including any padding bases"""
        raise NotImplementedError

    def alt_length(self):
        """Length of alternate allele including any padding bases"""
        raise NotImplementedError

    def length_change(self, allele: int | None = None):
        try:
            svlen = self._record.info.get("SVLEN", None)
        except ValueError: # PySAM seems to raise an error if field is not defined in VCF at all
            svlen = None
        if svlen is None:
            svlen = tuple(self.alt_length(i) - self.ref_length if alt != "*" else None for i, alt in enumerate(self._record.alts, start=1))
        elif isinstance(svlen, int):
            svlen = (svlen,)  # If SVLEN is Number=1, convert to sequence
        return svlen if allele is None else svlen[allele - 1]

    @cached_property
    def vg_variant_id(self):
        return vg_variant_id(self._record)

    @classmethod
    def from_pysam(cls, record: pysam.VariantRecord) -> "Variant":
        """Factory method for creating appropriate Variant objects"""
        if not _has_symbolic_allele(record):
            return _SequenceResolvedVariant(record)
        msg = "Symbolic variants not currently supported"
        raise NotImplementedError(msg)


class _SequenceResolvedVariant(Variant):
    def __init__(self, record):
        Variant.__init__(self, record)
        assert self._sequence_resolved

    @property
    def ref_length(self):
        return len(self._record.ref)

    def alt_length(self, allele) -> int | None:
        assert allele >= 1
        alt_allele = self._record.alleles[allele]
        return len(alt_allele) if alt_allele != "*" else None

    def alt_reference_region(self, allele) -> Range | None:
        assert allele >= 1
        alt_allele = self._record.alleles[allele]
        if alt_allele == "*":
            return None
        # Compute per-allele padding (since is may be different than the global padding)
        padding = len(os.path.commonprefix([self._record.ref, alt_allele]))
        return Range(self.contig, self.start + padding, self.end)

    def alt_seq(self, allele) -> str | None:
        assert allele >= 1
        alt_allele = self._record.alleles[allele]
        if alt_allele == "*":
            return None
        assert _VALID_BASES_RE.fullmatch(alt_allele), f"Unexpected base in sequence resolved allele {alt_allele} in region {self.reference_region}"
        # Compute per-allele padding (since is may be different than the global padding)
        padding = len(os.path.commonprefix([self._record.ref, alt_allele]))
        return alt_allele[padding:]


class _SymbolicDeletionVariant(Variant):
    def __init__(self, record):
        Variant.__init__(self, record)
        assert self.num_alt == 1, "Multi-allelic symbolic variants are not supported"

    @property
    def ref_length(self):
        return self.end - self.start

    def alt_length(self, allele=1):
        assert allele >= 1
        return 1

    def alt_reference_region(self, allele) -> Range:
        assert allele >= 1
        return Range(self.contig, self.start + self._padding, self.end)

    def alt_seq(self, allele):
        assert allele >= 1
        return ""


def overlapping_records(vcf_path: str, flank=0):
    # We assume VCF is in sorted order
    current_range = None
    current_records = []

    with pysam.VariantFile(vcf_path) as vcf_file:
        for record in vcf_file:
            variant = Variant.from_pysam(record)
            variant_range = variant.reference_region.expand(flank)
            if current_range is None:
                current_range = variant_range
                current_records = [record]
            elif current_range.overlaps(variant_range):
                current_range = current_range.union(variant_range)
                current_records.append(record)
            else:
                # Next variant doesn't overlap, so yield current records and then reset
                yield current_range, current_records
                current_range = variant_range
                current_records = [record]

        # yield any remaining records
        if current_records:
            yield current_range, current_records
