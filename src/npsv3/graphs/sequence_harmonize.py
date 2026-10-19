"""Truvari-style haplotype harmonization (the `truvari phab` step).

Given a reference sequence and haplotype sequences for the same region, align them all together (multiple sequence alignment)
and re-call every haplotype's variants against the reference from that single alignment. Variants are translated into a single
coordinate system with consistent gap placement, e.g., the same event in a tandem repeat is represented identically
in every haplotype. This is the representation `truvari refine` compares after harmonization, and it is the input to the
Truvari-style matching in `sequence_compare`.

Ported from Truvari 5.3.0 (`truvari/phab.py`, `truvari/msatovcf.py`; MIT license, https://github.com/ACEnglish/truvari)
and kept behavior-compatible so results can be checked against it (`tests/graphs/test_sequence_harmonize.py`). Needs the optional
`harmonize` extra (`edlib`, `pyabpoa`, `pywfa`).

Notes:
* abPOA (the Truvari default) is documented as non-deterministic across machines; `wfa` is deterministic but aligns each
  haplotype to the reference independently, which can give less parsimonious variants.
* Alignment is impractical for very long sequences.
"""

import logging
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

logger = logging.getLogger(__name__)

REF_PREFIX = "ref_"

# An aligned haplotype is either its gapped sequence (MSA aligners) or a (gapped reference, gapped haplotype) pair (pairwise
# aligners).
Aligned = str | tuple[str, str]
Aligner = Callable[[Mapping[str, str]], dict[str, Aligned]]


@dataclass(frozen=True, order=True)
class AlignedVariant:
    """A variant called from a harmonized alignment, in VCF conventions.

    After decomposition either `ref` or `alt` is a single base, i.e., the variant is a SNV, insertion or deletion.

    Attributes:
        pos: 1-based reference position of the first base of `ref` (the anchor base for an indel).
        ref: Reference allele, including the anchor base for an indel.
        alt: Alternate allele, including the anchor base for an indel.
    """

    pos: int
    ref: str
    alt: str

    @property
    def kind(self) -> str:
        if len(self.ref) == 1 and len(self.alt) == 1:
            return "SNP"
        return "INS" if len(self.alt) > len(self.ref) else "DEL"

    @property
    def size(self) -> int:
        """Length of the insertion/deletion in bp (1 for a SNV)."""
        return max(abs(len(self.alt) - len(self.ref)), 1) if self.kind != "SNP" else 1

    @property
    def end(self) -> int:
        """Last reference base covered (1-based, inclusive)."""
        return self.pos + len(self.ref) - 1


@dataclass(frozen=True)
class Harmonized:
    """Result of `harmonize`: each haplotype's variants called from a single alignment with the reference.

    Attributes:
        variants: Haplotype name -> its variants against the reference, for every input haplotype, including those not
            aligned (identical to the reference, or duplicates with `dedup`).
        aligner: Name of the aligner used.
    """

    variants: dict[str, tuple[AlignedVariant, ...]]
    aligner: str


def _expand_cigar(seq: str, ref: str, cigar) -> tuple[str, str]:
    """Put gaps '-' in the sequence and reference where the CIGAR requires them."""
    seq_pos = ref_pos = 0
    seq_list, ref_list = list(seq), list(ref)
    for code, span in cigar:
        if code == 2:  # noqa: PLR2004 (deletion from reference: gap in the haplotype)
            seq_list.insert(seq_pos, "-" * span)
            seq_pos += 1
            ref_pos += span
        elif code == 1:
            ref_list.insert(ref_pos, "-" * span)
            seq_pos += span
            ref_pos += 1
        else:
            seq_pos += span
            ref_pos += span
    return "".join(ref_list), "".join(seq_list)


def _reference_key(sequences: Mapping[str, str]) -> str:
    return next(k for k in sequences if k.startswith(REF_PREFIX))


def align_poa(sequences: Mapping[str, str]) -> dict[str, Aligned]:
    """Multiple sequence alignment with abPOA (Truvari's default). Sequences are aligned longest first, as in Truvari."""
    import pyabpoa

    parts = sorted(((len(seq), seq, name) for name, seq in sequences.items()), reverse=True)
    _, seqs, names = zip(*parts, strict=True)
    result = pyabpoa.msa_aligner().msa(list(seqs), out_cons=False, out_msa=True)
    return dict(zip(names, result.msa_seq, strict=True))


def align_wfa(sequences: Mapping[str, str]) -> dict[str, Aligned]:
    """Align each haplotype to the reference independently with WFA (fast, deterministic, less parsimonious)."""
    from pywfa import WavefrontAligner

    ref_key = _reference_key(sequences)
    reference = sequences[ref_key]
    aligner = WavefrontAligner(reference, distance="affine2p", span="end-to-end", heuristic="adaptive")
    aligned: dict[str, Aligned] = {ref_key: reference}
    for name, seq in sequences.items():
        if name == ref_key:
            continue
        aligner.wavefront_align(seq)
        aligned[name] = _expand_cigar(seq, reference, aligner.cigartuples)
    return aligned


ALIGNERS: dict[str, Aligner] = {"poa": align_poa, "wfa": align_wfa}


def _decompose(variant: AlignedVariant) -> list[AlignedVariant]:
    """Return decomposed AlignedVairant(s) by left-trimming variant splitting MNV/indels into a deletion plus an insertion (easier to compare)."""
    pos, ref, alt = variant.pos, variant.ref, variant.alt
    if ref == alt:
        return []

    # Move down while the anchor base is identical, stopping when one base is left or the anchor bases differ
    trim = 0
    while trim < len(ref) - 1 and trim < len(alt) - 1 and ref[trim] == alt[trim]:
        trim += 1
    pos, ref, alt = pos + trim, ref[trim:], alt[trim:]
    if len(ref) == 1 or len(alt) == 1:
        return [AlignedVariant(pos, ref, alt)]

    deletion = AlignedVariant(pos, ref[:-1], alt[0])
    insertion = AlignedVariant(pos + len(ref) - 1, ref[-1], alt[1:])
    return [deletion, insertion]


def alignment_to_variants(ref_aln: str, alt_aln: str, start_pos: int, anchor_base: str) -> list[AlignedVariant]:
    """Return variants from the bases of a pairwise alignment (against the gapped reference)"""
    variants: list[AlignedVariant] = []
    current: AlignedVariant | None = None  # variant being extended (anchored, not yet decomposed)
    cur_pos = start_pos
    for ref_char, alt_char in zip(ref_aln, alt_aln, strict=True):
        is_ref = ref_char != "-"
        ref_base = ref_char if is_ref else ""
        alt_base = alt_char if alt_char != "-" else ""
        if not ref_base and not alt_base:  # gap on gap
            continue

        if ref_base == alt_base:
            if current is not None and is_ref:  # back to matching the reference
                variants.extend(_decompose(current))
                current = None
        elif current is None:
            # -1 for the anchor base we force on
            current = AlignedVariant(cur_pos - 1, anchor_base + ref_base, anchor_base + alt_base)
        else:
            current = replace(current, ref=current.ref + ref_base, alt=current.alt + alt_base)
        if is_ref:
            cur_pos += 1
            anchor_base = ref_base
    if current is not None:
        variants.extend(_decompose(current))
    return variants


def msa_to_variants(
    msa: Mapping[str, Aligned], start_pos: int = 1, abs_anchor_base: str = "N"
) -> dict[str, tuple[AlignedVariant, ...]]:
    """Return variants for every non-reference entry of an alignment (the reference entry's key starts with `ref_`)."""
    ref_key = _reference_key(msa)  # type: ignore[arg-type]
    ref_seq = msa[ref_key]
    ref_seq = ref_seq.upper() if isinstance(ref_seq, str) else None

    variants: dict[str, tuple[AlignedVariant, ...]] = {}
    for key, aligned in msa.items():
        if key == ref_key:
            continue
        if isinstance(aligned, tuple):
            ref_aln, alt_aln = aligned[0].upper(), aligned[1].upper()
        else:
            ref_aln, alt_aln = ref_seq, aligned.upper()
        assert ref_aln is not None, f"Reference sequence must be a string for pairwise aligners (got {ref_aln!r})"
        anchor_base = ref_aln[0] if ref_aln[0] != "-" else abs_anchor_base
        variants[key] = tuple(alignment_to_variants(ref_aln, alt_aln, start_pos, anchor_base))
    return variants


def harmonize(
    reference: str,
    haplotypes: Mapping[str, str],
    *,
    aligner: str | Aligner = "poa",
    start_pos: int = 1,
    anchor_base: str = "N",
    dedup: bool = False,
) -> Harmonized:
    """Align `haplotypes` (name -> sequence) with `reference` and call each haplotype's variants against the reference.

    Args:
        reference: Reference sequence for the region.
        haplotypes: Haplotype name -> sequence for the same region. Names must not start with `ref_`.
        aligner: Name of a built-in aligner (`"poa"` or `"wfa"`, see `ALIGNERS`) or an `Aligner` callable.
        start_pos: 1-based position of the reference's first base, used for the variants' positions.
        anchor_base: Anchor base for a variant at the very start of the region when the alignment opens with a gap in the
            reference (e.g., the reference base preceding `start_pos`).
        dedup: Align identical haplotype sequences only once (Truvari's `--dedup`).

    Returns:
        Each haplotype's variants against the reference, in VCF conventions.
    """
    assert not any(name.startswith(REF_PREFIX) for name in haplotypes), (
        f"Haplotype names cannot start with {REF_PREFIX}"
    )
    reference = reference.upper()
    haplotypes = {name: seq.upper() for name, seq in haplotypes.items()}
    align = ALIGNERS[aligner] if isinstance(aligner, str) else aligner
    ref_key = f"{REF_PREFIX}region:{start_pos}-{start_pos + len(reference) - 1}"

    variants: dict[str, tuple[AlignedVariant, ...]] = {name: () for name, seq in haplotypes.items() if seq == reference}
    to_align = {name: seq for name, seq in haplotypes.items() if seq != reference}
    if not to_align:
        return Harmonized(variants, aligner if isinstance(aligner, str) else aligner.__name__)

    key_map = {name: name for name in to_align}
    if dedup:
        first_with_sequence: dict[str, str] = {}
        for name, seq in list(to_align.items()):
            if seq in first_with_sequence:
                key_map[name] = first_with_sequence[seq]
                del to_align[name]
            else:
                first_with_sequence[seq] = name

    msa = align({ref_key: reference, **to_align})
    msa_variants = msa_to_variants(msa, start_pos, anchor_base)
    variants.update({name: msa_variants[key] for name, key in key_map.items()})
    return Harmonized(variants, aligner if isinstance(aligner, str) else aligner.__name__)


def variant_carriers(harmonized: Harmonized) -> dict[AlignedVariant, list[str]]:
    """Group harmonized variants by the haplotypes that carry them.

    Because harmonization gives every haplotype the same representation of a shared event, the same variant called in
    different haplotypes is equal and is counted once, with all its carriers.

    Args:
        harmonized: Result of `harmonize`.

    Returns:
        Each distinct variant -> names of the haplotypes carrying it, in the order of `harmonized.variants`. Haplotypes
        without variants do not appear.
    """
    carriers: dict[AlignedVariant, list[str]] = defaultdict(list)
    for name, variants in harmonized.variants.items():
        for variant in variants:
            carriers[variant].append(name)
    return dict(carriers)


__all__ = [
    "ALIGNERS",
    "AlignedVariant",
    "Harmonized",
    "align_poa",
    "align_wfa",
    "alignment_to_variants",
    "harmonize",
    "msa_to_variants",
    "variant_carriers",
]
