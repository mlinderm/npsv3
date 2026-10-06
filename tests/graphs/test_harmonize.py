import json
import os
import random

import pytest

pytest.importorskip("pyabpoa")
pytest.importorskip("pywfa")

from npsv3.graphs.harmonize import (
    AlignedVariant,
    alignment_to_variants,
    harmonize,
    msa_to_variants,
    variant_counts,
)

ORACLE_PATH = os.path.join(os.path.dirname(__file__), "data", "truvari_harmonize_oracle.json")
with open(ORACLE_PATH) as _f:
    ORACLE = json.load(_f)
CASES = sorted(ORACLE["cases"])


def _carriers(variants_by_haplotype) -> set[tuple]:
    """{(pos, ref, alt, sorted carriers)} from harmonized variants."""
    carriers: dict[AlignedVariant, list[str]] = {}
    for name, variants in variants_by_haplotype.items():
        for variant in variants:
            carriers.setdefault(variant, []).append(name)
    return {(v.pos, v.ref, v.alt, tuple(sorted(names))) for v, names in carriers.items()}


def _oracle_variants(entry) -> set[tuple]:
    return {(v["pos"], v["ref"], v["alt"], tuple(sorted(v["carriers"]))) for v in entry["variants"]}


def _oracle_msa(entry):
    return {k: tuple(v) if isinstance(v, list) else v for k, v in entry["msa"].items()}


class TestAlignmentToVariants:
    def test_snv(self):
        [variant] = alignment_to_variants("ACGT", "ACTT", 1, "N")
        assert variant == AlignedVariant(3, "G", "T")
        assert variant.kind == "SNP"
        assert variant.size == 1

    def test_insertion_and_deletion(self):
        [insertion] = alignment_to_variants("AC--GT", "ACTTGT", 1, "N")
        assert insertion == AlignedVariant(2, "C", "CTT")
        assert (insertion.kind, insertion.size) == ("INS", 2)
        [deletion] = alignment_to_variants("ACTTGT", "AC--GT", 1, "N")
        assert deletion == AlignedVariant(2, "CTT", "C")
        assert (deletion.kind, deletion.size, deletion.end) == ("DEL", 2, 4)

    def test_replacement_is_decomposed(self):
        # Adjacent substitutions become separate SNVs
        assert alignment_to_variants("ACGGT", "ACTTT", 1, "N") == [
            AlignedVariant(3, "G", "T"),
            AlignedVariant(4, "G", "T"),
        ]
        # A replacement of unequal length becomes a deletion plus an SNV
        assert alignment_to_variants("ACGGGT", "ACTT-T", 1, "N") == [
            AlignedVariant(3, "GG", "T"),
            AlignedVariant(5, "G", "T"),
        ]

    def test_variant_at_start_uses_anchor_base(self):
        assert alignment_to_variants("--ACGT", "TTACGT", 1, "N") == [AlignedVariant(0, "N", "NTT")]

    def test_start_position_offsets_coordinates(self):
        assert alignment_to_variants("ACGT", "ACTT", 101, "N") == [AlignedVariant(103, "G", "T")]


class TestHarmonizeBasics:
    def test_identical_haplotypes_need_no_alignment(self):
        reference = "".join(random.Random(1).choices("ACGT", k=100))
        result = harmonize(reference, {"a": reference})
        assert result.variants == {"a": ()}
        assert result.msa == {}

    def test_dedup_gives_identical_variants(self):
        reference = "".join(random.Random(2).choices("ACGT", k=100))
        hap = reference[:50] + "AAAA" + reference[50:]
        result = harmonize(reference, {"a": hap, "b": hap}, dedup=True)
        assert result.variants["a"] == result.variants["b"]
        assert len(result.variants["a"]) == 1

    def test_unknown_aligner(self):
        with pytest.raises(KeyError):
            harmonize("ACGT", {"a": "ACGA"}, aligner="nope")

    def test_variant_counts(self):
        reference = "".join(random.Random(3).choices("ACGT", k=100))
        hap = reference[:50] + "AAAA" + reference[50:]
        carriers = variant_counts(harmonize(reference, {"a": hap, "b": hap}))
        assert sorted(next(iter(carriers.values()))) == ["a", "b"]


@pytest.mark.parametrize("aligner", ["poa", "wfa"])
@pytest.mark.parametrize("case", CASES)
class TestTruvariParity:
    """Our port against what Truvari 5.3.0 itself produced (see tests/graphs/data/make_truvari_harmonize_oracle.py)."""

    def test_variants_from_truvari_alignment(self, case, aligner):
        """Variant calling is deterministic: given Truvari's own MSA we must call exactly Truvari's variants."""
        entry = ORACLE["cases"][case][aligner]
        variants = msa_to_variants(_oracle_msa(entry))
        assert _carriers(variants) == _oracle_variants(entry)

    def test_end_to_end(self, case, aligner):
        """Our alignment + variant calling vs Truvari's (abPOA versions can differ in tie-breaking, so this can drift)."""
        data = ORACLE["cases"][case]
        result = harmonize(data["reference"], data["haplotypes"], aligner=aligner)
        assert _carriers(result.variants) == _oracle_variants(data[aligner])
