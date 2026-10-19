import itertools
import json
import os
import random

import numpy as np
import pytest

pytest.importorskip("edlib")
pytest.importorskip("pyabpoa")

from npsv3.graphs.sequence_compare import (
    ComparisonParams,
    best_similarity,
    compare_genotypes,
    compare_haplotype_set,
    compare_haplotypes,
    compare_variants,
    edit_similarity,
    haplotypes_differ,
    match_variants,
    reciprocal_overlap,
    roll_similarity,
    size_similarity,
    unroll_similarity,
    variant_size,
)
from npsv3.graphs.sequence_harmonize import AlignedVariant

DATA = os.path.join(os.path.dirname(__file__), "data")
with open(os.path.join(DATA, "truvari_match_oracle.json")) as _f:
    ORACLE = json.load(_f)

PARAM_SETS = ORACLE["param_sets"]
RUNS = [(case, key) for case, entry in sorted(ORACLE["cases"].items()) for key in sorted(entry["results"])]


def _random_sequence(length: int, seed: int) -> str:
    return "".join(random.Random(seed).choices("ACGT", k=length))


def _ins(pos: int, sequence: str) -> AlignedVariant:
    return AlignedVariant(pos, sequence[0], sequence)


def _dele(pos: int, sequence: str) -> AlignedVariant:
    return AlignedVariant(pos, sequence, sequence[0])


class TestPrimitives:
    def test_edit_similarity(self):
        assert edit_similarity("ACGT", "ACGT") == 1.0
        assert edit_similarity("ACGT", "ACGA") == pytest.approx(7 / 8)
        assert edit_similarity("acgt", "ACGT") == 1.0

    def test_size_similarity(self):
        assert size_similarity(35, 52) == (pytest.approx(35 / 52), -17)
        assert size_similarity(0, 0) == (1.0, 0)
        assert size_similarity(0, 10) == (pytest.approx(0.1), -9)  # Truvari treats a zero size as 1

    def test_reciprocal_overlap(self):
        assert reciprocal_overlap(0, 100, 50, 150) == pytest.approx(0.5)
        assert reciprocal_overlap(0, 10, 20, 30) == 0.0
        assert reciprocal_overlap(0, 100, 0, 50) == pytest.approx(0.5)

    def test_roll_similarity_is_rotation_invariant(self):
        unit = _random_sequence(40, seed=1)
        rotated = unit[13:] + unit[:13]
        assert roll_similarity(unit, rotated) == 1.0
        assert edit_similarity(unit, rotated) < 1.0

    def test_unroll_similarity(self):
        unit = _random_sequence(40, seed=2)
        # Sequence B inserted `offset` bases downstream is A rotated by `offset`
        offset = 7
        downstream = unit[-offset:] + unit[:-offset]
        assert unroll_similarity(downstream, unit, offset) == 1.0 or unroll_similarity(unit, downstream, offset) == 1.0
        assert best_similarity(unit, downstream, offset) == 1.0


class TestMatching:
    PARAMS = ComparisonParams(refdist=20, pctseq=0.7, pctsize=0.7, pctovl=0.6, sizemin=30, sizefilt=30)

    def test_defaults_are_the_project_criteria(self):
        defaults = ComparisonParams()
        assert (defaults.refdist, defaults.pctovl, defaults.sizemin, defaults.sizefilt) == (20, 0.6, 30, 30)
        stock = ComparisonParams.truvari_defaults()
        assert (stock.refdist, stock.pctovl, stock.sizemin, stock.sizefilt) == (500, 0.0, 50, 30)

    def test_identical_variants_match(self):
        variant = _ins(100, _random_sequence(60, seed=3))
        result = match_variants(variant, variant, self.PARAMS)
        assert result.state
        assert (result.seqsim, result.sizesim, result.ovlpct) == (1.0, 1.0, 1.0)
        assert result.score == pytest.approx(100.0)

    def test_failed_criteria_are_reported(self):
        base, comp = _ins(100, _random_sequence(60, seed=4)), _ins(300, _random_sequence(40, seed=5))
        result = match_variants(base, comp, self.PARAMS)
        assert not result.state
        assert {"refdist", "pctsize", "pctovl"} <= set(result.failed)
        assert result.refdist_needed == 200  # first refdist at which the 1bp boundaries at 100 and 300 are within range

    def test_type_must_match_unless_ignored(self):
        sequence = _random_sequence(60, seed=6)
        base, comp = _ins(100, sequence), _dele(100, sequence)
        assert "type" in match_variants(base, comp, self.PARAMS).failed
        assert "type" not in match_variants(base, comp, ComparisonParams(typeignore=True)).failed

    def test_size_filters_follow_truvari(self):
        small, big = _ins(100, _random_sequence(25, seed=7)), _ins(100, _random_sequence(60, seed=8))
        comparison = compare_variants([small, big], [small, big], self.PARAMS)
        assert comparison.ignored_base == (small,)
        assert comparison.ignored_comp == (small,)
        assert not comparison.differs

    def test_unmatched_comparison_between_sizefilt_and_sizemin_is_not_a_false_positive(self):
        mid = _ins(100, _random_sequence(40, seed=9))
        params = ComparisonParams(sizemin=50, sizefilt=30)
        comparison = compare_variants([], [mid], params)
        assert not comparison.fp
        assert not comparison.differs

    def test_one_to_one_pick(self):
        sequence = _random_sequence(60, seed=10)
        base = [_ins(100, sequence)]
        comp = [_ins(100, sequence), _ins(102, sequence)]
        comparison = compare_variants(base, comp, self.PARAMS)
        assert len(comparison.matches) == 1
        assert comparison.matches[0].comp == comp[0]  # closest start wins the tie on score
        assert comparison.fp == (comp[1],)

    def test_empty_sides(self):
        variant = _ins(100, _random_sequence(60, seed=11))
        assert compare_variants([variant], [], self.PARAMS).fn == (variant,)
        assert compare_variants([], [variant], self.PARAMS).fp == (variant,)
        assert not compare_variants([], [], self.PARAMS).differs

    def test_score(self):
        sequence = _random_sequence(60, seed=12)
        assert compare_variants([_ins(100, sequence)], [_ins(100, sequence)], self.PARAMS).score == 1.0
        assert compare_variants([_ins(100, sequence)], [], self.PARAMS).score == 0.0

    def test_explain_names_the_failed_criteria(self):
        base, comp = _ins(100, _random_sequence(60, seed=13)), _ins(300, _random_sequence(60, seed=14))
        [line, _] = compare_variants([base], [comp], self.PARAMS).explain()
        assert "refdist" in line


@pytest.mark.parametrize(("case", "key"), RUNS)
def test_matches_truvari_bench(case, key):
    """Our matching against what Truvari 5.3.0 `bench` itself reports for the same variants (tests/graphs/data)."""
    entry = ORACLE["cases"][case]
    pair, set_name = key.split("|")
    base_name, comp_name = pair.split(">")
    oracle = entry["results"][key]

    def variants(name):
        return [AlignedVariant(v["pos"], v["ref"], v["alt"]) for v in entry["variants"] if name in v["carriers"]]

    params = dict(PARAM_SETS[set_name])
    result = compare_variants(variants(base_name), variants(comp_name), ComparisonParams(**params))

    def keys(items):
        return {(v["pos"], v["ref"], v["alt"]) for v in items}

    def vkey(v):
        return (v.pos, v.ref, v.alt)

    assert {vkey(m.base) for m in result.matches} == keys(oracle["tp_base"])
    assert {vkey(m.comp) for m in result.matches} == keys(oracle["tp_comp"])
    assert {vkey(v) for v in result.fn} == keys(oracle["fn"])
    assert {vkey(v) for v in result.fp} == keys(oracle["fp"])

    # Truvari's annotations on each matched base call
    by_base = {vkey(m.base): m for m in result.matches}
    for item in oracle["tp_base"]:
        match, info = by_base[(item["pos"], item["ref"], item["alt"])], item["info"]
        assert round(match.seqsim, 4) == pytest.approx(info["PctSeqSimilarity"])
        assert round(match.sizesim, 4) == pytest.approx(info["PctSizeSimilarity"])
        assert round(match.ovlpct, 4) == pytest.approx(info["PctRecOverlap"])
        assert match.sizediff == info["SizeDiff"]
        assert (match.st_dist, match.ed_dist) == (info["StartDistance"], info["EndDistance"])
        assert int(match.score) == info["TruScore"]


class TestHaplotypes:
    @pytest.fixture
    def reference(self) -> str:
        return _random_sequence(300, seed=20)

    def test_identical_haplotypes_do_not_differ(self, reference):
        hap = reference[:100] + _random_sequence(60, seed=21) + reference[100:]
        result = compare_haplotypes(reference, hap, hap)
        assert not result.differs
        assert result.score == 1.0

    def test_sv_vs_reference_differs(self, reference):
        hap = reference[:100] + _random_sequence(60, seed=22) + reference[100:]
        assert haplotypes_differ(reference, reference, hap)

    def test_small_variants_are_ignored(self, reference):
        hap = reference[:100] + "ACGTA" + reference[100:]
        assert not haplotypes_differ(reference, reference, hap)

    def test_one_base_longer_insertion_does_not_differ(self, reference):
        insertion = _random_sequence(100, seed=23)
        sv, longer = reference[:150] + insertion + reference[150:], reference[:150] + insertion + "A" + reference[150:]
        assert not haplotypes_differ(reference, sv, longer)

    def test_thresholds_are_configurable(self, reference):
        short, long = _random_sequence(35, seed=24), _random_sequence(60, seed=24)
        hap_a, hap_b = reference[:150] + short + reference[150:], reference[:150] + long + reference[150:]
        # 35/60 fails pctsize (0.7) and, after inflating insertions by their size, pctovl (0.6)
        assert haplotypes_differ(reference, hap_a, hap_b)
        assert haplotypes_differ(reference, hap_a, hap_b, ComparisonParams(pctsize=0.5, pctseq=0.5))
        assert not haplotypes_differ(reference, hap_a, hap_b, ComparisonParams(pctsize=0.5, pctseq=0.5, pctovl=0.5))

    def test_sizemin_is_configurable(self, reference):
        hap = reference[:100] + _random_sequence(40, seed=25) + reference[100:]
        assert haplotypes_differ(reference, reference, hap, ComparisonParams(sizemin=30, sizefilt=30))
        assert not haplotypes_differ(reference, reference, hap, ComparisonParams(sizemin=50, sizefilt=50))

    def test_wfa_aligner(self, reference):
        pytest.importorskip("pywfa")
        hap = reference[:100] + _random_sequence(60, seed=26) + reference[100:]
        assert haplotypes_differ(reference, reference, hap, aligner="wfa")

    def test_haplotype_set(self, reference):
        insertion = _random_sequence(80, seed=27)
        haplotypes = {
            "ref": reference,
            "sv": reference[:150] + insertion + reference[150:],
            "sv_plus": reference[:150] + insertion + "A" + reference[150:],
        }
        for joint in (False, True):
            matrix = compare_haplotype_set(reference, haplotypes, joint=joint)
            assert matrix.names == ("ref", "sv", "sv_plus")
            assert matrix.differs.tolist() == [[False, True, True], [True, False, False], [True, False, False]]
            assert matrix.score[1, 2] == pytest.approx(1.0, abs=0.01)
            assert matrix.score[0, 1] == 0.0


def test_variant_size():
    assert variant_size(AlignedVariant(10, "A", "T")) == 0
    assert variant_size(_ins(10, "A" + "C" * 40)) == 40
    assert variant_size(_dele(10, "A" + "C" * 25)) == 25


def _genotype_haplotypes() -> tuple[str, list[str]]:
    """A reference and candidates: the reference, a 300bp deletion, a near-identical 290bp deletion and an insertion"""
    reference = _random_sequence(2000, 7)
    deletion = reference[:900] + reference[1200:]
    similar = reference[:905] + reference[1195:]
    insertion = reference[:1000] + _random_sequence(300, 8) + reference[1000:]
    return reference, [reference, deletion, similar, insertion]


@pytest.mark.parametrize("joint", [True, False])
def test_compare_genotypes(joint):
    reference, haplotypes = _genotype_haplotypes()
    result = compare_genotypes(reference, haplotypes, [0, 1], joint=joint)

    genotypes = list(itertools.combinations_with_replacement(range(len(haplotypes)), 2))
    assert len(result.differs) == len(result.score) == len(genotypes)
    # The label and the label with a near-identical deletion are the same genotype
    same = {genotypes[i] for i in np.flatnonzero(~result.differs)}
    assert same == {(0, 1), (0, 2)}
    assert result.score[genotypes.index((0, 1))] == 1.0
    assert result.score[genotypes.index((0, 2))] > result.score[genotypes.index((0, 3))]
    assert result.label_differs is None and result.label_score is None


def test_compare_genotypes_is_unphased():
    reference, haplotypes = _genotype_haplotypes()
    result = compare_genotypes(reference, haplotypes, [1, 0])
    genotypes = list(itertools.combinations_with_replacement(range(len(haplotypes)), 2))
    assert {genotypes[i] for i in np.flatnonzero(~result.differs)} == {(0, 1), (0, 2)}


def test_compare_genotypes_with_sample():
    reference, haplotypes = _genotype_haplotypes()
    # The sample's own haplotypes are the label, but with the near-identical deletion
    result = compare_genotypes(reference, haplotypes, [0, 1], [haplotypes[2], reference])
    assert result.label_differs is False
    assert 0 < result.label_score <= 1

    result = compare_genotypes(reference, haplotypes, [0, 1], [reference, haplotypes[3]])
    assert result.label_differs is True


def test_compare_genotypes_skips_identical_sequences(monkeypatch):
    import npsv3.graphs.sequence_compare as sequence_compare

    def fail(*args, **kwargs):
        raise AssertionError("Identical sequences should not be harmonized")

    monkeypatch.setattr(sequence_compare, "harmonize", fail)
    reference, haplotypes = _genotype_haplotypes()
    result = compare_genotypes(reference, haplotypes[1:2], [0, 0], [haplotypes[1], haplotypes[1]])
    assert result.differs.tolist() == [False]
    assert result.score.tolist() == [1.0]
    assert result.label_differs is False and result.label_score == 1.0
