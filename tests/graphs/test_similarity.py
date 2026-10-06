import random

import numpy as np
import pytest

from npsv3.graphs.similarity import (
    evidence_distance,
    evidence_distance_matrix,
    haplotype_edits,
    site_similarity,
    site_similarity_matrix,
)


def _random_sequence(length: int, seed: int) -> str:
    return "".join(random.Random(seed).choices("ACGT", k=length))


@pytest.fixture
def reference() -> str:
    return _random_sequence(400, seed=1)


def _insert(sequence: str, position: int, insertion: str) -> str:
    return sequence[:position] + insertion + sequence[position:]


class TestSiteSimilarity:
    def test_edits_ignore_small_background(self, reference):
        hap = _insert(_insert(reference, 100, "ACGTAC"), 250, _random_sequence(60, seed=2))
        [edit] = haplotype_edits(reference, hap, min_size=20)
        assert edit.kind == "INS"
        assert edit.size == 60

    def test_adjacent_blocks_are_merged(self, reference):
        insertion = _random_sequence(40, seed=3)
        hap = _insert(_insert(reference, 100, insertion[:15]), 105, insertion[15:])
        [edit] = haplotype_edits(reference, hap, min_size=20, merge_gap=10)
        assert edit.size == 40

    def test_one_base_longer_insertion_is_equivalent(self, reference):
        insertion = _random_sequence(171, seed=4)
        result = site_similarity(
            reference, _insert(reference, 150, insertion), _insert(reference, 150, insertion + "A")
        )
        assert result.equivalent
        assert result.score > 0.99
        assert len(result.matches) == 1

    def test_size_ratio_threshold(self, reference):
        short, long = _random_sequence(35, seed=5), _random_sequence(52, seed=5)
        hap_a, hap_b = _insert(reference, 150, short), _insert(reference, 150, long)
        assert not site_similarity(reference, hap_a, hap_b).equivalent  # 35/52 = 0.67 < 0.75
        loose = site_similarity(reference, hap_a, hap_b, min_size_ratio=0.6)
        assert loose.equivalent
        assert loose.score == pytest.approx(2 * 35 / (35 + 52))

    def test_distance_threshold(self, reference):
        insertion = _random_sequence(60, seed=6)
        hap_a, hap_b = _insert(reference, 100, insertion), _insert(reference, 180, insertion)
        assert not site_similarity(reference, hap_a, hap_b).equivalent
        assert site_similarity(reference, hap_a, hap_b, max_distance=100).equivalent

    def test_kind_must_match(self, reference):
        deletion = reference[:150] + reference[210:]
        result = site_similarity(reference, deletion, _insert(reference, 150, _random_sequence(60, seed=7)))
        assert not result.equivalent
        assert result.score == 0.0

    def test_sequence_similarity_filter(self, reference):
        hap_a = _insert(reference, 150, _random_sequence(60, seed=8))
        hap_b = _insert(reference, 150, _random_sequence(60, seed=9))
        assert site_similarity(reference, hap_a, hap_b).equivalent
        assert not site_similarity(reference, hap_a, hap_b, min_seq_similarity=0.9).equivalent

    def test_reference_like_haplotypes(self, reference):
        result = site_similarity(reference, reference, _insert(reference, 100, "ACG"))
        assert result.equivalent
        assert result.score == 1.0
        assert (
            site_similarity(reference, reference, _insert(reference, 100, _random_sequence(60, seed=10))).score == 0.0
        )

    def test_partial_match_scores_between_zero_and_one(self, reference):
        sv, other = _random_sequence(60, seed=11), _random_sequence(80, seed=12)
        hap_a = _insert(reference, 100, sv)
        hap_b = _insert(_insert(reference, 100, sv), 300, other)
        result = site_similarity(reference, hap_a, hap_b)
        assert not result.equivalent
        assert len(result.unmatched_b) == 1
        assert result.score == pytest.approx(2 * 60 / (60 + 60 + 80))

    def test_matrix(self, reference):
        insertion = _random_sequence(60, seed=13)
        haplotypes = [reference, _insert(reference, 150, insertion), _insert(reference, 150, insertion + "A")]
        matrix = site_similarity_matrix(reference, haplotypes)
        assert matrix.shape == (3, 3)
        np.testing.assert_allclose(matrix, matrix.T)
        assert matrix[1, 2] > 0.99
        assert matrix[0, 1] == 0.0


class TestEvidenceDistance:
    def test_identical_haplotypes(self, reference):
        result = evidence_distance(reference, reference)
        assert result.distance == 0.0
        assert result.n_kmers == 0

    def test_symmetric_and_positive(self, reference):
        other = _insert(reference, 200, _random_sequence(40, seed=14))
        forward, backward = evidence_distance(reference, other), evidence_distance(other, reference)
        assert forward.distance == pytest.approx(backward.distance)
        assert forward.distance > 0
        assert forward.n_kmers > 0

    def test_independent_of_shared_flanks(self, reference):
        other = _insert(reference, 200, _random_sequence(40, seed=15))
        flank = _random_sequence(300, seed=16)
        bare = evidence_distance(reference, other)
        flanked = evidence_distance(flank + reference + flank, flank + other + flank)
        assert flanked.distance == pytest.approx(bare.distance, rel=0.1)

    def test_strand_invariant(self, reference):
        other = _insert(reference, 200, _random_sequence(40, seed=17))
        rc = str.maketrans("ACGT", "TGCA")
        assert evidence_distance(reference, other).distance == pytest.approx(
            evidence_distance(reference, other[::-1].translate(rc)).distance
        )

    def test_larger_difference_is_farther(self, reference):
        small = _insert(reference, 200, _random_sequence(25, seed=18))
        large = _insert(reference, 200, _random_sequence(100, seed=19))
        assert evidence_distance(reference, large).distance > evidence_distance(reference, small).distance

    def test_one_base_difference_is_closer_than_new_insertion(self, reference):
        insertion = _random_sequence(100, seed=20)
        sv, longer = _insert(reference, 200, insertion), _insert(reference, 200, insertion + "A")
        assert evidence_distance(sv, longer).distance < evidence_distance(reference, sv).distance

    def test_matrix(self, reference):
        haplotypes = [
            reference,
            _insert(reference, 200, _random_sequence(40, seed=21)),
            _insert(reference, 100, "A" * 30),
        ]
        distances, n_kmers = evidence_distance_matrix(haplotypes)
        assert distances.shape == n_kmers.shape == (3, 3)
        np.testing.assert_allclose(distances, distances.T)
        assert np.all(np.diag(distances) == 0)
        assert distances[0, 1] == pytest.approx(evidence_distance(haplotypes[0], haplotypes[1]).distance)
