"""Haplotype-level similarity metrics (prototype).

Two complementary, graph-independent answers to "are these two haplotypes the same thing for genotyping purposes?":

* `site_similarity`: *same-site allele* similarity. Reduce each haplotype to its significant edits against the reference
  (insertions/deletions of at least `min_size` bp) and match edits between the haplotypes by type, position and size (and,
  optionally, sequence). Captures "the same variant, differing in detail", e.g. INS-171 vs INS-172 at one site.
* `evidence_distance`: sequencing-*evidence* distance. Poisson deviance between the expected k-mer counts of the two
  haplotypes, over the k-mers whose multiplicity differs. Captures "could short reads tell these apart", independent of
  how the edits are labeled.

They answer different questions and are meant to be combined: site similarity says the haplotypes *should* be treated as
equivalent; evidence distance says whether the data can separate them. Both take plain sequences (e.g. from
`Graph.path_sequence`), so they work for sampled haplotypes, assembled paths and synthetic counterfactuals alike.

Limitations of this prototype:
* `haplotype_edits` uses `difflib`, which is quadratic and not repeat-aware: in tandem repeats the same event can be placed
  at shifted coordinates, so keep `max_distance` generous there (or left-align first).
* `evidence_distance` is Poisson (no overdispersion/stutter/GC bias) and dominated by high-multiplicity repeat k-mers.
"""

import collections
import difflib
import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


# --------------------------------------------------------------------------------------------------------------------
# Same-site allele similarity
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Edit:
    """A significant difference between a haplotype and the reference, in reference coordinates (0-based, half-open)."""

    kind: (
        str  # "INS"/"DEL" if the haplotype is longer/shorter over the span (by net length), else "REPL" (equal length)
    )
    ref_start: int
    ref_end: int
    alt_seq: str  # sequence the haplotype carries over [ref_start, ref_end)
    ref_seq: str  # sequence the reference has over [ref_start, ref_end)

    @property
    def size(self) -> int:
        """Net length change for INS/DEL (a 1bp substitution beside a 35bp insertion is a 35bp INS), span length for REPL."""
        if self.kind == "REPL":
            return len(self.alt_seq)
        return abs(len(self.alt_seq) - len(self.ref_seq))

    @property
    def content(self) -> str:
        """The sequence that differs: the haplotype's for INS/REPL, the reference's for DEL."""
        return self.ref_seq if self.kind == "DEL" else self.alt_seq


def _kind(ref_seq: str, alt_seq: str) -> str:
    if len(alt_seq) > len(ref_seq):
        return "INS"
    if len(alt_seq) < len(ref_seq):
        return "DEL"
    return "REPL"


def haplotype_edits(reference: str, haplotype: str, *, min_size: int = 20, merge_gap: int = 10) -> list[Edit]:
    """Significant edits of `haplotype` relative to `reference`.

    Raw alignment blocks closer than `merge_gap` reference bases are merged first, since an event in a repeat is often split
    into several adjacent blocks (e.g. a 35bp insertion aligned as 1bp + 34bp). Merged edits smaller than `min_size`
    (by `Edit.size`) are dropped as "background".
    """
    blocks = [
        list(op[1:])  # i1, i2, j1, j2
        for op in difflib.SequenceMatcher(None, reference, haplotype, autojunk=False).get_opcodes()
        if op[0] != "equal"
    ]
    merged: list[list[int]] = []
    for block in blocks:
        if merged and block[0] - merged[-1][1] <= merge_gap:
            merged[-1][1], merged[-1][3] = block[1], block[3]
        else:
            merged.append(block)

    edits = []
    for i1, i2, j1, j2 in merged:
        ref_seq, alt_seq = reference[i1:i2], haplotype[j1:j2]
        edit = Edit(_kind(ref_seq, alt_seq), i1, i2, alt_seq, ref_seq)
        if edit.size >= min_size:
            edits.append(edit)
    return edits


@dataclass(frozen=True)
class SiteMatch:
    a: Edit
    b: Edit
    distance: int
    size_ratio: float
    seq_similarity: float | None


@dataclass(frozen=True)
class SiteSimilarity:
    score: float  # size-weighted fraction of edits matched, in [0, 1]; 1.0 if neither haplotype has significant edits
    equivalent: bool  # every significant edit in both haplotypes has a match
    matches: tuple[SiteMatch, ...]
    unmatched_a: tuple[Edit, ...]
    unmatched_b: tuple[Edit, ...]


def _seq_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def match_edits(
    edits_a: Sequence[Edit],
    edits_b: Sequence[Edit],
    *,
    max_distance: int = 10,
    min_size_ratio: float = 0.75,
    min_seq_similarity: float | None = None,
) -> SiteSimilarity:
    """One-to-one matching of two edit lists; see `site_similarity` for the criteria."""
    candidates = []
    for ia, ea in enumerate(edits_a):
        for ib, eb in enumerate(edits_b):
            if ea.kind != eb.kind:
                continue
            distance = max(abs(ea.ref_start - eb.ref_start), abs(ea.ref_end - eb.ref_end))
            ratio = min(ea.size, eb.size) / max(ea.size, eb.size)
            if distance > max_distance or ratio < min_size_ratio:
                continue
            similarity = None
            if min_seq_similarity is not None:
                similarity = _seq_similarity(ea.content, eb.content)
                if similarity < min_seq_similarity:
                    continue
            candidates.append((distance, -ratio, ia, ib, SiteMatch(ea, eb, distance, ratio, similarity)))

    # Greedy one-to-one assignment, closest then most similar in size first
    used_a, used_b, matches = set(), set(), []
    for _, _, ia, ib, match in sorted(candidates, key=lambda c: c[:4]):
        if ia in used_a or ib in used_b:
            continue
        used_a.add(ia)
        used_b.add(ib)
        matches.append(match)

    unmatched_a = tuple(e for i, e in enumerate(edits_a) if i not in used_a)
    unmatched_b = tuple(e for i, e in enumerate(edits_b) if i not in used_b)
    total = sum(e.size for e in edits_a) + sum(e.size for e in edits_b)
    score = 1.0 if total == 0 else 2 * sum(min(m.a.size, m.b.size) for m in matches) / total
    return SiteSimilarity(score, not unmatched_a and not unmatched_b, tuple(matches), unmatched_a, unmatched_b)


def site_similarity(
    reference: str,
    hap_a: str,
    hap_b: str,
    *,
    min_size: int = 20,
    merge_gap: int = 10,
    max_distance: int = 10,
    min_size_ratio: float = 0.75,
    min_seq_similarity: float | None = None,
) -> SiteSimilarity:
    """Same-site allele similarity of two haplotypes relative to a common `reference`.

    Two edits match if they are the same kind (INS/DEL/REPL), their reference spans start and end within `max_distance` bp,
    their sizes have min/max ratio of at least `min_size_ratio`, and, if `min_seq_similarity` is given, their differing
    sequences have `difflib` ratio at least that. Matching is one-to-one. `score` is the fraction of total edit size that is
    matched (2*sum(min(size_a, size_b)) / (total_a + total_b)); `equivalent` requires every significant edit to be matched.
    """
    return match_edits(
        haplotype_edits(reference, hap_a, min_size=min_size, merge_gap=merge_gap),
        haplotype_edits(reference, hap_b, min_size=min_size, merge_gap=merge_gap),
        max_distance=max_distance,
        min_size_ratio=min_size_ratio,
        min_seq_similarity=min_seq_similarity,
    )


def site_similarity_matrix(reference: str, haplotypes: Sequence[str], **kwargs) -> np.ndarray:
    """(H, H) matrix of `SiteSimilarity.score`; `kwargs` are split between `haplotype_edits` and `match_edits`."""
    edit_args = {k: kwargs.pop(k) for k in ("min_size", "merge_gap") if k in kwargs}
    edits = [haplotype_edits(reference, h, **edit_args) for h in haplotypes]
    matrix = np.ones((len(haplotypes), len(haplotypes)))
    for i, j in itertools.combinations(range(len(haplotypes)), 2):
        matrix[i, j] = matrix[j, i] = match_edits(edits[i], edits[j], **kwargs).score
    return matrix


# --------------------------------------------------------------------------------------------------------------------
# Sequencing-evidence distance
# --------------------------------------------------------------------------------------------------------------------


def _canonical(kmer: str) -> str:
    rc = kmer[::-1].translate(_COMPLEMENT)
    return min(kmer, rc)


def kmer_multiplicities(sequence: str, k: int = 21) -> collections.Counter:
    """Canonical k-mer -> number of occurrences in `sequence` (k-mers with ambiguous bases are skipped)."""
    return collections.Counter(
        _canonical(sequence[i : i + k]) for i in range(len(sequence) - k + 1) if "N" not in sequence[i : i + k]
    )


@dataclass(frozen=True)
class EvidenceDistance:
    distance: float  # symmetrized Poisson deviance of expected k-mer counts (per haplotype copy at `depth`)
    n_kmers: int  # k-mers whose multiplicity differs between the haplotypes (the statistic's degrees of freedom)

    # Note: each fully-differing k-mer contributes a near-constant deviance for a given depth (about 60 at depth 15), so the
    # distance is essentially proportional to `n_kmers`; the practical question "can reads tell them apart" is thus how many
    # k-mers differ, and any threshold should be calibrated against replicate simulations (A2), not read off analytically.


def _deviance(observed: np.ndarray, expected: np.ndarray) -> float:
    positive = observed > 0
    terms = np.where(positive, observed * np.log(np.where(positive, observed, 1.0) / expected), 0.0)
    return float(2 * np.sum(terms - (observed - expected)))


def _evidence_distance(counts_a: collections.Counter, counts_b: collections.Counter, depth: float, eps: float):
    keys = [k for k in counts_a.keys() | counts_b.keys() if counts_a.get(k, 0) != counts_b.get(k, 0)]
    if not keys:
        return EvidenceDistance(0.0, 0)
    a = depth * np.array([counts_a.get(k, 0) for k in keys], dtype=float)
    b = depth * np.array([counts_b.get(k, 0) for k in keys], dtype=float)
    distance = 0.5 * (_deviance(a, b + eps) + _deviance(b, a + eps))
    return EvidenceDistance(distance, len(keys))


def evidence_distance(
    hap_a: str, hap_b: str, *, k: int = 21, depth: float = 15.0, eps: float = 0.3
) -> EvidenceDistance:
    """Evidence distance between two haplotype sequences.

    `depth` is the expected k-mer count contributed by one copy of a haplotype (k-mer coverage / ploidy); `eps` is an additive
    error floor on expected counts. Only k-mers whose multiplicity differs are scored, so the distance is exactly zero for
    identical haplotypes and does not depend on any other candidate haplotypes or on shared flanking sequence. For
    homozygous genotypes the distance is about twice this value (deviance scales with depth); for hets compare via
    `evidence_distance` of the differing haplotype.
    """
    return _evidence_distance(kmer_multiplicities(hap_a, k), kmer_multiplicities(hap_b, k), depth, eps)


def evidence_distance_matrix(
    haplotypes: Sequence[str], *, k: int = 21, depth: float = 15.0, eps: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """(H, H) matrices of `EvidenceDistance.distance` and `.n_kmers`."""
    counts = [kmer_multiplicities(h, k) for h in haplotypes]
    distances = np.zeros((len(haplotypes), len(haplotypes)))
    n_kmers = np.zeros_like(distances, dtype=int)
    for i, j in itertools.combinations(range(len(haplotypes)), 2):
        result = _evidence_distance(counts[i], counts[j], depth, eps)
        distances[i, j] = distances[j, i] = result.distance
        n_kmers[i, j] = n_kmers[j, i] = result.n_kmers
    return distances, n_kmers
