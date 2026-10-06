"""Truvari-style comparison of harmonized haplotypes.

Answers two questions about a pair of haplotypes over a region:

* *Would Truvari consider them different?* `truvari_differ` / `HaplotypeComparison.differs`. The haplotypes are harmonized
  together with the reference (`harmonize`, the `truvari phab` step) and the variants of one are then benchmarked against the
  variants of the other with Truvari `bench` matching, as `truvari refine` does. They differ if any benchmarked variant is
  left unmatched (a false negative or false positive).
* *How similar are they?* `Comparison.matches` etc. carry per-pair sequence/size similarity, reciprocal overlap and distance
  (computed for every candidate pair, not short-circuited, so near misses are quantified), and `HaplotypeComparison.score`
  summarizes them. `MatchResult.required` gives, per criterion, the threshold at which a pair would start to match.

The matching is a port of Truvari 5.3.0 (`variant_record.py`, `comparisons.py`, `matching.py`, `bench.py`; MIT license,
https://github.com/ACEnglish/truvari): same-type check, `refdist` position test, `pctsize`, reciprocal overlap with insertion
inflation (`pctovl`), `pctseq` via edit distance with rotation/unrolling for tandem repeats, size filters (`sizemin` for the
base, `sizefilt` for the comparison, `sizemax`), and the greedy one-to-one `pick=single`. Genotype handling is omitted
(haplotypes are single-copy). Ties in the pick order are broken by position in the input, whereas Truvari's are arbitrary.

Defaults are *not* Truvari's `bench` defaults but the criteria used in this project (`refdist` 20, `pctovl` 0.6, `sizemin`
30); use `TruvariParams.truvari_defaults()` for the stock values. Needs the optional `harmonize` extra.
"""

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from npsv3.graphs.harmonize import AlignedVariant, Harmonized, harmonize


@dataclass(frozen=True)
class TruvariParams:
    refdist: int = 20  # max distance between the calls' boundaries
    pctseq: float = 0.7  # minimum sequence similarity (0 disables)
    pctsize: float = 0.7  # minimum min(size)/max(size)
    pctovl: float = 0.6  # minimum reciprocal overlap (insertions are inflated by their size)
    sizemin: int = 30  # minimum size of base calls
    sizefilt: int = 30  # minimum size of comparison calls to consider matching
    sizemax: int = 50_000  # maximum size of either call (-1 for no maximum)
    typeignore: bool = False  # don't require the same variant type
    roll: bool = True  # compare rotations/unrolled sequences to tolerate shifted placement in repeats (`bench` default)

    @classmethod
    def truvari_defaults(cls, **overrides) -> "TruvariParams":
        """Truvari 5.3.0 `bench` defaults (`refdist` 500, `pctovl` 0, `sizemin` 50, ...)."""
        return cls(refdist=500, pctovl=0.0, sizemin=50, **overrides)


DEFAULT_PARAMS = TruvariParams()


# --------------------------------------------------------------------------------------------------------------------
# Truvari primitives
# --------------------------------------------------------------------------------------------------------------------


def edit_similarity(a: str, b: str) -> float:
    """Truvari `seqsim`: (len_a + len_b - edit_distance) / (len_a + len_b)."""
    import edlib

    a, b = a.upper(), b.upper()
    total = len(a) + len(b)
    return (total - edlib.align(a, b)["editDistance"]) / total


def _smallest_rotation(sequence: str) -> str:
    doubled = sequence + sequence
    n = len(sequence)
    return min(doubled[i : i + n] for i in range(n))


def roll_similarity(a: str, b: str) -> float:
    """Similarity of the lexicographically smallest rotations of the sequences."""
    return edit_similarity(_smallest_rotation(a), _smallest_rotation(b))


def unroll_similarity(seq_a: str, seq_b: str, offset: int) -> float:
    """Similarity of `seq_a` and `seq_b` rotated by `offset`, how far `seq_a` is upstream of `seq_b`."""
    f = offset % len(seq_b)
    return edit_similarity(seq_a, seq_b[-f:] + seq_b[:-f])


def best_similarity(a: str, b: str, start_distance: int) -> float:
    """Best of rolled, unrolled (both directions from both ends) and direct similarity, as Truvari `best_seqsim`."""
    rolled = roll_similarity(a, b) if len(a) < 500 and len(b) < 500 else 0.0  # noqa: PLR2004
    return max(
        rolled,
        unroll_similarity(a, b, start_distance),
        unroll_similarity(a, b, -start_distance),
        unroll_similarity(b, a, start_distance),
        unroll_similarity(b, a, -start_distance),
        edit_similarity(a, b),
    )


def size_similarity(size_a: int, size_b: int) -> tuple[float, int]:
    """Truvari `sizesim`: min/max size ratio and the size difference (a - b)."""
    if size_a == 0 or size_b == 0:
        if size_a == size_b:
            return 1.0, 0
        size_a, size_b = max(size_a, 1), max(size_b, 1)
    return min(size_a, size_b) / max(size_a, size_b), size_a - size_b


def reciprocal_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    start, end = max(a_start, b_start), min(a_end, b_end)
    if start < end:
        return (end - start) / max(a_end - a_start, b_end - b_start)
    return 0.0


def _start(variant: AlignedVariant) -> int:
    return variant.pos - 1  # 0-based, like pysam


def _end(variant: AlignedVariant) -> int:
    return variant.pos - 1 + len(variant.ref)


def truvari_size(variant: AlignedVariant) -> int:
    """Truvari `var_size`: length difference for indels, 0 for SNVs."""
    return 0 if variant.kind == "SNP" else abs(len(variant.alt) - len(variant.ref))


def _boundaries(variant: AlignedVariant, *, ins_inflate: bool = False) -> tuple[int, int]:
    start, end = _start(variant), _end(variant)
    if ins_inflate and variant.kind == "INS":
        size = truvari_size(variant)
        start, end = start - size // 2, end + size // 2
    return start, end


def _sequence(variant: AlignedVariant) -> str:
    return (variant.ref if variant.kind == "DEL" else variant.alt).upper()


# --------------------------------------------------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchResult:
    base: AlignedVariant | None
    comp: AlignedVariant | None
    state: bool = False  # Truvari considers the pair a match
    seqsim: float | None = None
    sizesim: float | None = None
    ovlpct: float | None = None
    sizediff: int | None = None
    st_dist: int | None = None  # base start minus comp start
    ed_dist: int | None = None
    score: float | None = None  # Truvari's TruScore, 0-100
    failed: tuple[str, ...] = ()  # criteria not met: "type", "refdist", "pctsize", "pctovl", "pctseq"
    refdist_needed: int | None = None  # smallest `refdist` at which the boundaries would be within range

    @property
    def required(self) -> dict[str, float]:
        """The threshold of each criterion at which this pair would pass (`pctseq` etc. at or below, `refdist` at or above)."""
        return {
            "refdist": self.refdist_needed,
            "pctsize": self.sizesim,
            "pctovl": self.ovlpct,
            "pctseq": self.seqsim,
        }

    def sort_key(self) -> tuple:
        """Truvari's ordering: matches first, then score, then closeness of start and end."""
        return (
            self.state,
            self.score if self.score is not None else 0,
            -abs(self.st_dist) if self.st_dist is not None else -float("inf"),
            -abs(self.ed_dist) if self.ed_dist is not None else -float("inf"),
        )


def match_variants(base: AlignedVariant, comp: AlignedVariant, params: TruvariParams) -> MatchResult:
    """Truvari `VariantRecord.var_match` without short-circuiting (all measures are always computed)."""
    failed = []
    if not params.typeignore and base.kind != comp.kind:
        failed.append("type")

    b_start, b_end = _boundaries(base)
    c_start, c_end = _boundaries(comp)
    refdist_needed = max(0, max(b_start - c_end, c_start - b_end) + 1)
    if not max(b_start - params.refdist, c_start) < min(b_end + params.refdist, c_end):
        failed.append("refdist")

    sizesim, sizediff = size_similarity(truvari_size(base), truvari_size(comp))
    if sizesim < params.pctsize:
        failed.append("pctsize")

    ovlpct = reciprocal_overlap(*_boundaries(base, ins_inflate=True), *_boundaries(comp, ins_inflate=True))
    if ovlpct < params.pctovl:
        failed.append("pctovl")

    seqsim = 0.0
    st_dist, ed_dist = b_start - c_start, b_end - c_end
    if params.pctseq > 0:
        if base.ref == comp.ref and base.alt == comp.alt:
            seqsim = 1.0
        elif not params.roll or st_dist == 0 or ed_dist == 0:
            seqsim = edit_similarity(_sequence(base), _sequence(comp))
        else:
            seqsim = best_similarity(_sequence(base), _sequence(comp), st_dist)
        if seqsim < params.pctseq:
            failed.append("pctseq")

    score = (seqsim + sizesim + ovlpct) / 3.0 * 100
    return MatchResult(
        base,
        comp,
        not failed,
        seqsim,
        sizesim,
        ovlpct,
        sizediff,
        st_dist,
        ed_dist,
        score,
        tuple(failed),
        refdist_needed,
    )


@dataclass(frozen=True)
class Comparison:
    """Result of benchmarking `comp` variants against `base` variants (one direction)."""

    matches: tuple[MatchResult, ...]  # true positives
    fn: tuple[AlignedVariant, ...]  # base variants without a match
    fp: tuple[AlignedVariant, ...]  # comparison variants without a match (and at least `sizemin`, as in Truvari)
    ignored_base: tuple[AlignedVariant, ...]  # filtered by size
    ignored_comp: tuple[AlignedVariant, ...]
    nearest: Mapping[AlignedVariant, MatchResult]  # for each unmatched variant, its best (failed) candidate pair
    params: TruvariParams

    @property
    def differs(self) -> bool:
        return bool(self.fn or self.fp)

    @property
    def score(self) -> float:
        """Size-weighted fraction of benchmarked variants that matched, 2*sum(matched)/(sum(base)+sum(comp)); 1.0 if none."""
        matched = sum(min(truvari_size(m.base), truvari_size(m.comp)) for m in self.matches)  # type: ignore[arg-type]
        total = sum(truvari_size(v) for v in (*self.fn, *self.fp)) + sum(
            truvari_size(m.base) + truvari_size(m.comp)  # type: ignore[arg-type]
            for m in self.matches
        )
        return 1.0 if total == 0 else 2 * matched / total

    def explain(self) -> list[str]:
        """Human-readable reasons for each unmatched variant."""
        lines = []
        for label, variants in (("FN (base only)", self.fn), ("FP (comp only)", self.fp)):
            for variant in variants:
                near = self.nearest.get(variant)
                why = (
                    "no candidate"
                    if near is None
                    else f"nearest fails {','.join(near.failed) or 'nothing'}: {near.required}"
                )
                lines.append(f"{label} {variant.kind}{truvari_size(variant)}@{variant.pos}: {why}")
        return lines


def compare_variants(
    base_variants: Sequence[AlignedVariant],
    comp_variants: Sequence[AlignedVariant],
    params: TruvariParams = DEFAULT_PARAMS,
) -> Comparison:
    """Benchmark `comp_variants` against `base_variants` as `truvari bench` does (`pick=single`)."""

    def too_big(variant):
        return params.sizemax != -1 and truvari_size(variant) > params.sizemax

    base = [v for v in base_variants if not too_big(v) and truvari_size(v) >= params.sizemin]
    comp = [v for v in comp_variants if not too_big(v) and truvari_size(v) >= params.sizefilt]
    ignored_base = tuple(v for v in base_variants if v not in base)
    ignored_comp = tuple(v for v in comp_variants if v not in comp)

    results = [[match_variants(b, c, params) for c in comp] for b in base]
    candidates = sorted(
        ((i, j) for i in range(len(base)) for j in range(len(comp)) if results[i][j].state),
        key=lambda ij: (results[ij[0]][ij[1]].sort_key(), -ij[0], -ij[1]),
        reverse=True,
    )
    used_base, used_comp, matches = set(), set(), []
    for i, j in candidates:
        if i in used_base or j in used_comp:
            continue
        used_base.add(i)
        used_comp.add(j)
        matches.append(results[i][j])

    nearest: dict[AlignedVariant, MatchResult] = {}
    for i, b in enumerate(base):
        if i not in used_base and comp:
            nearest[b] = max(results[i], key=MatchResult.sort_key)
    for j, c in enumerate(comp):
        if j not in used_comp and base:
            nearest[c] = max((row[j] for row in results), key=MatchResult.sort_key)

    fn = tuple(b for i, b in enumerate(base) if i not in used_base)
    # Truvari does not count unmatched comparison calls between sizefilt and sizemin as false positives
    fp = tuple(c for j, c in enumerate(comp) if j not in used_comp and truvari_size(c) >= params.sizemin)
    return Comparison(tuple(matches), fn, fp, ignored_base, ignored_comp, nearest, params)


# --------------------------------------------------------------------------------------------------------------------
# Haplotype level
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HaplotypeComparison:
    forward: Comparison  # variants of `a` as base, `b` as comparison
    reverse: Comparison | None  # the other direction (symmetric comparisons)
    harmonized: Harmonized

    @property
    def differs(self) -> bool:
        return self.forward.differs or (self.reverse is not None and self.reverse.differs)

    @property
    def score(self) -> float:
        scores = [self.forward.score] + ([self.reverse.score] if self.reverse is not None else [])
        return min(scores)


def compare_harmonized(
    harmonized: Harmonized,
    name_a: str,
    name_b: str,
    params: TruvariParams = DEFAULT_PARAMS,
    *,
    symmetric: bool = True,
) -> HaplotypeComparison:
    """Compare two haplotypes that were harmonized together (pairwise or as part of a larger set)."""
    variants_a, variants_b = harmonized.variants[name_a], harmonized.variants[name_b]
    forward = compare_variants(variants_a, variants_b, params)
    reverse = compare_variants(variants_b, variants_a, params) if symmetric else None
    return HaplotypeComparison(forward, reverse, harmonized)


def compare_haplotypes(
    reference: str,
    hap_a: str,
    hap_b: str,
    params: TruvariParams = DEFAULT_PARAMS,
    *,
    aligner: str = "poa",
    symmetric: bool = True,
) -> HaplotypeComparison:
    """Harmonize `reference`, `hap_a` and `hap_b` together (as `truvari phab` does for a pair) and compare them.

    With `symmetric`, `hap_a` is also benchmarked as the comparison and the result differs if either direction does; with
    `sizemin == sizefilt` the two directions only differ in how ties are picked.
    """
    harmonized = harmonize(reference, {"a": hap_a, "b": hap_b}, aligner=aligner)
    return compare_harmonized(harmonized, "a", "b", params, symmetric=symmetric)


def truvari_differ(
    reference: str,
    hap_a: str,
    hap_b: str,
    params: TruvariParams = DEFAULT_PARAMS,
    *,
    aligner: str = "poa",
) -> bool:
    """Would Truvari (after `phab` harmonization, as in `refine`) consider the two haplotypes different?"""
    return compare_haplotypes(reference, hap_a, hap_b, params, aligner=aligner).differs


@dataclass(frozen=True)
class HaplotypeMatrix:
    differs: np.ndarray  # (H, H) bool
    score: np.ndarray  # (H, H) float
    names: tuple[str, ...]
    comparisons: Mapping[tuple[str, str], HaplotypeComparison] = field(repr=False)


def compare_haplotype_set(
    reference: str,
    haplotypes: Mapping[str, str],
    params: TruvariParams = DEFAULT_PARAMS,
    *,
    aligner: str = "poa",
    joint: bool = False,
) -> HaplotypeMatrix:
    """All pairwise comparisons of the haplotypes.

    By default each pair is harmonized on its own with the reference (what Truvari would do for those two call sets, the most
    faithful). With `joint` all haplotypes are aligned once, which is faster and gives one consistent coordinate system, but
    a pair's variants can then depend on the other haplotypes in the set.
    """
    names = tuple(haplotypes)
    joint_harmonized = harmonize(reference, haplotypes, aligner=aligner) if joint else None
    differs = np.zeros((len(names), len(names)), dtype=bool)
    score = np.ones((len(names), len(names)))
    comparisons = {}
    for (i, a), (j, b) in itertools.combinations(enumerate(names), 2):
        if joint_harmonized is not None:
            result = compare_harmonized(joint_harmonized, a, b, params)
        else:
            result = compare_haplotypes(reference, haplotypes[a], haplotypes[b], params, aligner=aligner)
        differs[i, j] = differs[j, i] = result.differs
        score[i, j] = score[j, i] = result.score
        comparisons[a, b] = result
    return HaplotypeMatrix(differs, score, names, comparisons)


__all__ = [
    "Comparison",
    "HaplotypeComparison",
    "HaplotypeMatrix",
    "MatchResult",
    "DEFAULT_PARAMS",
    "TruvariParams",
    "best_similarity",
    "compare_haplotype_set",
    "compare_haplotypes",
    "compare_harmonized",
    "compare_variants",
    "edit_similarity",
    "match_variants",
    "truvari_differ",
    "truvari_size",
]
