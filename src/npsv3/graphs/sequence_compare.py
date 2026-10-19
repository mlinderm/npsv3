"""Sequence comparison of harmonized haplotypes (Truvari-style variant matching).

Answers two questions about a pair of haplotypes over a region:

* *Are they different?* `haplotypes_differ` / `HaplotypeComparison.differs`. The haplotypes are harmonized
  together with the reference (`harmonize`, the `truvari phab` step) and the variants of one are then benchmarked against the
  variants of the other with Truvari `bench` matching, as `truvari refine` does. They differ if any benchmarked variant is
  left unmatched (a false negative or false positive).
* *How similar are they?* `Comparison.matches` etc. carry per-pair sequence/size similarity, reciprocal overlap and distance
  (computed for every candidate pair, not short-circuited, so near misses are quantified), and `HaplotypeComparison.score`
  summarizes them. `MatchResult.required` gives, per criterion, the threshold at which a pair would start to match.

`compare_genotypes` lifts both to genotypes: which candidate genotypes differ from a labeled genotype, and how similar they are.

The matching is a port of Truvari 5.3.0 (`variant_record.py`, `comparisons.py`, `matching.py`, `bench.py`; MIT license,
https://github.com/ACEnglish/truvari): same-type check, `refdist` position test, `pctsize`, reciprocal overlap with insertion
inflation (`pctovl`), `pctseq` via edit distance with rotation/unrolling for tandem repeats, size filters (`sizemin` for the
base, `sizefilt` for the comparison, `sizemax`), and the greedy one-to-one `pick=single`. Genotype handling is omitted
(haplotypes are single-copy). Ties in the pick order are broken by position in the input, whereas Truvari's are arbitrary.

Defaults are *not* Truvari's `bench` defaults but the criteria used in this project (`refdist` 20, `pctovl` 0.6, `sizemin`
30); use `ComparisonParams.truvari_defaults()` for the stock values. Needs the optional `harmonize` extra.
"""

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from npsv3.graphs.sequence_harmonize import AlignedVariant, Harmonized, harmonize


@dataclass(frozen=True)
class ComparisonParams:
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
    def truvari_defaults(cls, **overrides) -> "ComparisonParams":
        """Truvari 5.3.0 `bench` defaults (`refdist` 500, `pctovl` 0, `sizemin` 50, ...)."""
        return cls(refdist=500, pctovl=0.0, sizemin=50, **overrides)


DEFAULT_PARAMS = ComparisonParams()


def edit_similarity(seq_a: str, seq_b: str) -> float:
    """Return sequence simlarity (Truvari's `seqsim`), defined as (len_a + len_b - edit_distance) / (len_a + len_b)."""
    import edlib

    seq_a, seq_b = seq_a.upper(), seq_b.upper()
    total = len(seq_a) + len(seq_b)
    return (total - edlib.align(seq_a, seq_b)["editDistance"]) / total


def _smallest_rotation(sequence: str) -> str:
    doubled = sequence + sequence
    n = len(sequence)
    return min(doubled[i : i + n] for i in range(n))


def roll_similarity(seq_a: str, seq_b: str) -> float:
    """Return the similarity of the lexicographically smallest rotations of the sequences `seq_a` and `seq_b`."""
    return edit_similarity(_smallest_rotation(seq_a), _smallest_rotation(seq_b))


def unroll_similarity(seq_a: str, seq_b: str, offset: int) -> float:
    """Return the similarity of `seq_a` and `seq_b` rotated by `offset`, how far `seq_a` is upstream of `seq_b`."""
    f = offset % len(seq_b)
    return edit_similarity(seq_a, seq_b[-f:] + seq_b[:-f])


def best_similarity(seq_a: str, seq_b: str, start_distance: int) -> float:
    """Reutrn the best of rolled, unrolled (both directions from both ends) and direct similarity (Truvari's `best_seqsim`)."""
    rolled = roll_similarity(seq_a, seq_b) if len(seq_a) < 500 and len(seq_b) < 500 else 0.0  # noqa: PLR2004
    return max(
        rolled,
        unroll_similarity(seq_a, seq_b, start_distance),
        unroll_similarity(seq_a, seq_b, -start_distance),
        unroll_similarity(seq_b, seq_a, start_distance),
        unroll_similarity(seq_b, seq_a, -start_distance),
        edit_similarity(seq_a, seq_b),
    )


def size_similarity(size_a: int, size_b: int) -> tuple[float, int]:
    """Return min/max size ratio and the size difference (size_a - size_b) (Truvari's `sizesim`)."""
    if size_a == 0 or size_b == 0:
        if size_a == size_b:
            return 1.0, 0
        size_a, size_b = max(size_a, 1), max(size_b, 1)
    return min(size_a, size_b) / max(size_a, size_b), size_a - size_b


def reciprocal_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Return the reciprocal overlap of two intervals"""
    start, end = max(a_start, b_start), min(a_end, b_end)
    if start < end:
        return (end - start) / max(a_end - a_start, b_end - b_start)
    return 0.0


def _start(variant: AlignedVariant) -> int:
    return variant.pos - 1  # 0-based, like pysam


def _end(variant: AlignedVariant) -> int:
    return variant.pos - 1 + len(variant.ref)


def variant_size(variant: AlignedVariant) -> int:
    """Return the size of a variant as used by Truvari (`var_size`). The length difference for indels, 0 for SNVs."""
    return 0 if variant.kind == "SNP" else abs(len(variant.alt) - len(variant.ref))


def _boundaries(variant: AlignedVariant, *, ins_inflate: bool = False) -> tuple[int, int]:
    start, end = _start(variant), _end(variant)
    if ins_inflate and variant.kind == "INS":
        size = variant_size(variant)
        start, end = start - size // 2, end + size // 2
    return start, end


def _sequence(variant: AlignedVariant) -> str:
    return (variant.ref if variant.kind == "DEL" else variant.alt).upper()


@dataclass(frozen=True)
class MatchResult:
    """Comparison of one base variant with one comparison variant (from `match_variants`).

    Attributes:
        base: Variant being benchmarked against.
        comp: Variant being benchmarked.
        state: Whether the pair matches, i.e., no criterion failed.
        seqsim: Sequence similarity of the inserted/deleted sequences (0 if `pctseq` is disabled).
        sizesim: Size similarity, min(size)/max(size).
        ovlpct: Reciprocal overlap, with insertions inflated by their size.
        sizediff: Base size minus comparison size.
        st_dist: Base start minus comparison start.
        ed_dist: Base end minus comparison end.
        score: Truvari's TruScore, the mean of `seqsim`, `sizesim` and `ovlpct` scaled to 0-100.
        failed: Criteria not met, any of "type", "refdist", "pctsize", "pctovl" and "pctseq".
        refdist_needed: Smallest `refdist` at which the boundaries would be within range.
    """

    base: AlignedVariant
    comp: AlignedVariant
    state: bool
    seqsim: float
    sizesim: float
    ovlpct: float
    sizediff: int
    st_dist: int
    ed_dist: int
    score: float
    failed: tuple[str, ...]
    refdist_needed: int

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
        return (self.state, self.score, -abs(self.st_dist), -abs(self.ed_dist))


def match_variants(base: AlignedVariant, comp: AlignedVariant, params: ComparisonParams) -> MatchResult:
    """Truvari `VariantRecord.var_match` without short-circuiting (all measures are always computed)."""
    failed = []
    if not params.typeignore and base.kind != comp.kind:
        failed.append("type")

    b_start, b_end = _boundaries(base)
    c_start, c_end = _boundaries(comp)
    refdist_needed = max(0, max(b_start - c_end, c_start - b_end) + 1)
    if not max(b_start - params.refdist, c_start) < min(b_end + params.refdist, c_end):
        failed.append("refdist")

    sizesim, sizediff = size_similarity(variant_size(base), variant_size(comp))
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
    """Result of benchmarking comparison variants against base variants in one direction (from `compare_variants`).

    As in Truvari, unmatched comparison variants smaller than `sizemin` (but at least `sizefilt`) are in neither `fp` nor
    `ignored_comp`.

    Attributes:
        matches: True positives, one-to-one pairs of base and comparison variants.
        fn: Base variants without a match (false negatives).
        fp: Comparison variants of at least `sizemin` without a match (false positives).
        ignored_base: Base variants outside the size range (`sizemin` to `sizemax`), not benchmarked.
        ignored_comp: Comparison variants outside the size range (`sizefilt` to `sizemax`), not benchmarked.
        nearest: Each unmatched benchmarked variant -> its best-scoring (failed) pair, for explaining near misses. Absent
            if there were no variants on the other side.
        params: Thresholds used.
    """

    matches: tuple[MatchResult, ...]
    fn: tuple[AlignedVariant, ...]
    fp: tuple[AlignedVariant, ...]
    ignored_base: tuple[AlignedVariant, ...]
    ignored_comp: tuple[AlignedVariant, ...]
    nearest: Mapping[AlignedVariant, MatchResult]
    params: ComparisonParams

    @property
    def differs(self) -> bool:
        return bool(self.fn or self.fp)

    @property
    def score(self) -> float:
        """Size-weighted fraction of benchmarked variants that matched, 2*sum(matched)/(sum(base)+sum(comp)); 1.0 if none."""
        matched = sum(min(variant_size(m.base), variant_size(m.comp)) for m in self.matches)
        total = sum(variant_size(v) for v in (*self.fn, *self.fp)) + sum(
            variant_size(m.base) + variant_size(m.comp) for m in self.matches
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
                lines.append(f"{label} {variant.kind}{variant_size(variant)}@{variant.pos}: {why}")
        return lines


def compare_variants(
    base_variants: Sequence[AlignedVariant],
    comp_variants: Sequence[AlignedVariant],
    params: ComparisonParams = DEFAULT_PARAMS,
) -> Comparison:
    """Benchmark `comp_variants` against `base_variants` (as `truvari bench` does with `pick=single`)."""

    def too_big(variant):
        return params.sizemax != -1 and variant_size(variant) > params.sizemax

    base = [v for v in base_variants if not too_big(v) and variant_size(v) >= params.sizemin]
    comp = [v for v in comp_variants if not too_big(v) and variant_size(v) >= params.sizefilt]
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
    fp = tuple(c for j, c in enumerate(comp) if j not in used_comp and variant_size(c) >= params.sizemin)
    return Comparison(tuple(matches), fn, fp, ignored_base, ignored_comp, nearest, params)


@dataclass(frozen=True)
class HaplotypeComparison:
    """Comparison of two harmonized haplotypes, `a` and `b` (from `compare_harmonized` or `compare_haplotypes`).

    The haplotypes differ if either direction has an unmatched variant (`differs`), and their similarity is the lower of the
    two directions' scores (`score`).

    Attributes:
        forward: `a`'s variants as the base and `b`'s as the comparison.
        reverse: `b`'s variants as the base and `a`'s as the comparison, or None if the comparison is not symmetric.
        harmonized: Harmonization the variants were called from, which may include other haplotypes.
    """

    forward: Comparison
    reverse: Comparison | None
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
    params: ComparisonParams = DEFAULT_PARAMS,
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
    params: ComparisonParams = DEFAULT_PARAMS,
    *,
    aligner: str = "poa",
    symmetric: bool = True,
) -> HaplotypeComparison:
    """Return the comparison of `reference`, `hap_a` and `hap_b` harmonized together (as `truvari phab` does for a pair).

    With `symmetric`, `hap_a` is also benchmarked as the comparison and the result differs if either direction does; with
    `sizemin == sizefilt` the two directions only differ in how ties are picked.
    """
    harmonized = harmonize(reference, {"a": hap_a, "b": hap_b}, aligner=aligner)
    return compare_harmonized(harmonized, "a", "b", params, symmetric=symmetric)


def haplotypes_differ(
    reference: str,
    hap_a: str,
    hap_b: str,
    params: ComparisonParams = DEFAULT_PARAMS,
    *,
    aligner: str = "poa",
) -> bool:
    """Return true if the two haplotypes are different, i.e., would Truvari (after `phab` harmonization, as in `refine`) consider them so?"""
    return compare_haplotypes(reference, hap_a, hap_b, params, aligner=aligner).differs


@dataclass(frozen=True)
class HaplotypeMatrix:
    """All pairwise comparisons of a set of H haplotypes (from `compare_haplotype_set`).

    The matrices are symmetric and indexed in the order of `names`; a haplotype never differs from itself (score 1).

    Attributes:
        differs: (H, H) bool, whether each pair of haplotypes differs (`HaplotypeComparison.differs`).
        score: (H, H) float, similarity of each pair (`HaplotypeComparison.score`).
        names: Haplotype names, in matrix order.
        comparisons: (name_a, name_b) -> full comparison, for each pair with `name_a` before `name_b` in `names`.
    """

    differs: np.ndarray
    score: np.ndarray
    names: tuple[str, ...]
    comparisons: Mapping[tuple[str, str], HaplotypeComparison] = field(repr=False)


def compare_haplotype_set(
    reference: str,
    haplotypes: Mapping[str, str],
    params: ComparisonParams = DEFAULT_PARAMS,
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


@dataclass(frozen=True)
class GenotypeComparison:
    """Comparison of every candidate genotype with a labeled genotype (from `compare_genotypes`).

    A genotype matches another if its haplotypes can be paired with the other's (in any order) such that no pair differs
    (`HaplotypeComparison.differs`). Its score is the best, over those pairings, of the lowest pair score.

    Attributes:
        differs: (G,) bool, whether each genotype, in `combinations_with_replacement(range(H), ploidy)` order, differs
            from the label (False for the label itself).
        score: (G,) float, similarity of each genotype to the label (1 for the label itself).
        label_differs: Whether the label differs from the sample's own haplotypes, None if they were not provided.
        label_score: Similarity of the label to the sample's own haplotypes, None if they were not provided.
    """

    differs: np.ndarray
    score: np.ndarray
    label_differs: bool | None
    label_score: float | None


def compare_genotypes(
    reference: str,
    haplotypes: Sequence[str],
    label: Sequence[int],
    sample_haplotypes: Sequence[str] | None = None,
    params: ComparisonParams = DEFAULT_PARAMS,
    *,
    aligner: str = "poa",
    joint: bool = True,
) -> GenotypeComparison:
    """Compare each genotype of the candidate `haplotypes` with the `label` genotype (indices into `haplotypes`, one per
    ploidy index) and, if provided, the label with the sample's own haplotypes (e.g., its assembled paths).

    Only the comparisons needed are made, those of each candidate with each label haplotype (and of the label with the sample),
    and identical sequences are not compared. With `joint` (the default) all the sequences are harmonized together once,
    which is much faster than harmonizing each pair, but a pair's variants can then depend on the other sequences.
    """
    ploidy = len(label)
    sequences = {f"hap{i}": sequence for i, sequence in enumerate(haplotypes)}
    sample_names = None
    if sample_haplotypes is not None:
        sample_names = [f"sample{i}" for i in range(len(sample_haplotypes))]
        sequences.update(zip(sample_names, sample_haplotypes, strict=True))

    harmonized = None  # Harmonized on first use, so if every comparison is between identical sequences, none is needed
    pairs: dict[tuple[str, str], tuple[bool, float]] = {}

    def compare_pair(a: str, b: str) -> tuple[bool, float]:
        nonlocal harmonized
        key = (a, b) if a <= b else (b, a)
        if key not in pairs:
            if sequences[a] == sequences[b]:
                pairs[key] = (False, 1.0)
            else:
                if joint:
                    if harmonized is None:
                        harmonized = harmonize(reference, sequences, aligner=aligner, dedup=True)
                    result = compare_harmonized(harmonized, a, b, params)
                else:
                    result = compare_haplotypes(reference, sequences[a], sequences[b], params, aligner=aligner)
                pairs[key] = (result.differs, result.score)
        return pairs[key]

    def compare_genotype(genotype: Sequence[str], target: Sequence[str]) -> tuple[bool, float]:
        differs, score = True, 0.0
        for pairing in set(itertools.permutations(target)):
            results = [compare_pair(a, b) for a, b in zip(genotype, pairing, strict=True)]
            differs = differs and any(pair_differs for pair_differs, _ in results)
            score = max(score, min(pair_score for _, pair_score in results))
        return differs, score

    label_names = [f"hap{i}" for i in label]
    genotypes = list(itertools.combinations_with_replacement(range(len(haplotypes)), ploidy))
    results = [compare_genotype([f"hap{i}" for i in genotype], label_names) for genotype in genotypes]
    label_differs, label_score = compare_genotype(label_names, sample_names) if sample_names is not None else (None, None)
    return GenotypeComparison(
        np.array([differs for differs, _ in results], dtype=bool),
        np.array([score for _, score in results], dtype=float),
        label_differs,
        label_score,
    )


__all__ = [
    "Comparison",
    "ComparisonParams",
    "DEFAULT_PARAMS",
    "GenotypeComparison",
    "HaplotypeComparison",
    "HaplotypeMatrix",
    "MatchResult",
    "best_similarity",
    "compare_genotypes",
    "compare_haplotype_set",
    "compare_haplotypes",
    "compare_harmonized",
    "compare_variants",
    "edit_similarity",
    "haplotypes_differ",
    "match_variants",
    "variant_size",
]
