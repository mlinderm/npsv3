"""Dual-run BWA-vs-k-mer agreement harness.

Implements the "Dual-run agreement harness" described in the "Cheaper allele evidence for image
generation" plan's Verification section: with `pileup.allele_evidence=both`, `CoverageImageGenerator`
computes both engines' per-fragment result for every fragment it renders. This module collects those
per-fragment results into rows (via `AgreementSink`, hooked into `ImageGenerator.generate`/`_generate` at
their single call site), then reports how the two engines compare -- confusion matrix, quality-delta
correlation, Bland-Altman data, stratified disagreement rates, and pre/post-resize pixel correlation
between the old realignment channels (6/7) and the new k-mer channel (8).

Do **not** try to infer agreement from rendered images: the pileup subsample and the resize both destroy
per-fragment traceability. This module always works from the per-fragment rows and the pre-resize raw
tensor instead.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pysam
from scipy import stats

from npsv3.pileup import AlleleAssignment, Fragment
from npsv3.util.range import Range

FRAGMENT_CLASS_SOFT_CLIP = "soft_clip"
FRAGMENT_CLASS_BREAKPOINT_SPANNING = "breakpoint_spanning"
FRAGMENT_CLASS_DISCORDANT_PAIR_ONLY = "discordant_pair_only"
FRAGMENT_CLASS_BOTH_MATES_INTERNAL = "both_mates_internal"

VARIANT_CLASS_INS = "INS"
VARIANT_CLASS_DEL = "DEL"
VARIANT_CLASS_MIXED = "MIXED"
VARIANT_CLASS_OTHER = "SNV_OR_OTHER"
VARIANT_CLASS_NONE = "NONE"

ASSIGNMENT_LABELS = ("REF", "ALT", "AMB")


def _threshold_assignment(ref_value: float, alt_value: float, delta: float) -> AlleleAssignment:
    diff = alt_value - ref_value
    if diff > delta:
        return AlleleAssignment.ALT
    if diff < -delta:
        return AlleleAssignment.REF
    return AlleleAssignment.AMB


def _read_has_soft_clip(read: pysam.AlignedSegment) -> bool:
    cigar = read.cigartuples
    return bool(cigar) and any(op == pysam.CSOFT_CLIP for op, _ in cigar)


def _read_has_indel_op(read: pysam.AlignedSegment) -> bool:
    cigar = read.cigartuples
    return bool(cigar) and any(op in (pysam.CDEL, pysam.CINS) for op, _ in cigar)


def _read_touches_variant(read: pysam.AlignedSegment, variant_range: Range) -> bool:
    return variant_range.get_overlap(read) > 0


def classify_fragment(fragment: Fragment, analysis_variants: Sequence) -> str:
    """Classify a fragment for the agreement harness's stratified-disagreement report.

    Priority order mirrors how directly the fragment's *own* sequence carries the discriminating signal:
    a read that soft-clips at a junction sees the most novel sequence; a read whose own alignment touches
    a variant (an indel CIGAR op, or overlapping an SNV) sees some; a fragment that is properly paired and
    whose only anomaly is an insert-size signal spanning a variant while both mates align cleanly carries
    none by construction (plan 1c) -- explicitly predicted to be the k-mer path's weakest class; anything
    else is "background", away from any variant.
    """
    reads = fragment.reads
    if any(_read_has_soft_clip(r) for r in reads):
        return FRAGMENT_CLASS_SOFT_CLIP

    variant_ranges = [Range(v.contig, v.start, v.end) for v in analysis_variants]
    if any(_read_has_indel_op(r) for r in reads) or any(
        _read_touches_variant(r, vr) for r in reads for vr in variant_ranges
    ):
        return FRAGMENT_CLASS_BREAKPOINT_SPANNING

    if (
        fragment.is_properly_paired
        and fragment.read1.reference_end < fragment.read2.reference_start
        and any(fragment.insert_region.get_overlap(vr) > 0 for vr in variant_ranges)
    ):
        return FRAGMENT_CLASS_DISCORDANT_PAIR_ONLY

    return FRAGMENT_CLASS_BOTH_MATES_INTERNAL


def _variant_class(variant) -> str:
    length_changes = [lc for lc in variant.length_change() if lc is not None]
    if not length_changes or all(lc == 0 for lc in length_changes):
        return VARIANT_CLASS_OTHER
    if all(lc > 0 for lc in length_changes):
        return VARIANT_CLASS_INS
    if all(lc < 0 for lc in length_changes):
        return VARIANT_CLASS_DEL
    return VARIANT_CLASS_MIXED


def nearest_variant_class(fragment: Fragment, analysis_variants: Sequence) -> str:
    """Class (INS/DEL/MIXED/SNV_OR_OTHER) of the analysis variant nearest this fragment's own span."""
    if not analysis_variants:
        return VARIANT_CLASS_NONE

    reads = fragment.reads
    span_start = min(r.reference_start for r in reads)
    span_end = max(r.reference_end for r in reads)
    center = (span_start + span_end) / 2.0

    best, best_distance = None, None
    for v in analysis_variants:
        distance = 0.0
        if center < v.start:
            distance = v.start - center
        elif center > v.end:
            distance = center - v.end
        if best_distance is None or distance < best_distance:
            best, best_distance = v, distance
    return _variant_class(best) if best is not None else VARIANT_CLASS_NONE


def _fragment_min_baseq(fragment: Fragment) -> float | None:
    quals = [q for r in fragment.reads for q in (r.query_qualities or [])]
    return float(min(quals)) if quals else None


class AgreementSink:
    """Collects one row per (fragment, engine) as `ImageGenerator._generate` computes them.

    Constructed once per image (in practice, once for the "real" image per region -- see
    `images/example.py:make_graph_example`'s `agreement_sink` parameter), passed to
    `ImageGenerator.generate(..., agreement_sink=sink)`, and read back via `.rows` afterward. Pass with
    `pileup.allele_evidence=both` so both engines actually compute a result per fragment; if only one
    engine ran, `record()` still works, it just emits one-sided rows.
    """

    def __init__(self, assign_delta: float = 1.0):
        self.assign_delta = assign_delta
        self.region_label: str = ""
        self.image_index: int = 0
        self.analysis_variants: Sequence = ()
        self.num_alleles: int = 0
        self.raw_image = None  # Set by make_graph_example when return_raw=True; all channels, pre-resize.
        self.rows: list[dict] = []

    def record(self, fragment: Fragment, bwa_allele, bwa_read1, bwa_read2, kmer_read1, kmer_read2):
        fragment_class = classify_fragment(fragment, self.analysis_variants)
        variant_class = nearest_variant_class(fragment, self.analysis_variants)
        min_baseq = _fragment_min_baseq(fragment)
        common = {
            "region": self.region_label,
            "image_index": self.image_index,
            "fragment_qname": fragment.query_name,
            "fragment_class": fragment_class,
            "variant_class": variant_class,
            "min_baseq": min_baseq,
            "num_alleles": self.num_alleles,
        }

        if bwa_allele is not None:
            self.rows.append({
                **common,
                "engine": "bwa",
                "ref_quality": bwa_allele.ref_quality,
                "alt_quality": bwa_allele.alt_quality,
                "assignment": bwa_allele.allele.name,
                "read_scores": (
                    bwa_read1.ref_quality if bwa_read1 is not None else None,
                    bwa_read1.alt_quality if bwa_read1 is not None else None,
                    bwa_read2.ref_quality if bwa_read2 is not None else None,
                    bwa_read2.alt_quality if bwa_read2 is not None else None,
                ),
                "multi_occurrence_hits": None,
            })

        if kmer_read1 is not None or kmer_read2 is not None:
            ref1, alt1, multi1 = kmer_read1 if kmer_read1 is not None else (np.zeros(0), np.zeros(0), 0)
            ref2, alt2, multi2 = kmer_read2 if kmer_read2 is not None else (np.zeros(0), np.zeros(0), 0)
            # Per-read peak vote height (bounded ~[0, k]), summed across the fragment's reads -- NOT a sum
            # over every base position, which would integrate the tent's *area* (inflated by roughly
            # k * window_count) rather than report a magnitude comparable to BWA's bounded [0,40] quality
            # scale. This mirrors what the pixel encoding already treats as the informative quantity: peak
            # per-base contrast, not total base-position mass.
            max_ref1 = float(np.max(ref1)) if ref1.size else 0.0
            max_ref2 = float(np.max(ref2)) if ref2.size else 0.0
            max_alt1 = float(np.max(alt1)) if alt1.size else 0.0
            max_alt2 = float(np.max(alt2)) if alt2.size else 0.0
            ref_total = max_ref1 + max_ref2
            alt_total = max_alt1 + max_alt2
            self.rows.append({
                **common,
                "engine": "kmer",
                "ref_quality": ref_total,
                "alt_quality": alt_total,
                "assignment": _threshold_assignment(ref_total, alt_total, self.assign_delta).name,
                "read_scores": (
                    max_ref1 if kmer_read1 is not None else None,
                    max_alt1 if kmer_read1 is not None else None,
                    max_ref2 if kmer_read2 is not None else None,
                    max_alt2 if kmer_read2 is not None else None,
                ),
                "multi_occurrence_hits": multi1 + multi2,
            })


def to_dataframe(rows: Sequence[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _pivot_by_fragment(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (region, image_index, fragment_qname) with <col>_bwa / <col>_kmer columns, keeping
    only fragments both engines actually scored -- an inner join, not a diff of rendered pixels."""
    key = ["region", "image_index", "fragment_qname"]
    bwa = df[df.engine == "bwa"].set_index(key)
    kmer = df[df.engine == "kmer"].set_index(key)
    return bwa.join(kmer, lsuffix="_bwa", rsuffix="_kmer", how="inner")


def confusion_matrix(df: pd.DataFrame, assign_delta: float) -> pd.DataFrame:
    """3x3 REF/ALT/AMB confusion matrix (rows=bwa, columns=kmer) recomputed at `assign_delta`."""
    joined = _pivot_by_fragment(df)
    matrix = pd.DataFrame(0, index=list(ASSIGNMENT_LABELS), columns=list(ASSIGNMENT_LABELS))
    for ref_b, alt_b, ref_k, alt_k in zip(
        joined["ref_quality_bwa"], joined["alt_quality_bwa"], joined["ref_quality_kmer"], joined["alt_quality_kmer"]
    ):
        b = _threshold_assignment(ref_b, alt_b, assign_delta).name
        k = _threshold_assignment(ref_k, alt_k, assign_delta).name
        matrix.loc[b, k] += 1
    return matrix


def quality_correlations(df: pd.DataFrame) -> dict:
    """Pearson/Spearman correlation of `alt_quality - ref_quality` between engines -- the quantity that
    becomes a pixel. Both engines' qualities are already clipped ([0,40] for bwa, raw vote counts with no
    natural ceiling for k-mer); this is a plain correlation over whatever range each engine reports, not a
    unit-matched comparison."""
    joined = _pivot_by_fragment(df)
    if len(joined) < 2:  # noqa: PLR2004
        return {"n": len(joined), "pearson_r": float("nan"), "pearson_p": float("nan"),
                "spearman_r": float("nan"), "spearman_p": float("nan")}
    bwa_delta = joined["alt_quality_bwa"] - joined["ref_quality_bwa"]
    kmer_delta = joined["alt_quality_kmer"] - joined["ref_quality_kmer"]
    pearson_r, pearson_p = stats.pearsonr(bwa_delta, kmer_delta)
    spearman_r, spearman_p = stats.spearmanr(bwa_delta, kmer_delta)
    return {"n": len(joined), "pearson_r": pearson_r, "pearson_p": pearson_p,
            "spearman_r": spearman_r, "spearman_p": spearman_p}


def bland_altman_data(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/diff pairs for a Bland-Altman plot of `alt_quality - ref_quality`, bwa vs. kmer. Exposes
    systematic compression toward 0 for fragments with no informative k-mer, which a single correlation
    coefficient can hide."""
    joined = _pivot_by_fragment(df)
    bwa_delta = joined["alt_quality_bwa"] - joined["ref_quality_bwa"]
    kmer_delta = joined["alt_quality_kmer"] - joined["ref_quality_kmer"]
    return pd.DataFrame({
        "mean": (bwa_delta + kmer_delta) / 2.0,
        "diff": kmer_delta - bwa_delta,
        "fragment_class": joined["fragment_class_bwa"],
        "variant_class": joined["variant_class_bwa"],
    })


_STRATIFICATION_COLUMNS = {
    "fragment_class": "fragment_class_bwa",
    "variant_class": "variant_class_bwa",
    "num_alleles": "num_alleles_bwa",
    "region": None,  # index level, handled specially below
}


def stratified_disagreement(df: pd.DataFrame, by: str, assign_delta: float = 1.0) -> pd.DataFrame:
    """Per-stratum REF/ALT/AMB agreement rate at `assign_delta`.

    `by` is one of "fragment_class", "variant_class", "num_alleles", "region", "min_baseq" (bucketed into
    quality bands), or "multi_occurrence" (whether the k-mer scan hit any locally multi-occurrent key --
    only ever true under `evidence_kmer_set=all`, see plan constraint 2).
    """
    joined = _pivot_by_fragment(df)
    bwa_assign = [
        _threshold_assignment(r, a, assign_delta).name
        for r, a in zip(joined["ref_quality_bwa"], joined["alt_quality_bwa"])
    ]
    kmer_assign = [
        _threshold_assignment(r, a, assign_delta).name
        for r, a in zip(joined["ref_quality_kmer"], joined["alt_quality_kmer"])
    ]
    joined = joined.assign(
        bwa_assign=bwa_assign, kmer_assign=kmer_assign,
        agree=[b == k for b, k in zip(bwa_assign, kmer_assign)],
    )

    if by == "min_baseq":
        bins = [0, 10, 20, 30, 41]
        labels = ["<10", "10-19", "20-29", "30+"]
        key = pd.cut(joined["min_baseq_bwa"], bins=bins, labels=labels, right=False, include_lowest=True)
    elif by == "multi_occurrence":
        key = (joined["multi_occurrence_hits_kmer"].fillna(0) > 0).map(
            {True: "has_multi_occurrence", False: "none"}
        )
    elif by == "region":
        key = joined.index.get_level_values("region")
    elif by in _STRATIFICATION_COLUMNS:
        key = joined[_STRATIFICATION_COLUMNS[by]]
    else:
        raise ValueError(f"Unknown stratification axis: {by!r}")

    return (
        joined.groupby(key, observed=True)["agree"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "agreement_rate", "count": "n"})
        .sort_values("n", ascending=False)
    )


def pixel_correlation(
    raw_image: np.ndarray, generator, widths: Sequence[int], bwa_channels: Sequence[int] = (6, 7),
    kmer_channel: int = 8,
) -> dict:
    """Correlate the BWA-derived channels against the new k-mer channel, pre-resize (full `region.length`
    columns) and post-resize at each candidate output width, using the generator's own resize/pooling
    logic (`ImageGenerator._resize`) so the comparison matches production rendering exactly.

    Only compares pixels the k-mer channel actually rendered an opinion on (nonzero) -- comparing against
    the 0 = "no information" background on both sides would inflate the correlation with a large block of
    trivially-agreeing zeros that carries no signal.
    """
    channel_idx = {ch: i for i, ch in enumerate(generator._image_channels)}
    if kmer_channel not in channel_idx or any(ch not in channel_idx for ch in bwa_channels):
        raise ValueError(
            f"generator._image_channels ({generator._image_channels}) must include both bwa_channels "
            f"{bwa_channels} and kmer_channel {kmer_channel} (e.g. pileup.allele_evidence=both with "
            f"image_channels=[...,6,7,8])"
        )

    def _corr(tensor: np.ndarray) -> dict:
        kmer = tensor[:, :, channel_idx[kmer_channel]].astype(np.float64).ravel()
        informative = kmer != 0
        out = {}
        for ch in bwa_channels:
            bwa = tensor[:, :, channel_idx[ch]].astype(np.float64).ravel()
            if informative.sum() >= 2:  # noqa: PLR2004
                out[ch] = float(np.corrcoef(bwa[informative], kmer[informative])[0, 1])
            else:
                out[ch] = float("nan")
        return out

    selected = raw_image[:, :, generator._image_channels]
    results = {"pre_resize": _corr(selected), "n_informative_pixels": int(np.count_nonzero(
        selected[:, :, channel_idx[kmer_channel]]
    ))}

    original_width = generator._image_width
    try:
        for width in widths:
            generator._image_width = width
            resized = generator._resize(selected)
            results[f"width_{width}"] = _corr(resized)
    finally:
        generator._image_width = original_width

    return results


DEFAULT_DELTA_SWEEP = (0.5, 1.0, 2.0)


def format_report(
    df: pd.DataFrame, pixel_results_by_region: dict | None = None, assign_delta: float = 1.0,
    delta_sweep: Sequence[float] = DEFAULT_DELTA_SWEEP,
) -> str:
    """Human-readable summary of everything above, in the order the plan's Verification section lists it."""
    lines = [f"Dual-run agreement report ({len(df)} rows, {df.engine.value_counts().to_dict()})", ""]

    lines.append(
        f"Confusion matrices (bwa rows x kmer columns) at assign_delta in {list(delta_sweep)} -- sweeping "
        "separates genuine disagreement from threshold-boundary noise (plan Verification):"
    )
    for delta in delta_sweep:
        lines.append(f"  delta={delta}:")
        lines.append("    " + confusion_matrix(df, delta).to_string().replace("\n", "\n    "))
    lines.append("")

    corr = quality_correlations(df)
    lines.append(
        f"alt_quality - ref_quality correlation (n={corr['n']}): "
        f"Pearson r={corr['pearson_r']:.3f} (p={corr['pearson_p']:.2e}), "
        f"Spearman r={corr['spearman_r']:.3f} (p={corr['spearman_p']:.2e})"
    )
    lines.append("")

    for by in ("fragment_class", "variant_class", "min_baseq", "num_alleles", "multi_occurrence", "region"):
        try:
            strat = stratified_disagreement(df, by, assign_delta)
        except (KeyError, ValueError) as e:
            lines.append(f"Stratified disagreement by {by}: skipped ({e})")
            continue
        lines.append(f"Stratified agreement rate by {by}:")
        lines.append(strat.to_string())
        lines.append("")

    if pixel_results_by_region:
        lines.append("Pixel correlation (channel 6/7 vs. channel 8), by region:")
        for region, results in pixel_results_by_region.items():
            lines.append(f"  {region} ({results['n_informative_pixels']} informative pixels pre-resize):")
            for scope, corrs in results.items():
                if scope == "n_informative_pixels":
                    continue
                lines.append(f"    {scope}: " + ", ".join(f"ch{ch}={r:.3f}" for ch, r in corrs.items()))
        lines.append("")

    return "\n".join(lines)
