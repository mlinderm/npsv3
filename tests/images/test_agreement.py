import numpy as np
import pysam
import pytest

from npsv3.images import agreement
from npsv3.pileup import AlleleRealignment, Fragment

HEADER = pysam.AlignmentHeader.from_dict(
    {
        "HD": {"VN": "1.5", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 249250621}],
    }
)

QUAL = "I" * 150  # A uniform, high-quality 150bp read


def _read(name, pos, cigar, *, reverse=False, paired=False, mate_pos=None):
    read = pysam.AlignedSegment.fromstring(
        f"{name}\t0\tchr1\t{pos}\t60\t{cigar}\t=\t{mate_pos or pos}\t0\t{'A' * 150}\t{QUAL}",
        header=HEADER,
    )
    read.is_reverse = reverse
    read.is_paired = paired
    return read


class _FakeVariant:
    """Minimal duck-typed stand-in for npsv3._native_graph.Variant (contig/start/end/length_change())."""

    def __init__(self, contig, start, end, length_changes):
        self.contig = contig
        self.start = start
        self.end = end
        self._length_changes = length_changes

    def length_change(self):
        return self._length_changes


class TestClassifyFragment:
    def test_soft_clip(self):
        read = _read("r1", 1000, "100M50S")
        fragment = Fragment(read)
        assert agreement.classify_fragment(fragment, []) == agreement.FRAGMENT_CLASS_SOFT_CLIP

    def test_breakpoint_spanning_via_indel_op(self):
        read = _read("r1", 1000, "70M10D80M")  # 70+80=150 query-consuming bases, matching the 150bp read
        fragment = Fragment(read)
        assert agreement.classify_fragment(fragment, []) == agreement.FRAGMENT_CLASS_BREAKPOINT_SPANNING

    def test_breakpoint_spanning_via_variant_overlap(self):
        read = _read("r1", 1000, "150M")  # covers [999, 1149) 0-indexed
        variant = _FakeVariant("chr1", 1050, 1051, [0])
        assert agreement.classify_fragment(fragment := Fragment(read), [variant]) == agreement.FRAGMENT_CLASS_BREAKPOINT_SPANNING

    def test_discordant_pair_only(self):
        # Reads far apart (a large gap) that straddle a variant without either read's own span touching it.
        read1 = _read("r1", 1000, "150M", paired=True, mate_pos=2000)
        read2 = _read("r1", 2000, "150M", reverse=True, paired=True, mate_pos=1000)
        fragment = Fragment(read1)
        fragment.add_read(read2)
        assert fragment.is_properly_paired
        variant = _FakeVariant("chr1", 1500, 1501, [-50])  # sits in the insert gap, not in either read
        assert agreement.classify_fragment(fragment, [variant]) == agreement.FRAGMENT_CLASS_DISCORDANT_PAIR_ONLY

    def test_both_mates_internal(self):
        read1 = _read("r1", 1000, "150M", paired=True, mate_pos=1300)
        read2 = _read("r1", 1300, "150M", reverse=True, paired=True, mate_pos=1000)
        fragment = Fragment(read1)
        fragment.add_read(read2)
        # No variants at all -- nothing for this fragment to be near.
        assert agreement.classify_fragment(fragment, []) == agreement.FRAGMENT_CLASS_BOTH_MATES_INTERNAL

    def test_soft_clip_takes_priority_over_variant_overlap(self):
        read = _read("r1", 1000, "100M50S")
        variant = _FakeVariant("chr1", 1050, 1051, [0])
        assert agreement.classify_fragment(Fragment(read), [variant]) == agreement.FRAGMENT_CLASS_SOFT_CLIP


class TestNearestVariantClass:
    def test_insertion(self):
        read = _read("r1", 1000, "150M")
        variant = _FakeVariant("chr1", 1050, 1051, [30])
        assert agreement.nearest_variant_class(Fragment(read), [variant]) == agreement.VARIANT_CLASS_INS

    def test_deletion(self):
        read = _read("r1", 1000, "150M")
        variant = _FakeVariant("chr1", 1050, 1051, [-30])
        assert agreement.nearest_variant_class(Fragment(read), [variant]) == agreement.VARIANT_CLASS_DEL

    def test_mixed_multiallelic(self):
        read = _read("r1", 1000, "150M")
        variant = _FakeVariant("chr1", 1050, 1051, [30, -30])
        assert agreement.nearest_variant_class(Fragment(read), [variant]) == agreement.VARIANT_CLASS_MIXED

    def test_snv_or_other(self):
        read = _read("r1", 1000, "150M")
        variant = _FakeVariant("chr1", 1050, 1051, [0])
        assert agreement.nearest_variant_class(Fragment(read), [variant]) == agreement.VARIANT_CLASS_OTHER

    def test_no_variants(self):
        read = _read("r1", 1000, "150M")
        assert agreement.nearest_variant_class(Fragment(read), []) == agreement.VARIANT_CLASS_NONE

    def test_picks_nearest_of_several(self):
        read = _read("r1", 1000, "150M")  # spans [999, 1149)
        near = _FakeVariant("chr1", 1100, 1101, [-30])
        far = _FakeVariant("chr1", 5000, 5001, [30])
        assert agreement.nearest_variant_class(Fragment(read), [far, near]) == agreement.VARIANT_CLASS_DEL


class TestAgreementSinkRecord:
    def test_kmer_reduction_uses_peak_height_not_area(self):
        """Regression test for the sum-vs-max bug: summing every base position of the cumulative vote
        array integrates the tent's area (inflated by ~k * window_count) rather than reporting a per-read
        magnitude comparable to BWA's bounded [0,40] quality scale. The reduction must take the *peak*
        height reached, not the sum over positions."""
        sink = agreement.AgreementSink(assign_delta=1.0)
        read = _read("r1", 1000, "150M")
        fragment = Fragment(read)

        # A tent that ramps up to a peak of 21 (k=21) over many positions and back down -- summing every
        # position would give a huge number; taking the max should give exactly 21.
        ref_votes = np.concatenate([np.arange(0, 21), np.full(50, 21), np.arange(21, 0, -1)]).astype(np.float32)
        alt_votes = np.zeros_like(ref_votes)
        sink.record(fragment, None, None, None, (ref_votes, alt_votes, 0), None)

        row = sink.rows[0]
        assert row["engine"] == "kmer"
        assert row["ref_quality"] == pytest.approx(21.0)
        assert row["alt_quality"] == pytest.approx(0.0)
        assert row["assignment"] == "REF"

    def test_bwa_and_kmer_rows_share_common_fields(self):
        sink = agreement.AgreementSink(assign_delta=1.0)
        sink.region_label = "chr1:1-100"
        sink.image_index = 0
        sink.num_alleles = 2
        read = _read("r1", 1000, "150M")
        fragment = Fragment(read)

        bwa_allele = AlleleRealignment(5, 15, agreement.AlleleAssignment.ALT)
        bwa_read1 = AlleleRealignment(5, 15, agreement.AlleleAssignment.ALT)

        sink.record(fragment, bwa_allele, bwa_read1, None, None, None)
        row = sink.rows[0]
        assert row["region"] == "chr1:1-100"
        assert row["fragment_qname"] == "r1"
        assert row["engine"] == "bwa"
        assert row["assignment"] == "ALT"
        assert row["multi_occurrence_hits"] is None

    def test_no_row_emitted_when_engine_did_not_run(self):
        sink = agreement.AgreementSink()
        read = _read("r1", 1000, "150M")
        fragment = Fragment(read)
        sink.record(fragment, None, None, None, None, None)
        assert sink.rows == []


class TestAnalysisFunctions:
    @pytest.fixture
    def sample_df(self):
        rows = [
            # Fragment 1: both engines agree ALT
            {"region": "r", "image_index": 0, "fragment_qname": "f1", "engine": "bwa",
             "ref_quality": 0.0, "alt_quality": 30.0, "assignment": "ALT", "read_scores": None,
             "fragment_class": "soft_clip", "variant_class": "DEL", "min_baseq": 30.0, "num_alleles": 2,
             "multi_occurrence_hits": None},
            {"region": "r", "image_index": 0, "fragment_qname": "f1", "engine": "kmer",
             "ref_quality": 0.0, "alt_quality": 21.0, "assignment": "ALT", "read_scores": None,
             "fragment_class": "soft_clip", "variant_class": "DEL", "min_baseq": 30.0, "num_alleles": 2,
             "multi_occurrence_hits": 0},
            # Fragment 2: bwa says REF, kmer says AMB (disagreement)
            {"region": "r", "image_index": 0, "fragment_qname": "f2", "engine": "bwa",
             "ref_quality": 30.0, "alt_quality": 0.0, "assignment": "REF", "read_scores": None,
             "fragment_class": "both_mates_internal", "variant_class": "INS", "min_baseq": 15.0, "num_alleles": 2,
             "multi_occurrence_hits": None},
            {"region": "r", "image_index": 0, "fragment_qname": "f2", "engine": "kmer",
             "ref_quality": 0.5, "alt_quality": 0.0, "assignment": "AMB", "read_scores": None,
             "fragment_class": "both_mates_internal", "variant_class": "INS", "min_baseq": 15.0, "num_alleles": 2,
             "multi_occurrence_hits": 0},
        ]
        return agreement.to_dataframe(rows)

    def test_confusion_matrix(self, sample_df):
        matrix = agreement.confusion_matrix(sample_df, assign_delta=1.0)
        assert matrix.loc["ALT", "ALT"] == 1
        assert matrix.loc["REF", "AMB"] == 1
        assert matrix.values.sum() == 2

    def test_stratified_disagreement_by_fragment_class(self, sample_df):
        strat = agreement.stratified_disagreement(sample_df, "fragment_class", assign_delta=1.0)
        assert strat.loc["soft_clip", "agreement_rate"] == 1.0
        assert strat.loc["both_mates_internal", "agreement_rate"] == 0.0

    def test_stratified_disagreement_by_multi_occurrence(self, sample_df):
        strat = agreement.stratified_disagreement(sample_df, "multi_occurrence", assign_delta=1.0)
        assert "none" in strat.index
        assert strat.loc["none", "n"] == 2

    def test_quality_correlations_runs(self, sample_df):
        result = agreement.quality_correlations(sample_df)
        assert result["n"] == 2

    def test_bland_altman_data(self, sample_df):
        bland_altman = agreement.bland_altman_data(sample_df)
        assert len(bland_altman) == 2
        assert {"mean", "diff", "fragment_class", "variant_class"} <= set(bland_altman.columns)

    def test_format_report_runs(self, sample_df):
        report = agreement.format_report(sample_df, assign_delta=1.0)
        assert "Confusion matrices" in report
        assert "Stratified agreement rate by fragment_class" in report
