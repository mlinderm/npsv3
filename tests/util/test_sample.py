import logging
import os

import pandas as pd
import pytest

from npsv3.util.sample import Sample, _estimate_coverage_from_histogram, _estimate_kmer_coverage

from .. import _first_existing, data_path


class TestSampleDataclass:
    def test_default_optional_fields(self):
        sample = Sample(
            name="TEST",
            sequencer="HS25",
            read_length=150,
            mean_coverage=30.0,
            mean_insert_size=450.0,
            std_insert_size=100.0,
        )
        assert sample.bam is None
        assert sample.sex == 0
        assert sample.chrom_normalized_coverage == {}
        assert sample.gc_normalized_coverage == {}
        assert sample.kmc_prefix is None
        assert sample.kmer_coverage is None

    def test_chrom_mean_coverage_unknown_defaults_to_mean(self):
        sample = Sample("TEST", "HS25", 150, 30.0, 450.0, 100.0)
        assert sample.chrom_mean_coverage("chrX") == pytest.approx(30.0)

    def test_chrom_mean_coverage_known_chrom(self):
        sample = Sample(
            "TEST", "HS25", 150, 30.0, 450.0, 100.0,
            chrom_normalized_coverage={"chr1": 1.1},
        )
        assert sample.chrom_mean_coverage("chr1") == pytest.approx(33.0)
        assert sample.chrom_mean_coverage("chr2") == pytest.approx(30.0)

    def test_gc_normalized_coverage_unknown_defaults_to_one(self):
        sample = Sample("TEST", "HS25", 150, 30.0, 450.0, 100.0)
        assert sample.gc_normalized_coverage.get(50, 1.0) == pytest.approx(1.0)

    def test_gc_normalized_coverage_known_gc_fraction(self):
        sample = Sample(
            "TEST", "HS25", 150, 30.0, 450.0, 100.0,
            gc_normalized_coverage={50: 0.95},
        )
        assert sample.gc_normalized_coverage.get(50, 1.0) == pytest.approx(0.95)
        assert sample.gc_normalized_coverage.get(40, 1.0) == pytest.approx(1.0)


class TestSampleFromJson:
    def test_from_json_required_fields(self):
        sample = Sample.from_json(data_path("stats.json"))
        assert sample.name == "HG002"
        assert sample.sequencer == "HS25"
        assert sample.read_length == 148
        assert sample.mean_coverage == pytest.approx(25.46)

    def test_from_json_chrom_coverage(self):
        sample = Sample.from_json(data_path("stats.json"))
        assert "1" in sample.chrom_normalized_coverage
        assert sample.chrom_mean_coverage("1") == pytest.approx(
            sample.mean_coverage * sample.chrom_normalized_coverage["1"]
        )

    def test_from_json_chrom_mean_coverage_default(self):
        sample = Sample.from_json(data_path("stats.json"))
        assert sample.chrom_mean_coverage("chrUnknown") == pytest.approx(sample.mean_coverage)

    def test_from_json_gc_coverage_keys_are_integers(self):
        sample = Sample.from_json(data_path("stats.json"), min_gc_bin=100, max_gc_error=0.01)
        assert all(isinstance(k, int) for k in sample.gc_normalized_coverage.keys())

    def test_from_json_gc_coverage_unknown_defaults_to_one(self):
        sample = Sample.from_json(data_path("stats.json"))
        assert sample.gc_normalized_coverage.get(999, 1.0) == pytest.approx(1.0)


class TestEstimateKmerCoverage:
    def test_unimodal_distribution(self):
        # mode count (30 with 80k distinct kmers) >= median count (30 with cumulative 150k distinct kmers out of 235k total) 
        histogram = pd.Series({28: 40_000, 29: 70_000, 30: 80_000, 31: 60_000, 32: 30_000, 33: 15_000})
        assert _estimate_coverage_from_histogram(histogram) == 30

    def test_bimodal_returns_secondary_peak(self):
        # mode count (15 with 50k distinct kmers) <= median count (16 with cumulative 90k distinct kmers out of 180k total), 
        # secondary peak at 30 with 45k distinct kmers
        histogram = pd.Series({15: 50_000, 16: 40_000, 29: 25_000, 30: 45_000, 31: 20_000})
        assert _estimate_coverage_from_histogram(histogram) == 30

    def test_example_histogram(self):
        kmc_hist = _first_existing(
            data_path("HG00096.final.kmc.hist"),
            "/data/HG00096.final.kmc.hist"
        )
        if kmc_hist is None:
            pytest.skip("Example KMC histogram not found")
            return
        histogram = pd.read_csv(kmc_hist, sep="\t", index_col=0, names=["count", "freq"])["freq"]
        assert _estimate_coverage_from_histogram(histogram) == 29

    def test_example_histogram_unable_to_estimate(self, caplog):
        kmc_hist = _first_existing(
            data_path("HG00733.final.kmc.hist")
        )
        if kmc_hist is None:
            pytest.skip("Example KMC histogram not found")
            return
        histogram = pd.read_csv(kmc_hist, sep="\t", index_col=0, names=["count", "freq"])["freq"]
        # This sample has a mode of 13, but no clear secondary peak according to Sirén et al.'s criteria
        with caplog.at_level(logging.WARNING):
            _estimate = _estimate_coverage_from_histogram(histogram)
            assert "Failed to estimate k-mer coverage, instead setting to median coverage." in caplog.text

    @pytest.mark.skip(reason="Can be slow to generate KMC histogram")
    def test_example_kmc_database(self):
        kmc_db = _first_existing(
            "/data/HG00096.final.kmc.kmc_pre"
        )
        if kmc_db is None:
            pytest.skip("Example KMC database not found")
            return
        assert _estimate_kmer_coverage(os.path.splitext(kmc_db)[0]) == 29

