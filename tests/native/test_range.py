import pytest

from npsv3._native_graph import Range


class TestRangeRegionString:
    def test_basic_parsing(self):
        r = Range("chr1:1000-2000")
        assert r.contig == "chr1"
        assert r.start == 999   # 0-based
        assert r.end == 2000    # half-open

    def test_equivalent_to_explicit_constructor(self):
        assert Range("chr1:1000-2000") == Range("chr1", 999, 2000)

    def test_str_round_trip(self):
        region = "chr1:1000-2000"
        assert str(Range(region)) == region

    def test_single_base(self):
        r = Range("chr1:500-500")
        assert r.start == 499
        assert r.end == 500
        assert r.length == 1

    def test_contig_with_underscores_and_dots(self):
        r = Range("chr1_random:100-200")
        assert r.contig == "chr1_random"
        assert r.start == 99
        assert r.end == 200

    def test_invalid_reversed_range(self):
        with pytest.raises(Exception):
            Range("chr1:2000-1000")

    def test_invalid_string(self):
        with pytest.raises(Exception):
            Range("not:a:valid:region")
