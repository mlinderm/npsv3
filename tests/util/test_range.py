import pytest

from npsv3.util.range import Range


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

    def test_contig_with_underscores(self):
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


class TestRangeOverlap:
    def test_overlap(self):
        assert Range("chr1", 100, 200).overlaps(Range("chr1", 150, 250))
        assert not Range("chr1", 100, 200).overlaps(Range("chr1", 200, 250))
        assert not Range("chr1", 100, 200).overlaps(Range("chr2", 150, 250))
        assert Range("chr1", 100, 200).overlaps(Range("chr1", 50, 150))
        assert not Range("chr1", 100, 200).overlaps(Range("chr1", 50, 100))

    def test_insertion_overlap(self):
        assert Range("chr1", 100, 100).overlaps(Range("chr1", 99, 101)), "Contained in-between ranges should overlap"
        assert not Range("chr1", 100, 100).overlaps(Range("chr1", 100, 101))
        assert not Range("chr1", 100, 100).overlaps(Range("chr1", 99, 100))

    # TODO: Revisit the overlap of zero-length ranges


class TestRangeSlug:
    def test_slug_round_trip(self):
        r = Range("chr1", 100, 200)
        assert r.slug == "chr1_100_200"
        assert Range.parse_slug(r.slug) == r

    def test_parse_slug_with_contig_underscores(self):
        r = Range("chr1_random", 100, 200)
        slug = r.slug
        assert slug == "chr1_random_100_200"
        assert Range.parse_slug(slug) == r

    def test_parse_slug_invalid(self):
        with pytest.raises(ValueError, match="Invalid Range slug"):
            Range.parse_slug("chr1_100")

class TestRangePysamFetch:
    def test_pysam_fetch(self):
        r = Range("chr1", 100, 200)
        assert r.pysam_fetch == {"contig": "chr1", "start": 100, "stop": 200}


class TestRangeExpand:
    def test_expand_symmetric(self):
        r = Range("chr1", 100, 200)
        expanded = r.expand(50)
        assert expanded == Range("chr1", 50, 250)

    def test_expand_asymmetric(self):
        r = Range("chr1", 100, 200)
        expanded = r.expand(50, 75)
        assert expanded == Range("chr1", 50, 275)

    def test_expand_clamps_to_zero(self):
        r = Range("chr1", 10, 20)
        expanded = r.expand(15, 5)
        assert expanded == Range("chr1", 0, 25)


class TestRangeUnionIntersection:
    def test_union(self):
        a = Range("chr1", 100, 200)
        b = Range("chr1", 50, 150)
        assert a.union(b) == Range("chr1", 50, 200)
        # Non-mutating: `a` should be unchanged
        assert a == Range("chr1", 100, 200)

    def test_union_with_mutates(self):
        a = Range("chr1", 100, 200)
        a.union_with(Range("chr1", 50, 150))
        assert a == Range("chr1", 50, 200)

    def test_union_different_contigs_raises(self):
        with pytest.raises(Exception):
            Range("chr1", 0, 10).union(Range("chr2", 0, 10))

    def test_intersection(self):
        a = Range("chr1", 100, 200)
        b = Range("chr1", 150, 250)
        assert a.intersection(b) == Range("chr1", 150, 200)

    def test_intersection_no_overlap(self):
        a = Range("chr1", 100, 200)
        b = Range("chr1", 200, 300)
        result = a.intersection(b)
        assert result.length == 0

    def test_intersection_different_contigs(self):
        a = Range("chr1", 100, 200)
        b = Range("chr2", 100, 200)
        result = a.intersection(b)
        assert result.length == 0


class TestRangeContainsCenterWindow:
    def test_contains(self):
        r = Range("chr1", 100, 200)
        assert r.contains(100)
        assert r.contains(199)
        assert not r.contains(200)
        assert not r.contains(99)

    def test_center(self):
        r = Range("chr1", 100, 200)
        center = r.center
        assert center.contig == "chr1"
        assert center.start == center.end == 150
        assert center.length == 0

    def test_window(self):
        r = Range("chr1", 100, 200)
        windows = r.window(50)
        assert windows == [Range("chr1", 100, 150), Range("chr1", 150, 200)]

    def test_window_not_multiple_raises(self):
        r = Range("chr1", 100, 175)
        with pytest.raises(Exception):
            r.window(50)


class TestRangeGetOverlap:
    def test_get_overlap_range(self):
        a = Range("chr1", 100, 200)
        b = Range("chr1", 150, 250)
        assert a.get_overlap(b) == 50

    def test_get_overlap_range_no_overlap(self):
        a = Range("chr1", 100, 200)
        b = Range("chr1", 200, 300)
        assert a.get_overlap(b) == 0

    def test_get_overlap_range_different_contig(self):
        a = Range("chr1", 100, 200)
        b = Range("chr2", 100, 200)
        assert a.get_overlap(b) == 0

    def test_get_overlap_duck_typed_read(self):
        class FakeRead:
            reference_name = "chr1"

            def get_overlap(self, start, end):
                assert (start, end) == (100, 200)
                return 42

        r = Range("chr1", 100, 200)
        assert r.get_overlap(FakeRead()) == 42

    def test_get_overlap_duck_typed_read_different_contig(self):
        class FakeRead:
            reference_name = "chr2"

            def get_overlap(self, start, end):
                raise AssertionError("Should not be called for a different contig")

        r = Range("chr1", 100, 200)
        assert r.get_overlap(FakeRead()) == 0

    def test_get_overlap_duck_typed_read_unmapped(self):
        class FakeRead:
            reference_name = None

        r = Range("chr1", 100, 200)
        assert r.get_overlap(FakeRead()) == 0


class TestRangeComparisons:
    def test_eq(self):
        assert Range("chr1", 100, 200) == Range("chr1", 100, 200)
        assert Range("chr1", 100, 200) != Range("chr1", 100, 201)
        assert Range("chr1", 100, 200) != Range("chr2", 100, 200)

    def test_le_containment(self):
        outer = Range("chr1", 0, 100)
        inner = Range("chr1", 20, 80)
        assert inner <= outer
        assert not outer <= inner
        assert inner <= inner # noqa: PLR0124

    def test_lt_strict_containment(self):
        outer = Range("chr1", 0, 100)
        inner = Range("chr1", 20, 80)
        assert inner < outer
        assert not outer < inner
        assert not inner < inner # noqa: PLR0124

    def test_hash_consistent_with_eq(self):
        a = Range("chr1", 100, 200)
        b = Range("chr1", 100, 200)
        assert a == b
        assert hash(a) == hash(b)
