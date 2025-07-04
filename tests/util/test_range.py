from npsv3.util.range import Range


class TestRange:
    def test_overlap(self):
        assert Range("chr1", 100, 200).overlaps(Range("chr1", 150, 250))
        assert not Range("chr1", 100, 200).overlaps(Range("chr1", 200, 250))
        assert not Range("chr1", 100, 200).overlaps(Range("chr2", 150, 250))
        assert Range("chr1", 100, 200).overlaps(Range("chr1", 50, 150))
        assert not Range("chr1", 100, 200).overlaps(Range("chr1", 50, 100))

    def test_insertion_overlap(self):
        assert Range("chr1", 100, 100).overlaps(Range("chr1", 100, 100)), "Matching in-between ranges should overlap"
        assert Range("chr1", 100, 100).overlaps(Range("chr1", 99, 101)), "Contained in-between ranges should overlap"
        assert not Range("chr1", 100, 100).overlaps(Range("chr1", 100, 101))
        assert not Range("chr1", 100, 100).overlaps(Range("chr1", 99, 100))
