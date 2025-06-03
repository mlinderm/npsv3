import os
from collections import deque

import pysam
import pytest

from npsv3.graphs.graph_constructor import (
    AltPath,
    GraphConstructor,
    ReferenceSpan,
    add_haplotypes_to_gfa,
    variant_path_name,
    vcf_to_paths,
)
from npsv3.util.range import Range
from npsv3.util.vcf import index_variant_file

from .. import B37_REF_FASTA, HG00731_VCF, HG38_REF_FASTA, data_path


class TestGraphConstructor:
    def test_find_spans(self):
        region = Range("chr1", 0, 100)
        construct = GraphConstructor(
            region, data_path("empty.vcf.gz")
        )
        assert construct.num_spans == 1
        assert construct.get_span_region(0) == region

        # SNV in a single span
        start_idx, end_idx = construct.find_overlapping_spans(Range("chr1", 50, 51))
        assert start_idx == 0
        assert end_idx == 0

        # INS in a single span
        start_idx, end_idx = construct.find_overlapping_spans(Range("chr1", 50, 50))
        assert start_idx == 0
        assert end_idx == 0

        # INS matching existing span
        construct.spans = deque([
            ReferenceSpan(Range("chr1", 0, 50)),
            ReferenceSpan(Range("chr1", 50, 50)),
            ReferenceSpan(Range("chr1", 50, 100)),
        ])
        start_idx, end_idx = construct.find_overlapping_spans(Range("chr1", 50, 50))
        assert start_idx == 1, "Should match insertion span"
        assert end_idx == 1, "Should match insertion span"

        start_idx, end_idx = construct.find_overlapping_spans(Range("chr1", 50, 51))
        assert start_idx == 2, "Should match insertion span"
        assert end_idx == 2, "Should match after insertion span"

        # SNV in a series
        construct.spans = deque([
            ReferenceSpan(Range("chr1", 0, 50)),
            ReferenceSpan(Range("chr1", 50, 51)),
            ReferenceSpan(Range("chr1", 51, 52)),
            ReferenceSpan(Range("chr1", 52, 100)),
        ])
        for start, exp_idx in [(49, 0), (50, 1), (51, 2), (52, 3)]:
            start_idx, end_idx = construct.find_overlapping_spans(Range("chr1", start, start+1))
            assert start_idx == exp_idx
            assert end_idx == exp_idx

        # INS after an SNV
        construct.spans = deque([
            ReferenceSpan(Range("chr1", 0, 50)),
            ReferenceSpan(Range("chr1", 50, 51)),
            ReferenceSpan(Range("chr1", 51, 100)),
        ])
        start_idx, end_idx = construct.find_overlapping_spans(Range("chr1", 51, 51))
        assert start_idx == 2, "Should match after SNV"
        assert end_idx == 2, "Should match after SNV"


    def test_target_span(self):
        construct = GraphConstructor(Range("chr1", 0, 100), data_path("empty.vcf.gz"))

        # Positive interval
        construct.spans = deque([
            ReferenceSpan(Range("chr1", 0, 50)),
            ReferenceSpan(Range("chr1", 50, 51)),
            ReferenceSpan(Range("chr1", 51, 100)),
        ])
        assert construct.find_target_span(50) == 1
        assert construct.find_target_span(51) == 2

        # Null interval (i.e., INS)
        construct.spans = deque([
            ReferenceSpan(Range("chr1", 0, 50)),
            ReferenceSpan(Range("chr1", 50, 50)),
            ReferenceSpan(Range("chr1", 50, 100)),
        ])
        assert construct.find_target_span(50) == 1

    def test_span_splitting_positive(self):
        construct = GraphConstructor(Range("chr1", 0, 100), data_path("empty.vcf.gz"))

        variant_id = "c73c0e2d845e9b0e3c350f6c161a10b84252c108"
        alt = AltPath(variant_path_name(variant_id, 1), 51, "C")
        construct._split_spans(Range("chr1", 50, 51), variant_id, alt)
        assert [span.region for span in construct.spans] == [
            Range("chr1", 0, 50),
            Range("chr1", 50, 51),
            Range("chr1", 51, 100),
        ]
        assert construct.spans[1].names == { "chr1", variant_path_name(variant_id, 0)}
        assert [len(span.alts) for span in construct.spans] == [0, 1, 0]
        assert construct.spans[1].alts[0] == alt

    def test_span_splitting_null(self):
        construct = GraphConstructor(Range("chr1", 0, 100), data_path("empty.vcf.gz"))

        variant_id = "c73c0e2d845e9b0e3c350f6c161a10b84252c108"
        alt = AltPath(variant_path_name(variant_id, 1), 50, "CAA")
        construct._split_spans(Range("chr1", 50, 50), variant_id, alt)
        assert [span.region for span in construct.spans] == [
            Range("chr1", 0, 50),
            Range("chr1", 50, 50),
            Range("chr1", 50, 100),
        ]

    def test_span_splitting_at_start(self):
        construct = GraphConstructor(Range("chr1", 0, 100), data_path("empty.vcf.gz"))

        del_variant_id = "a"*40
        alt = AltPath(variant_path_name(del_variant_id, 1), 60, "")
        construct._split_spans(Range("chr1", 50, 60), del_variant_id, alt)

        snv_variant_id = "b"*40
        alt = AltPath(variant_path_name(snv_variant_id, 1), 51, "C")
        construct._split_spans(Range("chr1", 50, 51), snv_variant_id, alt)

        assert [span.region for span in construct.spans] == [
            Range("chr1", 0, 50),
            Range("chr1", 50, 51),
            Range("chr1", 51, 60),
            Range("chr1", 60, 100),
        ]
        assert construct.spans[1].names == { "chr1", variant_path_name(del_variant_id, 0), variant_path_name(snv_variant_id, 0)}
        assert len(construct.spans[1].alts) == 2

    # TODO: Test splitting across multiple spans and at span boundaries

    @pytest.mark.skipif(not os.path.exists(HG38_REF_FASTA), reason="HG38 reference required")
    @pytest.mark.parametrize(("region", "vcf_file"), [
        (Range("chr1", 8977700, 8977700), "chr1_8976700_8978700.vcf.gz"),
        (Range("chr1", 41824764, 41824818), "chr1_41823764_41825818.vcf.gz"),
    ])
    def test_graph_construction(self, tmp_path, cfg, region, vcf_file):
        region = region.expand(cfg.pileup.graph_flank)
        construct = GraphConstructor(
            region, data_path(vcf_file)
        )

        assert sum(len(span.region) for span in construct.spans) == len(region)
        for i, span in enumerate(construct.spans[1:]):
            assert span.region.start == construct.spans[i].region.end

        gfa_path = os.path.join(tmp_path, "test.gfa")
        construct.to_gfa(HG38_REF_FASTA, gfa_path)

        haplotype_paths = filter(lambda name: name.startswith("HG00731#"), construct.paths)
        assert set(haplotype_paths) == {
            f"HG00731#0#{region.contig}#0",
            f"HG00731#1#{region.contig}#0"
        }, "Expected two haplotypes for this sample"

    def test_colocated_snv_del(self, cfg):
        region = Range("chr1", 6012136, 6012573)
        construct = GraphConstructor(
            region.expand(cfg.pileup.graph_flank), data_path("chr1_6011136_6013135.vcf.gz")
        )
        #construct.to_gfa(HG38_REF_FASTA)

        colocated_span = construct.spans[35]
        assert colocated_span.region == Range("chr1", 6012521, 6012522)
        assert len(colocated_span.alts) == 2

    @pytest.mark.skipif(not os.path.exists(HG38_REF_FASTA), reason="HG38 reference required")
    @pytest.mark.skipif(not HG00731_VCF, reason="HG00731 VCF required")
    def test_gfa_generation(self, tmp_path, cfg):
        region = Range("chr1", 5246615, 5246691)
        construct = GraphConstructor(
            region.expand(cfg.pileup.graph_flank),
            HG00731_VCF,
        )

        # Generate complete GFA without error
        gfa_path = os.path.join(tmp_path, "test.gfa")
        construct.to_gfa(HG38_REF_FASTA, gfa_path)
        #add_haplotypes_to_gfa(gfa_path, HG00731_VCF, region.expand(cfg.pileup.graph_flank))


    @pytest.mark.skipif(not os.path.exists(B37_REF_FASTA), reason="B37 reference required")
    @pytest.mark.parametrize(("variant", "addl_paths"), [
    ("1	1000001	.	G	C	100	PASS	.	GT	0/1", { "Sample#0#1#1": [4, 6], "Sample#1#1#1": [5, 6] }),
    ("1	1000001	.	G	C	100	PASS	.	GT	0|1", { "Sample#0#1#1": [4, 6], "Sample#1#1#1": [5, 6] }),
    ("1	1000001	.	G	C	100	PASS	.	GT:PS	0|1:1000001", { "Sample#0#1#1": [4, 6], "Sample#1#1#1": [5, 6] }),
    ("1	1000001	.	G	C	100	PASS	.	GT	1/1", { "Sample#0#1#0": [1, 2, 5, 6], "Sample#1#1#0": [1, 3, 5, 6] }),
    ])
    def test_unphased_transitions(self, tmp_path, variant, addl_paths):
        vcf_path = os.path.join(tmp_path, "test.vcf.gz")
        with pysam.BGZFile(vcf_path, "wb") as vcf_file:
            vcf_file.write(
                f"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=1,length=249250621>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set identifier">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample
1	1000000	.	T	A	100	PASS	.	GT	0/1
{variant}
""".encode()
            )
        index_variant_file(vcf_path)

        region = Range.parse_literal ("1:1000000-1000001").expand(10)
        construct = GraphConstructor(region, vcf_path)
        #construct.to_gfa(B37_REF_FASTA)

        expected_paths = {
            "Sample#0#1#0": [1, 2],
            "Sample#1#1#0": [1, 3],
            **addl_paths
        }
        for name, nodes in expected_paths.items():
            assert construct.paths.get(name) == nodes, f"Path {name} does not match expected nodes"

    @pytest.mark.skipif(not os.path.exists(B37_REF_FASTA), reason="B37 reference required")
    @pytest.mark.parametrize(("variant", "addl_paths"), [
    ("1	1000001	.	G	C	100	PASS	.	GT	0/1", { "Sample#0#1#1": [4, 6], "Sample#1#1#1": [5, 6] }),
    ("1	1000001	.	G	C	100	PASS	.	GT	0|1", { "Sample#0#1#0": [1, 2, 4, 6], "Sample#1#1#0": [1, 3, 5, 6] }),
    ("1	1000001	.	G	C	100	PASS	.	GT:PS	0|1:1000001", { "Sample#0#1#1": [4, 6], "Sample#1#1#1": [5, 6] }),
    ("1	1000001	.	G	C	100	PASS	.	GT	1/1", { "Sample#0#1#0": [1, 2, 5, 6], "Sample#1#1#0": [1, 3, 5, 6] }),
    ])
    def test_global_transitions(self, tmp_path, variant, addl_paths):
        vcf_path = os.path.join(tmp_path, "test.vcf.gz")
        with pysam.BGZFile(vcf_path, "wb") as vcf_file:
            vcf_file.write(
                f"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=1,length=249250621>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set identifier">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample
1	1000000	.	T	A	100	PASS	.	GT	0|1
{variant}
""".encode()
            )
        index_variant_file(vcf_path)

        region = Range.parse_literal ("1:1000000-1000001").expand(10)
        construct = GraphConstructor(region, vcf_path)
        #construct.to_gfa(B37_REF_FASTA)

        expected_paths = {
            "Sample#0#1#0": [1, 2],
            "Sample#1#1#0": [1, 3],
            **addl_paths
        }
        for name, nodes in expected_paths.items():
            assert construct.paths.get(name) == nodes, f"Path {name} does not match expected nodes"

    @pytest.mark.skipif(not os.path.exists(B37_REF_FASTA), reason="B37 reference required")
    @pytest.mark.parametrize(("variant", "addl_paths"), [
    ("1	1000001	.	G	C	100	PASS	.	GT	0/1", { "Sample#0#1#1": [4, 6], "Sample#1#1#1": [5, 6] }),
    ("1	1000001	.	G	C	100	PASS	.	GT	0|1", { "Sample#0#1#1": [4, 6], "Sample#1#1#1": [5, 6] }),
    ("1	1000001	.	G	C	100	PASS	.	GT:PS	0|1:1000000", { "Sample#0#1#0": [1, 2, 4, 6], "Sample#1#1#0": [1, 3, 5, 6] }),
    ("1	1000001	.	G	C	100	PASS	.	GT:PS	0|1:1000001", { "Sample#0#1#1": [4, 6], "Sample#1#1#1": [5, 6] }),
    ("1	1000001	.	G	C	100	PASS	.	GT	1/1", { "Sample#0#1#0": [1, 2, 5, 6], "Sample#1#1#0": [1, 3, 5, 6] }),
    ])
    def test_local_transitions(self, tmp_path, variant, addl_paths):
        vcf_path = os.path.join(tmp_path, "test.vcf.gz")
        with pysam.BGZFile(vcf_path, "wb") as vcf_file:
            vcf_file.write(
                f"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=1,length=249250621>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set identifier">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample
1	1000000	.	T	A	100	PASS	.	GT:PS	0|1:1000000
{variant}
""".encode()
            )
        index_variant_file(vcf_path)

        region = Range.parse_literal ("1:1000000-1000001").expand(10)
        construct = GraphConstructor(region, vcf_path)
        #construct.to_gfa(B37_REF_FASTA)

        expected_paths = {
            "Sample#0#1#0": [1, 2],
            "Sample#1#1#0": [1, 3],
            **addl_paths
        }
        for name, nodes in expected_paths.items():
            assert construct.paths.get(name) == nodes, f"Path {name} does not match expected nodes"

    @pytest.mark.skipif(not os.path.exists(B37_REF_FASTA), reason="B37 reference required")
    def test_star_alleles(self, tmp_path):
        vcf_path = os.path.join(tmp_path, "test.vcf.gz")
        with pysam.BGZFile(vcf_path, "wb") as vcf_file:
            vcf_file.write(
                b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=14,length=107349540>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample
14	77187581	.	GCC	G	603.88	PASS	.	GT	0/1
14	77187582	.	C	CAAAAAAAAAA,*	344.04	PASS	.	GT	1/2
"""
            )
        index_variant_file(vcf_path)

        region = Range("14", 77187572, 77187592)

        construct = GraphConstructor(region, vcf_path)
        #construct.to_gfa(B37_REF_FASTA)

        assert construct.paths.get(f"Sample#0#{region.contig}#0") == [1,2,5,6,7]
        assert construct.paths.get(f"Sample#1#{region.contig}#0") == [1,3,7]

    @pytest.mark.skipif(not os.path.exists(HG38_REF_FASTA), reason="HG38 reference required")
    def test_implicit_overlap(self, tmp_path):
        vcf_path = os.path.join(tmp_path, "test.vcf.gz")
        with pysam.BGZFile(vcf_path, "wb") as vcf_file:
            vcf_file.write(
                b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set identifier">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample
chr1	8978661	.	AAAAAAAAAAAAAAC	A	.	PASS	.	GT	0|1
chr1	8978664	.	A	C	.	PASS	.	GT	0|0
"""
            )
        index_variant_file(vcf_path)

        region = Range.parse_literal ("chr1:8978661-8978675").expand(10)
        construct = GraphConstructor(region, vcf_path)
        #construct.to_gfa(HG38_REF_FASTA)

        expected_paths = {
            "Sample#0#chr1#0": [1,2,4,6,7],
            "Sample#1#chr1#0": [1,3,7],
        }
        for name, nodes in expected_paths.items():
            assert construct.paths.get(name) == nodes, f"Path {name} does not match expected nodes"

    @pytest.mark.skipif(not os.path.exists(HG38_REF_FASTA), reason="HG38 reference required")
    def test_implicit_ins_overlap(self, tmp_path):
        vcf_path = os.path.join(tmp_path, "test.vcf.gz")
        with pysam.BGZFile(vcf_path, "wb") as vcf_file:
            vcf_file.write(
                b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set identifier">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample
chr1	5246237	.	C	T	.	.	.	GT	0|0
chr1	5246237	.	C	CCTTCCTCTTCCTCCCTTCCTTCCTTCCTTT	.	.	.	GT	1|0
chr1	5246237	.	C	CCTTT	.	.	.	GT	0|1
"""
            )
        index_variant_file(vcf_path)

        region = Range.parse_literal ("chr1:5246237-5246237").expand(10)
        construct = GraphConstructor(region, vcf_path)
        #construct.to_gfa(HG38_REF_FASTA)

        expected_paths = {
            "Sample#0#chr1#0": [1,2,5,7],
            "Sample#1#chr1#0": [1,2,6,7],
        }
        for name, nodes in expected_paths.items():
            assert construct.paths.get(name) == nodes, f"Path {name} does not match expected nodes"


    @pytest.mark.skipif(not os.path.exists(B37_REF_FASTA), reason="B37 reference required")
    def test_mixed_alleles(self, tmp_path):
        vcf_path = os.path.join(tmp_path, "test.vcf.gz")
        with pysam.BGZFile(vcf_path, "wb") as vcf_file:
            vcf_file.write(
                b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=1,length=249250621>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set identifier">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample
1	1000000	.	T	A	100	PASS	.	GT	0/1
1	1000001	.	G	C	100	PASS	.	GT:PS	0|1:1000000
1	1000002	.	G	T	100	PASS	.	GT:PS	1|1:1000000
1	1000003	.	G	A	100	PASS	.	GT:PS	1|0:1000003
1	1000004	.	C	T	100	PASS	.	GT	0/0
1	1000005	.	A	G	100	PASS	.	GT	1/1
1	1000006	.	C	G	100	PASS	.	GT	0|1
"""
            )
        index_variant_file(vcf_path)

        region = Range.parse_literal ("1:1000000-1000005").expand(10)
        construct = GraphConstructor(region, vcf_path)
        #construct.to_gfa(B37_REF_FASTA)

        expected_paths = {
            "Sample#0#1#0": [1, 2],
            "Sample#1#1#0": [1, 3],
            "Sample#0#1#1": [4, 7],
            "Sample#1#1#1": [5, 7],
            "Sample#0#1#2": [9, 10, 13],
            "Sample#1#1#2": [8, 10, 13],
            "Sample#0#1#3": [14, 16],
            "Sample#1#1#3": [15, 16],
        }
        for name, nodes in expected_paths.items():
            assert construct.paths.get(name) == nodes, f"Path {name} does not match expected nodes"

        # # Use vg to test path generation
        # # Note: vg does not seem to take into account the PS tag
        # gfa_path = os.path.join(tmp_path, "test.gfa")
        # construct.to_gfa(B37_REF_FASTA, gfa_path)
        # with open(gfa_path, "r") as gfa_file:
        #     for name, _strand, nodes in vcf_to_paths(gfa_path, vcf_path, region):
        #         print(name, nodes)
        #         #assert construct.paths.get(name) == [int(n) for n in nodes], f"Path {name} does not match expected nodes"


