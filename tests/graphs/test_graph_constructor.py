import os
from collections import deque

import pytest

from npsv3.graphs.graph_constructor import GraphConstructor, ReferenceSpan, AltPath, variant_path_name, vcf_to_paths, gfa_to_xg, add_haplotypes_to_gfa
from npsv3.util.range import Range

from .. import B37_REF_FASTA, HG38_REF_FASTA, HG00731_VCF, data_path

class TestGraphConstructor:
    def test_find_spans(self):
        region = Range("chr1", 0, 100)
        construct = GraphConstructor(
            region, data_path("empty.vcf.gz")
        )
        assert construct.num_spans == 1 and construct.get_span_region(0) == region 

        # SNV in a single span
        start_idx, end_idx = construct.find_overlapping_spans(Range("chr1", 50, 51))
        assert start_idx == 0 and end_idx == 0

        # INS in a single span
        start_idx, end_idx = construct.find_overlapping_spans(Range("chr1", 50, 50))
        assert start_idx == 0 and end_idx == 0

        # INS matching existing span
        construct.spans = deque([
            ReferenceSpan(Range("chr1", 0, 50)),
            ReferenceSpan(Range("chr1", 50, 50)), 
            ReferenceSpan(Range("chr1", 50, 100)),
        ])
        start_idx, end_idx = construct.find_overlapping_spans(Range("chr1", 50, 50))
        assert start_idx == 1 and end_idx == 1, "Should match insertion span"
            
        start_idx, end_idx = construct.find_overlapping_spans(Range("chr1", 50, 51))
        assert start_idx == 2 and end_idx == 2, "Should match after insertion span"   

        # SNV in a series
        construct.spans = deque([
            ReferenceSpan(Range("chr1", 0, 50)),
            ReferenceSpan(Range("chr1", 50, 51)), 
            ReferenceSpan(Range("chr1", 51, 52)), 
            ReferenceSpan(Range("chr1", 52, 100)),
        ])
        for start, exp_idx in [(49, 0), (50, 1), (51, 2), (52, 3)]:
            start_idx, end_idx = construct.find_overlapping_spans(Range("chr1", start, start+1))
            assert start_idx == exp_idx and end_idx == exp_idx

        # INS after an SNV
        construct.spans = deque([
            ReferenceSpan(Range("chr1", 0, 50)),
            ReferenceSpan(Range("chr1", 50, 51)),
            ReferenceSpan(Range("chr1", 51, 100)),
        ])
        start_idx, end_idx = construct.find_overlapping_spans(Range("chr1", 51, 51))
        assert start_idx == 2 and end_idx == 2, "Should match after SNV"   

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
    @pytest.mark.parametrize("region,vcf_file", [
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

        haplotype_paths = vcf_to_paths(gfa_path, data_path(vcf_file), region)
        for i, (name, strand, nodes) in enumerate(haplotype_paths):
            assert name == f"HG00731#{i}#{region.contig}#0"
        assert i == 1, "Expected two haplotypes"


    def test_colocated_SNV_DEL(self, tmp_path, cfg):
        region = Range("chr1", 6012136, 6012573)
        construct = GraphConstructor(
            region.expand(cfg.pileup.graph_flank), data_path("chr1_6011136_6013135.vcf.gz")
        )
        construct.to_gfa(HG38_REF_FASTA)
        
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
        add_haplotypes_to_gfa(gfa_path, HG00731_VCF, region.expand(cfg.pileup.graph_flank))
