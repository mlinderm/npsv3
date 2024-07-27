import os
import pytest

import odgi

from npsv3.graphs.graph import Graph, variant_path_name
from npsv3.util.range import Range

from .. import B37_REF_FASTA, HG38_REF_FASTA, HG00731_VCF, HG00731_SV_VCF, data_path

def assert_topological_order(graph: odgi.graph):
    visited = set()
    
    def check_edge(handle):
        assert graph.get_id(handle) not in visited

    def check_node(handle):
        visited.add(graph.get_id(handle))
        # False to traverse "right" or "downstream" edges
        graph.follow_edges(handle, False, check_edge)

    graph.for_each_handle(check_node)

class TestGraphConstructionFromVCF:
    @pytest.mark.skipif(not os.path.exists(B37_REF_FASTA), reason="B37 reference required")
    def test_simple_haplotype(self):
        # Presentation example
        region = Range("12", 22127564, 22132387)
        graph = Graph.from_vcf(
            B37_REF_FASTA,
            data_path("12_22129565_22130387.background.vcf.gz"),
            region,
            inference_vcf=data_path("12_22129565_22130387.vcf.gz"),
        )

        for path in ("12", "HG002#0#12#0", "HG002#1#12#0"):
            assert graph.has_path(path)
        for haplotype in (0, 1):
            assert not graph.has_path(
                f"HG002#{haplotype}#12#1"
            ), "VCF should translate to a single path for each haplotype"

        # Use exhaustive generation
        haplotypes = graph.all_haplotypes(
            data_path("12_22129565_22130387.vcf.gz"), "HG002#0#12#0", region
        )
        assert len(haplotypes) == 2, "A single isolated variant should only have two haplotypes"
        for i in range(2):
            assert haplotypes[i].paths == {f"_alt_553e586e2a8e7c2fd70661fec7b529c5453a9b45_{i}"}

        # There is a 1bp deletion in base haplotype and a 822bp deletion in the SV
        assert len(haplotypes[0].sequence()) == region.length - 1
        assert len(haplotypes[1].sequence()) == region.length - 1 - 822

        # Variant path should match the background
        assert graph.nodes_on_path("HG002#0#12#0") == haplotypes[1].nodes

        variant_haplotypes = graph.all_haplotypes(data_path("12_22129565_22130387.vcf.gz"), "12", region)
        assert len(variant_haplotypes) == 2, "The reference background should generate the same number of haplotypes"

        assert len(variant_haplotypes[0].sequence()) == region.length
        assert len(variant_haplotypes[1].sequence()) == region.length - 822

    @pytest.mark.skipif(not os.path.exists(HG38_REF_FASTA), reason="HG38 reference required")
    def test_insertion_haplotype(self):
        region = Range("chrY", 56880140, 56880241)
        graph = Graph.from_vcf(
            HG38_REF_FASTA,
            data_path("chrY_56879191_56881191.vcf.gz"),
            region,
            inference_vcf=data_path("chrY_56879191_56881191.sv.vcf.gz"),
        )
        
        # Use exhaustive generation
        haplotypes = graph.all_haplotypes(data_path("chrY_56879191_56881191.sv.vcf.gz"), "chrY", region)
        
        assert len(haplotypes) == 2, "The reference background should generate the same number of haplotypes"

        assert len(haplotypes[0].sequence()) == region.length
        assert len(haplotypes[1].sequence()) == region.length + 73
        
    @pytest.mark.skip()
    @pytest.mark.skipif(not os.path.exists(HG38_REF_FASTA), reason="HG38 reference required")
    def test_complex_haplotype(self):
        region = Range("chr13", 29557413, 29560096)
        graph = Graph.from_vcf(
            HG38_REF_FASTA,
            data_path("chr13_29557414_29560096.vcf.gz"),
            region,
        )
        for path in ("chr13", "NA12878#0#chr13#0", "NA12878#1#chr13#0"):
            assert graph.has_path(path)

        haplotypes = graph.all_haplotypes(
            data_path("chr13_29557414_29560096.inference.vcf.gz"),
            "NA12878#0#chr13#0",
            region,
        )
        assert len(haplotypes) > 1000

    # For chr1_41823764_41825818.vcf.gz the "reference" path for the variant conflicts with "true" path
    # so we have 2 haplotypes for the absence of that variant, the explicit reference path and an 
    # "implicit" path that does not include the alternate allele

    @pytest.mark.skipif(not os.path.exists(HG38_REF_FASTA), reason="HG38 reference required")
    @pytest.mark.parametrize("region,background_vcf,inference_vcf,exp_haplotypes", [
        (Range("chr1", 8977700, 8977700), "chr1_8976700_8978700.vcf.gz", "chr1_8976700_8978700.sv.vcf.gz", (2,2)),
        (Range("chr1", 41824764, 41824818), "chr1_41823764_41825818.vcf.gz", "chr1_41823764_41825818.sv.vcf.gz", (3,3)),
    ])
    def test_adjacent_variant_haplotype(self, cfg, region, background_vcf, inference_vcf, exp_haplotypes):
        graph = Graph.from_vcf(
            HG38_REF_FASTA,
            data_path(background_vcf),
            region.expand(cfg.pileup.graph_flank),
            inference_vcf=data_path(inference_vcf),
        )
        graph._graph.to_gfa()
        for allele, exp_haplotypes_allele in enumerate(exp_haplotypes):
            haplotypes = graph.all_haplotypes(data_path(inference_vcf), f"HG00731#{allele}#{region.contig}#0", region.expand(cfg.pileup.variant_padding))
            assert len(haplotypes) == exp_haplotypes_allele, f"Unexpected number of haplotypes for allele {allele}"

    @pytest.mark.skipif(not os.path.exists(HG38_REF_FASTA), reason="HG38 reference required")
    def test_inconsistent_backbone_haplotype(self, cfg):
        region = Range("chr1", 6012136, 6012573)
        graph = Graph.from_vcf(
            HG38_REF_FASTA,
            data_path("chr1_6011136_6013135.vcf.gz"),
            region.expand(cfg.pileup.graph_flank),
            inference_vcf=data_path("chr1_6011136_6013135.sv.vcf.gz"),
        )

        # Since graph is optimized during construction, we should iterate over the nodes in topological order
        assert_topological_order(graph._graph)

        # HG00731#0#chr1#0 is complete path, so the shortest path should be the same as the path in 
        # the original graph
        first_chrom_path = graph.nodes_on_path("HG00731#0#chr1#0")
        assert graph.shortest_path("HG00731#0#chr1") == first_chrom_path

        # HG00731#1#chr1 is comprised of multiple paths, so shortest path should link these paths, minimizing
        # the number of reference bases required. In this case the only difference is now a 1|0 DEL.
        shortest_second_chrom_path = graph.shortest_path("HG00731#1#chr1")
        assert set(first_chrom_path) - set(shortest_second_chrom_path) == graph.path_nodes["_alt_962c481d8368e20de391e2a3265d6e6c9fd77761_1"]
        assert set(shortest_second_chrom_path) - set(first_chrom_path) == graph.path_nodes["_alt_962c481d8368e20de391e2a3265d6e6c9fd77761_0"]

        for backbone in ("chr1", "HG00731#0#chr1#0", "HG00731#1#chr1"):
            haplotypes = graph.all_haplotypes(
                data_path("chr1_6011136_6013135.sv.vcf.gz"),
                backbone,
                region.expand(cfg.pileup.variant_padding),
            )
            assert len(haplotypes) == 12, "4 possible SVs, but 2 are mutually exclusive, thus 12 haplotypes"
            
            # One of the paths should match the backbone
            assert sum(h.nodes == graph.shortest_path(backbone) for h in haplotypes) == 1

    @pytest.mark.skipif(not os.path.exists(HG38_REF_FASTA), reason="HG38 reference required")
    def test_overlapping_haplotype(self, cfg):
        region = Range("chr1", 853424, 853622)
        graph = Graph.from_vcf(
            HG38_REF_FASTA,
            data_path("chr1_853425_853622.vcf.gz"),
            region.expand(cfg.pileup.graph_flank),
            inference_vcf=data_path("chr1_853425_853622.sv.vcf.gz"),
        )
        graph._graph.to_gfa()
        haplotypes = graph.all_haplotypes(
            data_path("chr1_853425_853622.sv.vcf.gz"),
            "HG00731#0#chr1#0",
            region.expand(cfg.pileup.variant_padding),
        )
        assert len(haplotypes) == 4, "Two possible alleles in region, so 4 possible haplotypes"

        # The first two haplotypes should not have a larger deletion that spans a true shorter deletion, and thus
        # should have the deletion variant
        del_nodes = graph.path_nodes["_alt_ff413357d62f19b9ab324950d39208722aa44da0_1"]
        for h in (0, 1):
            assert len(del_nodes & set(haplotypes[h].nodes)) > 0
        for h in (2, 3):
            assert len(del_nodes & set(haplotypes[h].nodes)) == 0

    
    @pytest.mark.skipif(not os.path.exists(HG38_REF_FASTA), reason="HG38 reference required")
    @pytest.mark.skipif(
        not HG00731_VCF or not HG00731_SV_VCF, reason="HG00731 VCFs required"
    )
    @pytest.mark.parametrize(
        "region",
        [
            # Range("chr1", 831163, 833782),
            # Range("chr1", 859060, 864208),
            # Range("chr1", 1075570, 1075670),
            # Range("chr1", 1978993, 1979167),
            # Range("chr1", 2689931, 2689931),
            #Range("chr1", 6012135, 6012135),
            # Range("chr1", 12858834, 12858933),
            # Range("chr1", 29553648, 29553842), # Region overlaps N's, should generally be excluded
            # Range("chr1", 38618549, 38620153),
            Range("chr1", 5722418, 5722418),
            Range("chr21", 37122424, 37122424),
        ],
    )
    def test_observed_errors_haplotype(self, cfg, region):
        graph = Graph.from_vcf(
            HG38_REF_FASTA,
            HG00731_VCF,
            region.expand(cfg.pileup.graph_flank),
            inference_vcf=HG00731_SV_VCF,
        )
        #graph._graph.to_gfa()
        assert graph.is_bubble_path(region.contig), "Graph must form bubble for reference paths"

        haplotypes = graph.all_haplotypes(
            HG00731_SV_VCF,
            region.contig,
            region.expand(cfg.pileup.variant_padding),
        )
        assert len(haplotypes) > 1
        assert (
            len(haplotypes[0].sequence()) == graph.region.length
        ), "With reference background, first haplotype should match reference length"
        assert haplotypes[0].nodes == graph.nodes_on_path(region.contig)


        backgrounds = [
            graph.all_haplotypes(HG00731_SV_VCF, f"HG00731#{i}#{region.contig}", region.expand(cfg.pileup.variant_padding))
            for i in range(2)
        ]
        for allele, background in enumerate(backgrounds):
            base_path_nodes = graph.shortest_path(f"HG00731#{allele}#{region.contig}")
            assert sum(haplotype.nodes == base_path_nodes for haplotype in background) == 1, f"No path matches backbone for allele {allele}"
