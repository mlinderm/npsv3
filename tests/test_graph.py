import os

import pytest

from npsv3.graph import Graph, variant_path_name
from npsv3.util.range import Range

from . import B37_REF_FASTA, HG38_REF_FASTA, data_path

def image_region(cfg, region) -> Range:
    # Try to minimize compression by setting right padding to exact width...
    to_pad = cfg.pileup.image_width - region.length
    left_padding = max((to_pad + 1) // 2, cfg.pileup.variant_padding)
    right_padding = max(to_pad // 2, cfg.pileup.variant_padding)
    return region.expand(left_padding, right_padding)

class TestGraphConstructionFromVCF:
    @pytest.mark.skipif(
        not os.path.exists(B37_REF_FASTA), reason="B37 reference required"
    )
    def test_simple_haplotype_generator(self):
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
        haplotypes = graph.generate_possible_haplotypes(
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

        variant_haplotypes = graph.generate_possible_haplotypes(data_path("12_22129565_22130387.vcf.gz"), "12", region)
        assert len(variant_haplotypes) == 2, "The reference background should generate the same number of haplotypes"

        assert len(variant_haplotypes[0].sequence()) == region.length
        assert len(variant_haplotypes[1].sequence()) == region.length - 822

        # Use the query interface
        query_haplotypes = [
            graph.generate_haplotype("12", [f"_alt_553e586e2a8e7c2fd70661fec7b529c5453a9b45_{i}"]) for i in range(2)
        ]
        assert len(query_haplotypes[0].sequence()) == region.length
        assert len(query_haplotypes[1].sequence()) == region.length - 822

    @pytest.mark.skipif(not os.path.exists(HG38_REF_FASTA), reason="HG38 reference required")
    def test_insertion_haplotype_generation(self):
        region = Range("chrY", 56880140, 56880241)
        graph = Graph.from_vcf(
            HG38_REF_FASTA,
            data_path("chrY_56879191_56881191.vcf.gz"),
            region,
            inference_vcf=data_path("chrY_56879191_56881191.sv.vcf.gz"),
        )

        # Use exhaustive generation
        graph.generate_possible_haplotypes(data_path("chrY_56879191_56881191.sv.vcf.gz"), "chrY", region)

    @pytest.mark.skip()
    @pytest.mark.skipif(not os.path.exists(HG38_REF_FASTA), reason="HG38 reference required")
    def test_complex_haplotype_generation(self):
        region = Range("chr13", 29557413, 29560096)
        graph = Graph.from_vcf(
            HG38_REF_FASTA,
            data_path("chr13_29557414_29560096.vcf.gz"),
            region,
        )
        for path in ("chr13", "NA12878#0#chr13#0", "NA12878#1#chr13#0"):
            assert graph.has_path(path)

        haplotypes = graph.generate_possible_haplotypes(
            data_path("chr13_29557414_29560096.inference.vcf.gz"),
            "NA12878#0#chr13#0",
            region,
        )
        assert len(haplotypes) == 8848

    @pytest.mark.skipif(
        not os.path.exists("/storage/mlinderman/projects/sv/npsv3-experiments"), reason="Not running on cluster"
    )
    @pytest.mark.skipif(not os.path.exists(HG38_REF_FASTA), reason="HG38 reference required")
    def test_problem_graph_generation(self, cfg):
        region = Range("chr1", 853424, 853622)
        graph = Graph.from_vcf(
            HG38_REF_FASTA,
            "/storage/mlinderman/projects/sv/npsv3-experiments/training/HGSVC2_training_vcfs/HG00731.freeze4.alt.passing.training.hg38.vcf.gz",
            region.expand(cfg.pileup.graph_flank),
            inference_vcf="/storage/mlinderman/projects/sv/npsv3-experiments/training/HGSVC2_training_vcfs/HG00731.freeze4.sv.alt.passing.training.hg38.vcf.gz",
        )
        haplotypes = graph.generate_possible_haplotypes(
            "/storage/mlinderman/projects/sv/npsv3-experiments/training/HGSVC2_training_vcfs/HG00731.freeze4.sv.alt.passing.training.hg38.vcf.gz",
            "HG00731#0#chr1#0",
            region.expand(cfg.pileup.variant_padding),
        )

        assert len(haplotypes) == 4, "Two possible alleles in CIS, so 4 possible alleles"

        # The first two haplotypes should not have a larger deletion that spans a true shorter deletion, and thus
        # should have the deletion variant
        del_nodes = graph.path_nodes["_alt_ff413357d62f19b9ab324950d39208722aa44da0_1"]
        for h in (0, 1):
            assert len(del_nodes & set(haplotypes[h].nodes)) > 0
        for h in (2, 3):
            assert len(del_nodes & set(haplotypes[h].nodes)) == 0

    @pytest.mark.skipif(
        not os.path.exists("/storage/mlinderman/projects/sv/npsv3-experiments"), reason="Not running on cluster"
    )
    @pytest.mark.skipif(not os.path.exists(HG38_REF_FASTA), reason="HG38 reference required")
    @pytest.mark.parametrize(
        "region",
        [
            # Range("chr1", 831163, 833782),
            # Range("chr1", 859060, 864208),
            # Range("chr1", 1075570, 1075670),
            # Range("chr1", 1978993, 1979167),
            # Range("chr1", 2689931, 2689931),
            Range("chr1", 6012136, 6012135),
            # Range("chr1", 12858834, 12858933),   
            # Range("chr1", 29553648, 29553842), # Region overlaps N's, should generally be excluded
            # Range("chr1", 38618549, 38620153),
        ],
    )
    def test_problem_haplotype_generation(self, cfg, region):
        graph = Graph.from_vcf(
            HG38_REF_FASTA,
            "/storage/mlinderman/projects/sv/npsv3-experiments/training/HGSVC2_training_vcfs/HG00731.freeze4.alt.passing.training.hg38.vcf.gz",
            region.expand(cfg.pileup.graph_flank),
            inference_vcf="/storage/mlinderman/projects/sv/npsv3-experiments/training/HGSVC2_training_vcfs/HG00731.freeze4.sv.alt.passing.training.hg38.vcf.gz",
        )

        graph._graph.to_gfa()
        assert graph.is_bubble_path(region.contig), "Graph must form bubble for reference paths"
        assert all(graph.is_bubble_path(f"HG00731#{i}#{region.contig}#0") for i in range(2)), "Graph must form bubble for haplotype paths"

        haplotypes = graph.generate_possible_haplotypes(
            "/storage/mlinderman/projects/sv/npsv3-experiments/training/HGSVC2_training_vcfs/HG00731.freeze4.sv.alt.passing.training.hg38.vcf.gz",
            "chr1",
            image_region(cfg, region),
        )
        assert len(haplotypes) > 1
        assert (
            len(haplotypes[0].sequence()) == graph.region.length
        ), "With reference background, first haplotype should match reference length"
        assert haplotypes[0].nodes == graph.nodes_on_path("chr1")
    


class TestGraphQuery:
    @pytest.mark.skipif(not os.path.exists(HG38_REF_FASTA), reason="HG38 reference required")
    def test_complex_haplotype_demo(self):
        region = Range("chr13", 29557919, 29559563)
        graph = Graph.from_vcf(
            HG38_REF_FASTA,
            data_path("chr13_29557414_29560096.vcf.gz"),
            region,
        )

        with pytest.raises(ValueError):
            # These two variants can't be on the same haplotype
            graph.generate_haplotype(
                "chr13",
                [
                    variant_path_name("3ff82bc30d9789c4f2e2118a107db1483b15b459", 1),
                    variant_path_name("816a3a14ae47235e36f4c2cec09ded5147c4a391", 1),
                ],
            )
