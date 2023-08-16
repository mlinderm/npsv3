import os

import pytest

from npsv3.graph import Graph, variant_path_name
from npsv3.util.range import Range

from . import B37_REF_FASTA, HG38_REF_FASTA, data_path


class TestGraphConstructionFromVCF:
    @pytest.mark.skipif(not os.path.exists(B37_REF_FASTA), reason="B37 reference required")
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
        haplotypes = graph.generate_possible_haplotypes(data_path("12_22129565_22130387.vcf.gz"), "HG002#0#12#0")
        assert len(haplotypes) == 2, "A single isolated variant should only have two haplotypes"
        for i in range(2):
            assert haplotypes[i].paths == {f"_alt_553e586e2a8e7c2fd70661fec7b529c5453a9b45_{i}"}

        # There is a 1bp deletion in base haplotype and a 822bp deletion in the SV
        assert len(haplotypes[0].sequence()) == region.length - 1
        assert len(haplotypes[1].sequence()) == region.length - 1 - 822

        # Variant path should match the background
        assert graph.nodes_on_path("HG002#0#12#0") == haplotypes[1].nodes

        variant_haplotypes = graph.generate_possible_haplotypes(data_path("12_22129565_22130387.vcf.gz"), "12")
        assert len(variant_haplotypes) == 2, "The reference background should generate the same number of haplotypes"

        assert len(variant_haplotypes[0].sequence()) == region.length
        assert len(variant_haplotypes[1].sequence()) == region.length - 822

        # Use the query interface
        query_haplotypes = [
            graph.generate_haplotype("12", [f"_alt_553e586e2a8e7c2fd70661fec7b529c5453a9b45_{i}"]) for i in range(2)
        ]
        assert len(query_haplotypes[0].sequence()) == region.length
        assert len(query_haplotypes[1].sequence()) == region.length - 822

    @pytest.mark.skip()
    @pytest.mark.skipif(not os.path.exists(HG38_REF_FASTA), reason="HG38 reference required")
    def test_complex_haplotype_generation(self):
        graph = Graph.from_vcf(
            HG38_REF_FASTA, data_path("chr13_29557414_29560096.vcf.gz"), Range("chr13", 29557413, 29560096)
        )
        for path in ("chr13", "NA12878#0#chr13#0", "NA12878#1#chr13#0"):
            assert graph.has_path(path)

        haplotypes = graph.generate_possible_haplotypes(
            data_path("chr13_29557414_29560096.inference.vcf.gz"), "NA12878#0#chr13#0"
        )
        assert len(haplotypes) == 15608


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

        # # Get variant ids
        # ref_haplotype = graph.generate_haplotype("chr13", [])
        # ref_seq = ref_haplotype.sequence()
        # with pysam.VariantFile(data_path("chr13_29557414_29560096.inference.vcf.gz"), drop_samples=True) as inference_vcf_file:
        #     for record in inference_vcf_file.fetch(**region.pysam_fetch):
        #         variant_id = vg_variant_id(record)
        #         print(variant_id, record.contig, record.pos, record.info["SVLEN"])

        # base = variant_path_name("816a3a14ae47235e36f4c2cec09ded5147c4a391",1)
        # base_haplotype = graph.generate_haplotype("chr13", [base])
        # for variant_id in ("3ff82bc30d9789c4f2e2118a107db1483b15b459",): #("851cdc40e30e11df220f3ba566ca92fbfe19a791",): #("3bae2e6237b646bc6bd4aa878e089b4c94761334",):
        #     new_haplotype = graph.generate_haplotype("chr13", [variant_path_name(variant_id, 1)])
        #     #single_haplotype = graph.generate_haplotype("chr13", [base, variant_path_name(variant_id, 1)])
        #     alts = [base_haplotype.sequence(), new_haplotype.sequence()] #, single_haplotype.sequence()]
        #     print(record.contig, region.start, ".", ref_seq, ",".join(alts), ".", "PASS", f"SVTYPE=DEL;SVLEN={','.join(str(len(alt) - len(ref_seq)) for alt in alts)}", sep="\t")
