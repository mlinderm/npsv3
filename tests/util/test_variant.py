import os

import pysam

from npsv3.util.range import Range
from npsv3.util.vcf import index_variant_file
from npsv3.variant import Variant


def _write_vcf(tmp_path, vcf: bytes) -> str:
    """Return GraphConstructor for VCF as literal string in `expand`ed region."""
    vcf_path = os.path.join(tmp_path, "test.vcf.gz")
    with pysam.BGZFile(vcf_path, "wb") as vcf_file:
        vcf_file.write(vcf)
    index_variant_file(vcf_path)

    return vcf_path

class TestStarAlleleVariant:
    def test_star_allele(self, tmp_path):
        vcf_path = _write_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count in genotypes, for each ALT allele, in the same order as listed">
##contig=<ID=14,length=107349540>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
14	77187582	.	C	CAAAAAAAAAA,*	344.04	PASS	AC=2,4
"""
        )

        record = next(pysam.VariantFile(vcf_path, "r"))
        variant = Variant.from_pysam(record)

        assert variant._padding == 1
        assert variant.reference_region == Range("14", 77187582, 77187582)
        assert variant.length_change() == (10, None)
        assert variant.ref_length == 1
        assert variant.alt_length(1) == 11
        assert variant.alt_length(2) is None
        assert variant.alt_seq(1) == "AAAAAAAAAA"
        assert variant.alt_seq(2) is None


    def test_star_allele_with_svlen(self, tmp_path):
        vcf_path = _write_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422>
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="Difference in length between REF and ALT alleles">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	1140155	.	G	GGGGGTGTCAACATCGAACCGGGGGACCTGGGTCCTGGGGAGCTTCCTGGGGTCAGAAGGTGGGGGTGTCAGCATCGAACCGGGGGACCTGGGTCCTGGGGAGCTTCCTGGGGTCAGAAGGTAGGGGTGTCAGCATCGAACCGGGGGACCTGGGTCATGGGGAGCTTCCTGGGGTCAGAAGGTGGGGGTGTCAACGTCGAACCGGGGGGCCTGGGTCCTGGGGAGCTTCCTGGGGTCAGAAGGTAGGGGTGTCAACGTCGAACCGGGGGACCTGGGTCCTGGGGAGCTTCCTGGGGTCAGAAGGTGGGGGTGTCAACGTCGAACCGGGGGACCTGGGTCCTGGGGAGCTTCCTGGGGTCAGAAGGTGGGGGTGTCAACGTCGAACCGGGGGACCTGGGTCCTGGGGAGCTTCCTGGGTTCAGAAGGTGGGGGTGTCAGCATCGAACCGGGGGACCTGGGTCCTGGGGAGCTTCCTGGGGTCAGAAGGTGGGGGTGTCAGCATCGAACCGGGGGACCTGGGTCCTGGGGAGCTTCCTGGGGTCAGAAGGTGGGGGTGTCAACATCGAACCGGGGGACCTGGGTCCTGGGGAGCTTCCTGGGGTCAGAAGGTGGGGGTGTCAGCATCGAACCGGGGGACCTGGGTCCTGGGGAGCTTCCTGGGGTCAGAAGGTA,*	.	.	SVTYPE=INS;SVLEN=671,0	GT	1|2
"""
        )

        record = next(pysam.VariantFile(vcf_path, "r"))
        variant = Variant.from_pysam(record)

        assert variant.length_change() == (671,None), "Allele length for star alleles should be None" 