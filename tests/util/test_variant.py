import os

import pysam

from npsv3.util.range import Range
from npsv3.variant import Variant


class TestVCFVariantTypes:
    def test_star_allele(self, tmp_path):
        vcf_path = os.path.join(tmp_path, "test.vcf")
        with open(vcf_path, "w") as vcf_file:
            vcf_file.write(
                """##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count in genotypes, for each ALT allele, in the same order as listed">
##contig=<ID=14,length=107349540>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
14	77187582	.	C	CAAAAAAAAAA,*	344.04	PASS	AC=2,4
"""
            )

        with open(vcf_path) as vcf_file:
            for line in vcf_file:
                print(line.strip())
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

