from npsv3.util.range import Range
from npsv3.util.variant import Variant, VariantFileReader

from .. import create_vcf


def _first_variant(vcf_path: str) -> Variant:
    with VariantFileReader.open(vcf_path) as reader:
        return next(reader.fetch())


class TestVariantFileReader:
    def test_variant_reader(self, tmp_path):
        vcf_path = create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3	Sample4
chr1	3693767	.	C	CCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	.	.	GT	./.	0|1	./.	./.
chr1	3693767	.	C	CCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	.	.	GT	./.	./.	./.	0|1
chr1	3693767	.	C	G	30	.	.	GT	./.	./.	./.	./.
chr1	3693767	.	C	G,CCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	.	.	GT	./.	./.	1|2	./."""
        )  # fmt: skip

        reader = VariantFileReader.open(vcf_path)
        samples = reader.samples()
        assert samples == ["Sample1", "Sample2", "Sample3", "Sample4"]

        variants = list(reader.fetch())
        assert len(variants) == 4

        for i, ranges in enumerate(
            [
                [Range("chr1", 3693767, 3693767)] * 2,
                [Range("chr1", 3693767, 3693767)] * 2,
                [Range("chr1", 3693766, 3693767)] * 2,
                [Range("chr1", 3693766, 3693767), Range("chr1", 3693766, 3693767), Range("chr1", 3693767, 3693767)],
            ]
        ):
            assert [variants[i].allele_reference_region(a) for a in range(variants[i].num_alleles)] == ranges
        for i, lengths in enumerate([[None, 210], [None, 245], [None, 0], [None, 0, 210]]):
            assert [variants[i].allele_length_change(a) for a in range(variants[i].num_alleles)] == lengths

    def test_variant_star_allele(self, tmp_path):
        vcf_path = create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
chr1	3693767	.	C	G,*	30	.	.	GT	1/2"""
        )  # fmt: skip

        reader = VariantFileReader.open(vcf_path)
        [variant] = list(reader.fetch())
        assert variant.allele_reference_region(0) == Range("chr1", 3693766, 3693767)
        assert variant.allele_reference_region(2) is None


class TestStarAlleleVariant:
    def test_padded_allele(self, tmp_path):
        vcf_path = create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422>
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="Difference in length between REF and ALT alleles">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG002
chr1	1650664	.	CTTTTT	C,CTTTT	.	.	SVTYPE=DEL;SVLEN=-5,-1	GT	1|2
"""
        )  # fmt: skip
        variant = _first_variant(vcf_path)
        assert variant.allele_reference_region(2) == Range("chr1", 1650664, 1650665)

    def test_star_allele(self, tmp_path):
        vcf_path = create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count in genotypes, for each ALT allele, in the same order as listed">
##contig=<ID=14,length=107349540>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
14	77187582	.	C	CAAAAAAAAAA,*	344.04	PASS	AC=2,4
"""
        )  # fmt: skip

        variant = _first_variant(vcf_path)

        # `reference_region` should be trimmed to exclude the shared padding base(s) relative to the
        # untrimmed `record_reference_region`
        assert variant.reference_region().start - variant.record_reference_region().start == 1
        assert variant.reference_region() == Range("14", 77187582, 77187582)
        assert variant.length_change() == [10, None]
        assert variant.ref_length == 1
        assert variant.allele_length(1) == 11
        assert variant.allele_length(2) is None
        assert variant.allele_sequence(1) == "AAAAAAAAAA"
        assert variant.allele_sequence(2) is None

    def test_star_allele_with_svlen(self, tmp_path):
        vcf_path = create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422>
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="Difference in length between REF and ALT alleles">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	1140155	.	G	GGGGGTGTCAACATCGAACCGGGGGACCTGGGTCCTGGGGAGCTTCCTGGGGTCAGAAGGTGGGGGTGTCAGCATCGAACCGGGGGACCTGGGTCCTGGGGAGCTTCCTGGGGTCAGAAGGTAGGGGTGTCAGCATCGAACCGGGGGACCTGGGTCATGGGGAGCTTCCTGGGGTCAGAAGGTGGGGGTGTCAACGTCGAACCGGGGGGCCTGGGTCCTGGGGAGCTTCCTGGGGTCAGAAGGTAGGGGTGTCAACGTCGAACCGGGGGACCTGGGTCCTGGGGAGCTTCCTGGGGTCAGAAGGTGGGGGTGTCAACGTCGAACCGGGGGACCTGGGTCCTGGGGAGCTTCCTGGGTTCAGAAGGTGGGGGTGTCAGCATCGAACCGGGGGACCTGGGTCCTGGGGAGCTTCCTGGGGTCAGAAGGTGGGGGTGTCAGCATCGAACCGGGGGACCTGGGTCCTGGGGAGCTTCCTGGGGTCAGAAGGTGGGGGTGTCAACATCGAACCGGGGGACCTGGGTCCTGGGGAGCTTCCTGGGGTCAGAAGGTGGGGGTGTCAGCATCGAACCGGGGGACCTGGGTCCTGGGGAGCTTCCTGGGGTCAGAAGGTA,*	.	.	SVTYPE=INS;SVLEN=671,0	GT	1|2
"""
        )  # fmt: skip

        variant = _first_variant(vcf_path)
        assert variant.length_change() == [671, None], "Allele length for star alleles should be None"

    def test_allele_with_right_padding(self, tmp_path):
        vcf_path = create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="Difference in length between REF and ALT alleles">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG002
chr1	3999762	6281	ATGGAGTTGCAGGATACGCCACAGAGAGGGGAGGGGGCCACACTGCCGACGGGGCAGGCCTGGAGTTGCAGGACGTGTCACAGAGAGAGGAAGGGGCCACACTGCTGACGGGGCGGGCC	A,CTGGAGTTGCAGGATACGCCACAGAGAGGGGAGGGGGCCACACTGCCGACGGGGCAGGCCTGGAGTTGCAGGACGTGTCACAGAGAGAGGAAGGGGCCACACTGCTGACGGGGCGGGCC	.	.	SVTYPE=DEL;SVLEN=-118,0	GT	2|1
"""
        )  # fmt: skip

        variant = _first_variant(vcf_path)
        assert variant.length_change() == [-118, 0]
        assert variant.length_change(1) == -118
        assert variant.length_change(2) == 0

        assert variant.allele_reference_region(1) == Range("chr1", 3999762, 3999762 + 118)
        assert variant.allele_reference_region(2) == Range("chr1", 3999761, 3999761 + 1)  # Allele is effectively a SNV

        assert variant.allele_sequence(1) == ""
        assert variant.allele_sequence(2) == "C"

    def test_ins_with_right_padding(self, tmp_path):
        vcf_path = create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr4,length=190214555>
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="Difference in length between REF and ALT alleles">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample
chr4	99589036	.	C	TATATATATGTTCATATATATATTC	30	.	SVTYPE=INS;SVLEN=24	GT	1|0
"""
        )  # fmt: skip

        variant = _first_variant(vcf_path)
        assert variant.length_change() == [24]

        # We don't remove "right padding" if one of the alleles is just length 1
        assert variant.allele_reference_region(1) == Range("chr4", 99589035, 99589036)
        assert variant.allele_sequence(1) == "TATATATATGTTCATATATATATTC"

    def test_split_star_allele(self, tmp_path):
        vcf_path = create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=1,length=249250621>
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="Difference in length between REF and ALT alleles">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample
1	5474286	.	CG	*	.	.	.	GT	1/0
"""
        )  # fmt: skip
        variant = _first_variant(vcf_path)
        assert variant.reference_region() == Range("1", 5474285, 5474287), "With only * alternate allele, reference be entire REF allele"
