import os

import pysam

from npsv3._native_graph import Graph, Range, VariantFileReader
from npsv3.util.vcf import index_variant_file
from npsv3.variant import Variant

from .. import HG38_REF_FASTA


def _create_vcf(tmp_path: str, vcf: bytes) -> str:
    vcf_path = os.path.join(tmp_path, "test.vcf.gz")
    with pysam.BGZFile(vcf_path, "wb") as vcf_file:
        vcf_file.write(vcf)
    index_variant_file(vcf_path)
    return vcf_path


class TestVariantFileReader:
    def test_variant_reader(self, tmp_path):
        vcf_path = _create_vcf(tmp_path, b"""##fileformat=VCFv4.2
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
        vcf_path = _create_vcf(tmp_path, b"""##fileformat=VCFv4.2
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


class TestGraph:
    def test_graph_samples_including(self, tmp_path):
        vcf_path = _create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3	Sample4
chr1	3693767	.	C	CCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	.	.	GT	./.	0|1	./.	./.
chr1	3693767	.	C	CCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	.	.	GT	./.	./.	./.	0|1
chr1	3693767	.	C	G	30	.	.	GT	./.	./.	./.	./.
chr1	3693767	.	C	G,CCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	.	.	GT	./.	./.	1|2	./."""
        )  # fmt: skip

        region = Range("chr1", 3693757, 3693777)
        graph = Graph(HG38_REF_FASTA, vcf_path, region)
        assert graph.node_count() == 7
        graph.dump()

        reader = VariantFileReader.open(vcf_path)
        exclude_nodes = set()
        for variant in reader.fetch(region):
            sv_alleles = {
                i
                for i in range(1, variant.num_alleles)
                if abs(variant.allele_length_change(i) or 0) >= 50
            }
            if sv_alleles:
                variant_id = variant.variant_id
                ref_nodes = set(graph.path_nodes(f"_alt_{variant_id}_0"))
                exclude_nodes.update(*(graph.path_nodes(f"_alt_{variant_id}_{a}") for a in sv_alleles))
                exclude_nodes.difference_update(ref_nodes)  # Remove nodes shared with references paths to get nodes that distinguish ALT alleles

        ref_samples = set(reader.samples()) - set(graph.samples_including(list(exclude_nodes)))
        assert ref_samples == {"Sample1"}


