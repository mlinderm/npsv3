import os

import pysam

from npsv3._native_graph import Graph, Range
from npsv3.util.vcf import index_variant_file
from npsv3.variant import Variant

from .. import HG38_REF_FASTA


def _create_vcf(tmp_path: str, vcf: bytes) -> str:
    vcf_path = os.path.join(tmp_path, "test.vcf.gz")
    with pysam.BGZFile(vcf_path, "wb") as vcf_file:
        vcf_file.write(vcf)
    index_variant_file(vcf_path)
    return vcf_path


def _non_ref_genotype(sample: pysam.VariantRecordSample) -> bool:
    """Return True if the sample's genotype is defined and has non-reference allele"""
    non_ref = False
    for allele in sample.allele_indices:
        if allele is None:
            return False
        non_ref = non_ref or allele > 0
    return non_ref


class TestGraph:
    def test_graph_ref_samples(self, tmp_path):
        region = Range("chr1", 3693757, 3693777)
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

        graph = Graph(HG38_REF_FASTA, vcf_path, region)
        assert graph.node_count() == 7

        # Identify paths to exclude (ALT SV alleles)
        with pysam.VariantFile(vcf_path) as src_vcf_file:
            exclude_nodes = set()
            for record in src_vcf_file:
                variant = Variant.from_pysam(record)
                exclude_alleles = {
                    i
                    for i, length in enumerate(variant.length_change(), start=1)
                    if length is not None and abs(length) >= 50
                }
                exclude_nodes.update(*(graph.path_nodes(f"_alt_{variant.vg_variant_id}_{a}") for a in exclude_alleles))
            exclude_nodes -= set(graph.path_nodes(region.contig))

            assert len(exclude_nodes) == 2
            ref_samples = set(src_vcf_file.header.samples) - set(graph.samples_including(list(exclude_nodes)))
            assert ref_samples == {"Sample1"}
