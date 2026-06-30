import pytest

from npsv3._native_graph import Graph, Range, UniqueKmersOverlay, VariantFileReader

from .. import HG38_REF_FASTA, _create_vcf


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


@pytest.mark.skipif(HG38_REF_FASTA is None, reason="HG38 reference FASTA not available")
class TestSerializationBytes:
    def setup_vcf(self, tmp_path):
        vcf_path = _create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
chr1	52277191	.	TCTATTGTTAGTAAAATAC	T	.	PASS	.	GT	0/1"""
        )  # fmt: skip
        return vcf_path, Range("chr1", 52277181, 52277219)

    def test_graph_save_and_load_bytes(self, tmp_path):
        vcf_path, region = self.setup_vcf(tmp_path)
        graph = Graph(HG38_REF_FASTA, vcf_path, region)

        data = graph.save_bytes()
        assert isinstance(data, bytes)
        assert len(data) > 0

        # Verify that the return buffer matches the direct file serialization
        save_path = tmp_path / "graph.bin"
        graph.save(str(save_path))
        with open(save_path, "rb") as f:
            file_data = f.read()
            assert data == file_data

        # Verify we can load the graph back from the bytes (with same attribute)
        restored = Graph.load_bytes(data)
        assert restored.node_count() == graph.node_count()

    def test_unique_kmers_save_and_load_bytes(self, tmp_path):
        vcf_path, region = self.setup_vcf(tmp_path)
        graph = Graph(HG38_REF_FASTA, vcf_path, region)
        kmers = UniqueKmersOverlay(graph, 7, max_edges=5)
        assert len(kmers) > 0  # sanity: this graph has unique k-mers

        data = kmers.save_bytes()
        assert isinstance(data, bytes)
        assert len(data) > 0

        restored = UniqueKmersOverlay(graph, data)
        assert len(restored) == len(kmers)
        assert restored.sequences == kmers.sequences

