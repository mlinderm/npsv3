import pytest

from npsv3._native_graph import HaplotypePriorOverlay, UniqueKmersOverlay
from npsv3.graphs.graph import Graph
from npsv3.util.range import Range
from npsv3.util.variant import VariantFileReader

from .. import HG38_REF_FASTA, create_vcf


class TestGraph:
    @pytest.mark.skipif(HG38_REF_FASTA is None, reason="HG38 reference FASTA not available")
    def test_graph_samples_including(self, tmp_path):
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

        region = Range("chr1", 3693757, 3693777)
        graph = Graph(HG38_REF_FASTA, vcf_path, region)
        assert graph.node_count() == 7

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
        vcf_path = create_vcf(tmp_path, b"""##fileformat=VCFv4.2
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
        assert isinstance(data, bytearray)  # named for symmetry with load_bytes(), but returns bytearray
        assert len(data) > 0

        # Verify that the return buffer matches the direct file serialization
        save_path = tmp_path / "graph.bin"
        graph.save(str(save_path))
        with open(save_path, "rb") as f:
            file_data = f.read()
            assert data == file_data

        # load_bytes() accepts either bytes or bytearray -- exercise both overloads here.
        restored = Graph.load_bytes(data)
        assert restored.node_count() == graph.node_count()
        restored_from_bytes = Graph.load_bytes(bytes(data))
        assert restored_from_bytes.node_count() == graph.node_count()

    def test_unique_kmers_save_and_load_bytes(self, tmp_path):
        vcf_path, region = self.setup_vcf(tmp_path)
        graph = Graph(HG38_REF_FASTA, vcf_path, region)
        kmers = UniqueKmersOverlay(graph, 7, max_edges=5)
        assert len(kmers) > 0  # sanity: this graph has unique k-mers

        data = kmers.save_bytes()
        assert isinstance(data, bytearray)
        assert len(data) > 0

        restored = UniqueKmersOverlay(graph, data)
        assert len(restored) == len(kmers)
        assert restored.sequences == kmers.sequences

    def test_haplotype_gbwt_overlay_save_and_load_bytes(self, tmp_path):
        vcf_path, region = self.setup_vcf(tmp_path)
        graph = Graph(HG38_REF_FASTA, vcf_path, region)
        overlay = HaplotypePriorOverlay(graph)

        data = overlay.save_bytes()
        assert isinstance(data, bytearray)
        assert len(data) > 0

        # Walk Sample1's own hom-alt haplotype (Find() once, Extend() thereafter) -- the discipline the
        # overlay's own class docs require for Width()/ScoreToGo() to be meaningful.
        walk = graph.path_nodes("Sample1#1#chr1#0")
        assert len(walk) > 1

        def walk_and_collect(ov):
            state = ov.find(walk[0])
            widths = [ov.width(state)]
            score_to_gos = [ov.score_to_go(state)]
            for node in walk[1:]:
                state = ov.extend(state, node)
                widths.append(ov.width(state))
                score_to_gos.append(ov.score_to_go(state))
            return widths, score_to_gos

        widths, score_to_gos = walk_and_collect(overlay)
        assert not overlay.empty(overlay.find(walk[0]))
        assert widths[0] > 0

        restored = HaplotypePriorOverlay(graph, data)
        restored_widths, restored_score_to_gos = walk_and_collect(restored)
        assert restored_widths == widths
        assert restored_score_to_gos == score_to_gos

    def test_haplotype_gbwt_overlay_panel_state_round_trips_through_python(self, tmp_path):
        vcf_path, region = self.setup_vcf(tmp_path)
        graph = Graph(HG38_REF_FASTA, vcf_path, region)
        overlay = HaplotypePriorOverlay(graph)

        walk = graph.path_nodes("Sample1#1#chr1#0")
        # A PanelState returned by find()/extend() must be directly usable by every other method without any
        # Python-side introspection of its contents (it's deliberately opaque).
        state = overlay.find(walk[0])
        for node in walk[1:]:
            assert overlay.width(state) >= 0
            assert overlay.score_to_go(state) <= 0.0  # log-probabilities, unweighted units
            state = overlay.extend(state, node)
        assert not overlay.empty(state)

    def test_haplotype_transition_overlay_save_and_load_bytes(self, tmp_path):
        vcf_path, region = self.setup_vcf(tmp_path)
        graph = Graph(HG38_REF_FASTA, vcf_path, region)
        overlay = HaplotypePriorOverlay(graph)

        data = overlay.save_bytes()
        assert isinstance(data, bytearray)
        assert len(data) > 0

        walk = graph.path_nodes("Sample1#1#chr1#0")
        assert len(walk) > 1
        from_node, to_node = walk[0], walk[1]

        probability = overlay.transition_probability(from_node, to_node, 1.0)
        log_probability = overlay.log_transition_probability(from_node, to_node, 1.0)
        node_totals = overlay.node_totals

        restored = HaplotypePriorOverlay(graph, data)
        assert restored.transition_probability(from_node, to_node, 1.0) == probability
        assert restored.log_transition_probability(from_node, to_node, 1.0) == log_probability
        assert restored.node_totals == node_totals

