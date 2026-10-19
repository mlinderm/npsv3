import itertools
import os
from collections.abc import Sequence

import pytest

from npsv3.graphs.graph import Graph
from npsv3.graphs.haplotype import (
    ConstantKmerClassify,
    KmerZygosity,
    _create_graph_and_sampler,
    _filter_variants,
    _resolved_alleles,
    haplotype_alleles,
)
from npsv3.images.haplotypes import (
    InsufficientHaplotypesError,
    _variant_allele_paths,
    set_variant_allele,
    true_haplotype_neighbors,
)
from npsv3.util.range import Range
from npsv3.util.variant import Variant, VariantFileReader

from .. import HG38_REF_FASTA, create_vcf

def _is_walk(graph: Graph, haplotype: Sequence[int]) -> bool:
    return all(graph.has_edge(a, b) for a, b in itertools.pairwise(haplotype))


def _alleles(sampler, haplotype, analysis_variants):
    return _resolved_alleles(haplotype_alleles(sampler, [haplotype], analysis_variants)[0])


class TestTwoInsertions:
    """Two insertions (the analysis variants) with a homozygous SNV (not an analysis variant) between them"""

    # cSpell:disable
    # Matches "chr1:1134644-1134867" in HG00733.hgsvc3-hprc-2024-02-23.dipcall.passing.hg38.vcf.gz
    VCF = b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=chr1,length=248956422>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00733
chr1	1134739	.	A	ACATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGCACCGTGAGGGGTGTGGCTTCCTCGC	30	PASS	.	GT	1|0
chr1	1134748	.	T	G	30	PASS	.	GT	1|1
chr1	1134771	.	G	GCACCGTGAGGGGTGTGGCTTCCTCGCCATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGCACCGTGAGGGGTGTGGCTTCCTCGCCATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGA	30	PASS	.	GT	0|1"""  # fmt: skip
    # cSpell:enable

    REGION = Range("chr1:1134644-1134867")

    @pytest.fixture
    def graph_and_variants(self, tmp_path):
        if not HG38_REF_FASTA or not os.path.exists(HG38_REF_FASTA):
            pytest.skip("Missing reference")
        
        vcf_path = create_vcf(tmp_path, self.VCF)
        graph, _unique_kmers, sampler = _create_graph_and_sampler(HG38_REF_FASTA, vcf_path, self.REGION, k=31)
        sampler.initialize_scores(ConstantKmerClassify(KmerZygosity.ABSENT))
        
        variants: list[Variant] = []
        with VariantFileReader.open(vcf_path) as vcf_file:
            variants = list(vcf_file.fetch(self.REGION))
        analysis_variants = _filter_variants(variants, min_variant_size=50)
        assert len(analysis_variants) == 2

        return graph, sampler, variants, analysis_variants

    def test_set_variant_allele_round_trip(self, graph_and_variants):
        graph, sampler, _variants, analysis_variants = graph_and_variants
        ins1, ins2 = analysis_variants
        hap0 = graph.path_nodes(f"HG00733#0#{self.REGION.contig}#0")
        ins1_paths = _variant_allele_paths(graph, ins1)
        ins2_paths = _variant_allele_paths(graph, ins2)

        removed = set_variant_allele(graph, hap0, ins1_paths, 0)
        assert removed is not None
        assert _is_walk(graph, removed)
        assert _alleles(sampler, removed, analysis_variants) == {(ins1.variant_id, 0), (ins2.variant_id, 0)}

        # Restoring the insertion should recreate the true haplotype exactly
        assert set_variant_allele(graph, removed, ins1_paths, 1) == hap0

        added = set_variant_allele(graph, hap0, ins2_paths, 1)
        assert added is not None
        assert _is_walk(graph, added)
        assert _alleles(sampler, added, analysis_variants) == {(ins1.variant_id, 1), (ins2.variant_id, 1)}

    def test_true_haplotype_neighbors(self, graph_and_variants):
        graph, sampler, variants, analysis_variants = graph_and_variants
        ins1, ins2 = analysis_variants
        snv = next(v for v in variants if v not in analysis_variants)
        snv_alt_nodes = set(graph.path_nodes(f"_alt_{snv.variant_id}_1")) - set(graph.path_nodes(f"_alt_{snv.variant_id}_0"))

        haplotypes, alleles, true_idxs = true_haplotype_neighbors(
            graph, sampler, analysis_variants, self.REGION.contig, "HG00733", max_haplotypes=8
        )
        assert true_idxs == [1, 2]
        assert haplotypes[1] == graph.path_nodes(f"HG00733#0#{self.REGION.contig}#0")
        assert haplotypes[2] == graph.path_nodes(f"HG00733#1#{self.REGION.contig}#0")
        assert alleles == [
            {(ins1.variant_id, 0), (ins2.variant_id, 0)},
            {(ins1.variant_id, 1), (ins2.variant_id, 0)},
            {(ins1.variant_id, 0), (ins2.variant_id, 1)},
            {(ins1.variant_id, 1), (ins2.variant_id, 1)},
        ]
        for haplotype, haplotype_alleles_ in zip(haplotypes, alleles, strict=True):
            assert _is_walk(graph, haplotype)
            assert _alleles(sampler, haplotype, analysis_variants) == haplotype_alleles_
            # All haplotypes retain the sample's (homozygous) background SNV
            assert snv_alt_nodes <= set(haplotype)

    def test_true_haplotype_neighbors_exact_reference(self, graph_and_variants):
        graph, sampler, _variants, analysis_variants = graph_and_variants
        haplotypes, _alleles, true_idxs = true_haplotype_neighbors(
            graph, sampler, analysis_variants, self.REGION.contig, "HG00733", max_haplotypes=8, exact_reference=True
        )
        assert haplotypes[0] == graph.path_nodes(self.REGION.contig)
        assert true_idxs == [1, 2]
        assert len(haplotypes) == 4

    def test_true_haplotype_neighbors_max_haplotypes(self, graph_and_variants):
        graph, sampler, _variants, analysis_variants = graph_and_variants
        haplotypes, _alleles, true_idxs = true_haplotype_neighbors(
            graph, sampler, analysis_variants, self.REGION.contig, "HG00733", max_haplotypes=3
        )
        # The reference and true haplotypes are always included
        assert len(haplotypes) == 3
        assert true_idxs == [1, 2]

    def test_true_haplotype_neighbors_missing_true_haplotypes(self, graph_and_variants):
        graph, sampler, _variants, analysis_variants = graph_and_variants
        with pytest.raises(InsufficientHaplotypesError):
            true_haplotype_neighbors(graph, sampler, analysis_variants, self.REGION.contig, "OTHER", max_haplotypes=8)

    @pytest.mark.parametrize("zygosity", [KmerZygosity.ABSENT, KmerZygosity.HETEROZYGOUS, KmerZygosity.HOMOZYGOUS])
    def test_find_best_backbone_paths_matches_brute_force(self, graph_and_variants, zygosity):
        """The backbone-constrained DP finds the optimal haplotype for every combination of analysis-variant alleles"""
        graph, sampler, _variants, analysis_variants = graph_and_variants
        sampler.initialize_scores(ConstantKmerClassify(zygosity))
        variant_paths = [_variant_allele_paths(graph, variant) for variant in analysis_variants]

        for i in range(2):
            backbone = graph.path_nodes(f"HG00733#{i}#{self.REGION.contig}#0")

            # Brute-force: apply every combination of alleles to the backbone
            expected: dict[frozenset, float] = {}
            for combination in itertools.product(*(sorted(paths) for paths in variant_paths)):
                haplotype = backbone
                for paths, allele in zip(variant_paths, combination, strict=True):
                    haplotype = set_variant_allele(graph, haplotype, paths, allele)
                    assert haplotype is not None
                key = frozenset(_alleles(sampler, haplotype, analysis_variants))
                expected[key] = max(expected.get(key, float("-inf")), sampler.score(haplotype))
            assert len(expected) == 4

            found = sampler.find_best_backbone_paths(backbone, n=10)
            assert backbone in found
            found_scores = [sampler.score(path) for path in found]
            assert found_scores == sorted(found_scores, reverse=True)
            assert {frozenset(_alleles(sampler, path, analysis_variants)): score for path, score in zip(found, found_scores, strict=True)} == pytest.approx(expected)
            for path in found:
                assert _is_walk(graph, path)
                assert path[0] == backbone[0]
                assert path[-1] == backbone[-1]

            # Width limits the number of paths to the highest-scoring classes
            top = sampler.find_best_backbone_paths(backbone, n=2)
            assert [sampler.score(path) for path in top] == pytest.approx(sorted(expected.values(), reverse=True)[:2])

    def test_find_best_backbone_paths_invalid_backbone(self, graph_and_variants):
        graph, sampler, _variants, _analysis_variants = graph_and_variants
        backbone = graph.path_nodes(f"HG00733#0#{self.REGION.contig}#0")
        with pytest.raises(ValueError, match="Backbone"):
            sampler.find_best_backbone_paths(backbone[1:], n=2)


class TestNestedVariant:
    """An insertion nested within a (homozygous) deletion"""

    # cSpell:disable
    VCF = b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=chr1,length=248956422>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00733
chr1	1134740	.	CATCTGAGTCTGCAGCCCCTCTCAGGAGGGGGCACCGTGAGGGGTGTGGCTTCCTCGCCATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGACACCGTGAGG	C	30	PASS	.	GT	1|1
chr1	1134771	.	G	GCACCGTGAGGGGTGTGGCTTCCTCGCCATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGCACCGTGAGGGGTGTGGCTTCCTCGCCATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGA	30	PASS	.	GT	0|0"""  # fmt: skip
    # cSpell:enable

    REGION = Range("chr1:1134644-1134867")

    def test_find_best_backbone_paths_nested_variant(self, tmp_path):
        """Replacing the backbone's deletion with REF exposes the alleles of an SV nested within the deletion"""
        if not HG38_REF_FASTA or not os.path.exists(HG38_REF_FASTA):
            pytest.skip("Missing reference")
        vcf_path = create_vcf(tmp_path, self.VCF)
        graph, _unique_kmers, sampler = _create_graph_and_sampler(HG38_REF_FASTA, vcf_path, self.REGION, k=31)
        sampler.initialize_scores(ConstantKmerClassify(KmerZygosity.ABSENT))

        variants: list[Variant] = []
        with VariantFileReader.open(vcf_path) as vcf_file:
            variants = list(vcf_file.fetch(self.REGION))
        analysis_variants = _filter_variants(variants, min_variant_size=50)
        assert len(analysis_variants) == 2
        deletion, insertion = analysis_variants
        
        backbone = graph.path_nodes(f"HG00733#0#{self.REGION.contig}#0")
        found = sampler.find_best_backbone_paths(backbone, n=10)
        found_alleles = [_alleles(sampler, path, analysis_variants) for path in found]

        # The nested insertion can't be combined with the deletion (and the backbone's '*' allele for the insertion is
        # equivalent to REF once resolved).
        assert len(found) == 3
        assert {(deletion.variant_id, 0), (insertion.variant_id, 1)} in found_alleles
        assert {(deletion.variant_id, 0), (insertion.variant_id, 0)} in found_alleles
        assert all(_is_walk(graph, path) for path in found)
       

class TestHGSVC3HPRCPAVPopulationSimilarVariants:
    """A single analysis variant (a 53bp insertion, homozygous in HG00096) among many similar small variants from other
    population samples that are not on the HG00096 haplotypes, but that the sampler can pull into the haplotypes."""

    # cSpell:disable
    VCF = b"""##fileformat=VCFv4.2
##source=PAV 2.4.6.0
##FILTER=<ID=PASS,Description="All filters passed">
##INFO=<ID=ID,Number=1,Type=String,Description="Variant ID">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Variant type">
##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="Variant length">
##INFO=<ID=HAP,Number=.,Type=String,Description="List of haplotype names variant was identified in">
##INFO=<ID=HAP_VARIANTS,Number=.,Type=String,Description="List of variant IDs identifiying the variant merged in for each haplotype (INFO/HAP order)">
##INFO=<ID=COV_MEAN,Number=.,Type=String,Description="Mean coverage for each haplotype under the whole variant (not just breakpoints, INFO/HAP order)">
##INFO=<ID=COV_PROP,Number=.,Type=String,Description="Proportion of reference bases under the whole variant with at least one aligned query (assembly sequence) (INFO/HAP order)">
##INFO=<ID=QRY_REGION,Number=.,Type=String,Description="Region of the query (assembly sequence) where this variant was found (chrom:pos-end, 1-based, closed coordinates, not BED) (INFO/HAP order)">
##INFO=<ID=QRY_STRAND,Number=.,Type=String,Description="Orientation of the aligned query (assembly sequence) at this site (INFO/HAP order)">
##INFO=<ID=CALL_SOURCE,Number=.,Type=String,Description="How variant was called - CIGAR, ALIGN_TRUNC, etc (INFO/HAP order)">
##INFO=<ID=COMPOUND,Number=.,Type=String,Description="ID of the variant covering this event if FILTER is COMPOUND. May be multiple IDs. For all variants covering others, if they are removed, then all COMPOUND filtered variants matching those IDs might be safely recovered unless they were removed by other filters.">
##INFO=<ID=INNER_REF,Number=.,Type=String,Description="Inversion inner breakpoint in reference coordinates (INFO/HAP order)">
##INFO=<ID=INNER_TIG,Number=.,Type=String,Description="Inversion inner breakpoint in contig coordinates (INFO/HAP order)">
##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count in genotypes">
##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles in called genotypes">
##FILTER=<ID=QRY_FILTER,Description="Query filter region">
##FILTER=<ID=COMPOUND,Description="Inside larger variant">
##FILTER=<ID=SVLEN,Description="Variant size out of bounds">
##FILTER=<ID=TRIM,Description="Alignment trimming removed variant region">
##FILTER=<ID=CALLABLE,Description="Record masked by region">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=FT,Number=1,Type=String,Description="Genotype-level filter">
##ALT=<ID=INV,Description="Inversion">
##contig=<ID=chr4,length=190214555>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00096	HG00171	HG00268	HG00358
chr4	109850414	chr4-109850415-INS-18	C	CTTCTTTCTTTCTTTCTTT	0	PASS	ID=chr4-109850415-INS-18;SVTYPE=INS;SVLEN=18;HAP=h1;HAP_VARIANTS=chr4-109850415-INS-18;COV_MEAN=1.0;COV_PROP=1.0;QRY_REGION=haplotype1-0000012:80484396-80484413;QRY_STRAND=-;CALL_SOURCE=CIGAR;AC=1;AN=2	GT	.	1|0	.	.
chr4	109850415	chr4-109850415-SNV-CT	C	T	0	PASS	ID=chr4-109850415-SNV-CT;SVTYPE=SNV;HAP=h2;HAP_VARIANTS=chr4-109850415-SNV-CT;COV_MEAN=1.0;COV_PROP=1.0;QRY_REGION=haplotype2-0000062:59658386-59658386;QRY_STRAND=+;CALL_SOURCE=CIGAR;AC=1;AN=2	GT	.	.	0|1	.
chr4	109850418	chr4-109850419-INS-2	C	CTT	0	PASS	ID=chr4-109850419-INS-2;SVTYPE=INS;SVLEN=2;HAP=h1,h2;HAP_VARIANTS=chr4-109850419-INS-2,chr4-109850419-INS-2;COV_MEAN=1.0,1.0;COV_PROP=1.0,1.0;QRY_REGION=haplotype1-0000031:80295058-80295059,haplotype2-0000125:80694436-80694437;QRY_STRAND=-,-;CALL_SOURCE=CIGAR,CIGAR;AC=2;AN=2	GT	1|1	.	.	.
chr4	109850418	chr4-109850418-SNV-CT	C	T	0	PASS	ID=chr4-109850418-SNV-CT;SVTYPE=SNV;HAP=h1;HAP_VARIANTS=chr4-109850418-SNV-CT;COV_MEAN=1.0;COV_PROP=1.0;QRY_REGION=haplotype1-0000012:80484392-80484392;QRY_STRAND=-;CALL_SOURCE=CIGAR;AC=1;AN=2	GT	.	1|0	.	.
chr4	109850422	chr4-109850423-INS-35	C	CCTTCCTTCCTTCCTTCTTTCTTTCTTTCTTTCTTT	0	PASS	ID=chr4-109850423-INS-35;SVTYPE=INS;SVLEN=35;HAP=h2;HAP_VARIANTS=chr4-109850423-INS-35;COV_MEAN=1.0;COV_PROP=1.0;QRY_REGION=haplotype2-0000062:59658394-59658428;QRY_STRAND=+;CALL_SOURCE=CIGAR;AC=1;AN=2	GT	.	.	0|1	.
chr4	109850422	chr4-109850423-INS-26;chr4-109850422-SNV-CT	C	CCTTCCTTCTTCTTTCTTTCTTTCTTT,T	0	PASS	ID=chr4-109850423-INS-26;SVTYPE=INS;SVLEN=26;HAP=h2;HAP_VARIANTS=chr4-109850423-INS-26;COV_MEAN=1.0;COV_PROP=1.0;QRY_REGION=haplotype2-0000147:80269784-80269809;QRY_STRAND=-;CALL_SOURCE=CIGAR;AC=1,1;AN=2	GT	.	2|1	.	.
chr4	109850422	chr4-109850423-INS-11;chr4-109850423-INS-7	C	CCTTCTTTCTTT,CCTTCTTT	0	PASS	ID=chr4-109850423-INS-11;SVTYPE=INS;SVLEN=11;HAP=h1;HAP_VARIANTS=chr4-109850423-INS-11;COV_MEAN=1.0;COV_PROP=1.0;QRY_REGION=haplotype1-0000024:110818070-110818080;QRY_STRAND=+;CALL_SOURCE=CIGAR;AC=1,1;AN=2	GT	.	.	.	1|2
chr4	109850422	chr4-109850422-SNV-CT	C	T	0	PASS	ID=chr4-109850422-SNV-CT;SVTYPE=SNV;HAP=h1,h2;HAP_VARIANTS=chr4-109850422-SNV-CT,chr4-109850422-SNV-CT;COV_MEAN=1.0,1.0;COV_PROP=1.0,1.0;QRY_REGION=haplotype1-0000031:80295054-80295054,haplotype2-0000125:80694432-80694432;QRY_STRAND=-,-;CALL_SOURCE=CIGAR,CIGAR;AC=2;AN=2	GT	1|1	.	.	.
chr4	109850424	chr4-109850425-INS-15	T	TTCCTTCCTTCCTTCC	0	PASS	ID=chr4-109850425-INS-15;SVTYPE=INS;SVLEN=15;HAP=h1;HAP_VARIANTS=chr4-109850425-INS-15;COV_MEAN=1.0;COV_PROP=1.0;QRY_REGION=haplotype1-0000012:111550184-111550198;QRY_STRAND=+;CALL_SOURCE=CIGAR;AC=1;AN=2	GT	.	.	1|0	.
chr4	109850486	chr4-109850487-INS-52	T	TCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTCTTTTTTC	0	PASS	ID=chr4-109850487-INS-52;SVTYPE=INS;SVLEN=52;HAP=h1,h2;HAP_VARIANTS=chr4-109850487-INS-52,chr4-109850487-INS-52;COV_MEAN=1.0,1.0;COV_PROP=1.0,1.0;QRY_REGION=haplotype1-0000031:80294938-80294989,haplotype2-0000125:80694316-80694367;QRY_STRAND=-,-;CALL_SOURCE=CIGAR,CIGAR;AC=2;AN=2	GT	1|1	.	.	."""  # fmt: skip
    #cSpell:enable

    REGION = Range("chr4:109850391-109850582")
    SAMPLE = "HG00096"

    @pytest.fixture
    def graph_and_variants(self, tmp_path):
        if not HG38_REF_FASTA or not os.path.exists(HG38_REF_FASTA):
            pytest.skip("Missing reference")

        vcf_path = create_vcf(tmp_path, self.VCF)
        graph, _unique_kmers, sampler = _create_graph_and_sampler(
            HG38_REF_FASTA, vcf_path, self.REGION, k=31
        )

        variants: list[Variant] = []
        with VariantFileReader.open(vcf_path) as vcf_file:
            variants = list(vcf_file.fetch(self.REGION))
        analysis_variants = _filter_variants(variants, min_variant_size=50)
        assert len(analysis_variants) == 1
        assert len(variants) > len(analysis_variants)
        return graph, sampler, variants, analysis_variants

    def _backbone_nodes(self, graph: Graph, analysis_variants) -> set[int]:
        """Nodes on the sample's true haplotypes or the analysis variants' allele paths"""
        nodes = set()
        for i in range(2):
            nodes.update(graph.path_nodes(f"{self.SAMPLE}#{i}#{self.REGION.contig}#0"))
        for variant in analysis_variants:
            for path in _variant_allele_paths(graph, variant).values():
                nodes.update(path)
        return nodes

    @pytest.mark.parametrize("zygosity", [KmerZygosity.ABSENT, KmerZygosity.HETEROZYGOUS, KmerZygosity.HOMOZYGOUS])
    def test_sampled_haplotypes_include_other_variants(self, graph_and_variants, zygosity):
        """Unconstrained sampling pulls in variants not present in the sample's true haplotypes"""
        graph, sampler, _variants, analysis_variants = graph_and_variants
        sampler.initialize_scores(ConstantKmerClassify(zygosity))
        backbone_nodes = self._backbone_nodes(graph, analysis_variants)

        sampled = sampler.sample_haplotypes(n=8)
        assert any(set(haplotype) - backbone_nodes for haplotype in sampled)

    @pytest.mark.parametrize("zygosity", [KmerZygosity.ABSENT, KmerZygosity.HETEROZYGOUS, KmerZygosity.HOMOZYGOUS])
    def test_true_haplotype_neighbors(self, graph_and_variants, zygosity):
        """The neighbors differ from the sample's true haplotypes only at the analysis variant"""
        graph, sampler, _variants, analysis_variants = graph_and_variants
        sampler.initialize_scores(ConstantKmerClassify(zygosity))
        (insertion,) = analysis_variants
        true_path = graph.path_nodes(f"{self.SAMPLE}#0#{self.REGION.contig}#0")
        assert graph.path_nodes(f"{self.SAMPLE}#1#{self.REGION.contig}#0") == true_path

        haplotypes, alleles, true_idxs = true_haplotype_neighbors(
            graph, sampler, analysis_variants, self.REGION.contig, self.SAMPLE, max_haplotypes=8
        )
        # Homozygous ALT so the identical true haplotypes are only included once
        assert true_idxs == [1, 1]
        assert alleles == [{(insertion.variant_id, 0)}, {(insertion.variant_id, 1)}]
        assert haplotypes[1] == true_path
        # The reference haplotype is the true haplotype with the insertion removed (i.e., none of the other variants)
        assert set_variant_allele(graph, haplotypes[0], _variant_allele_paths(graph, insertion), 1) == true_path
        backbone_nodes = self._backbone_nodes(graph, analysis_variants)
        for haplotype in haplotypes:
            assert _is_walk(graph, haplotype)
            assert set(haplotype) <= backbone_nodes

    def test_true_haplotype_neighbors_exact_reference(self, graph_and_variants):
        graph, sampler, _variants, analysis_variants = graph_and_variants
        sampler.initialize_scores(ConstantKmerClassify(KmerZygosity.ABSENT))
        haplotypes, _alleles, true_idxs = true_haplotype_neighbors(
            graph, sampler, analysis_variants, self.REGION.contig, self.SAMPLE, max_haplotypes=8, exact_reference=True
        )
        assert haplotypes[0] == graph.path_nodes(self.REGION.contig)
        assert haplotypes[1] == graph.path_nodes(f"{self.SAMPLE}#0#{self.REGION.contig}#0")
        assert true_idxs == [1, 1]
        assert len(haplotypes) == 2
