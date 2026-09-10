import gc
import os
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from shlex import quote
from typing import ClassVar, cast

import hydra
import pandas as pd
import pytest
import ray
import webdataset as wds
from omegaconf import OmegaConf

from npsv3.graphs.graph import Graph
from npsv3.graphs.haplotype import (
    ConstantKmerClassify,
    HaplotypePriorOverlay,
    HaplotypeSamplerOverlay,
    KmerClassify,
    KmerCounts,
    KmerZygosity,
    UniqueKmersOverlay,
    _create_graph_and_sampler,
    _diplotypes_in_topk_shard,
    _sample_diplotypes_from_counts,
    add_population_haplotypes,
    diplotypes_in_topk,
    prepare_genotyping_haplotypes,
    sample_diplotypes,
    serialize_graph_and_unique_kmers,
)
from npsv3.util.range import Range
from npsv3.util.variant import Variant, VariantFileReader

from .. import (
    HG38_REF_FASTA,
    cache_filter_kmc_database,
    cache_graph_and_filter_kmc_database,
    create_vcf,
    data_path,
)


def _path_edits(path1: Sequence[int], path2: Sequence[int]) -> int:
    """Return the edit distance between node paths path1 and path2"""
    i, j = 0, 0
    distance = 0
    len1, len2 = len(path1), len(path2)
    while i < len1 and j < len2:
        if path1[i] == path2[j]:
            i += 1
            j += 1
        elif path1[i] < path2[j]:
            distance += 1
            i += 1
        else:
            distance += 1
            j += 1
    return distance + (len1 - i) + (len2 - j)

@dataclass(frozen=True)
class _MockVariant:
    """Minimal mock `Variant` class for testing"""

    variant_id: str
    num_alleles: int = 2

    def allele_length_change(self, allele_idx: int) -> int | None:
        return None if allele_idx == 0 else -50


class _MockGraph:
    """Minimal mock `Graph` class for testing."""
    def __init__(self, contig: str, reference_nodes: list[int], paths: dict[str, list[int]] | None = None):
        self.contig = contig
        self._reference_nodes = reference_nodes
        self._paths = dict(paths or {})

    def path_nodes(self, name: str) -> list[int]:
        if name == self.contig:
            return list(self._reference_nodes)
        return list(self._paths[name])

    def has_path(self, name: str) -> bool:
        return name == self.contig or name in self._paths

    def samples_including(self, nodes: Sequence[int]) -> list[str]:
        node_set = set(nodes)
        return sorted({
            name.split("#", 1)[0]
            for name, path in self._paths.items()
            if "#" in name and node_set & set(path)
        })


class _MockHaplotypeSampler:
    """Minimal mock `HaplotypeSamplerOverlay` class for testing."""
    def __init__(self, node_alleles: dict[int, tuple[str, int]], scores: dict[tuple[int, ...], float] | None = None):
        self._node_alleles = node_alleles
        self._scores = scores or {}

    def decode_haplotype(self, haplotype: list[int]) -> set[tuple[str, int]]:
        return {self._node_alleles[node] for node in haplotype if node in self._node_alleles}

    def score(self, haplotype: Sequence[int]) -> float:
        return self._scores.get(tuple(haplotype), 0.0)


class TestPrepareGenotypingHaplotypes:
    # Single bi-allelic variant with variant ID "v1"
    region = Range("chr1", 100, 200)
    variants: ClassVar = [cast(Variant, _MockVariant("v1"))]
    sampler = cast(HaplotypeSamplerOverlay, _MockHaplotypeSampler({2: ("v1", 0), 4: ("v1", 1)}))

    REF: ClassVar = [1, 2, 3]
    ALT: ClassVar = [1, 4, 3]

    def test_adds_missing_reference_haplotype(self):
        """The reference haplotype should be added at index 0 if it wasn't already sampled."""
        graph = cast(Graph, _MockGraph(self.region.contig, self.REF))  # No named "true" haplotype paths
        haplotypes, alleles, true_idxs = prepare_genotyping_haplotypes(
            graph, self.sampler, [self.ALT], self.variants, self.region.contig, "SAMPLE"
        )
        assert haplotypes == [self.REF, self.ALT]
        assert alleles == [{("v1", 0)}, {("v1", 1)}]
        assert true_idxs == []

    def test_moves_existing_reference_haplotype_to_index_0(self):
        """An already-sampled reference haplotype should be moved to index 0, not duplicated."""
        graph = cast(Graph, _MockGraph(self.region.contig, self.REF))
        haplotypes, alleles, _true_idxs = prepare_genotyping_haplotypes(
            graph, self.sampler, [self.ALT, self.REF], self.variants, self.region.contig, "SAMPLE"
        )
        assert haplotypes == [self.REF, self.ALT]
        assert alleles == [{("v1", 0)}, {("v1", 1)}]

    def test_reference_haplotype_already_first_is_untouched(self):
        """If the reference haplotype is already at index 0, it should remain there."""
        graph = cast(Graph, _MockGraph(self.region.contig, self.REF))
        haplotypes, _alleles, _true_idxs = prepare_genotyping_haplotypes(
            graph, self.sampler, [self.REF, self.ALT], self.variants, self.region.contig, "SAMPLE"
        )
        assert haplotypes == [self.REF, self.ALT]

    def test_true_haplotypes_reuse_matching_sampled_haplotype(self):
        """If a "true" haplotype path matches an already-sampled haplotype, its (possibly reordered)
        index should be reused rather than the haplotype being duplicated."""
        graph = cast(Graph, _MockGraph(
            self.region.contig, self.REF,
            paths={"SAMPLE#0#chr1#0": self.REF, "SAMPLE#1#chr1#0": self.ALT},
        ))
        haplotypes, _alleles, true_idxs = prepare_genotyping_haplotypes(
            graph, self.sampler, [self.ALT, self.REF], self.variants, self.region.contig, "SAMPLE"
        )
        assert haplotypes == [self.REF, self.ALT]
        assert true_idxs == [0, 1]

    def test_true_haplotypes_appended_when_not_sampled(self):
        """A homozygous sample should report homozygous halotype indices"""
        graph = cast(Graph, _MockGraph(
            self.region.contig, self.REF,
            paths={"SAMPLE#0#chr1#0": self.ALT, "SAMPLE#1#chr1#0": self.ALT},
        ))
        haplotypes, _alleles, true_idxs = prepare_genotyping_haplotypes(
            graph, self.sampler, [], self.variants, self.region.contig, "SAMPLE",
        )
        assert haplotypes == [self.REF, self.ALT]
        assert true_idxs == [1, 1]

    def test_true_heterozygous_haplotype(self):
        """A heterozygous sample should report heterozygous haplotype indices, but not duplicate haplotypes."""
        graph = cast(Graph, _MockGraph(self.region.contig, self.REF, paths={"SAMPLE#0#chr1#0": self.ALT, "SAMPLE#1#chr1#0": self.REF}))
        haplotypes, alleles, true_idxs = prepare_genotyping_haplotypes(
            graph, self.sampler, [], self.variants, self.region.contig, "SAMPLE",
        )
        assert haplotypes == [self.REF, self.ALT]
        assert alleles == [{("v1", 0)}, {("v1", 1)}]
        assert true_idxs == [1, 0]

    def test_missing_true_haplotype_path_is_skipped(self):
        """A ploidy index without a corresponding path in the graph (i.e. not fully resolved) should
        be silently skipped rather than raising an error."""
        graph = cast(Graph, _MockGraph(self.region.contig, self.REF, paths={"SAMPLE#1#chr1#0": self.ALT}))
        # No path for "SAMPLE#0#chr1#0"
        haplotypes, _alleles, true_idxs = prepare_genotyping_haplotypes(
            graph, self.sampler, [], self.variants, self.region.contig, "SAMPLE"
        )
        assert haplotypes == [self.REF, self.ALT]
        assert true_idxs == [None, 1]

    def test_no_true_haplotype_paths_present(self):
        """If none of the sample's "true" haplotype paths are in the graph, `true_haplotype_idxs` is
        empty and only the reference-haplotype invariant is enforced."""
        graph = cast(Graph, _MockGraph(self.region.contig, self.REF))  # No named paths at all
        haplotypes, _alleles, true_idxs = prepare_genotyping_haplotypes(
            graph, self.sampler, [self.ALT], self.variants, self.region.contig, "SAMPLE"
        )
        assert haplotypes == [self.REF, self.ALT]
        assert true_idxs == []

    def test_does_not_mutate_input_haplotypes_list(self):
        original = [self.ALT]
        prepare_genotyping_haplotypes(
            cast(Graph, _MockGraph(self.region.contig, self.REF)), self.sampler, original, self.variants, self.region.contig, "SAMPLE"
        )
        assert original == [self.ALT]


class TestAddPopulationHaplotypes:
    # Single bi-allelic variant "v1"; node 4 distinguishes its ALT allele from REF.
    contig = "chr1"
    REF: ClassVar = [1, 2, 3]
    ALT: ClassVar = [1, 4, 3]
    # A second variant "v2" and its ALT-distinguishing node 14, for multi-variant fixtures.
    REF2: ClassVar = [11, 12, 13]
    ALT2: ClassVar = [11, 14, 13]

    def _sampler(self, scores: dict[tuple[int, ...], float] | None = None) -> HaplotypeSamplerOverlay:
        return cast(HaplotypeSamplerOverlay, _MockHaplotypeSampler(
            {2: ("v1", 0), 4: ("v1", 1), 12: ("v2", 0), 14: ("v2", 1)}, scores=scores,
        ))

    def test_adds_missing_alt_allele_from_population_carrier(self):
        variants = [cast(Variant, _MockVariant("v1"))]
        graph = cast(Graph, _MockGraph(
            self.contig, self.REF,
            paths={
                "_alt_v1_0": self.REF, "_alt_v1_1": self.ALT,
                "OTHER#0#chr1#0": self.ALT, "OTHER#1#chr1#0": self.REF,
            },
        ))
        haplotypes, alleles = add_population_haplotypes(
            graph, self._sampler(), [self.REF], [{("v1", 0)}], variants, self.contig, { "SAMPLE" }, max_haplotypes=8,
        )
        assert haplotypes == [self.REF, self.ALT]
        assert alleles == [{("v1", 0)}, {("v1", 1)}]

    def test_does_not_duplicate_already_represented_allele(self):
        variants = [cast(Variant, _MockVariant("v1"))]
        graph = cast(Graph, _MockGraph(
            self.contig, self.REF,
            paths={"_alt_v1_0": self.REF, "_alt_v1_1": self.ALT, "OTHER#0#chr1#0": self.ALT},
        ))
        haplotypes, alleles = add_population_haplotypes(
            graph, self._sampler(), [self.REF, self.ALT], [{("v1", 0)}, {("v1", 1)}], variants,
            self.contig, { "SAMPLE" }, max_haplotypes=8,
        )
        assert haplotypes == [self.REF, self.ALT]

    def test_excludes_sample_itself_as_carrier(self):
        variants = [cast(Variant, _MockVariant("v1"))]
        graph = cast(Graph, _MockGraph(
            self.contig, self.REF,
            paths={"_alt_v1_0": self.REF, "_alt_v1_1": self.ALT, "SAMPLE#0#chr1#0": self.ALT},
        ))
        haplotypes, _alleles = add_population_haplotypes(
            graph, self._sampler(), [self.REF], [{("v1", 0)}], variants, self.contig, { "SAMPLE" }, max_haplotypes=8,
        )
        assert haplotypes == [self.REF]

    def test_returns_unchanged_when_no_carrier_exists(self):
        variants = [cast(Variant, _MockVariant("v1"))]
        graph = cast(Graph, _MockGraph(
            self.contig, self.REF,
            paths={"_alt_v1_0": self.REF, "_alt_v1_1": self.ALT},  # no individual paths embedded at all
        ))
        haplotypes, _alleles = add_population_haplotypes(
            graph, self._sampler(), [self.REF], [{("v1", 0)}], variants, self.contig, { "SAMPLE" }, max_haplotypes=8,
        )
        assert haplotypes == [self.REF]

    def test_multiallelic_variant_considers_each_allele_independently(self):
        variant = cast(Variant, _MockVariant("v1", num_alleles=3))
        alt_allele_2 = [1, 5, 3]
        sampler = cast(HaplotypeSamplerOverlay, _MockHaplotypeSampler({2: ("v1", 0), 4: ("v1", 1), 5: ("v1", 2)}))
        graph = cast(Graph, _MockGraph(
            self.contig, self.REF,
            paths={
                "_alt_v1_0": self.REF, "_alt_v1_1": self.ALT, "_alt_v1_2": alt_allele_2,
                "OTHER1#0#chr1#0": self.ALT, "OTHER2#0#chr1#0": alt_allele_2,
            },
        ))
        haplotypes, alleles = add_population_haplotypes(
            graph, sampler, [self.REF], [{("v1", 0)}], [variant], self.contig, { "SAMPLE" }, max_haplotypes=8,
        )
        assert haplotypes == [self.REF, self.ALT, alt_allele_2]
        assert alleles == [{("v1", 0)}, {("v1", 1)}, {("v1", 2)}]

    def test_does_not_mutate_input_lists(self):
        variants = [cast(Variant, _MockVariant("v1"))]
        graph = cast(Graph, _MockGraph(
            self.contig, self.REF,
            paths={"_alt_v1_0": self.REF, "_alt_v1_1": self.ALT, "OTHER#0#chr1#0": self.ALT},
        ))
        original_haplotypes = [self.REF]
        original_alleles = [{("v1", 0)}]
        add_population_haplotypes(
            graph, self._sampler(), original_haplotypes, original_alleles, variants, self.contig, { "SAMPLE" },
            max_haplotypes=8,
        )
        assert original_haplotypes == [self.REF]
        assert original_alleles == [{("v1", 0)}]

    def test_caps_at_max_haplotypes_keeping_highest_scoring_candidates(self):
        variant = cast(Variant, _MockVariant("v1", num_alleles=3))
        alt_allele_2 = [1, 5, 3]
        sampler = cast(HaplotypeSamplerOverlay, _MockHaplotypeSampler(
            {2: ("v1", 0), 4: ("v1", 1), 5: ("v1", 2)},
            scores={tuple(self.ALT): 1.0, tuple(alt_allele_2): 5.0},
        ))
        graph = cast(Graph, _MockGraph(
            self.contig, self.REF,
            paths={
                "_alt_v1_0": self.REF, "_alt_v1_1": self.ALT, "_alt_v1_2": alt_allele_2,
                "OTHER1#0#chr1#0": self.ALT, "OTHER2#0#chr1#0": alt_allele_2,
            },
        ))
        # Budget only allows one addition beyond REF -- the higher-scoring candidate (alt_allele_2) should
        # win over the lower-scoring one (ALT), regardless of discovery order.
        haplotypes, alleles = add_population_haplotypes(
            graph, sampler, [self.REF], [{("v1", 0)}], [variant], self.contig, { "SAMPLE" }, max_haplotypes=2,
        )
        assert haplotypes == [self.REF, alt_allele_2]
        assert alleles == [{("v1", 0)}, {("v1", 2)}]

    def test_same_candidate_found_via_two_targets_is_only_added_once(self):
        """A carrier whose single path covers ALT alleles of two different analysis variants is discovered
        independently while searching for each variant's missing allele, but must only be added once."""
        variants = [cast(Variant, _MockVariant("v1")), cast(Variant, _MockVariant("v2"))]
        sampler = self._sampler()
        ref_combined = self.REF + self.REF2
        multi = self.ALT + self.ALT2  # carries the ALT allele of both v1 and v2 on one path
        graph = cast(Graph, _MockGraph(
            self.contig, ref_combined,
            paths={
                "_alt_v1_0": self.REF, "_alt_v1_1": self.ALT,
                "_alt_v2_0": self.REF2, "_alt_v2_1": self.ALT2,
                "MULTI#0#chr1#0": multi,
            },
        ))
        haplotypes, alleles = add_population_haplotypes(
            graph, sampler, [ref_combined], [{("v1", 0), ("v2", 0)}], variants, self.contig, { "SAMPLE" },
            max_haplotypes=8,
        )
        assert haplotypes == [ref_combined, multi]
        assert alleles == [{("v1", 0), ("v2", 0)}, {("v1", 1), ("v2", 1)}]


@pytest.mark.skipif(not HG38_REF_FASTA, reason="HG38 reference FASTA not found")
class TestTopkHaplotypeSampling:
    def test_correct_diplotype_homalt(self, cfg, tmp_path):
        vcf_path = create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=chr12,length=133275309>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00096
chr12	21976631	.	CAGGGGCATACTGTGAAGAACTTGACCTCTAATTAATAGCTAAGGCCGATCCTAAGAGAGCCAATTGTGGGAGATTGTCAGCTACTATATTCCTCATAGCTGGGTAGAAAGCCCTCTTGAAGGAAGATCTGAGCAGTACATCTTAGTGTCTGTCACAGACACACAGAGCTTGGATGACTCAAAAAAAGAAAAAGAGAAATAATTCTTCTGATTCTAAATATGTAACCCTCATTCCCTGAGGCGCAGTACTTCAAATTTAAGAACAAAGTTATAAAAACAACTAGTTAAGAAAAAAAGATCTGTAATCCTACTTACTCCTCAAGCAATATAACCCCCAGAAGTTCTTCTCGAGTAAATTTATGAATATCCAGTGGGTGTCTCACAAGAGTTCTAATAACATGCTGTTGACTACCATCGGGGATTCTACCAATTTTCCTATCTCCTAATCTAGATCACTGGATAATGTGTCTAATTGCTCCTAAGTTAAGAGTGGTAGCTATGCCAAACCATTGGCAGTTTCACTTCCCAGACACTACTCCTGAGGATGCTACATAGCCCAAGACTGAGGGTTCTGACTTCTATTCAGGGGTTCTGATGTTTTATATCCAGAGAATACAAGGCACTGAAATCAGCATTTTATCATTTTATCAATAACACAACTCATCAACATTGCTAACATTCTGTCCCTGTGTCATCAATGTCATCACTTCTAAGAGGACTCAATGTCTCATGAAGGTTATAGAACAACAGCTTTTTGAGATTTTACTTACTTTTTTGTTGCAGCTTTCTTGCTCTCAGATTGAGAATGGCTGGTCTAATTGAT	C	30	PASS	.	GT	1|1"""
) # fmt: skip

        region = Range("chr12", 21976130, 21977953)

        with tempfile.TemporaryDirectory(dir=tmp_path) as kmc_dir:
            kmc_prefix = os.path.join(kmc_dir, "kmers")
            fasta_file = os.path.join(kmc_dir, "kmers.fa")

            # The unique k-mers, which occur at the end of the event, are not present in the true data.
            # We simulate that here with an empty KMC database to speed up the test.
            with open(fasta_file, "w") as f:
                for i, kmer in enumerate(["A" * cfg.graph.kmer_size]):
                    f.write(f">{i}\n{kmer}\n")
            subprocess.check_call(
                f"kmc -t1 -k{cfg.graph.kmer_size} -b -ci1 -fa {quote(fasta_file)} {quote(kmc_prefix)} {quote(kmc_dir)}",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            graph, haplotypes, diplotypes, *_ = sample_diplotypes(
                HG38_REF_FASTA, # type: ignore
                vcf_path,
                region,
                kmc_prefix,
                k=cfg.graph.kmer_size,
                kmer_coverage=29,
                min_variant_size=50,
                filter_kmers=False,
            )

            true_hap0_nodes = graph.path_nodes(f"HG00096#0#{region.contig}#0")
            true_hap1_nodes = graph.path_nodes(f"HG00096#1#{region.contig}#0")
            assert true_hap0_nodes == true_hap1_nodes, "Variant is hom. alt."

            assert len(haplotypes) == 2, "A bi-allelic variant should have two haplotypes"
            assert haplotypes[0] == true_hap0_nodes, "The true haplotype should be the top-ranked"

            assert len(diplotypes) >= 3, "A bi-allelic variant should have >=3 diplotypes"
            assert diplotypes[0].haplotypes == (0, 0), "Top rank is the true diplotype"
            assert diplotypes[1].haplotypes.count(0) >= 1, "The second-ranked diplotype should contain true haplotype"

    def test_correct_diplotype_het(self, cfg, tmp_path):
        vcf_path = create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=chr12,length=133275309>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00096
chr12	21976631	.	CAGGGGCATACTGTGAAGAACTTGACCTCTAATTAATAGCTAAGGCCGATCCTAAGAGAGCCAATTGTGGGAGATTGTCAGCTACTATATTCCTCATAGCTGGGTAGAAAGCCCTCTTGAAGGAAGATCTGAGCAGTACATCTTAGTGTCTGTCACAGACACACAGAGCTTGGATGACTCAAAAAAAGAAAAAGAGAAATAATTCTTCTGATTCTAAATATGTAACCCTCATTCCCTGAGGCGCAGTACTTCAAATTTAAGAACAAAGTTATAAAAACAACTAGTTAAGAAAAAAAGATCTGTAATCCTACTTACTCCTCAAGCAATATAACCCCCAGAAGTTCTTCTCGAGTAAATTTATGAATATCCAGTGGGTGTCTCACAAGAGTTCTAATAACATGCTGTTGACTACCATCGGGGATTCTACCAATTTTCCTATCTCCTAATCTAGATCACTGGATAATGTGTCTAATTGCTCCTAAGTTAAGAGTGGTAGCTATGCCAAACCATTGGCAGTTTCACTTCCCAGACACTACTCCTGAGGATGCTACATAGCCCAAGACTGAGGGTTCTGACTTCTATTCAGGGGTTCTGATGTTTTATATCCAGAGAATACAAGGCACTGAAATCAGCATTTTATCATTTTATCAATAACACAACTCATCAACATTGCTAACATTCTGTCCCTGTGTCATCAATGTCATCACTTCTAAGAGGACTCAATGTCTCATGAAGGTTATAGAACAACAGCTTTTTGAGATTTTACTTACTTTTTTGTTGCAGCTTTCTTGCTCTCAGATTGAGAATGGCTGGTCTAATTGAT	C	30	PASS	.	GT	1|1"""
) # fmt: skip

        region = Range("chr12", 21976130, 21978875)

        kmc_prefix = tmp_path / "kmers"
        fasta_file = tmp_path / "kmers.fa"

        # Create heterozygous counts (15) for the unique k-mers spanning the end of the event
        sequence ="TCTCAGATTGAGAATGGCTGGTCTAATTGATAGGGGCATACTGTGAAGAACTTGACCTCTA"
        kmers = [sequence[i:i + cfg.graph.kmer_size] for i in range(0, len(sequence) - cfg.graph.kmer_size + 1)]
        with open(fasta_file, "w") as f:
            for i, kmer in enumerate(kmers * 15):
                f.write(f">{i}\n{kmer}\n")
        subprocess.check_call(
            f"kmc -t1 -k{cfg.graph.kmer_size} -b -ci1 -fa {quote(str(fasta_file))} {quote(str(kmc_prefix))} {quote(str(tmp_path))}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        graph, haplotypes, diplotypes, sampler, *_ = sample_diplotypes(
            HG38_REF_FASTA, # type: ignore
            vcf_path,
            region,
            kmc_prefix,
            k=cfg.graph.kmer_size,
            kmer_coverage=29,
            min_variant_size=50,
            filter_kmers=False,
        )
        assert sampler.num_kmers() > 1, "There should be multiple k-mers to distinguish the haplotypes"

        true_hap0_idx = haplotypes.index(graph.path_nodes(f"HG00096#0#{region.contig}#0"))
        assert set(diplotypes[0].haplotypes) > { true_hap0_idx }, "The most likely diplotype should be het."

    @pytest.mark.usefixtures("ray_setup")
    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
    )
    def test_topk_genotype(self, cfg, hg00096_sample, tmp_path):
        vcf_path = create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=chr1,length=248956422>
##contig=<ID=chr12,length=133275309>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00096
chr12	21976631	.	CAGGGGCATACTGTGAAGAACTTGACCTCTAATTAATAGCTAAGGCCGATCCTAAGAGAGCCAATTGTGGGAGATTGTCAGCTACTATATTCCTCATAGCTGGGTAGAAAGCCCTCTTGAAGGAAGATCTGAGCAGTACATCTTAGTGTCTGTCACAGACACACAGAGCTTGGATGACTCAAAAAAAGAAAAAGAGAAATAATTCTTCTGATTCTAAATATGTAACCCTCATTCCCTGAGGCGCAGTACTTCAAATTTAAGAACAAAGTTATAAAAACAACTAGTTAAGAAAAAAAGATCTGTAATCCTACTTACTCCTCAAGCAATATAACCCCCAGAAGTTCTTCTCGAGTAAATTTATGAATATCCAGTGGGTGTCTCACAAGAGTTCTAATAACATGCTGTTGACTACCATCGGGGATTCTACCAATTTTCCTATCTCCTAATCTAGATCACTGGATAATGTGTCTAATTGCTCCTAAGTTAAGAGTGGTAGCTATGCCAAACCATTGGCAGTTTCACTTCCCAGACACTACTCCTGAGGATGCTACATAGCCCAAGACTGAGGGTTCTGACTTCTATTCAGGGGTTCTGATGTTTTATATCCAGAGAATACAAGGCACTGAAATCAGCATTTTATCATTTTATCAATAACACAACTCATCAACATTGCTAACATTCTGTCCCTGTGTCATCAATGTCATCACTTCTAAGAGGACTCAATGTCTCATGAAGGTTATAGAACAACAGCTTTTTGAGATTTTACTTACTTTTTTGTTGCAGCTTTCTTGCTCTCAGATTGAGAATGGCTGGTCTAATTGAT	C	30	PASS	.	GT	1|1"""
) # fmt: skip

        kmc_prefix = tmp_path / "kmers"
        fasta_file = tmp_path / "kmers.fa"

        # The unique k-mers, which occur at the end of the event, are not present in the true data.
        # We simulate that here with a synthetic KMC database containing only off-target k-mers.
        with open(fasta_file, "w") as f:
            for i, kmer in enumerate(["A" * cfg.graph.kmer_size]):
                f.write(f">{i}\n{kmer}\n")
        subprocess.check_call(
            f"kmc -t1 -k{cfg.graph.kmer_size} -b -ci1 -fa {quote(str(fasta_file))} {quote(str(kmc_prefix))} {quote(str(tmp_path))}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # # Although not used here, to filter the actual data:
        # kmc_prefix_path = _first_existing(
        #     "/data/HG00096.final.kmc.kmc_pre",
        # )
        # if kmc_prefix_path is None:
        #     pytest.skip("Example KMC database not found")
        # kmc_prefix = os.path.splitext(kmc_prefix_path)[0]

        hg00096_sample.kmc_prefix = kmc_prefix

        statistics = diplotypes_in_topk(cfg, vcf_path, hg00096_sample, filtered_kmer_path=kmc_prefix)
        pd.testing.assert_frame_equal(
            statistics,
            pd.DataFrame({
                "region": [str(Range("chr12", 21976631, 21977453).expand(cfg.pileup.variant_padding))],
                "variant": ["fe58cd4ae772afe360ddf77af9ff2297f4b2e809"],
                "sample": ["HG00096"],
                "haplotypes": [2],
                "haplotype_idxs": [(0, 0)],
                "diplotypes": [3],
                "diplotype_idx": [0],
                "all_haplotype_idxs": [(0, 0)],
                "all_diplotype_idx": [0],
                "true_haplotype_idxs": [(0, 0)],
                "true_diplotype_idx": [0],
            }),
        )

    @pytest.mark.usefixtures("ray_setup")
    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
        "graph.kmer_size=31",  # Test files were generated with k=31, so we need to use that here to get the expected results
    )
    def test_topk_genotype_multiallelic(self, cfg, hg00096_sample, tmp_path):
        vcf_path = create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=chr1,length=248956422>
##contig=<ID=chr12,length=133275309>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00096
chr1	1924223	.	G	GACCACCCCCCAGCTCACAGCCCACCCCCCCATCTCACCGCCCAGCCCCCCCATCTCACCAGCTGCCCCCTCCCGGGCACACCGCCCACCCCCCCATCTCACCA,GACCACCCCCCAGCTCACAGCCCACCCCCCCATCTCACCGCCCAGCCCCCCCATCTCACCAGCTGCCCCCTCCCCGACACACCGCCCACCCCCCCATCTCACCA	30	PASS	.	GT	1|2"""
        ) # fmt: skip

        # Pre-intersected set of k-mers in this region for testing
        kmc_prefix = tmp_path / "kmers"
        fasta_file = tmp_path / "kmers.fa"
        with open(data_path("chr1_1924223_1924223.kmc.hist")) as kmer_hist, open(fasta_file, "w") as f:
            fasta_row = 0
            for line in kmer_hist:
                kmer, count = line.strip().split()
                for _ in range(int(count)):
                    f.write(f">{fasta_row}\n{kmer}\n")
                    fasta_row += 1
        subprocess.check_call(
            f"kmc -t1 -k{cfg.graph.kmer_size} -b -ci1 -fa {quote(str(fasta_file))} {quote(str(kmc_prefix))} {quote(str(tmp_path))}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        hg00096_sample.kmc_prefix = kmc_prefix

        statistics = diplotypes_in_topk(cfg, vcf_path, hg00096_sample, filtered_kmer_path=kmc_prefix)
        assert len(statistics) == 1, "There should be one row of statistics for the single variant"
        assert all(h > 0 for h in statistics.iloc[0]["haplotype_idxs"]), "The true haplotypes should be found, but not necessarily top-ranked"
        assert statistics.iloc[0]["diplotype_idx"] >= 0, "The true diplotype should be found, but not necessarily top-ranked"

@pytest.mark.skipif(not HG38_REF_FASTA, reason="HG38 reference FASTA not found")
class TestHaplotypeSamplerOverlayPopulationPrior:
    """Test basic application of population prior"""

    def _build_graph(self, tmp_path):
        vcf_path = create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3	Sample4	Sample5	Sample6	Sample7
chr1	1000000	.	G	A	100	PASS	.	GT	0|0	0|0	0|0	1|1	1|1	1|1	0|1
chr1	1000001	.	G	C,T	100	PASS	.	GT	0|0	0|0	0|0	1|1	1|1	1|1	1|0"""
        )  # fmt: skip
        region = Range("chr1", 999989, 1000010)
        return Graph(HG38_REF_FASTA, vcf_path, region)  # type: ignore

    @pytest.mark.cfg_overrides(
        "graph.haplotype_sampler_params.haplotype_prior_weight=5.0",
        "graph.haplotype_sampler_params.panel_fallback_penalty=2.0",
    )
    def test_accepts_population_prior_kwargs_and_changes_ranking(self, cfg, tmp_path):
        graph = self._build_graph(tmp_path)
        unique_kmers = UniqueKmersOverlay(graph, cfg.graph.kmer_size, max_edges=5)
        # Every graph-unique k-mer uniformly ABSENT, so the k-mer term is a fixed function of the graph
        # rather than of any query reads. The population-prior term is the only source of ranking differences.
        counts = ConstantKmerClassify(KmerZygosity.ABSENT)

        prior_overlay = HaplotypePriorOverlay(graph)

        mixed_haplotype = graph.path_nodes("Sample7#0#chr1#0")  # r1, a2: rare (1/7) crossed combination
        pure_ref = graph.path_nodes("Sample1#0#chr1#0")  # r1, r2: common (6/7-ish) combination
        pure_alt = graph.path_nodes("Sample4#0#chr1#0")  # a1, a2: common (6/7-ish) combination

        def top_haplotypes(prior=None, params=None):
            sampler = HaplotypeSamplerOverlay(graph, unique_kmers, prior=prior, params=params)
            sampler.initialize_scores(counts)
            return sampler.find_best_paths(4)

        disabled = top_haplotypes()
        enabled = top_haplotypes(prior=prior_overlay, params=hydra.utils.instantiate(cfg.graph.haplotype_sampler_params))

        assert disabled != enabled, "A nonzero population prior should change the ranked haplotype set"
        assert mixed_haplotype in disabled, "Sanity check: the rare/crossed haplotype is reachable at all"
        assert mixed_haplotype not in enabled, "A strong population prior should disfavor the rare/crossed combination"
        assert all(nodes in enabled for nodes in (pure_ref, pure_alt)), "The panel-favored combinations should survive"

    @pytest.mark.cfg_overrides(
        "graph.haplotype_sampler_params.haplotype_prior_weight=5.0",
        "graph.haplotype_sampler_params.panel_fallback_penalty=2.0",
    )
    def test_overlay_outlives_python_reference(self, cfg, tmp_path):
        """Regression test for the nb::keep_alive binding design: the sampler must keep the overlay objects
        it was constructed with alive even after every other Python reference to them is dropped."""
        graph = self._build_graph(tmp_path)
        unique_kmers = UniqueKmersOverlay(graph, cfg.graph.kmer_size, max_edges=5)
        counts = ConstantKmerClassify(KmerZygosity.ABSENT)

        prior_overlay = HaplotypePriorOverlay(graph)

        sampler = HaplotypeSamplerOverlay(
            graph, unique_kmers, prior=prior_overlay, params=hydra.utils.instantiate(cfg.graph.haplotype_sampler_params)
        )
        del prior_overlay
        gc.collect()

        # Must not crash/UB despite the caller's own references being gone -- keep_alive should have tied
        # the overlays' lifetimes to the sampler's.
        sampler.initialize_scores(counts)
        haplotypes = sampler.find_best_paths(4)
        assert len(haplotypes) > 0
        for haplotype in haplotypes:
            sampler.score(haplotype)


@pytest.mark.skipif(not HG38_REF_FASTA, reason="HG38 reference FASTA not found")
@pytest.mark.usefixtures("ray_setup")
class TestPopulationPriorShardSchema:
    """Shard-schema opt-in behavior for serialize_graph_and_unique_kmers/_diplotypes_in_topk_shard."""

    VCF_BYTES = b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3	Sample4	Sample5	Sample6	Sample7
chr1	1000000	.	G	A	100	PASS	.	GT	0|0	0|0	0|0	1|1	1|1	1|1	0|1
chr1	1000001	.	G	C,T	100	PASS	.	GT	0|0	0|0	0|0	1|1	1|1	1|1	1|0"""  # fmt: skip
    REGION = Range("chr1", 999989, 1000010)

    @pytest.mark.cfg_overrides(f"reference={HG38_REF_FASTA}")
    def test_shard_keys_present_only_when_enabled(self, cfg, tmp_path):
        vcf_path = create_vcf(tmp_path, self.VCF_BYTES)

        for population_prior in (False, True):
            local_cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"graph.population_prior={population_prior}"]))
            output_dir = tmp_path / f"shards-{population_prior}"
            output_dir.mkdir()
            graph_shards, _unique_kmer_path, region_count = serialize_graph_and_unique_kmers(
                local_cfg,
                vcf_path,
                output_dir=str(output_dir),
                pool_kmers=False,
                region=self.REGION,
                min_variant_size=0,
            )
            assert region_count == 1
            records = list(wds.WebDataset(graph_shards, shardshuffle=False)) # type: ignore
            assert len(records) == 1
            record = records[0]
            if population_prior:
                assert len(record["haplotype_prior_overlay.bytes"]) > 0
            else:
                assert "haplotype_prior_overlay.bytes" not in record

    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
        "graph.population_prior=false",
        "graph.max_haplotypes=4",
        "graph.max_diplotypes=6",
    )
    def test_diplotypes_in_topk_shard_handles_shard_without_population_prior_keys(self, cfg, tmp_path):
        vcf_path = create_vcf(tmp_path, self.VCF_BYTES)

        output_dir = tmp_path / "shards"
        output_dir.mkdir()
        # With population_prior as False this shard shouldn't have any population-related keys
        graph_shards, _unique_kmer_path, _region_count = serialize_graph_and_unique_kmers(
            cfg, vcf_path, output_dir=str(output_dir), pool_kmers=False, region=self.REGION, min_variant_size=0,
        )

        kmc_dir = tmp_path / "kmc"
        kmc_dir.mkdir()
        fasta_file = kmc_dir / "kmers.fa"
        fasta_file.write_text(f">0\n{'A' * cfg.graph.kmer_size}\n")
        kmc_prefix = kmc_dir / "kmers"
        subprocess.check_call(
            f"kmc -t1 -k{cfg.graph.kmer_size} -b -ci1 -fa {quote(str(fasta_file))} {quote(str(kmc_prefix))} {quote(str(kmc_dir))}",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        # A shard that has no population-prior overlay keys must degrade gracefully (the overlay resolves to
        # None and the prior is inert) rather than raising KeyError.
        rows = ray.get(_diplotypes_in_topk_shard.remote(  # type: ignore
            cfg, graph_shards[0], vcf_path, "Sample1", str(kmc_prefix),
            kmer_coverage=29, min_variant_size=0,
        ))
        assert isinstance(rows, list)

    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
        "graph.population_prior=false",
        "graph.max_haplotypes=4",
        "graph.max_diplotypes=6",
    )
    def test_diplotypes_in_topk_shard_skips_filtered_genotype(self, cfg, tmp_path):
        # Sample1's genotype is explicitly filtered (FT != PASS/.) at the first variant only
        vcf_bytes = b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FILTER=<ID=LowQual,Description="Low quality">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=FT,Number=1,Type=String,Description="Genotype-level filter.">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3	Sample4	Sample5	Sample6	Sample7
chr1	1000000	.	G	A	100	PASS	.	GT:FT	0|1:LowQual	0|0:PASS	0|0:PASS	1|1:PASS	1|1:PASS	1|1:PASS	0|1:PASS
chr1	1000001	.	G	C,T	100	PASS	.	GT:FT	0|1:PASS	0|0:PASS	0|0:PASS	1|1:PASS	1|1:PASS	1|1:PASS	1|0:PASS"""  # fmt: skip
        vcf_path = create_vcf(tmp_path, vcf_bytes)

        output_dir = tmp_path / "shards"
        output_dir.mkdir()
        graph_shards, _unique_kmer_path, _region_count = serialize_graph_and_unique_kmers(
            cfg, vcf_path, output_dir=str(output_dir), pool_kmers=False, region=self.REGION, min_variant_size=0,
        )

        kmc_dir = tmp_path / "kmc"
        kmc_dir.mkdir()
        fasta_file = kmc_dir / "kmers.fa"
        fasta_file.write_text(f">0\n{'A' * cfg.graph.kmer_size}\n")
        kmc_prefix = kmc_dir / "kmers"
        subprocess.check_call(
            f"kmc -t1 -k{cfg.graph.kmer_size} -b -ci1 -fa {quote(str(fasta_file))} {quote(str(kmc_prefix))} {quote(str(kmc_dir))}",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        rows = ray.get(_diplotypes_in_topk_shard.remote(  # type: ignore
            cfg, graph_shards[0], vcf_path, "Sample1", str(kmc_prefix),
            kmer_coverage=29, min_variant_size=0,
        ))

        reported_variant_ids = {row["variant"] for row in rows}
        with VariantFileReader.open(vcf_path) as vcf_file:
            filtered_variant, passing_variant = list(vcf_file.fetch(self.REGION))
            assert filtered_variant.variant_id not in reported_variant_ids
            assert passing_variant.variant_id in reported_variant_ids

@pytest.mark.skipif(not HG38_REF_FASTA, reason="HG38 reference FASTA not found")
class TestTopkHaplotypeSamplingInHG00733:
    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
        "input=/storage/mlinderman/projects/sv/npsv3-experiments/resources/hgsvc3-hprc-2024-02-23.dipcall.population.passing.eval.hg38.vcf.gz",
        "graph.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
    )
    @pytest.mark.parametrize("region_str", ["chr1:148531170-148577610"])
    def test_graph_construction_errors(self, cfg, region_str):
        if not all(os.path.exists(f) for f in (cfg.reference, cfg.input, f"{cfg.graph.ref_kmer_counts_kmc_prefix}.kmc_pre")):
            pytest.skip("Missing necessary inputs")

        region = Range(region_str)
        ref_kmer_counts = KmerCounts(cfg.graph.ref_kmer_counts_kmc_prefix)
        _create_graph_and_sampler(
            cfg.reference,
            cfg.input,
            region,
            k=cfg.graph.kmer_size,
            ref_kmer_counts=ref_kmer_counts,
        )

    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
        "input=/storage/mlinderman/projects/sv/npsv3-experiments/resources/HG00733.hgsvc3-hprc-2024-02-23.dipcall.passing.hg38.vcf.gz",
        "graph.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
    )
    def test_downstream_small_del(self, cfg, hg00733_sample, tmp_path):
        if not all(os.path.exists(f) for f in (cfg.reference, cfg.input, f"{cfg.graph.ref_kmer_counts_kmc_prefix}.kmc_pre")):
            pytest.skip("Missing necessary inputs")

        ref_kmer_counts = KmerCounts(cfg.graph.ref_kmer_counts_kmc_prefix)

        region = Range("chr1:789386-789670")
        graph, unique_kmers, haplotype_sampler = _create_graph_and_sampler(
            cfg.reference,
            cfg.input,
            region,
            k=cfg.graph.kmer_size,
            ref_kmer_counts=ref_kmer_counts,
        )

        # Unique k-mers with out downstream DEL (which all span the breakpoint in the reference, i.e., not alt k-mers)
        # AATGGAATGGACTCCAATGGAATGTGGTG  chr1:789478-789506
        # ACTGGAATGGAATGGAATGGACTCCAATG  chr1:789468-789496
        # ATGGAATGGACTCCAATGGAATGTGGTGG  chr1:789479-789507
        # ATGGACTGGAATGGAATGGAATGGACTCC  chr1:789464-789492
        # GGAATGGACTCCAATGGAATGTGGTGGGA  chr1:789481-789509
        # GGACTGGAATGGAATGGAATGGACTCCAA  chr1:789466-789494
        # TGGACTGGAATGGAATGGAATGGACTCCA  chr1:789465-789493

        # TGGACTGGAATGGAATGGAATGGACTCCA has a count of 13, and thus classified as hom. biasing towards the reference haplotype. Three
        # other k-mers also have counts classified as het. Since 4 het+ kmers, and ony 3 absent, the most likely diplotype will be het.
        # That k-mers seems to originate elsewhere, e.g., chr9, and thus are off-target here.

        # Using the non-SV input VCF includes a downstream 5bp deletion, reported as 1|1. It induces an additional unique k-mer in the
        # reference genome spanning the event, AATGGAATGGAATGGAATGAAATGGACTA, with a count of 31 (thus classified as hom. (10.9,35)).
        # With that k-mer as hom. alt. the most likely haplotype will incorrectly include the reference allele for that deletion. The
        # hits are mapped to chr4, chr10 (mostly), chr20, etc. The count is not quite high enough to be ignored. The same issue is
        # observed with k=31.

        # Possible mitigations:
        # 1. Don't use a single k-mer to distinguish a haplotype, but require multiple k-mers to be present.
        # 2. Require uniqueness to just not be exact matches elsewhere in the genome, but "fuzzy" matches within some number of edits.
        # 2. Identify/implement a background model of off-target k-mer counts in the SRS data to have k-mer specific
        # thresholds for different zygosity classifications.

        # Use cached files to speed up repeated runs of the test
        filtered_kmer_path = cache_filter_kmc_database(cfg, hg00733_sample, cfg.input, region, unique_kmers, tmp_path=tmp_path)

        haplotypes, diplotypes = _sample_diplotypes_from_counts(
            haplotype_sampler,
            unique_kmers,
            filtered_kmer_path,
            k=cfg.graph.kmer_size,
            kmer_coverage=hg00733_sample.kmer_coverage,
            filter_kmers=False,
        )

    @pytest.mark.usefixtures("ray_setup")
    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
        "input=/storage/mlinderman/projects/sv/npsv3-experiments/resources/HG00733.hgsvc3-hprc-2024-02-23.dipcall.passing.hg38.vcf.gz",
        "graph.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
    )
    def test_multi_variant_with_star_alleles(self, cfg, hg00733_sample):
        """Test region with multiple variants and star alleles"""
        if not all(os.path.exists(f) for f in (cfg.reference, cfg.input, f"{cfg.graph.ref_kmer_counts_kmc_prefix}.kmc_pre")):
            pytest.skip("Missing necessary inputs")

        region = Range("chr7:104833216-104833994")

        # Use cached files to speed up repeated runs of the test
        graph_shards, filtered_kmer_path = cache_graph_and_filter_kmc_database(cfg, hg00733_sample, cfg.input, region)
        statistics = diplotypes_in_topk(cfg, cfg.input, hg00733_sample, region=region, graph_shards=graph_shards, filtered_kmer_path=filtered_kmer_path)

        # The first 3 haplotypes: [[1, 2, 6, 7, 9, 13, 14, 16], [1, 3, 5, 6, 7, 9, 13, 14, 16], [1, 3, 4, 7, 8, 11, 13, 14, 16]
        # and have identical scores and thus reported in an implementation-defined order.

        # The first variant is 0/1, but overlaps the second (genotype of 1/2 with a star allele), so we correctly
        # sample haplotypes that don't include the full reference allele for the first variant (nodes [3,5]), just node [3].
        # But node [3] is sufficient to uniquely identify that we didn't call the alternate allele (node [2]) for the
        # first variant, so we want to recognize that a haplotype of [..., 3, 4, ...] is a correct match for the reference
        # allele of the first variant. But with the star allele we want to recognize that a haplotype of [..., 2, 6, ...]
        # is compatible with the star allele, even though the haplotype is not disjoint with reference nodes [5, 6] for the
        # second variant, i.e., [..., 2, 6, ...] is "labeled" with both reference and star allele for variant 2.

        assert len(statistics) == 3, "There should be 3 'inference' variants in the region"
        assert all(statistics["haplotypes"] == 6), "With overlapping variants, there should be 6 haplotypes in the region"

        # Since there are haplotypes with identical scores, and implementation-defined tiebreaking, we assert on
        # the expected zygosity, not the specific haplotype indices.
        assert all(idx != -1 for idxs in statistics["haplotype_idxs"] for idx in idxs), \
            "Every allele should have a compatible sampled haplotype"
        distinct_pattern = [idxs[0] != idxs[1] for idxs in statistics["haplotype_idxs"]]
        assert distinct_pattern == [True, True, False], (
            "Variants 1/2 (heterozygous) should match two distinct haplotypes; variant 3 (homozygous) the same one"
        )
        assert all((statistics["all_diplotype_idx"] == -1) | (statistics["diplotype_idx"] <= statistics["all_diplotype_idx"]))

    @pytest.mark.usefixtures("ray_setup")
    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
        "input=/storage/mlinderman/projects/sv/npsv3-experiments/resources/HG00733.hgsvc3-hprc-2024-02-23.dipcall.passing.hg38.vcf.gz",
        "graph.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
    )
    def test_unexpected_all_diplotype_idx(self, cfg, hg00733_sample):
        if not all(os.path.exists(f) for f in (cfg.reference, cfg.input, f"{cfg.graph.ref_kmer_counts_kmc_prefix}.kmc_pre")):
            pytest.skip("Missing necessary inputs")

        region = Range("chr9:91796648-91799004")

        # Use cached files to speed up repeated runs of the test
        graph_shards, filtered_kmer_path = cache_graph_and_filter_kmc_database(cfg, hg00733_sample, cfg.input, region)
        statistics = diplotypes_in_topk(cfg, cfg.input, hg00733_sample, region=region, graph_shards=graph_shards, filtered_kmer_path=filtered_kmer_path)
        assert all((statistics["all_diplotype_idx"] == -1) | (statistics["diplotype_idx"] <= statistics["all_diplotype_idx"]))

    @pytest.mark.usefixtures("ray_setup")
    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
        "input=/storage/mlinderman/projects/sv/npsv3-experiments/resources/HG00733.hgsvc3-hprc-2024-02-23.dipcall.passing.hg38.vcf.gz",
        "graph.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
    )
    def test_missing_matching_halotypes(self, cfg, hg00733_sample):
        if not all(os.path.exists(f) for f in (cfg.reference, cfg.input, f"{cfg.graph.ref_kmer_counts_kmc_prefix}.kmc_pre")):
            pytest.skip("Missing necessary inputs")

        region = Range("chr1:6006213-6006792")

        # Use cached files to speed up repeated runs of the test
        graph_shards, filtered_kmer_path = cache_graph_and_filter_kmc_database(cfg, hg00733_sample, cfg.input, region)
        statistics = diplotypes_in_topk(cfg, cfg.input, hg00733_sample, region=region, graph_shards=graph_shards, filtered_kmer_path=filtered_kmer_path)
        assert len(statistics) == 0, "There should be no fully genotyped analysis variants in this region"
        # TODO: Test that an info message was logged about this region

    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
        "input=/storage/mlinderman/projects/sv/npsv3-experiments/resources/HG00733.hgsvc3-hprc-2024-02-23.dipcall.passing.hg38.vcf.gz",
        "graph.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
    )
    def test_no_haplotypes_sampled(self, cfg, hg00733_sample, tmp_path):
        if not all(os.path.exists(f) for f in (cfg.reference, cfg.input, f"{cfg.graph.ref_kmer_counts_kmc_prefix}.kmc_pre")):
            pytest.skip("Missing necessary inputs")

        ref_kmer_counts = KmerCounts(cfg.graph.ref_kmer_counts_kmc_prefix)

        region = Range("chr6:32249973-32250644")
        _graph, unique_kmers, haplotype_sampler = _create_graph_and_sampler(
            cfg.reference,
            cfg.input,
            region,
            k=cfg.graph.kmer_size,
            ref_kmer_counts=ref_kmer_counts,
        )

        filtered_kmer_path = cache_filter_kmc_database(cfg, hg00733_sample, cfg.input, region, unique_kmers, tmp_path=tmp_path)
        haplotypes, _diplotypes = _sample_diplotypes_from_counts(
            haplotype_sampler,
            unique_kmers,
            filtered_kmer_path,
            k=cfg.graph.kmer_size,
            kmer_coverage=hg00733_sample.kmer_coverage,
            filter_kmers=False,
        )
        assert len(haplotypes) > 0

    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
    )
    def test_overlapping_alleles(self, cfg, hg00733_sample, tmp_path):
        """Test region with overlapping variants, not all of genotyped in this sample"""
        # cSpell:disable
        vcf_path = create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=chr1,length=248956422>
##contig=<ID=chr12,length=133275309>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00733
chr1	789481	.	G	GGAATGGAATGCAATGGAATGCACTCGAACGGATTGGAATGGAATGGACTCGAATAGAATGGAATAAAATGAAATGGACTCCAATGGATTGGAATGGAATTGACTCCAATGGAATTGAATGGAGTGGAACCGAATGGAACGGATTGGAATGGAATGCACTCGAAATGAATTTGAATGGAATGGATTGGGCTCAAATGGAATGGAATGGAATGGAATGGAATGGAATGAACTCAAATGGATTAGCATGGAATGAAGTGGACTCGAATACAATGGAATGGAATGGACTCGAATGGAATGGAACGGACTTGAACGGAATGGAGTGGAATGGACTCGAATGGAATGGAGTTGAATGGACTCGAATGGAATGGAATGTAAAGGAATGGAATGAACTCGAAAGGAGTGGAATGTAATGGAATGAAATGGACTCGAATGGAATTAAATGGAATGGAACGGAATGGACTGGGATGGAATGGAACGGAACGGAACGCAGTTGAATTGAACGGACCCGGAATGGAATGGAATGGAATGGAATGAAATGGAATGAAGTGGACTCTAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATAGACTCGAATGAAATGGGATGGACTCGAATGGAATGGAACGGAATGGAATCGACTCGAGTGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATAGAATGGAATTGACTCAATTTAAATGGAATGAAGTGGAATGAACTCGAATGGCATGCAATGGAATGGAATAGAATCGAATGGAATAGAATGGACCCAAATGGAATGGAACGGAATGGAATGGAATGGAACGGAATGGAATAGAACGGAACGGAATGGAATGGATTGGAATGAACTCCAACGGAATGGAATGGACTCGAATGCAATGGAATGGAATGGAATGGAATGGAGTGGACTGGAATGGAATAGAATGGAATGGAATGGATAGGACTGGAATGAAATGGAATGGAATGGACTCGAATGGAATGGAATGGAATGGAATGGACTCAAATGGAATGGAATGGAATGGAATGGACACGAATGGAATGGAATTGAATGGAATGGAATGGACTGTATGAAAAGGAATGGATTGGAAAGGAATGGAATAGAACGGAATGGACTCGAATGGAATGGAAAGGACTCGAGTGGAATGGAATGGAATGGAATGGACTCGAATGGAATGGAGTGGAATGTATGCGAATGGAATGGAATTGAATGGATTCGAGTCTAACGGAATGTATGGAATGGACTCGAATGGAATGTAATGTAATGGAATGAAATGGACGCGAATGGAATGGAATGGAATGGAATGGAATGGAGTGGAATGGAATGGACTCGAATGGTATGGAATGGAATTGAATGGACTCGATAGGAATGGAATGGAATGGATTGGACTCGAAAGGAATGTAATGGAATGAAATGTGCTGGAATGGAATGGAATGGAATGGAATAAAATGTAATGGAATGGACTCGAATGGAATACAGTTGAATTGAATGGACCCGAAAGCAATGGAATGGAATGGAACGGATTGGAAGGGAATGGAATGGAATGAAATGGAAAAGACTCGAATGGAATGGAATGGACTCGAATGAAATGGAGTGGACTAGAATGGAATGGAATGGACTTGAAAGAAATGGAATGCAGTGGAATGGACTCGAATGGAATGCAATGGAATGGAATAGACTCGAACGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAACGGAATGGACGCGAATGAAATGGAACGGAACGGAATGGACTCGGATGGAACGGAATGGAACGGAATGGAATGGAATTTACTCGAATGGGATGGAATGGAATGGAATTTACTCGAATGGAATGGAATGGAATGGACTGAAATGGAATAGCATGGAATGGAATGGACTCGAATGCAATGGAATGGAATGGACTCGAATGGAACGGAATGGACTCGAACGGAGTGGAGTCGAATGGATTCGAATGGAATGCAATGGAATGGAACGGAATGCAATGTACTCGAATGGAATGGAATGTAATAGAATGAAAATTACTCGAATGGAATGGAATGGAATGGACTCCAATGGAATGGAATCGAACGGACTCGAATAGAATGCAGTTGAATTGAATGGACCTGAAAGAATGCAATGGAATGGAATGAAATGGACTCGAATGGAATGGAATAGACTGAAATGAAATGGAATGTACTGGAATGGAATGGAATGGAATGTACTGGAATGGAATGGAATGGACTCGAATGATATGCAATTGAATGGACTCGCATGGATTGGAATGGACTCTAGTGGAATGGAATGGAATA	30	PASS	.	GT	1|1
chr1	789481	.	G	GGAATGGAATGCAATGGAATGCACTCGAACGGATTGGAATGGAATGGACTCGAATAGAATGGAATAAAATGAAATGGACTCCAATGGATTGGAATGGAATTGACTCCAATGGAATTGAATGGAGTGGAACCGAATGGAACGGATTGGAATGGAATGCACTCGAAATGAATTTGAATGGAATGGATTGGGCTCAAATGGAATGGAATGGAATGGAATGGAATGGAATGAACTCAAATGGATTAGCATGGAATGAAGTGGACTCGAATACAATGGAATGGAATGGACTCGAATGGAATGGAACGGACTTGAACGGAATGGAGTGGAATGGACTCAAATGGAATGGAATGGAGTTGAATGGACTCGAATGGAATGGAATGTAAAGGAATGGAATGAACTCGAAAGGAGTGGAATGTAATGGAATGAAATGGACTCGAATGGAATTAAATGGAATGGAACGGAATGGACTGGGATGGAATGGAACGGAACGGAACGCAGTTGAATTGAACGGACCCGGAATGGAATGGAATGGAATGGAATGAAATGGAATGAAGTGGACTCTAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATAGACTCGAATGAAATGGGATGGACTCGAATGGAATGGAACGGAATGGAATCGACTCGAGTGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATAGAATGGAATTGACTCAATTTAAATGGAATGAAGTGGAATGAACTCGAATGGCATGCAATGGAATGGAATAGAATCGAATGGAATAGAATGGACCCAAATGGAATGGAACGGAATGGAATGGAATGGAACGGAATGGAATAGAACGGAACGGAATGGAATGGATTGGAATGAACTCCAACGGAATGGAATGGACTCGAATGCAATGGAATGGAATGGAATGGAATGGAGTGGACTGGAATGGAATAGAATGGAATGGAATGGATAGGACTGGAATGAAATGGAATGGAATGGACTCGAATGGAATGGAATGGAATGGAATGGAATGGACTCAAATGGAATGGAATGGAATGGAATGGACACGAATGGAATGGAATTGAATGGAATGGAATGGACTGTATGAAAAGGAATGGATTGGAAAGGAATGGAATAGAACGGAATGGACTCGAATGGAATGGAAAGGACTCGAGTGGAATGGAATGGAATGGAATGGACTCGAATGGAATGGAGTGGAATGTATGCGAATGGAATGGAATTGAATGGATTCGAGTCTAACGGAATGTATGGAATGGACTCGAATGGAATGGAATGTAATGGAATGAAATGGACGCGAATGGAATGGAATGGAATGGAATGGAGTGGAATGGAATGGACTCGAATGGTATGGAATGGAATTGAATGGACTCGATAGGAATGGAATGGAATGGATTGGACTCGAAAGGAATGTAATGGAATGAAATGTGCTGGAATGGAATGGAATGGAATGGAATAAAATGTAATGGAATGGACTCGAATGGAATACAGTTGAATTGAATGGACCCGAAAGCAATGGAATGGAATGGAACGGATTGGAAGGGAATGGAATGGAATGAAATGGAAAAGACTCGAATGGAATGGAATGGACTCGAATGAAATGGAGTGGACTAGAATGGAATGGAATGGACTTGAAAGAAATGGAATGCAGTGGAATGGACTCGAATGGAATGCAATGGAATGGAATAGACTCGAACGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAACGGAATGGACGCGAATGAAATGGAACGGAACGGAATGGACTCGGATGGAACGGAACGGAACGGAATGGAATGGAATTTACTCGAATGGGATGGAATGGAATGGAATTTACTCGAATGGAATGGAATGGAATGGACTGAAATGGAATAGCATGGAATGGAATGGACTCGAATGCAATGGAATGGAATGGACTCGAATGGAACGGAATGGACTCGAACGGAGTGGAGTCGAATGGATTCGAATGGAATGCAATGGAATGGAACGGAATGCAATGTACTCGAATGGAATGGAATGTAATAGAATGAAAATTACTCGAATGGAATGGAATGGAATGGACTCCAATGGAATGGAATCGAACGGACTCGAATAGAATGCAGTTGAATTGAATGGACCTGAAAGAATGCAATGGAATGGAATGAAATGGACTCGAATGGAATGGAATAGACTGAAATGAAATGGAATGTACTGGAATGGAATGGAATGGAATGTACTGGAATGGAATGGAATGGACTCGAATGATATGCAATTGAATGGACTCGCATGGATTGGAATGGACTCTAGTGGAATGGAATGGAATA,GGAATGGAATGCAATGGAATGCACTCGAACGGATTGGAATGGAATGGACTCGAATAGAATGGAATAAAATGAAATGGACTCCAATGGATTGGAATGGAATTGACTCCAATGGAATTGAATGGAGTGGAACCGAATGGAACGGATTGGAATGGAATGCACTCGAAATGAATTTGAATGGAATGGATTGGGCTCAAATGGAATGGAATGGAATGGAATGGAATGGAATGAACTCAAATGGATTAGCATGGAATGAAGTGGACTCGAATACAATGGAATGGAATGGACTCGAATGGAATGGAACGGACTTGAACGGAATGGAGTGGAATGGACTCGAATGGAATGGAGTTGAATGGACTCGAATGGAATGGAATGTAAAGGAATGGAATGAACTCGAAAGGAGTGGAATGTAATGGAATGAAATGGACTCGAATGGAATTAAATGGAATGGAACGGAATGGACTGGGATGGAATGGAACGGAACGGAACGCAGTTGAATTGAACGGACCCGGAATGGAATGGAATGGAATGGAATGAAATGGAATGAAGTGGACTCTAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATAGACTCGAATGAAATGGGATGGACTCGAATGGAATGGAACGGAATGGAATCGACTCGAGTGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATAGAATGGAATTGACTCAATTTAAATGGAATGAAGTGGAATGAACTCGAATGGCATGCAATGGAATGGAATAGAATCGAATGGAATAGAATGGACCCAAATGGAATGGAACGGAATGGAATGGAATGGAACGGAATGGAATAGAACGGAACGGAATGGAATGGATTGGAATGAACTCCAACGGAATGGAATGGACTCGAATGCAATGGAATGGAATGGAATGGAATGGAGTGGACTGGAATGGAATAGAATGGAATGGAATGGATAGGACTGGAATGAAATGGAATGGAATGGACTCGAATGGAATGGAATGGAATGGAATGGACTCAAATGGAATGGAATGGAATGGAATGGACACGAATGGAATGGAATTGAATGGAATGGAATGGACTGTATGAAAAGGAATGGATTGGAAAGGAATGGAATAGAACGGAATGGACTCGAATGGAATGGAAAGGACTCGAGTGGAATGGAATGGAATGGAATGGACTCGAATGGAATGGAGTGGAATGTATGCGAATGGAATGGAATTGAATGGATTCGAGTCTAACGGAATGTATGGAATGGACTCGAATGGAATGTAATGTAATGGAATGAAATGGACGCGAATGGAATGGAATGGAATGGAATGGAATGGAGTGGAATGGAATGGACTCGAATGGTATGGAATGGAATTGAATGGACTCGATAGGAATGGAATGGAATGGATTGGACTCGAAAGGAATGTAATGGAATGAAATGTGCTGGAATGGAATGGAATGGAATGGAATAAAATGTAATGGAATGGACTCGAATGGAATACAGTTGAATTGAATGGACCCGAAAGCAATGGAATGGAATGGAACGGATTGGAAGGGAATGGAATGGAATGAAATGGAAAAGACTCGAATGGAATGGAATGGACTCGAATGAAATGGAGTGGACTAGAATGGAATGGAATGGACTTGAAAGAAATGGAATGCAGTGGAATGGACTCGAATGGAATGCAATGGAATGGAATAGACTCGAACGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAACGGAATGGACGCGAATGAAATGGAACGGAACGGAATGGACTCGGATGGAACGGAATGGAACGGAATGGAATGGAATTTACTCGAATGGGATGGAATGGAATGGAATTTACTCGAATGGAATGGAATGGAATGGACTGAAATGGAATAGCATGGAATGGAATGGACTCGAATGCAATGGAATGGAATGGACTCGAATGGAACGGAATGGACTCGAACGGAGTGGAGTCGAATGGATTCGAATGGAATGCAATGGAATGGAACGGAATGCAATGTACTCGAATGGAATGGAATGTAATAGAATGAAAATTACTCGAATGGAATGGAATGGAATGGACTCCAATGGAATGGAATCGAACGGACTCGAATAGAATGCAGTTGAATTGAATGGACCTGAAAGAATGCAATGGAATGGAATGAAATGGACTCGAATGGAATGGAATAGACTGAAATGAAATGGAATGTACTGGAATGGAATGGAATGGAATGTACTGGAATGGAATGGAATGGACTCGAATGATATGCAATTGAATGGACTCGCATGGATTGGAATGGACTCTAGTGGAATGGAATGGAATA	30	PASS	.	GT	."""
        ) # fmt: skip
        # cSpell:enable

        ref_kmer_counts_prefix = f"/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k{cfg.graph.kmer_size}"
        if not os.path.exists(f"{ref_kmer_counts_prefix}.kmc_pre"):
            pytest.skip("Reference kmer counts not found")
        ref_kmer_counts = KmerCounts(ref_kmer_counts_prefix)

        region = Range("chr1:789386-789670")
        graph, unique_kmers, haplotype_sampler = _create_graph_and_sampler(
            cfg.reference,
            vcf_path,
            region,
            k=cfg.graph.kmer_size,
            ref_kmer_counts=ref_kmer_counts, # type: ignore
        )
        # The graph construction automatically "deduplicates" the overlapping insertions, so the graph should have 2
        # alternate alleles

        filtered_kmer_path = cache_filter_kmc_database(cfg, hg00733_sample, vcf_path, region, unique_kmers, tmp_path=tmp_path)
        haplotypes, diplotypes = _sample_diplotypes_from_counts(
            haplotype_sampler,
            unique_kmers,
            filtered_kmer_path,
            k=cfg.graph.kmer_size,
            kmer_coverage=hg00733_sample.kmer_coverage,
            filter_kmers=False,
        )
        assert len(haplotypes) == 3, "There should be 3 haplotypes, since there are only 2 unique alternate alleles"

        # `sample_haplotypes` mutates k-mer scores in place as it greedily selects haplotypes, so re-initialize
        # scores from the same k-mer counts before scoring the true and sampled haplotypes on equal footing.
        counts = KmerClassify(filtered_kmer_path, hg00733_sample.kmer_coverage)
        haplotype_sampler.initialize_scores(counts)

        true_scores = [haplotype_sampler.score(graph.path_nodes(f"{hg00733_sample.name}#{h}#{region.contig}#0")) for h in range(2)]
        sampled_scores = [haplotype_sampler.score(haplotype) for haplotype in haplotypes]

        assert sampled_scores[0] >= max(sampled_scores), "The first sampled haplotype should be the best-scoring sampled haplotype"
        assert sampled_scores[0] >= max(true_scores), "The first sampled haplotype should be the best-scoring sampled haplotype"

    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
        "input=/storage/mlinderman/projects/sv/npsv3-experiments/resources/hgsvc3-hprc-2024-02-23.dipcall.population.passing.eval.hg38.vcf.gz",
        "graph.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
    )
    def test_population_scale_overlapping_alleles(self, cfg, hg00733_sample, tmp_path):
        if not all(os.path.exists(f) for f in (cfg.reference, cfg.input, f"{cfg.graph.ref_kmer_counts_kmc_prefix}.kmc_pre")):
            pytest.skip("Missing necessary inputs")
        ref_kmer_counts = KmerCounts(cfg.graph.ref_kmer_counts_kmc_prefix)

        region = Range("chr1:789386-789670")
        graph, unique_kmers, haplotype_sampler = _create_graph_and_sampler(
            cfg.reference,
            cfg.input,
            region,
            k=cfg.graph.kmer_size,
            ref_kmer_counts=ref_kmer_counts,  # type: ignore
        )

        filtered_kmer_path = cache_filter_kmc_database(cfg, hg00733_sample, cfg.input, region, unique_kmers, tmp_path=tmp_path)
        haplotypes, diplotypes = _sample_diplotypes_from_counts(
            haplotype_sampler,
            unique_kmers,
            filtered_kmer_path,
            k=cfg.graph.kmer_size,
            kmer_coverage=hg00733_sample.kmer_coverage,
            filter_kmers=False,
        )

        # The "correct" insertion allele is node ~105, but is not necessarily sampled. The sampled vs. true insertion alleles have 238
        # mismatches out of a 2369bp insertion, i.e., are very similar. We would want the correct allele to be ranked higher, but not
        # necessarily penalize the other alleles much given the similarity. While there is a downstream deletion, is not clearly correlated
        # with one specific insertion (just an insertion) so help bias the choice towards the correct allele.

        # `sample_haplotypes` mutates k-mer scores in place as it greedily selects haplotypes, so re-initialize
        # scores from the same k-mer counts before scoring the true and sampled haplotypes on equal footing.
        counts = KmerClassify(filtered_kmer_path, hg00733_sample.kmer_coverage)
        haplotype_sampler.initialize_scores(counts)

        true_scores = [haplotype_sampler.score(graph.path_nodes(f"{hg00733_sample.name}#{h}#{region.contig}#0")) for h in range(2)]
        sampled_scores = [haplotype_sampler.score(haplotype) for haplotype in haplotypes]

        assert sampled_scores[0] >= max(sampled_scores), "The first sampled haplotype should be the best-scoring sampled haplotype"
        assert sampled_scores[0] >= max(true_scores), "The first sampled haplotype should be the best-scoring sampled haplotype"

        # TODO: Introduce a threshold, analogous to how Truvari, etc. match variant calls to determine if a haplotype should be considered
        # distinct or not? For exmaple, if the haplotype is 90% similar to another haplotype, that probably wouldn't be distinct enough to
        # detect.

    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
        "input=/storage/mlinderman/projects/sv/npsv3-experiments/resources/hgsvc3-hprc-2024-02-23.dipcall.population.passing.eval.hg38.vcf.gz",
        "graph.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
        "graph.population_prior=true",
        'graph.population_excluded_samples=["HG00733","HG00514","NA19240","NA24385"]',
        "graph.haplotype_sampler_params.haplotype_prior_weight=5.0",
    )
    def test_population_prior_from_upstream_snv(self, cfg, hg00733_sample, tmp_path):
        if not all(os.path.exists(f) for f in (cfg.reference, cfg.input, f"{cfg.graph.ref_kmer_counts_kmc_prefix}.kmc_pre")):
            pytest.skip("Missing necessary inputs")

        ref_kmer_counts = KmerCounts(cfg.graph.ref_kmer_counts_kmc_prefix)

        # Region reported in https://pmc.ncbi.nlm.nih.gov/articles/PMC10475782/ as having multi-population bi-allelic deletion.
        # DEL variant 16735b1fe61f031933ad09f1dc3fad42b8bd78f5 is highly correlated with upstream SNVs. Use of population priors
        # improves fidelity of sampled haptlotypes to the true haplotypes, in this case surrounding SNVs. At development time we
        # observed reduction of 8 -> 2 and 6 -> 0 edits for k-mer only vs. k-mer + population prior sampled haplotypes.

        region = Range("chr18:71411065-71411922")
        graph, unique_kmers, haplotype_sampler = _create_graph_and_sampler(
            cfg.reference,
            cfg.input,
            region,
            k=cfg.graph.kmer_size,
            ref_kmer_counts=ref_kmer_counts,
        )
        # Use cached files to speed up repeated runs of the test
        filtered_kmer_path = cache_filter_kmc_database(cfg, hg00733_sample, cfg.input, region, unique_kmers, tmp_path=tmp_path)

        counts = KmerClassify(filtered_kmer_path, hg00733_sample.kmer_coverage)
        haplotype_sampler.initialize_scores(counts)
        haplotypes = haplotype_sampler.sample_haplotypes(n=2) # Two unique SV haplotypes in this region

        haplotype_prior = HaplotypePriorOverlay(graph)

        params = hydra.utils.instantiate(cfg.graph.haplotype_sampler_params)
        assert params.haplotype_prior_weight == 5.0, "The haplotype prior weight should be set from the config"
        prior_haplotype_sampler = HaplotypeSamplerOverlay(
            graph,
            unique_kmers,
            str(cfg.input),
            region,
            min_size=50,
            prior=haplotype_prior,
            params=params,
        )
        prior_haplotype_sampler.initialize_scores(counts)
        prior_haplotypes = prior_haplotype_sampler.sample_haplotypes(n=2) # Two unique SV haplotypes in this region

        # Compute the distance between the true haplotypes and the sampled haplotypes
        for h in range(2):
            path_name = f"{hg00733_sample.name}#{h}#{region.contig}#0"
            nodes = graph.path_nodes(path_name)
            kmer_only_distance = min(_path_edits(nodes, haplotype) for haplotype in haplotypes)
            prior_distance = min(_path_edits(nodes, haplotype) for haplotype in prior_haplotypes)
            assert prior_distance <= kmer_only_distance, "The prior should improve the distance to the true haplotype"

@pytest.mark.skipif(not HG38_REF_FASTA, reason="HG38 reference FASTA not found")
class TestTopkHaplotypeSamplingInPopulation:
    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
        "input=/storage/mlinderman/projects/sv/npsv3-experiments/resources/hgsvc3-hprc-2024-02-23.dipcall.hg38.vcf.gz",
        "graph.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
    )
    def test_graph_construction_errors(self, cfg):
        if not all(os.path.exists(f) for f in (cfg.reference, cfg.input, f"{cfg.graph.ref_kmer_counts_kmc_prefix}.kmc_pre")):
            pytest.skip("Missing necessary inputs")

        region = Range("chr7:102594140-102690887")
        ref_kmer_counts = KmerCounts(cfg.graph.ref_kmer_counts_kmc_prefix)
        _create_graph_and_sampler(
            cfg.reference,
            cfg.input,
            region,
            k=cfg.graph.kmer_size,
            ref_kmer_counts=ref_kmer_counts,
        )

    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
        "input=/storage/mlinderman/projects/sv/npsv3-experiments/resources/hgsvc3-hprc-2024-02-23.dipcall.population.passing.eval.hg38.vcf.gz",
        "graph.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
        "graph.population_prior=true",
        'graph.population_excluded_samples=["HG00733","HG00514","NA19240","NA24385"]',
        "graph.haplotype_sampler_params.haplotype_prior_weight=5.0",
    )
    @pytest.mark.parametrize(("region_str", "weight", "penalty"), [
        ("chr1:161443672-161443863", 5.0, 0),
        ("chr5:54723110-54723301", 5.0, 0),
        ("chr13:59605032-59605223", 5.0, 0),
        ("chr21:40867003-40867194", 0.5, -30)
    ])
    def test_no_haplotypes_sampled(self, cfg, hg00733_sample, tmp_path, region_str, weight, penalty):
        if not all(os.path.exists(f) for f in (cfg.reference, cfg.input, f"{cfg.graph.ref_kmer_counts_kmc_prefix}.kmc_pre")):
            pytest.skip("Missing necessary inputs")

        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([
            f"graph.haplotype_sampler_params.haplotype_prior_weight={weight}",
            f"graph.haplotype_sampler_params.panel_fallback_penalty={penalty}",
        ]))

        region = Range(region_str)
        ref_kmer_counts = KmerCounts(cfg.graph.ref_kmer_counts_kmc_prefix)
        graph, unique_kmers, haplotype_sampler = _create_graph_and_sampler(
            cfg.reference,
            cfg.input,
            region,
            k=cfg.graph.kmer_size,
            ref_kmer_counts=ref_kmer_counts,
        )

        filtered_kmer_path = cache_filter_kmc_database(cfg, hg00733_sample, cfg.input, region, unique_kmers, tmp_path=tmp_path)
        counts = KmerClassify(filtered_kmer_path, hg00733_sample.kmer_coverage)

        haplotype_sampler.initialize_scores(counts)
        haplotypes = haplotype_sampler.sample_haplotypes(n=8)
        assert len(haplotypes) > 0, "There should be haplotypes sampled in this region"

        haplotype_prior = HaplotypePriorOverlay(graph, cfg.graph.population_excluded_samples)
        params = hydra.utils.instantiate(cfg.graph.haplotype_sampler_params)
        prior_haplotype_sampler = HaplotypeSamplerOverlay(
            graph,
            unique_kmers,
            str(cfg.input),
            region,
            min_size=50,
            prior=haplotype_prior,
            params=params,
        )
        prior_haplotype_sampler.initialize_scores(counts)
        prior_haplotypes = prior_haplotype_sampler.sample_haplotypes(n=8)
        assert len(prior_haplotypes) > 0, "There should be haplotypes sampled in this region"


    @pytest.mark.usefixtures("ray_setup")
    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
        "input=/storage/mlinderman/projects/sv/npsv3-experiments/resources/hgsvc3-hprc-2024-02-23.dipcall.population.passing.eval.hg38.vcf.gz",
        "graph.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
        "graph.population_prior=true",
        'graph.population_excluded_samples=["HG00733","HG00514","NA19240","NA24385"]',
        "graph.haplotype_sampler_params.haplotype_prior_weight=5.0",
        "graph.missing_are_ref=true"
    )
    def test_missing_genotypes(self, cfg, hg00733_sample):
        if not all(os.path.exists(f) for f in (cfg.reference, cfg.input, f"{cfg.graph.ref_kmer_counts_kmc_prefix}.kmc_pre")):
            pytest.skip("Missing necessary inputs")

        region = Range("chr11:801646-801889")

        graph_shards, filtered_kmer_path = cache_graph_and_filter_kmc_database(cfg, hg00733_sample, cfg.input, region)
        statistics = diplotypes_in_topk(cfg, cfg.input, hg00733_sample, region=region, graph_shards=graph_shards, filtered_kmer_path=filtered_kmer_path)
        assert len(statistics) == 1 # 1 SV in this region, but not present in HG00733
