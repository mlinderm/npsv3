import os
import subprocess
import tempfile
from shlex import quote

import numpy as np
import pandas as pd
import pytest

from npsv3._native_graph import Graph, KmerClassify, KmerCounts, Range, VariantFileReader
from npsv3.graphs.genotype import (
    _create_graph_and_sampler,
    _sample_diplotypes_from_counts,
    _serialize_graph_and_unique_kmers,
    genotypes_in_topk,
    sample_diplotypes,
)
from npsv3.util.sample import Sample, filter_kmc_database

from .. import HG38_REF_FASTA, _create_vcf, data_path, result_path


def _hash_vcf_file(vcf_path: str) -> str:
    """Compute a hash of the VCF file contents, ignoring header lines."""
    import hashlib
    with open(vcf_path, "rb") as f:
        digest = hashlib.file_digest(f, "sha256")
        return digest.hexdigest()

def _cache_filter_kmc_database( cfg, sample: Sample, vcf_path: str, region: Range, unique_kmers, tmp_path) -> str:
    # Incorporate hash into custom VCF file into cache directory to avoid collisions
    results_directory = result_path(f"{region.slug}.{_hash_vcf_file(vcf_path)}.{sample.name}.k{cfg.kmer.kmer_size}")
    os.makedirs(results_directory, exist_ok=True)
    filtered_kmer_path = os.path.join(results_directory, "filtered_kmers")
    if not os.path.exists(filtered_kmer_path + ".kmc_pre"):
        # Generate filtered k-mers if not already available (a slow step, so we cache the results)
        if not sample.kmc_prefix:
            pytest.skip(f"KMC database for {sample.name} not found")
        filter_kmc_database(
            sample.kmc_prefix, unique_kmers, cfg.kmer.kmer_size, filtered_kmer_path, tmp_dir=tmp_path
        )
    return filtered_kmer_path

@pytest.mark.skipif(not HG38_REF_FASTA, reason="HG38 reference FASTA not found")
class TestTopkHaplotypeSampling:
    def test_correct_diplotype_homalt(self, cfg, tmp_path):
        vcf_path = _create_vcf(tmp_path, b"""##fileformat=VCFv4.2
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
                for i, kmer in enumerate(["A" * cfg.kmer.kmer_size]):
                    f.write(f">{i}\n{kmer}\n")
            subprocess.check_call(
                f"kmc -t1 -k{cfg.kmer.kmer_size} -b -ci1 -fa {quote(fasta_file)} {quote(kmc_prefix)} {quote(kmc_dir)}",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            graph, haplotypes, diplotypes, *_ = sample_diplotypes(
                HG38_REF_FASTA, # type: ignore
                vcf_path,
                region,
                kmc_prefix,
                k=cfg.kmer.kmer_size,
                kmer_coverage=29,
                min_variant_size=50,
                filter_kmers=False,
            )

            true_hap0_nodes = graph.haplotype_paths(f"HG00096#0#{region.contig}")
            true_hap1_nodes = graph.haplotype_paths(f"HG00096#1#{region.contig}")
            assert true_hap0_nodes == true_hap1_nodes, "Variant is hom. alt."

            assert len(haplotypes) == 2, "A bi-allelic variant should have two haplotypes"
            assert haplotypes[0] == true_hap0_nodes, "The true haplotype should be the top-ranked"

            assert len(diplotypes) >= 3, "A bi-allelic variant should have >=3 diplotypes"
            assert diplotypes[0].haplotypes == (0, 0), "Top rank is the true diplotype"
            assert diplotypes[1].haplotypes.count(0) >= 1, "The second-ranked diplotype should contain true haplotype"

    def test_correct_diplotype_het(self, cfg, tmp_path):
        vcf_path = _create_vcf(tmp_path, b"""##fileformat=VCFv4.2
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
        kmers = [sequence[i:i + cfg.kmer.kmer_size] for i in range(0, len(sequence) - cfg.kmer.kmer_size + 1)]
        with open(fasta_file, "w") as f:
            for i, kmer in enumerate(kmers * 15):
                f.write(f">{i}\n{kmer}\n")
        subprocess.check_call(
            f"kmc -t1 -k{cfg.kmer.kmer_size} -b -ci1 -fa {quote(str(fasta_file))} {quote(str(kmc_prefix))} {quote(str(tmp_path))}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        graph, haplotypes, diplotypes, sampler, *_ = sample_diplotypes(
            HG38_REF_FASTA, # type: ignore
            vcf_path,
            region,
            kmc_prefix,
            k=cfg.kmer.kmer_size,
            kmer_coverage=29,
            min_variant_size=50,
            filter_kmers=False,
        )
        assert sampler.num_kmers() > 1, "There should be multiple k-mers to distinguish the haplotypes"

        true_hap0_idx = haplotypes.index(graph.haplotype_paths(f"HG00096#0#{region.contig}"))
        assert set(diplotypes[0].haplotypes) > { true_hap0_idx }, "The most likely diplotype should be het."

    @pytest.mark.usefixtures("ray_setup")
    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
    )
    def test_topk_genotype(self, cfg, hg00096_sample, tmp_path):
        vcf_path = _create_vcf(tmp_path, b"""##fileformat=VCFv4.2
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
            for i, kmer in enumerate(["A" * cfg.kmer.kmer_size]):
                f.write(f">{i}\n{kmer}\n")
        subprocess.check_call(
            f"kmc -t1 -k{cfg.kmer.kmer_size} -b -ci1 -fa {quote(str(fasta_file))} {quote(str(kmc_prefix))} {quote(str(tmp_path))}",
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

        statistics = genotypes_in_topk(cfg, vcf_path, hg00096_sample, filter_kmers=False)
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
        "kmer.kmer_size=31",  # Test files were generated with k=31, so we need to use that here to get the expected results
    )
    def test_topk_genotype_multiallelic(self, cfg, hg00096_sample, tmp_path):
        vcf_path = _create_vcf(tmp_path, b"""##fileformat=VCFv4.2
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
            f"kmc -t1 -k{cfg.kmer.kmer_size} -b -ci1 -fa {quote(str(fasta_file))} {quote(str(kmc_prefix))} {quote(str(tmp_path))}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        hg00096_sample.kmc_prefix = kmc_prefix

        statistics = genotypes_in_topk(cfg, vcf_path, hg00096_sample, filter_kmers=False)
        assert len(statistics) == 1, "There should be one row of statistics for the single variant"
        assert all(h > 0 for h in statistics.iloc[0]["haplotype_idxs"]), "The true haplotypes should be found, but not necessarily top-ranked"
        assert statistics.iloc[0]["diplotype_idx"] >= 0, "The true diplotype should be found, but not necessarily top-ranked"

@pytest.mark.skipif(not HG38_REF_FASTA, reason="HG38 reference FASTA not found")
class TestTopkHaplotypeSamplingInHG00733:
    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
    )
    def test_downstream_small_del(self, cfg, hg00733_sample, tmp_path):
        vcf_path = "/storage/mlinderman/projects/sv/npsv3-experiments/resources/HG00733.hgsvc3-hprc-2024-02-23.dipcall.passing.hg38.vcf.gz"
        if not os.path.exists(vcf_path):
            pytest.skip("HG00733 VCF not found")
        # chr1:789481G->[INS] 1|1

        ref_kmer_counts_prefix = f"/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k{cfg.kmer.kmer_size}"
        if not os.path.exists(f"{ref_kmer_counts_prefix}.kmc_pre"):
            pytest.skip("Reference kmer counts not found")
        ref_kmer_counts = KmerCounts(ref_kmer_counts_prefix)

        region = Range("chr1:789386-789670")
        graph, unique_kmers, haplotype_sampler = _create_graph_and_sampler(
            cfg.reference,
            vcf_path,
            region,
            k=cfg.kmer.kmer_size,
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
        filtered_kmer_path = _cache_filter_kmc_database(cfg, hg00733_sample, vcf_path, region, unique_kmers, tmp_path=tmp_path)

        haplotypes, diplotypes = _sample_diplotypes_from_counts(
            haplotype_sampler,
            unique_kmers,
            filtered_kmer_path,
            k=cfg.kmer.kmer_size,
            kmer_coverage=hg00733_sample.kmer_coverage,
            filter_kmers=False,
        )


    @pytest.mark.skip(reason="Skip unless debugging")
    @pytest.mark.usefixtures("ray_setup")
    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
        "kmer.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${kmer.kmer_size}",
    )
    def test_multi_variant_with_star_alleles(self, cfg, hg00733_sample):
        """Test region with multiple variants and star alleles"""
        vcf_path = "/storage/mlinderman/projects/sv/npsv3-experiments/resources/HG00733.hgsvc3-hprc-2024-02-23.dipcall.passing.hg38.vcf.gz"
        if not os.path.exists(vcf_path):
            pytest.skip("HG00733 VCF not found")

        if not os.path.exists(f"{cfg.kmer.ref_kmer_counts_kmc_prefix}.kmc_pre"):
            pytest.skip("Reference kmer counts not found")

        region = Range("chr7:104833216-104833994")

        # Use cached files to speed up repeated runs of the test
        results_directory = result_path(f"{region.slug}.HG00733.k{cfg.kmer.kmer_size}")
        os.makedirs(results_directory, exist_ok=True)
        filtered_kmer_path = os.path.join(results_directory, "filtered_kmers")
        graph_shards = [os.path.join(results_directory, "graphs-00000.tar.gz")]
        if not os.path.exists(f"{filtered_kmer_path}.kmc_pre") or not all(os.path.exists(shard) for shard in graph_shards):
            graph_shards, filtered_kmer_path, _region_count = _serialize_graph_and_unique_kmers(
                cfg,
                vcf_path,
                hg00733_sample,
                ref_kmer_counts_path=cfg.kmer.ref_kmer_counts_kmc_prefix,
                output_dir=results_directory,
                filter_kmers=True,
                region=region,
            )

        statistics = genotypes_in_topk(cfg, vcf_path, hg00733_sample, filter_kmers=True, region=region, graph_shards=graph_shards, filtered_kmer_path=filtered_kmer_path)

        # The first variant is 0/1, but overlaps the second (genotype of 1/2 with a star allele), so we correctly
        # sample haplotypes that don't include the full reference allele for the first variant (nodes [3,5]), just node [3].
        # But node [3] is sufficient to uniquely identify that we didn't call the alternate allele (node [2]) for the
        # first variant, so we want to recognize that a haplotype of [..., 3, 4, ...] is a correct match for the reference
        # allele of the first variant. But with the star allele we want to recognize that a haplotype of [..., 2, 6, ...]
        # is compatible with the star allele, even though the haplotype is not disjoint with reference nodes [5, 6] for the
        # second variant, i.e., [..., 2, 6, ...] is "labeled" with both reference and star allele for variant 2.

        assert len(statistics) == 3, "There should be 3 'inference' variants in the region"
        assert all(statistics["haplotypes"] == 6), "With overlapping variants, there should be 6 haplotypes in the region"
        assert statistics["haplotype_idxs"].equals(pd.Series([(0, 1), (0, 1), (0, 0)]))
        assert all((statistics["all_diplotype_idx"] == -1) | (statistics["diplotype_idx"] <= statistics["all_diplotype_idx"]))

    @pytest.mark.skip(reason="Skip unless debugging")
    @pytest.mark.usefixtures("ray_setup")
    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
        "kmer.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${kmer.kmer_size}",
    )
    def test_unexpected_all_diplotype_idx(self, cfg, hg00733_sample):
        vcf_path = "/storage/mlinderman/projects/sv/npsv3-experiments/resources/HG00733.hgsvc3-hprc-2024-02-23.dipcall.passing.hg38.vcf.gz"
        if not os.path.exists(vcf_path):
            pytest.skip("HG00733 VCF not found")

        if not os.path.exists(f"{cfg.kmer.ref_kmer_counts_kmc_prefix}.kmc_pre"):
            pytest.skip("Reference kmer counts not found")

        region = Range("chr9:91796648-91799004")

        # Use cached files to speed up repeated runs of the test
        results_directory = result_path(f"{region.slug}.HG00733.k{cfg.kmer.kmer_size}")
        os.makedirs(results_directory, exist_ok=True)
        filtered_kmer_path = os.path.join(results_directory, "filtered_kmers")
        graph_shards = [os.path.join(results_directory, "graphs-00000.tar.gz")]
        if not os.path.exists(f"{filtered_kmer_path}.kmc_pre") or not all(os.path.exists(shard) for shard in graph_shards):
            graph_shards, filtered_kmer_path, _region_count = _serialize_graph_and_unique_kmers(
                cfg,
                vcf_path,
                hg00733_sample,
                ref_kmer_counts_path=cfg.kmer.ref_kmer_counts_kmc_prefix,
                output_dir=results_directory,
                filter_kmers=True,
                region=region,
            )

        statistics = genotypes_in_topk(cfg, vcf_path, hg00733_sample, filter_kmers=True, region=region, graph_shards=graph_shards, filtered_kmer_path=filtered_kmer_path)
        assert all((statistics["all_diplotype_idx"] == -1) | (statistics["diplotype_idx"] <= statistics["all_diplotype_idx"]))

    # @pytest.mark.skip(reason="Skip unless debugging")
    @pytest.mark.usefixtures("ray_setup")
    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
        "kmer.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${kmer.kmer_size}",
    )
    def test_missing_matching_halotypes(self, cfg, hg00733_sample):
        vcf_path = "/storage/mlinderman/projects/sv/npsv3-experiments/resources/HG00733.hgsvc3-hprc-2024-02-23.dipcall.passing.hg38.vcf.gz"
        if not os.path.exists(vcf_path):
            pytest.skip("HG00733 VCF not found")

        if not os.path.exists(f"{cfg.kmer.ref_kmer_counts_kmc_prefix}.kmc_pre"):
            pytest.skip("Reference kmer counts not found")

        region = Range("chr1:6006213-6006792")

        # Use cached files to speed up repeated runs of the test
        results_directory = result_path(f"{region.slug}.HG00733.k{cfg.kmer.kmer_size}")
        os.makedirs(results_directory, exist_ok=True)
        filtered_kmer_path = os.path.join(results_directory, "filtered_kmers")
        graph_shards = [os.path.join(results_directory, "graphs-00000.tar.gz")]
        if not os.path.exists(f"{filtered_kmer_path}.kmc_pre") or not all(os.path.exists(shard) for shard in graph_shards):
            graph_shards, filtered_kmer_path, _region_count = _serialize_graph_and_unique_kmers(
                cfg,
                vcf_path,
                hg00733_sample,
                ref_kmer_counts_path=cfg.kmer.ref_kmer_counts_kmc_prefix,
                output_dir=results_directory,
                filter_kmers=True,
                region=region,
            )

        statistics = genotypes_in_topk(cfg, vcf_path, hg00733_sample, filter_kmers=True, region=region, graph_shards=graph_shards, filtered_kmer_path=filtered_kmer_path)
        assert len(statistics) == 0, "There should be no fully genotyped analysis variants in this region"
        # TODO: Test that an info message was logged about this region


    #@pytest.mark.skip(reason="Skip unless debugging")
    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
    )
    def test_no_haplotypes_sampled(self, cfg, hg00733_sample, tmp_path):
        vcf_path = "/storage/mlinderman/projects/sv/npsv3-experiments/resources/HG00733.hgsvc3-hprc-2024-02-23.dipcall.passing.hg38.vcf.gz"
        if not os.path.exists(vcf_path):
            pytest.skip("HG00733 VCF not found")

        ref_kmer_counts_prefix = f"/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k{cfg.kmer.kmer_size}"
        if not os.path.exists(f"{ref_kmer_counts_prefix}.kmc_pre"):
            pytest.skip("Reference kmer counts not found")
        ref_kmer_counts = KmerCounts(ref_kmer_counts_prefix)

        region = Range("chr6:32249973-32250644")
        graph, unique_kmers, haplotype_sampler = _create_graph_and_sampler(
            cfg.reference,
            vcf_path,
            region,
            k=cfg.kmer.kmer_size,
            ref_kmer_counts=ref_kmer_counts,
        )

        filtered_kmer_path = _cache_filter_kmc_database(cfg, hg00733_sample, vcf_path, region, unique_kmers, tmp_path=tmp_path)
        haplotypes, diplotypes = _sample_diplotypes_from_counts(
            haplotype_sampler,
            unique_kmers,
            filtered_kmer_path,
            k=cfg.kmer.kmer_size,
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
        vcf_path = _create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=chr1,length=248956422>
##contig=<ID=chr12,length=133275309>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00733
chr1	789481	.	G	GGAATGGAATGCAATGGAATGCACTCGAACGGATTGGAATGGAATGGACTCGAATAGAATGGAATAAAATGAAATGGACTCCAATGGATTGGAATGGAATTGACTCCAATGGAATTGAATGGAGTGGAACCGAATGGAACGGATTGGAATGGAATGCACTCGAAATGAATTTGAATGGAATGGATTGGGCTCAAATGGAATGGAATGGAATGGAATGGAATGGAATGAACTCAAATGGATTAGCATGGAATGAAGTGGACTCGAATACAATGGAATGGAATGGACTCGAATGGAATGGAACGGACTTGAACGGAATGGAGTGGAATGGACTCGAATGGAATGGAGTTGAATGGACTCGAATGGAATGGAATGTAAAGGAATGGAATGAACTCGAAAGGAGTGGAATGTAATGGAATGAAATGGACTCGAATGGAATTAAATGGAATGGAACGGAATGGACTGGGATGGAATGGAACGGAACGGAACGCAGTTGAATTGAACGGACCCGGAATGGAATGGAATGGAATGGAATGAAATGGAATGAAGTGGACTCTAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATAGACTCGAATGAAATGGGATGGACTCGAATGGAATGGAACGGAATGGAATCGACTCGAGTGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATAGAATGGAATTGACTCAATTTAAATGGAATGAAGTGGAATGAACTCGAATGGCATGCAATGGAATGGAATAGAATCGAATGGAATAGAATGGACCCAAATGGAATGGAACGGAATGGAATGGAATGGAACGGAATGGAATAGAACGGAACGGAATGGAATGGATTGGAATGAACTCCAACGGAATGGAATGGACTCGAATGCAATGGAATGGAATGGAATGGAATGGAGTGGACTGGAATGGAATAGAATGGAATGGAATGGATAGGACTGGAATGAAATGGAATGGAATGGACTCGAATGGAATGGAATGGAATGGAATGGACTCAAATGGAATGGAATGGAATGGAATGGACACGAATGGAATGGAATTGAATGGAATGGAATGGACTGTATGAAAAGGAATGGATTGGAAAGGAATGGAATAGAACGGAATGGACTCGAATGGAATGGAAAGGACTCGAGTGGAATGGAATGGAATGGAATGGACTCGAATGGAATGGAGTGGAATGTATGCGAATGGAATGGAATTGAATGGATTCGAGTCTAACGGAATGTATGGAATGGACTCGAATGGAATGTAATGTAATGGAATGAAATGGACGCGAATGGAATGGAATGGAATGGAATGGAATGGAGTGGAATGGAATGGACTCGAATGGTATGGAATGGAATTGAATGGACTCGATAGGAATGGAATGGAATGGATTGGACTCGAAAGGAATGTAATGGAATGAAATGTGCTGGAATGGAATGGAATGGAATGGAATAAAATGTAATGGAATGGACTCGAATGGAATACAGTTGAATTGAATGGACCCGAAAGCAATGGAATGGAATGGAACGGATTGGAAGGGAATGGAATGGAATGAAATGGAAAAGACTCGAATGGAATGGAATGGACTCGAATGAAATGGAGTGGACTAGAATGGAATGGAATGGACTTGAAAGAAATGGAATGCAGTGGAATGGACTCGAATGGAATGCAATGGAATGGAATAGACTCGAACGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAACGGAATGGACGCGAATGAAATGGAACGGAACGGAATGGACTCGGATGGAACGGAATGGAACGGAATGGAATGGAATTTACTCGAATGGGATGGAATGGAATGGAATTTACTCGAATGGAATGGAATGGAATGGACTGAAATGGAATAGCATGGAATGGAATGGACTCGAATGCAATGGAATGGAATGGACTCGAATGGAACGGAATGGACTCGAACGGAGTGGAGTCGAATGGATTCGAATGGAATGCAATGGAATGGAACGGAATGCAATGTACTCGAATGGAATGGAATGTAATAGAATGAAAATTACTCGAATGGAATGGAATGGAATGGACTCCAATGGAATGGAATCGAACGGACTCGAATAGAATGCAGTTGAATTGAATGGACCTGAAAGAATGCAATGGAATGGAATGAAATGGACTCGAATGGAATGGAATAGACTGAAATGAAATGGAATGTACTGGAATGGAATGGAATGGAATGTACTGGAATGGAATGGAATGGACTCGAATGATATGCAATTGAATGGACTCGCATGGATTGGAATGGACTCTAGTGGAATGGAATGGAATA	30	PASS	.	GT	1|1
chr1	789481	.	G	GGAATGGAATGCAATGGAATGCACTCGAACGGATTGGAATGGAATGGACTCGAATAGAATGGAATAAAATGAAATGGACTCCAATGGATTGGAATGGAATTGACTCCAATGGAATTGAATGGAGTGGAACCGAATGGAACGGATTGGAATGGAATGCACTCGAAATGAATTTGAATGGAATGGATTGGGCTCAAATGGAATGGAATGGAATGGAATGGAATGGAATGAACTCAAATGGATTAGCATGGAATGAAGTGGACTCGAATACAATGGAATGGAATGGACTCGAATGGAATGGAACGGACTTGAACGGAATGGAGTGGAATGGACTCAAATGGAATGGAATGGAGTTGAATGGACTCGAATGGAATGGAATGTAAAGGAATGGAATGAACTCGAAAGGAGTGGAATGTAATGGAATGAAATGGACTCGAATGGAATTAAATGGAATGGAACGGAATGGACTGGGATGGAATGGAACGGAACGGAACGCAGTTGAATTGAACGGACCCGGAATGGAATGGAATGGAATGGAATGAAATGGAATGAAGTGGACTCTAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATAGACTCGAATGAAATGGGATGGACTCGAATGGAATGGAACGGAATGGAATCGACTCGAGTGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATAGAATGGAATTGACTCAATTTAAATGGAATGAAGTGGAATGAACTCGAATGGCATGCAATGGAATGGAATAGAATCGAATGGAATAGAATGGACCCAAATGGAATGGAACGGAATGGAATGGAATGGAACGGAATGGAATAGAACGGAACGGAATGGAATGGATTGGAATGAACTCCAACGGAATGGAATGGACTCGAATGCAATGGAATGGAATGGAATGGAATGGAGTGGACTGGAATGGAATAGAATGGAATGGAATGGATAGGACTGGAATGAAATGGAATGGAATGGACTCGAATGGAATGGAATGGAATGGAATGGAATGGACTCAAATGGAATGGAATGGAATGGAATGGACACGAATGGAATGGAATTGAATGGAATGGAATGGACTGTATGAAAAGGAATGGATTGGAAAGGAATGGAATAGAACGGAATGGACTCGAATGGAATGGAAAGGACTCGAGTGGAATGGAATGGAATGGAATGGACTCGAATGGAATGGAGTGGAATGTATGCGAATGGAATGGAATTGAATGGATTCGAGTCTAACGGAATGTATGGAATGGACTCGAATGGAATGGAATGTAATGGAATGAAATGGACGCGAATGGAATGGAATGGAATGGAATGGAGTGGAATGGAATGGACTCGAATGGTATGGAATGGAATTGAATGGACTCGATAGGAATGGAATGGAATGGATTGGACTCGAAAGGAATGTAATGGAATGAAATGTGCTGGAATGGAATGGAATGGAATGGAATAAAATGTAATGGAATGGACTCGAATGGAATACAGTTGAATTGAATGGACCCGAAAGCAATGGAATGGAATGGAACGGATTGGAAGGGAATGGAATGGAATGAAATGGAAAAGACTCGAATGGAATGGAATGGACTCGAATGAAATGGAGTGGACTAGAATGGAATGGAATGGACTTGAAAGAAATGGAATGCAGTGGAATGGACTCGAATGGAATGCAATGGAATGGAATAGACTCGAACGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAACGGAATGGACGCGAATGAAATGGAACGGAACGGAATGGACTCGGATGGAACGGAACGGAACGGAATGGAATGGAATTTACTCGAATGGGATGGAATGGAATGGAATTTACTCGAATGGAATGGAATGGAATGGACTGAAATGGAATAGCATGGAATGGAATGGACTCGAATGCAATGGAATGGAATGGACTCGAATGGAACGGAATGGACTCGAACGGAGTGGAGTCGAATGGATTCGAATGGAATGCAATGGAATGGAACGGAATGCAATGTACTCGAATGGAATGGAATGTAATAGAATGAAAATTACTCGAATGGAATGGAATGGAATGGACTCCAATGGAATGGAATCGAACGGACTCGAATAGAATGCAGTTGAATTGAATGGACCTGAAAGAATGCAATGGAATGGAATGAAATGGACTCGAATGGAATGGAATAGACTGAAATGAAATGGAATGTACTGGAATGGAATGGAATGGAATGTACTGGAATGGAATGGAATGGACTCGAATGATATGCAATTGAATGGACTCGCATGGATTGGAATGGACTCTAGTGGAATGGAATGGAATA,GGAATGGAATGCAATGGAATGCACTCGAACGGATTGGAATGGAATGGACTCGAATAGAATGGAATAAAATGAAATGGACTCCAATGGATTGGAATGGAATTGACTCCAATGGAATTGAATGGAGTGGAACCGAATGGAACGGATTGGAATGGAATGCACTCGAAATGAATTTGAATGGAATGGATTGGGCTCAAATGGAATGGAATGGAATGGAATGGAATGGAATGAACTCAAATGGATTAGCATGGAATGAAGTGGACTCGAATACAATGGAATGGAATGGACTCGAATGGAATGGAACGGACTTGAACGGAATGGAGTGGAATGGACTCGAATGGAATGGAGTTGAATGGACTCGAATGGAATGGAATGTAAAGGAATGGAATGAACTCGAAAGGAGTGGAATGTAATGGAATGAAATGGACTCGAATGGAATTAAATGGAATGGAACGGAATGGACTGGGATGGAATGGAACGGAACGGAACGCAGTTGAATTGAACGGACCCGGAATGGAATGGAATGGAATGGAATGAAATGGAATGAAGTGGACTCTAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATAGACTCGAATGAAATGGGATGGACTCGAATGGAATGGAACGGAATGGAATCGACTCGAGTGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATAGAATGGAATTGACTCAATTTAAATGGAATGAAGTGGAATGAACTCGAATGGCATGCAATGGAATGGAATAGAATCGAATGGAATAGAATGGACCCAAATGGAATGGAACGGAATGGAATGGAATGGAACGGAATGGAATAGAACGGAACGGAATGGAATGGATTGGAATGAACTCCAACGGAATGGAATGGACTCGAATGCAATGGAATGGAATGGAATGGAATGGAGTGGACTGGAATGGAATAGAATGGAATGGAATGGATAGGACTGGAATGAAATGGAATGGAATGGACTCGAATGGAATGGAATGGAATGGAATGGACTCAAATGGAATGGAATGGAATGGAATGGACACGAATGGAATGGAATTGAATGGAATGGAATGGACTGTATGAAAAGGAATGGATTGGAAAGGAATGGAATAGAACGGAATGGACTCGAATGGAATGGAAAGGACTCGAGTGGAATGGAATGGAATGGAATGGACTCGAATGGAATGGAGTGGAATGTATGCGAATGGAATGGAATTGAATGGATTCGAGTCTAACGGAATGTATGGAATGGACTCGAATGGAATGTAATGTAATGGAATGAAATGGACGCGAATGGAATGGAATGGAATGGAATGGAATGGAGTGGAATGGAATGGACTCGAATGGTATGGAATGGAATTGAATGGACTCGATAGGAATGGAATGGAATGGATTGGACTCGAAAGGAATGTAATGGAATGAAATGTGCTGGAATGGAATGGAATGGAATGGAATAAAATGTAATGGAATGGACTCGAATGGAATACAGTTGAATTGAATGGACCCGAAAGCAATGGAATGGAATGGAACGGATTGGAAGGGAATGGAATGGAATGAAATGGAAAAGACTCGAATGGAATGGAATGGACTCGAATGAAATGGAGTGGACTAGAATGGAATGGAATGGACTTGAAAGAAATGGAATGCAGTGGAATGGACTCGAATGGAATGCAATGGAATGGAATAGACTCGAACGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAATGGAACGGAATGGACGCGAATGAAATGGAACGGAACGGAATGGACTCGGATGGAACGGAATGGAACGGAATGGAATGGAATTTACTCGAATGGGATGGAATGGAATGGAATTTACTCGAATGGAATGGAATGGAATGGACTGAAATGGAATAGCATGGAATGGAATGGACTCGAATGCAATGGAATGGAATGGACTCGAATGGAACGGAATGGACTCGAACGGAGTGGAGTCGAATGGATTCGAATGGAATGCAATGGAATGGAACGGAATGCAATGTACTCGAATGGAATGGAATGTAATAGAATGAAAATTACTCGAATGGAATGGAATGGAATGGACTCCAATGGAATGGAATCGAACGGACTCGAATAGAATGCAGTTGAATTGAATGGACCTGAAAGAATGCAATGGAATGGAATGAAATGGACTCGAATGGAATGGAATAGACTGAAATGAAATGGAATGTACTGGAATGGAATGGAATGGAATGTACTGGAATGGAATGGAATGGACTCGAATGATATGCAATTGAATGGACTCGCATGGATTGGAATGGACTCTAGTGGAATGGAATGGAATA	30	PASS	.	GT	."""
        ) # fmt: skip
        # cSpell:enable

        ref_kmer_counts_prefix = f"/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k{cfg.kmer.kmer_size}"
        if not os.path.exists(f"{ref_kmer_counts_prefix}.kmc_pre"):
            pytest.skip("Reference kmer counts not found")
        ref_kmer_counts = KmerCounts(ref_kmer_counts_prefix)

        region = Range("chr1:789386-789670")
        graph, unique_kmers, haplotype_sampler = _create_graph_and_sampler(
            cfg.reference,
            vcf_path,
            region,
            k=cfg.kmer.kmer_size,
            ref_kmer_counts=ref_kmer_counts,
        )
        # The graph construction automatically "deduplicates" the overlapping insertions, so the graph should have 2
        # alternate alleles

        filtered_kmer_path = _cache_filter_kmc_database(cfg, hg00733_sample, vcf_path, region, unique_kmers, tmp_path=tmp_path)
        haplotypes, diplotypes = _sample_diplotypes_from_counts(
            haplotype_sampler,
            unique_kmers,
            filtered_kmer_path,
            k=cfg.kmer.kmer_size,
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
    )
    def test_population_scale_overlapping_alleles(self, cfg, hg00733_sample, tmp_path):
        vcf_path = "/storage/mlinderman/projects/sv/npsv3-experiments/resources/HG00733.hgsvc3-hprc-2024-02-23.dipcall.population.passing.hg38.vcf.gz"
        if not os.path.exists(vcf_path):
            pytest.skip("HG00733 VCF not found")

        ref_kmer_counts_prefix = f"/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k{cfg.kmer.kmer_size}"
        if not os.path.exists(f"{ref_kmer_counts_prefix}.kmc_pre"):
            pytest.skip("Reference kmer counts not found")
        ref_kmer_counts = KmerCounts(ref_kmer_counts_prefix)

        region = Range("chr1:789386-789670")
        graph, unique_kmers, haplotype_sampler = _create_graph_and_sampler(
            cfg.reference,
            vcf_path,
            region,
            k=cfg.kmer.kmer_size,
            ref_kmer_counts=ref_kmer_counts,
        )

        filtered_kmer_path = _cache_filter_kmc_database(cfg, hg00733_sample, vcf_path, region, unique_kmers, tmp_path=tmp_path)
        haplotypes, diplotypes = _sample_diplotypes_from_counts(
            haplotype_sampler,
            unique_kmers,
            filtered_kmer_path,
            k=cfg.kmer.kmer_size,
            kmer_coverage=hg00733_sample.kmer_coverage,
            filter_kmers=False,
        )

        # The "correct" insertion allele is node 105, but is not necessarily sampled. The sampled vs. true insertion alleles have 238
        # mismatches out of a 2369bp insertion, i.e., are very similar. We would want the correct allele to be ranked higher, but not
        # necessarily penalize the other alleles much given the similarity.

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
