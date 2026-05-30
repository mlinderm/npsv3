import os
import subprocess
from shlex import quote
import tempfile

import pytest
import pandas as pd

from npsv3._native_graph import Range
from npsv3.graphs.genotype import genotypes_in_topk, sample_diplotypes, genotypes_in_topk
from npsv3.util.sample import Sample

from .. import HG38_REF_FASTA, data_path, _create_vcf

@pytest.mark.skipif(not HG38_REF_FASTA, reason="HG38 reference FASTA not found")
class TestTopkHaplotypeSampling:
    def test_correct_diplotype(self, cfg, tmp_path):
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
            pd.DataFrame(
                {
                    "region": [str(Range("chr12", 21976631, 21977453).expand(cfg.pileup.variant_padding))],
                    "variant": ["fe58cd4ae772afe360ddf77af9ff2297f4b2e809"],
                    "sample": ["HG00096"],
                    "haplotypes": [2],
                    "haplotype_idxs": [(0, 0)],
                    "diplotypes": [3],
                    "diplotype_idx": [0],
                }
            ),
        )


    @pytest.mark.cfg_overrides(
        f"reference={HG38_REF_FASTA}",
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
        assert statistics.iloc[0]["diplotype_idx"] > 0, "The true diplotype should be found, but not top-ranked"