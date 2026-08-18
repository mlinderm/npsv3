import os

import numpy as np
import pytest
import webdataset as wds
from omegaconf import OmegaConf

from npsv3._native_graph import KmerCounts
from npsv3.graphs.haplotype import _create_graph_and_sampler, _filter_variants, serialize_graph_and_unique_kmers
from npsv3.images.example import (
    example_to_image,
    make_graph_example,
    vcf_to_graph_examples,
)
from npsv3.simulation import bwa_index_loaded
from npsv3.util.range import Range
from npsv3.util.sample import kmc_filter
from npsv3.util.variant import VariantFileReader

from .. import (
    HG00731_HG38_BAM,
    HG38_REF_FASTA,
    cache_filter_kmc_database,
    create_vcf,
    hash_vcf_file,
    result_path,
)


@pytest.mark.cfg_overrides(
    f"reference={HG38_REF_FASTA}",
    "input=/storage/mlinderman/projects/sv/npsv3-experiments/resources/HG00733.hgsvc3-hprc-2024-02-23.dipcall.passing.hg38.vcf.gz",
    f"reads={HG00731_HG38_BAM}",
    "kmer.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${kmer.kmer_size}",
    "simulation.replicates=1",
)
class TestHG00733GraphExamples:
    def test_graph_to_example(self, cfg, hg00733_sample, tmp_path):
        # cSpell:disable
        vcf_path = create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=chr1,length=248956422>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00733
chr1	1134739	.	A	ACATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGCACCGTGAGGGGTGTGGCTTCCTCGC	30	PASS	.	GT	1|0
chr1	1134748	.	T	G	30	PASS	.	GT	1|1
chr1	1134771	.	G	GCACCGTGAGGGGTGTGGCTTCCTCGCCATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGCACCGTGAGGGGTGTGGCTTCCTCGCCATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGA	30	PASS	.	GT	0|1"""
        ) # fmt: skip
        # cSpell:enable

        region = Range("chr1:1134644-1134867")
        ref_kmer_counts = KmerCounts(cfg.kmer.ref_kmer_counts_kmc_prefix)
        graph, unique_kmers, sampler = _create_graph_and_sampler(
            cfg.reference,
            vcf_path,
            region,
            k=cfg.kmer.kmer_size,
            ref_kmer_counts=ref_kmer_counts,
        )

        filtered_kmer_path = cache_filter_kmc_database(cfg, hg00733_sample, vcf_path, region, unique_kmers, tmp_path=tmp_path)
        with VariantFileReader.open(vcf_path) as vcf_file:
            analysis_variants = _filter_variants(list(vcf_file.fetch(region)), min_variant_size=50)

        example = make_graph_example(
            cfg,
            graph,
            sampler,
            region,
            vcf_path,
            hg00733_sample,
            cfg.reads,
            filtered_kmer_path,
            analysis_variants,
        )

        assert example["region"] == str(region)

        assert "image" in example
        assert example["image"].shape == (cfg.pileup.image_height, cfg.pileup.image_width, len(cfg.pileup.image_channels))

        assert example["label"] > 0

        assert "sim.images" in example
        # This region has 10 possible genotypes for the available haplotypes (4 haplotypes, so 4*(4+1)/2 genotypes)
        assert example["sim.images"].shape == (cfg.simulation.replicates, 10, *example["image"].shape)

        png_path = tmp_path / "test.png"
        #png_path = result_path("test.png")
        example_to_image(cfg, example, png_path, select_channels=[0, 1, 4], with_simulations=True) # ALIGNED, PAIRED, BASEQ
        assert os.path.exists(png_path)


    @pytest.mark.usefixtures("ray_setup")
    def test_vcf_to_dataset(self, cfg, hg00733_sample, tmp_path):
        # cSpell:disable
        vcf_path = create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=chr1,length=248956422>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00733
chr1	1134739	.	A	ACATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGCACCGTGAGGGGTGTGGCTTCCTCGC	30	PASS	.	GT	1|0
chr1	1134748	.	T	G	30	PASS	.	GT	1|1
chr1	1134771	.	G	GCACCGTGAGGGGTGTGGCTTCCTCGCCATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGCACCGTGAGGGGTGTGGCTTCCTCGCCATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGA	30	PASS	.	GT	0|1"""
        ) # fmt: skip
        # cSpell:enable

        region = Range("chr1:1134644-1134867")

        # Use cached graph/unique-kmer shards and filtered k-mers (keyed on the VCF contents) to speed up repeated
        # runs of the test
        results_directory = result_path(f"{region.slug}.{hash_vcf_file(vcf_path)}.{hg00733_sample.name}.k{cfg.kmer.kmer_size}")
        os.makedirs(results_directory, exist_ok=True)
        filtered_kmer_path = os.path.join(results_directory, "filtered_kmers")
        graph_shards = [os.path.join(results_directory, "graphs-00000.tar.gz")]
        if not os.path.exists(f"{filtered_kmer_path}.kmc_pre") or not all(os.path.exists(shard) for shard in graph_shards):
            if not hg00733_sample.kmc_prefix:
                pytest.skip(f"KMC database for {hg00733_sample.name} not found")
            graph_shards, unique_kmer_path, _region_count = serialize_graph_and_unique_kmers(
                cfg,
                vcf_path,
                ref_kmer_counts_path=cfg.kmer.ref_kmer_counts_kmc_prefix,
                output_dir=results_directory,
                pool_kmers=True,
                region=region,
            )
            kmc_filter(hg00733_sample.kmc_prefix, unique_kmer_path, filtered_kmer_path, threads=cfg.threads)  # type: ignore

        output_dir = str(tmp_path / "shards")
        vcf_to_graph_examples(
            cfg,
            cfg.reads,
            hg00733_sample,
            vcf_path,
            output_dir,
            graph_shards=graph_shards,
            filtered_kmer_path=filtered_kmer_path,
        )
        assert os.path.exists(output_dir)

        dataset = wds.WebDataset(os.path.join(output_dir, "images-0000.tar.gz"), shardshuffle=False).decode()
        sample_count = 0
        for sample in dataset:
            assert sample["image.npy.gz"].shape == (cfg.pileup.image_height, cfg.pileup.image_width, len(cfg.pileup.image_channels))
            assert sample["label.cls"] > 0
            assert sample["sim.images.npy.gz"].shape == (
                cfg.simulation.replicates,
                10, # This region has 10 possible genotypes
                *sample["image.npy.gz"].shape
            )
            sample_count += 1
        assert sample_count == 1, "Only one sample in dataset"

# @pytest.mark.skipif(not os.path.exists(B37_REF_FASTA), reason="B37 reference required")
# @pytest.mark.cfg_overrides(
#     f"reference={B37_REF_FASTA}", "pileup=region",
# )
# @pytest.mark.usefixtures("ray_setup")
# class TestRegionToExample:
#     def test_single_region(self, tmp_path, cfg, hg002_sample):
#         region = Range("12", 22129564, 22130387)
#         example = make_example_from_region(
#             cfg,
#             region,
#             data_path("12_22127565_22132387.bam"),
#             hg002_sample,
#         )

#         assert example["region"] == str(region)
#         assert example["image"].shape == (cfg.pileup.image_height, region.length, len(cfg.pileup.image_channels))

#         png_path = str(tmp_path / "test.png")
#         # png_path = result_path("test.png")
#         example_to_image(cfg, example, png_path, select_channels=[0, 1, 4]) # ALIGNED, PAIRED, BASEQ
#         assert os.path.exists(png_path)

#     def test_vcf_to_shards(self, tmp_path, cfg, hg002_sample):
#         output_dir = str(tmp_path / "shards")
#         vcf_to_region_examples(
#             cfg,
#             data_path("12_22127565_22132387.bam"),
#             hg002_sample,
#             data_path("12_22129565_22130387.vcf.gz"),
#             output_dir,
#         )
#         assert os.path.exists(output_dir)

#         dataset = wds.WebDataset(os.path.join(output_dir, "images-0000.tar"), shardshuffle=False).decode()
#         for _i, sample in enumerate(dataset):
#             region = Range.parse_slug(sample["__key__"])
#             assert sample["image.npy.gz"].shape == (cfg.pileup.image_height, region.length, len(cfg.pileup.image_channels))
#         assert _i == 0, "Only one sample in dataset"


# @pytest.mark.skipif(
#     not B37_REF_FASTA or not bwa_index_loaded(B37_REF_FASTA), reason="B37 reference in SHM required"
# )
# @pytest.mark.cfg_overrides(
#     f"reference={B37_REF_FASTA}", "simulation.replicates=1", "pileup=unphased",
# )
# class TestB37GraphToExample:
#     def test_single_del(self, tmp_path, cfg, hg002_sample):
#         region = Range("12", 22129564, 22130387)
#         example = make_graph_example_from_region(
#             cfg,
#             region,
#             data_path("12_22127565_22132387.bam"),
#             hg002_sample,
#             data_path("12_22129565_22130387.background.vcf.gz"),
#             data_path("12_22129565_22130387.vcf.gz"),
#         )

#         assert Range.parse_literal(example["region"]) >= region
#         assert example["image"].shape == (cfg.pileup.image_height, cfg.pileup.image_width, len(cfg.pileup.image_channels))
#         assert example["label"] == 3  # 1/1 genotype

#         assert example["sim.images"].shape == (
#             4,  # 4 genotypes possible: 0/0, 0|1, 1|0, 1/1
#             cfg.simulation.replicates,
#             cfg.pileup.image_height,
#             cfg.pileup.image_width,
#             len(cfg.pileup.image_channels),
#         ), "Bi-allelic variant with single background should have 4 phased diploid genotypes"

#         png_path = str(tmp_path / "test.png")
#         example_to_image(cfg, example, png_path, with_simulations=True, select_channels=[0, 1, 5]) # ALIGNED, PAIRED, ALLELE
#         assert os.path.exists(png_path)

#     @pytest.mark.usefixtures("ray_setup")
#     def test_vcf_to_shards(self, tmp_path, cfg, hg002_sample):
#         output_dir = str(tmp_path / "shards")
#         vcf_to_graph_examples(
#             cfg,
#             data_path("12_22127565_22132387.bam"),
#             hg002_sample,
#             data_path("12_22129565_22130387.vcf.gz"),
#             output_dir,
#             background_vcf=data_path("12_22129565_22130387.background.vcf.gz"),
#         )
#         assert os.path.exists(output_dir)

#         dataset = wds.WebDataset(os.path.join(output_dir, "images-0000.tar"), shardshuffle=False).decode()
#         for _i, sample in enumerate(dataset):
#             assert sample["image.npy.gz"].shape == (cfg.pileup.image_height, cfg.pileup.image_width, len(cfg.pileup.image_channels))
#             assert sample["label.cls"] == 3
#             assert sample["sim.images.npy.gz"].shape == (
#                 4,
#                 cfg.simulation.replicates,
#                 cfg.pileup.image_height,
#                 cfg.pileup.image_width,
#                 len(cfg.pileup.image_channels),
#             )
#             np.testing.assert_array_equal(sample["label.inf.npy"], [0, 0, 0, 1]) # Label is 1/1, which has only one inference positive (itself)
#         assert _i == 0, "Only one sample in dataset"

#     @pytest.mark.skipif(
#         not SYNDIP_BAM or not SYNDIP_VCF or not SYNDIP_SV_VCF, reason="syndip dataset required"
#     )
#     @pytest.mark.parametrize(
#         "region",
#         [
#             #Range("1", 1259552, 1259552),
#             #Range("1", 2765871, 2765903),
#             #Range("1", 1501010, 1501010),
#             Range("1", 33439881, 33440008),
#         ],
#     )
#     def test_syndip_images(self, cfg, syndip_sample, region):
#         local_conf = OmegaConf.from_dotlist([
#             #f"simulation.save_sim_bam_dir={RESULT_DIR}",
#         ])
#         cfg = OmegaConf.merge(cfg, local_conf)

#         example = make_graph_example_from_region(
#             cfg,
#             region,
#             SYNDIP_BAM,
#             syndip_sample,
#             SYNDIP_VCF,
#             SYNDIP_SV_VCF,
#         )
#         #png_path = str(tmp_path / "test.png")
#         png_path = result_path("test.png")
#         example_to_image(cfg, example, png_path, with_simulations=True, render_channels=True, select_channels=[0, 1, 5]) # ALIGNED, PAIRED, ALLELE
#         assert os.path.exists(png_path)

#     @pytest.mark.skipif(
#         not HG002_B37_BAM or not HG002_GIAB_VCF or not HG002_GIAB_SV_VCF, reason="HG002 dataset required"
#     )
#     @pytest.mark.parametrize(
#         "region",
#         [
#             Range("1",16490549,16490549) # A hom. ref. INS that can be a FP with incomplete GATK-produced background (missing 10bp DEL, SNV)
#         ],
#     )
#     def test_hg002_images(self, cfg, hg002_sample, region):
#         local_conf = OmegaConf.from_dotlist([
#             #f"simulation.save_sim_bam_dir={RESULT_DIR}",
#         ])
#         cfg = OmegaConf.merge(cfg, local_conf)

#         example = make_graph_example_from_region(
#             cfg,
#             region,
#             HG002_B37_BAM,
#             hg002_sample,
#             HG002_GIAB_VCF,
#             HG002_GIAB_SV_VCF,
#         )
#         #png_path = str(tmp_path / "test.png")
#         png_path = result_path("test.png")
#         example_to_image(cfg, example, png_path, with_simulations=True, render_channels=False, select_channels=[0, 1, 5]) # ALIGNED, PAIRED, ALLELE
#         assert os.path.exists(png_path)

#     @pytest.mark.skipif(not HG002_B37_BAM, reason="HG002 dataset required")
#     def test_ins_background_images(self, cfg, hg002_sample):
#         local_conf = OmegaConf.from_dotlist([
#             #f"simulation.save_sim_bam_dir={RESULT_DIR}",
#         ])
#         cfg = OmegaConf.merge(cfg, local_conf)

#         region = Range("1", 16490549, 16490549)
#         example = make_graph_example_from_region(
#             cfg,
#             region,
#             HG002_B37_BAM,
#             hg002_sample,
#             data_path("1_16490549_16490549.vcf.gz"),
#             data_path("1_16490549_16490549.sv.vcf.gz"),
#         )
#         #png_path = str(tmp_path / "test.png")
#         png_path = result_path("test.png")
#         example_to_image(cfg, example, png_path, with_simulations=True, render_channels=False, select_channels=[0, 1, 5]) # ALIGNED, PAIRED, ALLELE
#         assert os.path.exists(png_path)



# @pytest.mark.skipif(
#     not all((HG38_REF_FASTA, bwa_index_loaded(HG38_REF_FASTA))), reason="HG38 reference in SHM required"
# )
# @pytest.mark.cfg_overrides(
#     f"reference={HG38_REF_FASTA}", "simulation.replicates=1", "pileup=unphased",
# )
# @pytest.mark.usefixtures("ray_setup")
# class TestHG38GraphToExample:
#     @pytest.mark.skipif(
#         not all((NA12878_BAM, NA12878_VCF, NA12878_SV_VCF)), reason="NA12878 dataset required"
#     )
#     @pytest.mark.parametrize(
#         ("region", "label", "ranked_label"),
#         [
#             (Range("chr1", 150506578,150506578), 0, [1] + [0] * 8), # 0/0 only has a true positive
#             (Range("chr1", 2994972, 2995286), 1, [0, 1, 2, 3]),  # 0/1 genotype, with other phase as "second-rank" and 1/1 as "third-rank"
#             (Range("chr1", 977943, 978002), 2, None) # 0/1 DEL, with complex upstream region
#         ],
#     )
#     def test_na12878_images(self, cfg, na12878_sample, region, label, ranked_label):
#         local_conf = OmegaConf.from_dotlist([ ])
#         cfg = OmegaConf.merge(cfg, local_conf)

#         example = make_graph_example_from_region(
#             cfg,
#             region,
#             NA12878_BAM,
#             na12878_sample,
#             NA12878_VCF,
#             NA12878_SV_VCF,
#         )

#         if label is not None:
#             assert example["label"] == label
#         if ranked_label is not None:
#             np.testing.assert_array_equal(example["label.rank"], ranked_label)

#         #png_path = str(tmp_path / "test.png")
#         png_path = result_path("test.png")
#         example_to_image(cfg, example, png_path, with_simulations=True, render_channels=False, select_channels=[0, 1, 5]) # ALIGNED, PAIRED, ALLELE
#         assert os.path.exists(png_path)


#     @pytest.mark.skipif(
#         not HG002_HG38_BAM or not HG002_DIPCALL_VCF or not HG002_DIPCALL_SV_VCF, reason="HG002 dataset required"
#     )
#     @pytest.mark.parametrize(
#         "region",
#         [
#             #Range("chr11",24919131,24919131),
#             Range("chr1",3693946,3693946), # FP INS (called as 1/1 instead of 0/1)
#         ],
#     )
#     def test_hg002_dipcall_images(self, cfg, hg002_hg38_sample, region):
#         local_conf = OmegaConf.from_dotlist([
#             f"simulation.save_sim_bam_dir={RESULT_DIR}",
#             f"pileup.variant_padding=256",
#         ])
#         cfg = OmegaConf.merge(cfg, local_conf)

#         example = make_graph_example_from_region(
#             cfg,
#             region,
#             HG002_HG38_BAM,
#             hg002_hg38_sample,
#             HG002_DIPCALL_VCF,
#             HG002_DIPCALL_SV_VCF,
#         )
#         #png_path = str(tmp_path / "test.png")
#         png_path = result_path("test.png")
#         example_to_image(cfg, example, png_path, with_simulations=True, render_channels=True, select_channels=[0, 1, 5]) # ALIGNED, PAIRED, ALLELE
#         assert os.path.exists(png_path)

#     @pytest.mark.skipif(
#         not HG00733_HG38_BAM or not HG00733_DIPCALL_VCF or not HG00733_DIPCALL_SV_VCF, reason="HG00733 dataset required"
#     )
#     @pytest.mark.parametrize(
#         "region",
#         [
#             Range("chr13",58895789,58895790),
#         ],
#     )
#     def test_hg00733_dipcall_images(self, cfg, hg00733_sample, region):
#         local_conf = OmegaConf.from_dotlist([
#         ])
#         cfg = OmegaConf.merge(cfg, local_conf)

#         example = make_graph_example_from_region(
#             cfg,
#             region,
#             HG00733_HG38_BAM,
#             hg00733_sample,
#             HG00733_DIPCALL_VCF,
#             HG00733_DIPCALL_SV_VCF,
#         )
#         print(example)
#         #png_path = str(tmp_path / "test.png")
#         png_path = result_path("test.png")
#         example_to_image(cfg, example, png_path, with_simulations=True, render_channels=True, select_channels=[0, 1, 5]) # ALIGNED, PAIRED, ALLELE
#         assert os.path.exists(png_path)

#     @pytest.mark.skipif(
#         not HG00731_HG38_BAM or not HG00731_TRAINING_VCF or not HG00731_TRAINING_SV_VCF, reason="HG00731 dataset required"
#     )
#     @pytest.mark.parametrize(
#         "region",
#         [
#             Range("chr15", 68409525, 68409557),
#         ],
#     )
#     def test_hg00731_training_images(self, cfg, hg00731_sample, region):
#         local_conf = OmegaConf.from_dotlist([
#         ])
#         cfg = OmegaConf.merge(cfg, local_conf)

#         example = make_graph_example_from_region(
#             cfg,
#             region,
#             HG00731_HG38_BAM,
#             hg00731_sample,
#             HG00731_TRAINING_VCF,
#             HG00731_TRAINING_SV_VCF,
#         )
#         print(example)
#         #png_path = str(tmp_path / "test.png")
#         png_path = result_path("test.png")
#         example_to_image(cfg, example, png_path, with_simulations=True, render_channels=True, select_channels=[0, 1, 5]) # ALIGNED, PAIRED, ALLELE
#         assert os.path.exists(png_path)


# @pytest.mark.skipif(
#     not os.path.exists(B37_REF_FASTA) or not bwa_index_loaded(B37_REF_FASTA), reason="B37 reference in SHM required"
# )
# @pytest.mark.cfg_overrides(
#     f"reference={B37_REF_FASTA}", "simulation.replicates=1", "pileup=unphased_variant",
# )
# @pytest.mark.usefixtures("ray_setup")
# class TestVariantToExample:
#     def test_vcf_to_shards(self, tmp_path, cfg, hg002_sample):
#         output_dir = str(tmp_path / "shards")
#         vcf_to_variant_examples(
#             cfg,
#             data_path("12_22127565_22132387.bam"),
#             hg002_sample,
#             data_path("12_22129565_22130387.vcf.gz"),
#             output_dir,
#             background_vcf=data_path("12_22129565_22130387.background.vcf.gz"),
#         )
#         assert os.path.exists(output_dir)

#         dataset = wds.WebDataset(os.path.join(output_dir, "images-0000.tar"), shardshuffle=False).decode()
#         for _i, sample in enumerate(dataset):
#             assert sample["__key__"] == "553e586e2a8e7c2fd70661fec7b529c5453a9b45"
#             assert sample["region.txt"] == str(Range("12",22129565,22130387).expand(cfg.pileup.variant_padding))
#             assert sample["image.npy.gz"].shape == (cfg.pileup.image_height, cfg.pileup.image_width, len(cfg.pileup.image_channels))
#             assert sample["label.cls"] == 3
#             assert sample["sim.images.npy.gz"].shape == (
#                 4,
#                 cfg.simulation.replicates,
#                 cfg.pileup.image_height,
#                 cfg.pileup.image_width,
#                 len(cfg.pileup.image_channels),
#             )

#             png_path = result_path("test.png")

#             example_to_image(
#                 cfg,
#                 {"image": sample["image.npy.gz"], "sim.images": sample["sim.images.npy.gz"] },
#                 png_path, with_simulations=True, render_channels=True, select_channels=[0, 1, 5], # ALIGNED, PAIRED, ALLELE
#             )
#             assert os.path.exists(png_path)
#         assert _i == 0, "Only one sample in dataset"


