import os

import pytest
import webdataset as wds
from omegaconf import OmegaConf

from npsv3.images.example import (
    example_to_image,
    make_example_from_region,
    make_graph_example_from_region,
    vcf_to_graph_examples,
    vcf_to_region_examples,
    vcf_to_variant_examples,
)
from npsv3.simulation import bwa_index_loaded
from npsv3.util.range import Range

from .. import (
    B37_REF_FASTA,
    HG002_DIPCALL_SV_VCF,
    HG002_DIPCALL_VCF,
    HG002_HG38_BAM,
    HG38_REF_FASTA,
    NA12878_BAM,
    NA12878_SV_VCF,
    NA12878_VCF,
    SYNDIP_BAM,
    SYNDIP_SV_VCF,
    SYNDIP_VCF,
    data_path,
    result_path,
)


@pytest.mark.skipif(not os.path.exists(B37_REF_FASTA), reason="B37 reference required")
@pytest.mark.cfg_overrides(
    f"reference={B37_REF_FASTA}", "pileup=region",
)
@pytest.mark.usefixtures("ray_setup")
class TestRegionToExample:
    def test_single_region(self, tmp_path, cfg, hg002_sample):
        region = Range("12", 22129564, 22130387)
        example = make_example_from_region(
            cfg,
            region,
            data_path("12_22127565_22132387.bam"),
            hg002_sample,
        )

        assert example["region"] == str(region)
        assert example["image"].shape == (cfg.pileup.image_height, region.length, len(cfg.pileup.image_channels))

        png_path = str(tmp_path / "test.png")
        # png_path = result_path("test.png")
        example_to_image(cfg, example, png_path, select_channels=[0, 1, 4]) # ALIGNED, PAIRED, BASEQ
        assert os.path.exists(png_path)

    def test_vcf_to_shards(self, tmp_path, cfg, hg002_sample):
        output_dir = str(tmp_path / "shards")
        vcf_to_region_examples(
            cfg,
            data_path("12_22127565_22132387.bam"),
            hg002_sample,
            data_path("12_22129565_22130387.vcf.gz"),
            output_dir,
        )
        assert os.path.exists(output_dir)

        dataset = wds.WebDataset(os.path.join(output_dir, "images-0000.tar"), shardshuffle=False).decode()
        for _i, sample in enumerate(dataset):
            region = Range.parse_slug(sample["__key__"])
            assert sample["image.npy.gz"].shape == (cfg.pileup.image_height, region.length, len(cfg.pileup.image_channels))
        assert _i == 0, "Only one sample in dataset"


@pytest.mark.skipif(
    not os.path.exists(B37_REF_FASTA) or not bwa_index_loaded(B37_REF_FASTA), reason="B37 reference in SHM required"
)
@pytest.mark.cfg_overrides(
    f"reference={B37_REF_FASTA}", "simulation.replicates=1", "pileup=unphased",
)
@pytest.mark.usefixtures("ray_setup")
class TestB37GraphToExample:
    def test_single_del(self, tmp_path, cfg, hg002_sample):
        region = Range("12", 22129564, 22130387)
        example = make_graph_example_from_region(
            cfg,
            region,
            data_path("12_22127565_22132387.bam"),
            hg002_sample,
            data_path("12_22129565_22130387.background.vcf.gz"),
            data_path("12_22129565_22130387.vcf.gz"),
        )

        assert example["region"] == str(region)
        assert example["image"].shape == (cfg.pileup.image_height, cfg.pileup.image_width, len(cfg.pileup.image_channels))
        assert example["label"] == 3  # 1/1 genotype

        assert example["sim.images"].shape == (
            4,  # 4 genotypes possible: 0/0, 0|1, 1|0, 1/1
            cfg.simulation.replicates,
            cfg.pileup.image_height,
            cfg.pileup.image_width,
            len(cfg.pileup.image_channels),
        ), "Bi-allelic variant with single background should have 4 phased diploid genotypes"

        png_path = str(tmp_path / "test.png")
        # png_path = result_path("test.png")
        example_to_image(cfg, example, png_path, with_simulations=True, select_channels=[0, 1, 5]) # ALIGNED, PAIRED, ALLELE
        assert os.path.exists(png_path)

    def test_vcf_to_shards(self, tmp_path, cfg, hg002_sample):
        output_dir = str(tmp_path / "shards")
        vcf_to_graph_examples(
            cfg,
            data_path("12_22127565_22132387.bam"),
            hg002_sample,
            data_path("12_22129565_22130387.vcf.gz"),
            output_dir,
            background_vcf=data_path("12_22129565_22130387.background.vcf.gz"),
        )
        assert os.path.exists(output_dir)

        dataset = wds.WebDataset(os.path.join(output_dir, "images-0000.tar"), shardshuffle=False).decode()
        for _i, sample in enumerate(dataset):
            assert sample["image.npy.gz"].shape == (cfg.pileup.image_height, cfg.pileup.image_width, len(cfg.pileup.image_channels))
            assert sample["label.cls"] == 3
            assert sample["sim.images.npy.gz"].shape == (
                4,
                cfg.simulation.replicates,
                cfg.pileup.image_height,
                cfg.pileup.image_width,
                len(cfg.pileup.image_channels),
            )
        assert _i == 0, "Only one sample in dataset"

    @pytest.mark.skipif(
        not SYNDIP_BAM or not SYNDIP_VCF or not SYNDIP_SV_VCF, reason="syndip dataset required"
    )
    @pytest.mark.parametrize(
        "region",
        [
            #Range("1", 1259552, 1259552),
            #Range("1", 2765871, 2765903),
            #Range("1", 1501010, 1501010),
            Range("1", 33439881, 33440008),
        ],
    )
    def test_syndip_images(self, cfg, syndip_sample, region):
        local_conf = OmegaConf.from_dotlist([
            #f"simulation.save_sim_bam_dir={RESULT_DIR}",
        ])
        cfg = OmegaConf.merge(cfg, local_conf)

        example = make_graph_example_from_region(
            cfg,
            region,
            SYNDIP_BAM,
            syndip_sample,
            SYNDIP_VCF,
            SYNDIP_SV_VCF,
        )
        #png_path = str(tmp_path / "test.png")
        png_path = result_path("test.png")
        example_to_image(cfg, example, png_path, with_simulations=True, render_channels=True, select_channels=[0, 1, 5]) # ALIGNED, PAIRED, ALLELE
        assert os.path.exists(png_path)

@pytest.mark.skipif(
    not all((HG38_REF_FASTA, bwa_index_loaded(HG38_REF_FASTA))), reason="HG38 reference in SHM required"
)
@pytest.mark.cfg_overrides(
    f"reference={HG38_REF_FASTA}", "simulation.replicates=1", "pileup=unphased",
)
@pytest.mark.usefixtures("ray_setup")
class TestHG38GraphToExample:
    @pytest.mark.skipif(
        not all((NA12878_BAM, NA12878_VCF, NA12878_SV_VCF)), reason="NA12878 dataset required"
    )
    @pytest.mark.parametrize(
        "region",
        [
            #Range("chr1",150506578,150506578),
            Range("chr1", 2994972, 2995286),
        ],
    )
    def test_na12878_images(self, cfg, na12878_sample, region):
        local_conf = OmegaConf.from_dotlist([ ])
        cfg = OmegaConf.merge(cfg, local_conf)

        example = make_graph_example_from_region(
            cfg,
            region,
            NA12878_BAM,
            na12878_sample,
            NA12878_VCF,
            NA12878_SV_VCF,
        )
        #png_path = str(tmp_path / "test.png")
        png_path = result_path("test.png")
        example_to_image(cfg, example, png_path, with_simulations=True, render_channels=False, select_channels=[0, 1, 5]) # ALIGNED, PAIRED, ALLELE
        assert os.path.exists(png_path)

    def test_hg002_dipcall_images(self, cfg, hg002_hg38_sample):
        region = Range.parse_literal("chr11:24919131-24919131")
        local_conf = OmegaConf.from_dotlist([ ])
        cfg = OmegaConf.merge(cfg, local_conf)

        example = make_graph_example_from_region(
            cfg,
            region,
            HG002_HG38_BAM,
            hg002_hg38_sample,
            HG002_DIPCALL_VCF,
            HG002_DIPCALL_SV_VCF,
        )
        #png_path = str(tmp_path / "test.png")
        png_path = result_path("test.png")
        example_to_image(cfg, example, png_path, with_simulations=True, render_channels=False, select_channels=[0, 1, 5]) # ALIGNED, PAIRED, ALLELE
        assert os.path.exists(png_path)


@pytest.mark.skipif(
    not os.path.exists(B37_REF_FASTA) or not bwa_index_loaded(B37_REF_FASTA), reason="B37 reference in SHM required"
)
@pytest.mark.cfg_overrides(
    f"reference={B37_REF_FASTA}", "simulation.replicates=1", "pileup=unphased_variant",
)
@pytest.mark.usefixtures("ray_setup")
class TestVariantToExample:
    def test_vcf_to_shards(self, tmp_path, cfg, hg002_sample):
        output_dir = str(tmp_path / "shards")
        vcf_to_variant_examples(
            cfg,
            data_path("12_22127565_22132387.bam"),
            hg002_sample,
            data_path("12_22129565_22130387.vcf.gz"),
            output_dir,
            background_vcf=data_path("12_22129565_22130387.background.vcf.gz"),
        )
        assert os.path.exists(output_dir)

        dataset = wds.WebDataset(os.path.join(output_dir, "images-0000.tar"), shardshuffle=False).decode()
        for _i, sample in enumerate(dataset):
            assert sample["__key__"] == "553e586e2a8e7c2fd70661fec7b529c5453a9b45"
            assert sample["region.txt"] == str(Range("12",22129565,22130387).expand(cfg.pileup.variant_padding))
            assert sample["image.npy.gz"].shape == (cfg.pileup.image_height, cfg.pileup.image_width, len(cfg.pileup.image_channels))
            assert sample["label.cls"] == 3
            assert sample["sim.images.npy.gz"].shape == (
                4,
                cfg.simulation.replicates,
                cfg.pileup.image_height,
                cfg.pileup.image_width,
                len(cfg.pileup.image_channels),
            )

            png_path = result_path("test.png")

            example_to_image(
                cfg,
                {"image": sample["image.npy.gz"], "sim.images": sample["sim.images.npy.gz"] },
                png_path, with_simulations=True, render_channels=True, select_channels=[0, 1, 5], # ALIGNED, PAIRED, ALLELE
            )
            assert os.path.exists(png_path)
        assert _i == 0, "Only one sample in dataset"
