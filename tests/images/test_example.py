import os

import pytest
from omegaconf import OmegaConf
import webdataset as wds

from npsv3.simulation import bwa_index_loaded
from npsv3.images.example import (
    example_to_image,
    make_example_from_region,
    make_graph_example_from_region,
    vcf_to_region_examples,
    vcf_to_graph_examples,
)
from npsv3.util.range import Range

from .. import B37_REF_FASTA, data_path, result_path


@pytest.mark.skipif(not os.path.exists(B37_REF_FASTA), reason="B37 reference required")
@pytest.mark.cfg_overrides(
    f"reference={B37_REF_FASTA}", "generator._target_=npsv3.images.generator.CoverageImageGenerator"
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
        assert example["image"].shape == (cfg.pileup.image_height, region.length, 6)

        png_path = str(tmp_path / "test.png")
        # png_path = result_path("test.png")
        example_to_image(cfg, example, png_path)
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

        dataset = wds.WebDataset(os.path.join(output_dir, "images-0000.tar")).decode()
        for _i, sample in enumerate(dataset):
            region = Range.parse_slug(sample["__key__"])
            assert sample["image.npy.gz"].shape == (cfg.pileup.image_height, region.length, 6)
        assert _i == 0, "Only one sample in dataset"


@pytest.mark.skipif(
    not os.path.exists(B37_REF_FASTA) or not bwa_index_loaded(B37_REF_FASTA), reason="B37 reference in SHM required"
)
@pytest.mark.cfg_overrides(
    f"reference={B37_REF_FASTA}", "simulation.replicates=1", "pileup.image_channels=[0,1,2,3,4,5,6,7]",
)
@pytest.mark.usefixtures("ray_setup")
class TestGraphToExample:
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
            8,
        ), "Bi-allelic variant with single background should have 4 phased diploid genotypes"

        png_path = str(tmp_path / "test.png")
        # png_path = result_path("test.png")
        example_to_image(cfg, example, png_path, with_simulations=True)
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

        dataset = wds.WebDataset(os.path.join(output_dir, "images-0000.tar")).decode()
        for _i, sample in enumerate(dataset):
            region = Range.parse_slug(sample["__key__"])
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

    # def test_number_of_support(self):
    #     dataset = wds.WebDataset(result_path("images-0000.tar")).decode()
    #     for _i, sample in enumerate(dataset):
    #         if sample["sim.images.npy.gz"].shape[0] == 1:
    #             print(sample["__key__"], sample["sim.images.npy.gz"].shape)
    #         #assert sample["sim.images.npy.gz"].shape[0] > 1

    #         assert _i < 100