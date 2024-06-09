import os

import pytest
import webdataset as wds

from npsv3.images.example import example_to_image, make_example_from_region, vcf_to_region_examples
from npsv3.util.range import Range

from .. import B37_REF_FASTA, RESULT_DIR, data_path, result_path


@pytest.mark.skipif(not os.path.exists(B37_REF_FASTA), reason="B37 reference required")
@pytest.mark.cfg_overrides(
    f"reference={B37_REF_FASTA}", "generator._target_=npsv3.images.generator.CoverageImageGenerator"
)
class TestRegionToExample:
    @pytest.mark.skip
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
        for i, sample in enumerate(dataset):
            region = Range.parse_slug(sample["__key__"])
            assert sample["image.npy.gz"].shape == (cfg.pileup.image_height, region.length, 6)
        assert i == 0, "Only one sample in dataset"
