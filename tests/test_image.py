import os

import pytest

from npsv3 import image
from npsv3.simulation import bwa_index_loaded
from npsv3.util.range import Range
from npsv3.util.sample import Sample

from . import B37_REF_FASTA, RESULT_DIR, data_path, result_path


@pytest.mark.skipif(
    not os.path.exists(B37_REF_FASTA) or not bwa_index_loaded(B37_REF_FASTA), reason="B37 reference required"
)
@pytest.mark.cfg_overrides(f"reference={B37_REF_FASTA}", "simulation.replicates=1")
class TestGraphToExample:
    def test_single_del(self, tmp_path, cfg, hg002_sample):
        example = image.make_example_from_region(
            cfg,
            Range("12", 22129564, 22130387),
            data_path("12_22129565_22130387.background.vcf.gz"),
            data_path("12_22129565_22130387.vcf.gz"),
            data_path("12_22127565_22132387.bam"),
            hg002_sample,
        )

        assert image._example_image_shape(example) == (
            cfg.pileup.image_height,
            cfg.pileup.image_width,
            len(cfg.pileup.image_channels),
        )
        assert image._example_label(example) == 3 # 1/1 genotype
        assert image._example_addl_attribute(example, "fisher_strand") is not None
        assert image._example_addl_attribute(example, "strand_orientation_bias") is not None

        assert image._example_sim_images_shape(example) == (
            4, # 4 genotypes possible: 0/0, 0|1, 1|0, 1/1
            cfg.simulation.replicates,
            cfg.pileup.image_height,
            cfg.pileup.image_width,
            len(cfg.pileup.image_channels),
        ), "Bi-allelic variant with single background should have 4 phased diploid genotypes"

        png_path = str(tmp_path / "test.png")
        image.example_to_image(cfg, example, png_path, with_simulations=True)
        assert os.path.exists(png_path)

class TestVCFToExamples:
    @pytest.mark.cfg_overrides(f"reference={B37_REF_FASTA}", "simulation.replicates=1")
    @pytest.mark.skipif(
        not os.path.exists(B37_REF_FASTA) or not bwa_index_loaded(B37_REF_FASTA), reason="B37 reference required"
    )
    def test_single_del(self, tmp_path, cfg, hg002_sample):
        output_dir = RESULT_DIR  # str(tmp_path / "tfrecords")
        image.vcf_to_tfrecords(
            cfg,
            data_path("12_22129565_22130387.background.vcf.gz"),
            data_path("12_22129565_22130387.vcf.gz"),
            data_path("12_22127565_22132387.bam"),
            hg002_sample,
            output_dir,
        )

        assert os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, "000.tfrecords.gz"))

        # Load dataset with simulated data
        dataset = image.load_example_dataset(
            os.path.join(output_dir, "000.tfrecords.gz"), with_label=True, with_simulations=True
        )
        for features, label in dataset:
            assert features["image"].shape == (
                cfg.pileup.image_height,
                cfg.pileup.image_width,
                len(cfg.pileup.image_channels),
            )
            assert label == 3
            assert features["sim/images"].shape == (
                4,
                cfg.simulation.replicates,
                *features["image"].shape,
            ), "Bi-allelic variant with single background should have 4 phased diploid genotypes"

            png_path = str(tmp_path / "test.png")
            image.features_to_image(cfg, features, png_path, with_simulations=True)
            assert os.path.exists(png_path)

    # @pytest.mark.skipif(not os.path.exists(HG38_REF_FASTA), reason="B37 reference required")
    # def test_complex_region(self, cfg):
    #     image.make_examples_from_vcf(cfg, data_path("chr13_29557414_29560096.inference.vcf.gz"), data_path("chr13_29557414_29560096.vcf.gz"))

    # @pytest.mark.skipif(not os.path.exists("/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.fasta"), reason="HG38 reference required")
    # @pytest.mark.cfg_overrides(f"reference=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.fasta", "simulation.replicates=1")
    # @pytest.mark.parametrize("region", [Range("chr1", 853424, 853622)])
    # def test_problem_regions(self, tmp_path, cfg, region):
    #     example = image.make_example_from_region(
    #         cfg,
    #         region,
    #         "/storage/mlinderman/projects/sv/npsv3-experiments/training/HGSVC2_training_vcfs/HG00731.freeze4.alt.passing.training.hg38.vcf.gz",
    #         "/storage/mlinderman/projects/sv/npsv3-experiments/training/HGSVC2_training_vcfs/HG00731.freeze4.sv.alt.passing.training.hg38.vcf.gz",
    #         "/storage/mlinderman/projects/sv/npsv3-experiments/resources/sequence/HG00731.final.cram",
    #         Sample.from_json("/storage/mlinderman/projects/sv/npsv3-experiments/resources/sequence/HG00731.final.stats.json"),
    #     )
