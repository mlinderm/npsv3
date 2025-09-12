import os

import pysam
import pytest
import torch

from npsv3.genotype import OnlinePackedImageDataModule, genotype
from npsv3.simulation import bwa_index_loaded
from npsv3.util.range import Range

from . import B37_REF_FASTA, EXPERIMENTS_DIR, data_path


@pytest.mark.skipif(not B37_REF_FASTA or not bwa_index_loaded(B37_REF_FASTA), reason="B37 reference in SHM required")
@pytest.mark.cfg_overrides(
    f"reference={B37_REF_FASTA}",
    "simulation.replicates=1",
    "pileup=unphased_variant",
)
@pytest.mark.usefixtures("ray_setup")
class TestOnlinePackedImageDataLoader:
    def test_packed_loader(self, cfg, hg002_sample):
        # TODO: Enable genotyping without a background
        dm = OnlinePackedImageDataModule(
            cfg,
            read_path=data_path("12_22127565_22132387.bam"),
            sample=hg002_sample,
            inference_vcf=data_path("12_22129565_22130387.vcf.gz"),
            background_vcf=data_path("12_22129565_22130387.background.vcf.gz"),
        )
        dm.prepare_data()

        dm.setup(stage="predict")

        for _i, batch in enumerate(dm.predict_dataloader()):
            images, labels, offsets, regions, *addl_fields = batch
            assert images.size(0) == 4+1, "Batch size must be 1 replicate of 4 genotypes + 1 query image"
            assert labels.equal(torch.tensor([0,3,3,1], dtype=torch.long)), "All replicates of correct genotype (3) should have 1 label"
            assert offsets.equal(torch.tensor([0, 4], dtype=torch.long)), "Offsets should match the support images"
            assert regions == [str(Range("12",22129565,22130387).expand(cfg.pileup.variant_padding))]
        dm.teardown(stage="predict")


@pytest.mark.skipif(not B37_REF_FASTA or not bwa_index_loaded(B37_REF_FASTA), reason="B37 reference in SHM required")
@pytest.mark.cfg_overrides(
    f"reference={B37_REF_FASTA}",
    "simulation.replicates=1",
    "pileup=unphased_variant",
    "model=paired_packed_inception_infonce",
    f'+model.checkpoint="{os.path.join(EXPERIMENTS_DIR, "training/hgsvc3-hprc-2024-02-23-mc-chm13.GRCh38.vcfbub.a100k.wave.passing.training.hg38.models/data.batch_size=1024,data=packed_images,model=paired_packed_inception_infonce,trainer.max_epochs=10/model.ckpt")}"',
)
@pytest.mark.usefixtures("ray_setup")
class TestGenotyping:
    def test_genotype_to_vcf(self, tmp_path, cfg, hg002_sample):
        output_path = tmp_path / "genotypes.vcf"
        genotype(
            cfg,
            data_path("12_22127565_22132387.bam"),
            hg002_sample,
            inference_vcf=data_path("12_22129565_22130387.vcf.gz"),
            output_path=str(output_path),
            background_vcf=data_path(
                "12_22129565_22130387.background.vcf.gz"
            ),  # TODO: Enable genotyping without a background
        )

        assert output_path.exists()
        with pysam.VariantFile(output_path) as vcf_file:
            for _i, record in enumerate(vcf_file):
                print(record)
                sample = record.samples["HG002"]
                assert "GT" in sample, f"Missing GT in sample entry for {record}"
                assert "MT" in sample, f"Missing MT in sample entry for {record}"
            assert _i == 0, f"Expected 1 record, found {_i + 1}"
