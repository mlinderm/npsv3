import itertools
import os

import pysam
import pytest
import torch

from npsv3.genotype import OnlinePackedImageDataModule, genotype
from npsv3.simulation import bwa_index_loaded
from npsv3.util.range import Range
from npsv3.util.vcf import index_variant_file

from . import B37_REF_FASTA, EXPERIMENTS_DIR, HG002_DIPCALL_VCF, HG002_HG38_BAM, HG38_REF_FASTA, data_path

def _indexed_vcf(tmp_path, vcf: bytes) -> str:
    """Create indexed VCF from literal string"""
    vcf_path = os.path.join(tmp_path, "test.vcf.gz")
    with pysam.BGZFile(vcf_path, "wb") as vcf_file:
        vcf_file.write(vcf)
    index_variant_file(vcf_path)
    return vcf_path

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
            images, labels, offsets, [region], [alleles], *addl_fields = batch
            assert images.size(0) == 4 + 1, "Batch size must be 1 replicate of 4 genotypes + 1 query image"
            assert labels.equal(torch.tensor([0, 3, 3, 1], dtype=torch.long)), (
                "All replicates of correct genotype (3) should have 1 label"
            )
            assert offsets.equal(torch.tensor([0, 4], dtype=torch.long)), "Offsets should match the support images"
            assert region == str(Range("12", 22129565, 22130387).expand(cfg.pileup.variant_padding))
            assert len(alleles) == 4 and all(len(a) == 2 for a in alleles), "There should be 4 genotypes with 2 alleles"  # noqa: PT018
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
class TestB37Genotyping:
    def test_genotype_to_vcf(self, tmp_path, cfg, hg002_sample):
        output_path = tmp_path / "genotypes.vcf"
        # TODO: Enable genotyping without a background
        genotype(
            cfg,
            data_path("12_22127565_22132387.bam"),
            hg002_sample,
            inference_vcf=data_path("12_22129565_22130387.vcf.gz"),
            output_path=str(output_path),
            background_vcf=data_path("12_22129565_22130387.background.vcf.gz"),
        )

        assert output_path.exists()
        with pysam.VariantFile(output_path) as vcf_file:
            for _i, record in enumerate(vcf_file):
                sample = record.samples["HG002"]
                assert sample["GT"] in itertools.product((0, 1), repeat=2), (
                    f"Missing or invalid GT in sample entry for {record}"
                )
                assert len(sample["MT"]) == 4, f"Missing or invalid MT in sample entry for {record}"
            assert _i == 0, f"Expected 1 record, found {_i + 1}"

@pytest.mark.skipif(not HG38_REF_FASTA or not bwa_index_loaded(HG38_REF_FASTA), reason="HG38 reference in SHM required")
@pytest.mark.cfg_overrides(
    f"reference={HG38_REF_FASTA}",
    "simulation.replicates=1",
    "pileup=unphased_variant",
    "model=paired_packed_inception_infonce",
    f'+model.checkpoint="{os.path.join(EXPERIMENTS_DIR, "training/hgsvc3-hprc-2024-02-23-mc-chm13.GRCh38.vcfbub.a100k.wave.passing.training.hg38.models/data.batch_size=1024,data=packed_images,model=paired_packed_inception_infonce,trainer.max_epochs=10/model.ckpt")}"',
)
@pytest.mark.usefixtures("ray_setup")
class TestHG38Genotyping:
    def test_correct_ploidy(self, tmp_path, cfg, hg002_hg38_sample):
        input_path = _indexed_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="Difference in length between REF and ALT alleles">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	3999762	6281	ATGGAGTTGCAGGATACGCCACAGAGAGGGGAGGGGGCCACACTGCCGACGGGGCAGGCCTGGAGTTGCAGGACGTGTCACAGAGAGAGGAAGGGGCCACACTGCTGACGGGGCGGGCC	A,CTGGAGTTGCAGGATACGCCACAGAGAGGGGAGGGGGCCACACTGCCGACGGGGCAGGCCTGGAGTTGCAGGACGTGTCACAGAGAGAGGAAGGGGCCACACTGCTGACGGGGCGGGCC	30	.	SVTYPE=DEL;SVLEN=-118,0
"""
        )  # fmt: skip

        output_path = tmp_path / "genotypes.vcf"
        # TODO: Enable genotyping without a background
        genotype(
            cfg,
            HG002_HG38_BAM,
            hg002_hg38_sample,
            inference_vcf=input_path,
            output_path=str(output_path),
            background_vcf=HG002_DIPCALL_VCF,
        )

        assert output_path.exists()
        with pysam.VariantFile(output_path) as vcf_file:
            for _i, record in enumerate(vcf_file):
                print(record)