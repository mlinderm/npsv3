import os

import pysam
import pytest

from npsv3.images.population import split_and_filter_vcf
from npsv3.util.vcf import index_variant_file

from .. import EXPERIMENTS_DIR, HG38_REF_FASTA

def _create_vcf(tmp_path: str, vcf: bytes) -> str:
    vcf_path = os.path.join(tmp_path, "test.vcf.gz")
    with pysam.BGZFile(vcf_path, "wb") as vcf_file:
        vcf_file.write(vcf)
    index_variant_file(vcf_path)
    return vcf_path

@pytest.mark.skipif(not os.path.exists(HG38_REF_FASTA), reason="HG38 reference fasta not found")
@pytest.mark.cfg_overrides(
    f"reference={HG38_REF_FASTA}",
)
class TestMakeTrainingVCFsFromPopulation:
    def test_update_filter(self, tmp_path, cfg):
        vcf_path = _create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FILTER=<ID=GAP1,Description="Uncalled in the first haplotype">
##FILTER=<ID=GAP2,Description="Uncalled in the second haplotype"> 
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=FT,Number=1,Type=String,Description="Genotype-level filter."> 
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3	Sample4
chr1	3693767	.	C	G	30	GAP1	.	GT:FT	.|1:GAP1	.:.	0|1:.	.:.
chr1	3693768	.	C	G	30	GAP1	.	GT:FT	.|1:GAP1	.:.	0|.:GAP2	.:."""
        )  # fmt: skip
        from npsv3.images.population import update_filter
        output_path = os.path.join(tmp_path, "output.vcf.gz")
        update_filter(cfg, vcf_path, output_path)

    def test_identify_reference_samples(self, cfg):
        population_training_vcf = os.path.join(EXPERIMENTS_DIR, "resources", "hgsvc3-hprc-2024-02-23.dipcall.training.sv.hg38.vcf.gz")
        split_and_filter_vcf(
            cfg,
            inference_vcf=population_training_vcf,
            output_dir="tests/data/images/population/output",
        )