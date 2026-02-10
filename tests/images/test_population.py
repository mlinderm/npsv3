import glob
import os

import pysam
import pytest

from omegaconf import OmegaConf
from npsv3._native_graph import VariantFileReader
from npsv3.images.population import split_and_filter_vcf
from npsv3.util.vcf import index_variant_file

from .. import EXPERIMENTS_DIR, HG38_REF_FASTA, _first_existing

# A ~210bp insertion allele used across tests (SV since >=50bp)
_SV_INS_ALLELE = "CCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG"

def _create_vcf(tmp_path: str, vcf: bytes, name: str = "test.vcf.gz") -> str:
    vcf_path = os.path.join(tmp_path, name)
    with pysam.BGZFile(vcf_path, "wb") as vcf_file:
        vcf_file.write(vcf)
    index_variant_file(vcf_path)
    return vcf_path

def _sample_vcf_paths(output_dir: str) -> dict[str, str]:
    """Return dict of sample name -> output VCF path for files created by split_and_filter_vcf."""
    paths = {}
    for path in glob.glob(os.path.join(output_dir, "*.vcf.gz")):
        sample = os.path.basename(path).removesuffix(".vcf.gz")
        paths[sample] = path
    return paths

def _count_variants(vcf_path: str) -> int:
    reader = VariantFileReader.open(vcf_path)
    return sum(1 for _ in reader.fetch())


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

    def test_identify_training_regions(self, tmp_path, cfg):
        input_vcf = _first_existing(
            "/data/hgsvc3-hprc-2024-02-23.dipcall.training.sv.hg38.vcf.gz",
            os.path.join(EXPERIMENTS_DIR, 'resources', 'hgsvc3-hprc-2024-02-23.dipcall.training.sv.hg38.vcf.gz')
        )
        if input_vcf is None:
            pytest.skip("Input VCF hgsvc3-hprc-2024-02-23.dipcall.training.sv.hg38.vcf.gz not found")

        local_conf = OmegaConf.from_dotlist([f"input={input_vcf}"])
        cfg = OmegaConf.merge(cfg, local_conf)

        split_and_filter_vcf(
            cfg,
            inference_vcf=cfg.input,
            output_dir=tmp_path,
        )

    def test_split_passing_sv_creates_sample_vcfs(self, tmp_path, cfg):
        """Passing SV: carrier is positive, non-carriers are negative, all get output VCFs."""
        vcf_path = _create_vcf(tmp_path, f"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3
chr1	3693767	.	C	C{_SV_INS_ALLELE}	30	PASS	.	GT	0|1	./.	./.
""".encode())  # fmt: skip

        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(output_dir)
        split_and_filter_vcf(cfg, vcf_path, output_dir)

        sample_vcfs = _sample_vcf_paths(output_dir)
        # Sample1 is positive (has passing SV), Sample2 and Sample3 are negative (no SV alleles)
        assert "Sample1" in sample_vcfs
        assert "Sample2" in sample_vcfs
        assert "Sample3" in sample_vcfs
        # Each sample VCF should contain the one passing variant
        for sample in sample_vcfs:
            assert _count_variants(sample_vcfs[sample]) == 1

    def test_split_filtered_sv_excluded_from_positive(self, tmp_path, cfg):
        """Sample with only a filtered SV should not be positive or negative."""
        vcf_path = _create_vcf(tmp_path, f"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FILTER=<ID=GAP1,Description="Uncalled in the first haplotype">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3
chr1	3693767	.	C	C{_SV_INS_ALLELE}	30	PASS	.	GT	0|1	./.	./.
chr1	3693767	.	C	C{_SV_INS_ALLELE}CCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	GAP1	.	GT	./.	0|1	./.
""".encode())  # fmt: skip

        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(output_dir)
        split_and_filter_vcf(cfg, vcf_path, output_dir)

        sample_vcfs = _sample_vcf_paths(output_dir)
        # Sample1 is positive (passing SV only)
        assert "Sample1" in sample_vcfs
        # Sample2 has a filtered SV, so it is neither positive nor negative
        assert "Sample2" not in sample_vcfs
        # Sample3 has no SVs at all, so it is negative
        assert "Sample3" in sample_vcfs

    def test_split_no_passing_variants_skips_region(self, tmp_path, cfg):
        """Region with only filtered variants should be skipped entirely."""
        vcf_path = _create_vcf(tmp_path, f"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FILTER=<ID=GAP1,Description="Uncalled in the first haplotype">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2
chr1	3693767	.	C	C{_SV_INS_ALLELE}	30	GAP1	.	GT	0|1	./.
""".encode())  # fmt: skip

        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(output_dir)
        split_and_filter_vcf(cfg, vcf_path, output_dir)

        sample_vcfs = _sample_vcf_paths(output_dir)
        # No output files since the only variant is filtered
        assert len(sample_vcfs) == 0

    def test_split_negative_excludes_filtered_sv_carriers(self, tmp_path, cfg):
        """Samples carrying any SV allele (even filtered) should not be negative."""
        vcf_path = _create_vcf(tmp_path, f"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FILTER=<ID=GAP1,Description="Uncalled in the first haplotype">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3
chr1	3693767	.	C	C{_SV_INS_ALLELE}	30	PASS	.	GT	0|1	./.	./.
chr1	3693767	.	C	C{_SV_INS_ALLELE}CCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	GAP1	.	GT	./.	./.	0|1
""".encode())  # fmt: skip

        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(output_dir)
        split_and_filter_vcf(cfg, vcf_path, output_dir)

        sample_vcfs = _sample_vcf_paths(output_dir)
        # Sample1 is positive (passing SV)
        assert "Sample1" in sample_vcfs
        # Sample2 has no SVs at all -> negative
        assert "Sample2" in sample_vcfs
        # Sample3 has a filtered SV -> excluded from both positive and negative
        assert "Sample3" not in sample_vcfs

    def test_split_only_writes_passing_variants(self, tmp_path, cfg):
        """Output VCFs should only contain the passing variants, not filtered ones."""
        vcf_path = _create_vcf(tmp_path, f"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FILTER=<ID=GAP1,Description="Uncalled in the first haplotype">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2
chr1	3693767	.	C	C{_SV_INS_ALLELE}	30	PASS	.	GT	0|1	./.
chr1	3693768	.	C	G	30	GAP1	.	GT	./.	./.
""".encode())  # fmt: skip

        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(output_dir)
        split_and_filter_vcf(cfg, vcf_path, output_dir)

        sample_vcfs = _sample_vcf_paths(output_dir)
        # Both samples should get output (Sample1 positive, Sample2 negative)
        assert len(sample_vcfs) == 2
        # Only the passing SV variant should be written, not the filtered SNV
        for sample in sample_vcfs:
            assert _count_variants(sample_vcfs[sample]) == 1