import glob
import pathlib
import os

import pysam
import pytest

from omegaconf import OmegaConf
from npsv3._native_graph import VariantFileReader
from npsv3.images.population import split_and_filter_vcf
from npsv3.util.vcf import index_variant_file

from .. import EXPERIMENTS_DIR, HG38_REF_FASTA, _first_existing


def _create_vcf(tmp_path: str, vcf: bytes, name: str = "test.vcf.gz") -> str:
    vcf_path = os.path.join(tmp_path, name)
    with pysam.BGZFile(vcf_path, "wb") as vcf_file:
        vcf_file.write(vcf)
    index_variant_file(vcf_path)
    return vcf_path

def _sample_vcf_paths(output_dir: str|pathlib.Path) -> dict[str, str]:
    sample_to_vcf = {file.name.removesuffix(".vcf.gz"): str(file) for file in pathlib.Path(output_dir).glob("*.vcf.gz")}
    return sample_to_vcf

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

    #@pytest.mark.skip(reason="Lengthy integration test that we run manually as needed")
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
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="Difference in length between REF and ALT alleles">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3
chr1	3999762	.	ATGGAGTTGCAGGATACGCCACAGAGAGGGGAGGGGGCCACACTGCCGACGGGGCAGGCCTGGAGTTGCAGGACGTGTCACAGAGAGAGGAAGGGGCCACACTGCTGACGGGGCGGGCC	A	.	.	SVTYPE=DEL;SVLEN=-118	GT	0/1	0/0	0/0
""".encode())  # fmt: skip

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        split_and_filter_vcf(cfg, vcf_path, output_dir)

        sample_to_vcf = _sample_vcf_paths(output_dir)
        assert len(sample_to_vcf) == 3, "Expected 3 sample VCFs, got {len(sample_to_vcf)}"
        assert all(sample in sample_to_vcf for sample in ["Sample1", "Sample2", "Sample3"]), f"Expected VCFs for Sample1, Sample2, Sample3, got {list(sample_to_vcf.keys())}"
        
        for sample, vcf_path in sample_to_vcf.items():
            reader = VariantFileReader.open(vcf_path)
            samples = reader.samples()
            assert samples == [sample], f"Expected only {sample} in VCF samples, got {samples}"
            variants = list(reader.fetch())
            assert len(variants) == 1, f"Expected exactly 1 variant in {sample}'s VCF, got {len(variants)}"

    def test_split_no_passing_variants_skips_region(self, tmp_path, cfg):
        """Passing SV: carrier is positive, non-carriers are negative, all get output VCFs."""
        vcf_path = _create_vcf(tmp_path, f"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FILTER=<ID=GAP1,Description="Uncalled in the first haplotype">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="Difference in length between REF and ALT alleles">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3
chr1	3999762	.	ATGGAGTTGCAGGATACGCCACAGAGAGGGGAGGGGGCCACACTGCCGACGGGGCAGGCCTGGAGTTGCAGGACGTGTCACAGAGAGAGGAAGGGGCCACACTGCTGACGGGGCGGGCC	A	.	GAP1	SVTYPE=DEL;SVLEN=-118	GT	0/1	0/0	0/0
""".encode())  # fmt: skip

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        split_and_filter_vcf(cfg, vcf_path, output_dir)

        sample_to_vcf = _sample_vcf_paths(output_dir)
        assert len(sample_to_vcf) == 0


    def test_split_filtered_sv_excluded_from_positive(self, tmp_path, cfg):
        """Sample with only a filtered SV should not be positive or negative."""
        vcf_path = _create_vcf(tmp_path, f"""##fileformat=VCFv4.2
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FILTER=<ID=PASS,Description="All filters passed">
##FILTER=<ID=GAP1,Description="Uncalled in the first haplotype">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="Difference in length between REF and ALT alleles">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3
chr1	3999762	.	ATGGAGTTGCAGGATACGCCACAGAGAGGGGAGGGGGCCACACTGCCGACGGGGCAGGCCTGGAGTTGCAGGACGTGTCACAGAGAGAGGAAGGGGCCACACTGCTGACGGGGCGGGCC	A	.	.	SVTYPE=DEL;SVLEN=-118	GT	0/1	0/0	0/0
chr1	3999763	.	TGGAGTTGCAGGATACGCCACAGAGAGGGGAGGGGGCCACACTGCCGACGGGGCAGGCCTGGAGTTGCAGGACGTGTCACAGAGAGAGGAAGGGGCCACACTGCTGACGGGGCGGGCC	T	.	GAP1	SVTYPE=DEL;SVLEN=-117	GT	0/0	0/1	0/0
""".encode())  # fmt: skip

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        split_and_filter_vcf(cfg, vcf_path, output_dir)

        sample_to_vcf = _sample_vcf_paths(output_dir)
       
        assert "Sample1" in sample_to_vcf  # Sample1 is positive (passing SV only)
        assert "Sample2" not in sample_to_vcf # Sample2 has a filtered SV, so it is neither positive nor negative
        assert "Sample3" in sample_to_vcf  # Sample3 has no SVs at all, so it is negative

        for sample, vcf_path in sample_to_vcf.items():
            reader = VariantFileReader.open(vcf_path)
            samples = reader.samples()
            assert samples == [sample], f"Expected only {sample} in VCF samples, got {samples}"
            variants = list(reader.fetch())
            assert len(variants) == 1, f"Expected exactly 1 variant in {sample}'s VCF, got {len(variants)}"
