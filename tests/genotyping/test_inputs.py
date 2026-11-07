import pysam
import pytest

from npsv3.genotyping.inputs import (
    check_provenance,
    file_provenance,
    genotyping_regions,
    prepare_genotyping_vcf,
    write_provenance,
)
from npsv3.util.range import Range
from npsv3.util.variant import VariantFileReader

from .. import create_vcf

HEADER = b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422>
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SV type">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
"""

DEL_REF = b"A" * 61
INS_ALT = b"A" + b"C" * 60


def _population_vcf(tmp_path):
    return create_vcf(
        tmp_path,
        HEADER
        + b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
        + b"chr1\t1000\tdel1\t" + DEL_REF + b"\tA\t.\tPASS\tSVTYPE=DEL\tGT\t0|1\t1|1\n"
        + b"chr1\t1050\tsnv1\tC\tT\t.\tPASS\t.\tGT\t1|0\t0|0\n"
        + b"chr1\t5000\tins1\tA\t" + INS_ALT + b"\t.\tPASS\tSVTYPE=INS\tGT\t0|0\t0|1\n"
        + b"chr1\t9000\tdel2\t" + DEL_REF + b"\tA\t.\tPASS\tSVTYPE=DEL\tGT\t1|1\t0|1\n",
        name="population.vcf.gz",
    )


def _inference_vcf(tmp_path):
    # A shared record (identical alleles to del1, but a different ID), a novel SV and a novel small variant in del2's region.
    # The inference VCF's own genotypes must be dropped.
    return create_vcf(
        tmp_path,
        HEADER
        + b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tINF\n"
        + b"chr1\t1000\tinf_del1\t" + DEL_REF + b"\tA\t30\tPASS\tSVTYPE=DEL\tGT\t1/1\n"
        + b"chr1\t7000\tinf_novel\t" + DEL_REF + b"\tA\t30\tPASS\tSVTYPE=DEL\tGT\t1/1\n"
        + b"chr1\t9020\tinf_snv\tA\tG\t30\tPASS\t.\tGT\t0/1\n",
        name="inference.vcf.gz",
    )


def _records(vcf_path):
    with pysam.VariantFile(vcf_path) as vcf_file:
        return list(vcf_file.header.samples), list(vcf_file)


def _ids(vcf_path):
    with VariantFileReader.open(vcf_path) as vcf_file:
        return [variant.variant_id for variant in vcf_file.fetch()]


class TestPrepareGenotypingVCF:
    def test_unchanged_population(self, tmp_path):
        population = _population_vcf(tmp_path)
        inputs = prepare_genotyping_vcf(population, tmp_path / "inputs", sample_name="OTHER")
        assert inputs.vcf_path == population, "Shouldn't rewrite the population VCF if nothing changes"
        assert inputs.inference_ids is None
        assert not inputs.excluded_sample

    def test_excludes_sample(self, tmp_path):
        population = _population_vcf(tmp_path)
        inputs = prepare_genotyping_vcf(population, tmp_path / "inputs", sample_name="S1")
        assert inputs.excluded_sample
        assert inputs.inference_ids is None

        samples, records = _records(inputs.vcf_path)
        assert samples == ["S2"]
        assert _ids(inputs.vcf_path) == _ids(population), "Excluding the sample must not change the variants"
        assert records[0].samples["S2"]["GT"] == (1, 1) and records[0].samples["S2"].phased  # noqa: PT018

    def test_merges_inference_sites(self, tmp_path):
        population = _population_vcf(tmp_path)
        inference = _inference_vcf(tmp_path)
        inputs = prepare_genotyping_vcf(population, tmp_path / "inputs", sample_name="S1", inference_vcf=inference)
        assert inputs.excluded_sample
        assert inputs.inference_ids == frozenset(_ids(inference))

        samples, records = _records(inputs.vcf_path)
        assert samples == ["S2"], "Neither the excluded sample nor the inference VCF's samples should be present"
        assert [record.pos for record in records] == [1000, 1050, 5000, 7000, 9000, 9020], "Identical records should merge"
        assert set(_ids(inputs.vcf_path)) == set(_ids(population)) | inputs.inference_ids, "Merging must preserve variant IDs"

        by_pos = {record.pos: record for record in records}
        assert by_pos[1000].samples["S2"]["GT"] == (1, 1), "Population genotypes are kept for shared records"
        for pos in (7000, 9020):
            assert by_pos[pos].samples["S2"]["GT"] == (None, None), "Population genotypes are missing at novel sites"

        # The native reader (and so graph construction) requires a GT field in every record
        with VariantFileReader.open(inputs.vcf_path) as vcf_file:
            for variant in vcf_file.fetch():
                variant.genotype(0)


class TestGenotypingRegions:
    def test_all_analysis_regions(self, tmp_path):
        population = _population_vcf(tmp_path)
        regions = genotyping_regions(population, flank=96, min_variant_size=50)
        assert [region.start for region in regions] == pytest.approx([1000 - 96, 5000 - 96, 9000 - 96], abs=2)

    def test_inference_regions(self, tmp_path):
        population = _population_vcf(tmp_path)
        inference = _inference_vcf(tmp_path)
        inputs = prepare_genotyping_vcf(population, tmp_path / "inputs", sample_name="S1", inference_vcf=inference)

        regions = genotyping_regions(inputs.vcf_path, flank=96, min_variant_size=50, inference_vcf=inference)
        # The region with only the population's ins1 isn't genotyped. The region with del2 is, since it contains the
        # small inference variant.
        assert len(regions) == 3
        assert [region.contains(pos) for region, pos in zip(regions, (1000, 7000, 9020), strict=True)] == [True] * 3
        assert not any(region.contains(5000) for region in regions)

    def test_restrict_to_region(self, tmp_path):
        population = _population_vcf(tmp_path)
        regions = genotyping_regions(population, flank=96, min_variant_size=50, region=Range("chr1", 4000, 6000))
        assert len(regions) == 1
        assert regions[0].contains(5000)


class TestProvenance:
    def test_check_provenance(self, tmp_path):
        population = _population_vcf(tmp_path)
        provenance = {"population": file_provenance(population), "sample": "S1", "inference": file_provenance(None)}
        write_provenance(tmp_path, provenance)
        check_provenance(tmp_path, provenance)
        with pytest.raises(ValueError, match="sample"):
            check_provenance(tmp_path, {**provenance, "sample": "S2"})
        check_provenance(tmp_path, {**provenance, "sample": "S2"}, keys=["population"])
