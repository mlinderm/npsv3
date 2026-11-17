import collections

import pysam
import pytest

from npsv3.genotyping.inputs import (
    INFERENCE_ALLELE_SEPARATOR,
    _allele_key,
    check_provenance,
    file_provenance,
    genotyping_regions,
    inference_allele_id,
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
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">
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


def _tags(record, key="INFERENCE_ALLELE"):
    """The record's Number=A INFO `key` with None for missing values"""
    return [None if value in (None, ".") else value for value in record.info.get(key, (None,) * len(record.alts))]


class TestPrepareGenotypingVCF:
    def test_unchanged_population(self, tmp_path):
        population = _population_vcf(tmp_path)
        inputs = prepare_genotyping_vcf(population, tmp_path / "inputs", sample_name="OTHER")
        assert inputs.vcf_path == population, "Shouldn't rewrite the population VCF if nothing changes"
        assert inputs.inference_sites is None

    def test_unchanged_population_without_inference_vcf(self, tmp_path):
        """Without an inference VCF, the population is likely the truth set, so it is used unchanged, keeping the sample's
        private alleles (its genotypes are excluded by the genotyping configuration instead)"""
        population = create_vcf(
            tmp_path,
            HEADER
            + b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
            + b"chr1\t1000\tprivate\tC\tT\t.\tPASS\t.\tGT\t0|1\t0|0\n"
            + b"chr1\t2000\tshared\tC\tT\t.\tPASS\t.\tGT\t0|1\t0|1\n"
            + b"chr1\t3000\tmulti\tC\tA,T\t.\tPASS\t.\tGT\t0|1\t0|2\n",
            name="population.vcf.gz",
        )
        inputs = prepare_genotyping_vcf(population, tmp_path / "inputs", sample_name="S1")
        assert inputs.vcf_path == population, "Shouldn't rewrite the population VCF"
        assert inputs.inference_sites is None
        assert not (tmp_path / "inputs").exists(), "Nothing should be written"

    def test_removes_alleles_private_to_sample_with_inference_vcf(self, tmp_path):
        population = create_vcf(
            tmp_path,
            HEADER
            + b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
            + b"chr1\t1000\tprivate\tC\tT\t.\tPASS\t.\tGT\t0|1\t0|0\n"
            + b"chr1\t2000\tshared\tC\tT\t.\tPASS\t.\tGT\t0|1\t0|1\n"
            + b"chr1\t3000\tmulti\tC\tA,T\t.\tPASS\t.\tGT\t0|1\t0|2\n",
            name="population.vcf.gz",
        )
        inference = create_vcf(
            tmp_path,
            HEADER + b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n" + b"chr1\t4000\tinf\tC\tT\t30\tPASS\t.\n",
            name="inference.vcf.gz",
        )
        inputs = prepare_genotyping_vcf(population, tmp_path / "inputs", sample_name="S1", inference_vcf=inference)

        _, records = _records(inputs.vcf_path)
        assert [record.pos for record in records] == [2000, 3000, 4000]
        assert records[1].alts == ("T",), "The ALT allele only carried by S1 should be trimmed"
        assert records[1].samples["S2"]["GT"] == (0, 1), "The genotypes should be updated for the trimmed alleles"

    def test_inference_vcf_restores_private_allele(self, tmp_path):
        population = create_vcf(
            tmp_path,
            HEADER
            + b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
            + b"chr1\t1000\tprivate\tC\tT\t.\tPASS\t.\tGT\t0|1\t0|0\n"
            + b"chr1\t2000\tshared\tC\tT\t.\tPASS\t.\tGT\t0|1\t0|1\n",
            name="population.vcf.gz",
        )
        inference = create_vcf(
            tmp_path,
            HEADER + b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n" + b"chr1\t1000\tinf\tC\tT\t30\tPASS\t.\n",
            name="inference.vcf.gz",
        )
        inputs = prepare_genotyping_vcf(population, tmp_path / "inputs", sample_name="S1", inference_vcf=inference)

        _, records = _records(inputs.vcf_path)
        assert [record.pos for record in records] == [1000, 2000]
        assert not records[0].format, "The private allele has no population support"
        assert _tags(records[0]) == [inference_allele_id(_records(inference)[1][0], 1)]

    def test_merges_inference_sites(self, tmp_path):
        population = _population_vcf(tmp_path)
        inference = _inference_vcf(tmp_path)
        inputs = prepare_genotyping_vcf(population, tmp_path / "inputs", sample_name="S1", inference_vcf=inference)

        samples, records = _records(inputs.vcf_path)
        assert samples == ["S2"], "Neither the excluded sample nor the inference VCF's samples should be present"
        assert [record.pos for record in records] == [1000, 5000, 7000, 9000, 9020], "Identical records should merge"
        population_ids = _ids(population)
        assert set(_ids(inputs.vcf_path)) == set(population_ids[:1] + population_ids[2:]) | set(_ids(inference))

        by_pos = {record.pos: record for record in records}
        assert by_pos[1000].id == "del1;inf_del1"
        assert by_pos[1000].samples["S2"]["GT"] == (1, 1), "Population genotypes are kept for shared records"
        for pos in (7000, 9020):
            assert not by_pos[pos].format, "Novel sites should have no genotype fields"

        _, inference_records = _records(inference)
        for record, pos in zip(inference_records, (1000, 7000, 9020), strict=True):
            assert _tags(by_pos[pos]) == [inference_allele_id(record, 1)]
            assert by_pos[pos].info["INFERENCE_QUAL"] == (30,)
        assert "INFERENCE_ALLELE" not in by_pos[5000].info

        # The native reader (and so graph construction) reports the sites-only records as missing genotypes
        with VariantFileReader.open(inputs.vcf_path) as vcf_file:
            alleles = [tuple(variant.genotype(0).alleles) for variant in vcf_file.fetch()]
        assert alleles == [(1, 1), (0, 1), (-1, -1), (0, 1), (-1, -1)]

        # The sites file has exactly the records with inference alleles
        sites_samples, sites = _records(inputs.inference_sites)
        assert sites_samples == []
        assert [(site.pos, site.id, site.alleles) for site in sites] == [
            (by_pos[pos].pos, by_pos[pos].id, by_pos[pos].alleles) for pos in (1000, 7000, 9020)
        ]

    def test_conflicting_inference_info(self, tmp_path):
        population = _population_vcf(tmp_path)
        inference = create_vcf(
            tmp_path,
            HEADER
            + b'##INFO=<ID=INFERENCE_ALLELE,Number=1,Type=String,Description="Something else">\n'
            + b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            + b"chr1\t4000\tinf\tC\tT\t30\tPASS\t.\n",
            name="inference.vcf.gz",
        )
        with pytest.raises(ValueError, match="INFERENCE_ALLELE"):
            prepare_genotyping_vcf(population, tmp_path / "inputs", sample_name="S1", inference_vcf=inference)


def _merge(tmp_path, population, inference):
    """Prepare the genotyping VCF from `population` and `inference`, lists of (POS, ID, REF, ALT[, INFO, GT]) and
    (POS, ID, REF, ALT[, QUAL]) tuples, without excluding the population's sample.

    Returns:
        The merged records as (POS, REF, ALTs, IDs, tags, GT), where tags are the inference record ID and allele index (not
        the vg variant ID) of each ALT, and the inference records
    """
    population_lines = [
        f"chr1\t{pos}\t{record_id}\t{ref}\t{alt}\t.\tPASS\t{rest[0] if rest else '.'}\tGT\t{rest[1] if len(rest) > 1 else '0|1'}"
        for pos, record_id, ref, alt, *rest in population
    ]
    inference_lines = [
        f"chr1\t{pos}\t{record_id}\t{ref}\t{alt}\t{rest[0] if rest else 30}\tPASS\t."
        for pos, record_id, ref, alt, *rest in inference
    ]
    population_vcf = create_vcf(
        tmp_path,
        HEADER + b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n" + "\n".join(population_lines).encode() + b"\n",
        name="population.vcf.gz",
    )
    inference_vcf = create_vcf(
        tmp_path,
        HEADER + b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n" + "\n".join(inference_lines).encode() + b"\n",
        name="inference.vcf.gz",
    )
    inputs = prepare_genotyping_vcf(population_vcf, tmp_path / "inputs", sample_name="OTHER", inference_vcf=inference_vcf)
    assert _records(inputs.vcf_path)[0] == _records(population_vcf)[0], "No samples should be removed"

    _, inference_records = _records(inference_vcf)
    names = {
        inference_allele_id(record, allele): f"{record.id}_{allele}"
        for record in inference_records
        for allele in range(1, len(record.alleles))
    }
    _, records = _records(inputs.vcf_path)
    _check_invariants(records, inference_records)

    merged = []
    for record in records:
        tags = [
            None if tag is None else INFERENCE_ALLELE_SEPARATOR.join(names[t] for t in tag.split(INFERENCE_ALLELE_SEPARATOR))
            for tag in _tags(record)
        ]
        genotype = record.samples["S"]["GT"] if record.format else None
        merged.append((record.pos, record.ref, record.alts, record.id, tags, genotype))
    return merged, records


def _check_invariants(records, inference_records):
    """Every inference allele is the same allele as in exactly one record, and no allele with an inference allele is in two
    records at the same position"""
    found = collections.Counter()
    for record in records:
        for alt, tag in zip(record.alts, _tags(record), strict=True):
            for allele_id in (tag or "").split(INFERENCE_ALLELE_SEPARATOR) if tag else ():
                found[allele_id] += 1
                original = next(
                    r for r in inference_records for i in range(1, len(r.alleles)) if inference_allele_id(r, i) == allele_id
                )
                index = next(i for i in range(1, len(original.alleles)) if inference_allele_id(original, i) == allele_id)
                assert original.pos == record.pos
                assert _allele_key(original.ref, original.alleles[index]) == _allele_key(record.ref, alt)
    expected = {inference_allele_id(r, i) for r in inference_records for i in range(1, len(r.alleles))}
    assert set(found) == expected, "Every inference allele should be in a record"
    assert set(found.values()) == {1}, "No inference allele should be in more than one record"

    tagged = collections.Counter()
    for record in records:
        for alt, tag in zip(record.alts, _tags(record), strict=True):
            if tag is not None:
                tagged[(record.pos, _allele_key(record.ref, alt))] += 1
    for record in records:
        for alt in record.alts:
            key = (record.pos, _allele_key(record.ref, alt))
            if key in tagged:
                occurrences = sum(_allele_key(r.ref, a) == key[1] for r in records if r.pos == record.pos for a in r.alts)
                assert occurrences == 1, f"Allele {key} with an inference allele is in more than one record"


class TestMergeRelationships:
    """The possible relationships between the population and inference VCFs' records at the same position"""

    def test_identical(self, tmp_path):
        merged, _ = _merge(tmp_path, [(100, "p", "ACGT", "A")], [(100, "i", "ACGT", "A")])
        assert merged == [(100, "ACGT", ("A",), "p;i", ["i_1"], (0, 1))]

    def test_identical_with_different_ref_padding(self, tmp_path):
        merged, _ = _merge(tmp_path, [(100, "p", "ACGT", "A")], [(100, "i", "ACGTA", "AA")])
        assert merged == [(100, "ACGT", ("A",), "p;i", ["i_1"], (0, 1))], "The population's representation"

    def test_identical_with_longer_population_padding(self, tmp_path):
        merged, _ = _merge(tmp_path, [(100, "p", "ACGTA", "AA")], [(100, "i", "ACGT", "A")])
        assert len(merged) == 1
        _pos, ref, alts, ids, tags, genotype = merged[0]
        assert _allele_key(ref, alts[0]) == ("ACGT", "A")
        assert (ids, tags, genotype) == ("p;i", ["i_1"], (0, 1))

    def test_inference_is_subset(self, tmp_path):
        """The inference record is contained in the population's multi-allelic record"""
        merged, _ = _merge(tmp_path, [(100, "p", "ACGT", "A,AC", "AF=0.1,0.2", "1|2")], [(100, "i", "ACGT", "AC")])
        assert merged == [(100, "ACGT", ("A", "AC"), "p;i", [None, "i_1"], (1, 2))]

    def test_inference_is_superset(self, tmp_path):
        merged, records = _merge(tmp_path, [(100, "p", "ACGT", "A", "AF=0.1")], [(100, "i", "ACGT", "A,ACG", 25)])
        assert merged == [(100, "ACGT", ("A", "ACG"), "p;i", ["i_1", "i_2"], (0, 1))]
        assert records[0].info["INFERENCE_QUAL"] == (25, 25)
        assert records[0].info["AF"][0] == pytest.approx(0.1)

    def test_partial_overlap(self, tmp_path):
        """The merged record has the union of the alleles"""
        merged, _ = _merge(tmp_path, [(100, "p", "ACGT", "A,AC", ".", "1|2")], [(100, "i", "ACGT", "AC,ACG")])
        assert merged == [(100, "ACGT", ("A", "AC", "ACG"), "p;i", [None, "i_1", "i_2"], (1, 2))]

    def test_no_shared_alleles_at_same_position(self, tmp_path):
        merged, _ = _merge(tmp_path, [(100, "p", "ACGT", "A")], [(100, "i", "ACGT", "AG")])
        assert merged == [(100, "ACGT", ("A",), "p", [None], (0, 1)), (100, "ACGT", ("AG",), "i", ["i_1"], None)]

    def test_different_positions(self, tmp_path):
        merged, _ = _merge(tmp_path, [(100, "p", "ACGT", "A")], [(200, "i", "ACGT", "A")])
        assert merged == [(100, "ACGT", ("A",), "p", [None], (0, 1)), (200, "ACGT", ("A",), "i", ["i_1"], None)]

    def test_split_inference_and_multiallelic_population(self, tmp_path):
        """bcftools merges only one of the inference records, and the other's allele is moved into the merged record"""
        merged, records = _merge(
            tmp_path, [(100, "p", "ACGT", "A,AC", ".", "1|2")], [(100, "ia", "ACGT", "A", 30), (100, "ib", "ACGT", "AC", 40)]
        )
        assert merged == [(100, "ACGT", ("A", "AC"), "p;ia", ["ia_1", "ib_1"], (1, 2))]
        assert records[0].info["INFERENCE_QUAL"] == (30, 40)

    def test_multiallelic_inference_and_split_population(self, tmp_path):
        """The inference allele stays in the population record whose genotypes have it, so the inference record is split"""
        merged, _ = _merge(
            tmp_path,
            [(100, "pa", "ACGT", "A", "AF=0.3", "0|1"), (100, "pb", "ACGT", "AC", "AF=0.4", "1|0")],
            [(100, "i", "ACGT", "A,AC")],
        )
        assert merged == [
            (100, "ACGT", ("A",), "pa;i", ["i_1"], (0, 1)),
            (100, "ACGT", ("AC",), "pb", ["i_2"], (1, 0)),
        ]

    def test_removed_allele_updates_genotypes_and_info(self, tmp_path):
        """Removing an allele no genotype has renumbers the remaining alleles and subsets the Number=A INFO fields"""
        merged, records = _merge(
            tmp_path,
            [(100, "px", "ACGT", "A,AC", "AF=0,0.5", "0|2"), (100, "py", "ACGT", "A", "AF=0.6", "1|0")],
            [(100, "i", "ACGT", "A,AG", 60)],
        )
        assert merged == [
            (100, "ACGT", ("AC", "AG"), "px;i", [None, "i_2"], (0, 1)),
            (100, "ACGT", ("A",), "py", ["i_1"], (1, 0)),
        ]
        assert records[0].samples["S"].phased
        assert records[0].info["AF"][0] == pytest.approx(0.5)
        assert records[0].info["AF"][1] is None, "The inference allele has no population frequency"
        assert records[1].info["INFERENCE_QUAL"] == (60,)

    def test_inference_shares_allele_between_its_own_records(self, tmp_path):
        merged, _ = _merge(tmp_path, [(500, "p", "ACGT", "A")], [(100, "i1", "ACGT", "A,AC"), (100, "i2", "ACGT", "AC")])
        assert merged[0] == (100, "ACGT", ("A", "AC"), "i1", ["i1_1", f"i1_2{INFERENCE_ALLELE_SEPARATOR}i2_1"], None)
        assert len(merged) == 2

    def test_population_only_shared_alleles_are_unchanged(self, tmp_path):
        merged, _ = _merge(
            tmp_path, [(100, "pa", "ACGT", "A,AC", ".", "1|0"), (100, "pb", "ACGT", "AC", ".", "0|1")], [(200, "i", "C", "T")]
        )
        assert merged[:2] == [
            (100, "ACGT", ("A", "AC"), "pa", [None, None], (1, 0)),
            (100, "ACGT", ("AC",), "pb", [None], (0, 1)),
        ]


class TestGenotypingRegions:
    def test_all_analysis_regions(self, tmp_path):
        population = _population_vcf(tmp_path)
        regions = genotyping_regions(population, flank=96, min_variant_size=50)
        assert [region.start for region in regions] == pytest.approx([1000 - 96, 5000 - 96, 9000 - 96], abs=2)

    def test_inference_regions(self, tmp_path):
        population = _population_vcf(tmp_path)
        inference = _inference_vcf(tmp_path)
        inputs = prepare_genotyping_vcf(population, tmp_path / "inputs", sample_name="S1", inference_vcf=inference)

        regions = genotyping_regions(inputs.vcf_path, flank=96, min_variant_size=50, inference_sites=inputs.inference_sites)
        # The regions with the shared and novel deletions. Neither the region with only the population's ins1, nor the region
        # with del2 is, since the inference variant in it isn't an analysis variant.
        assert len(regions) == 2
        assert [region.contains(pos) for region, pos in zip(regions, (1000, 7000), strict=True)] == [True] * 2

    def test_all_regions_with_inference(self, tmp_path):
        population = _population_vcf(tmp_path)
        inference = _inference_vcf(tmp_path)
        inputs = prepare_genotyping_vcf(population, tmp_path / "inputs", sample_name="S1", inference_vcf=inference)

        regions = genotyping_regions(inputs.vcf_path, flank=96, min_variant_size=50)
        assert [region.start for region in regions] == pytest.approx([1000 - 96, 5000 - 96, 7000 - 96, 9000 - 96], abs=2)

    def test_restrict_to_region(self, tmp_path):
        population = _population_vcf(tmp_path)
        regions = genotyping_regions(population, flank=96, min_variant_size=50, region=Range("chr1", 4000, 6000))
        assert len(regions) == 1
        assert regions[0].contains(5000)

    def test_restrict_inference_regions_to_region(self, tmp_path):
        inputs = prepare_genotyping_vcf(
            _population_vcf(tmp_path), tmp_path / "inputs", sample_name="S1", inference_vcf=_inference_vcf(tmp_path)
        )
        regions = genotyping_regions(
            inputs.vcf_path, flank=96, min_variant_size=50, inference_sites=inputs.inference_sites,
            region=Range("chr1", 6000, 8000),
        )
        assert len(regions) == 1
        assert regions[0].contains(7000)


class TestProvenance:
    def test_check_provenance(self, tmp_path):
        population = _population_vcf(tmp_path)
        provenance = {"population": file_provenance(population), "sample": "S1", "inference": file_provenance(None)}
        write_provenance(tmp_path, provenance)
        check_provenance(tmp_path, provenance)
        with pytest.raises(ValueError, match="sample"):
            check_provenance(tmp_path, {**provenance, "sample": "S2"})
        check_provenance(tmp_path, {**provenance, "sample": "S2"}, keys=["population"])
