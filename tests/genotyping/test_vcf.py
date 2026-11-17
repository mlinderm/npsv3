"""Stage 5 of the genotyper: scoring and writing the genotyped VCF (`npsv3.genotyping.vcf`)"""

import math
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pysam
import pytest
from pysam import bcftools

from npsv3.genotyping.inputs import genotyping_regions, prepare_genotyping_vcf
from npsv3.genotyping.vcf import (
    MAX_PHRED,
    ScoringOptions,
    _RegionIndex,
    genotype_variant,
    score_diplotypes,
    scoring_options,
    write_genotypes_vcf,
)
from npsv3.images.example import EnergonExampleWriter, energon_key
from npsv3.models.energon import prepare_energon_dataset
from npsv3.util.range import Range
from npsv3.util.variant import VariantFileReader, vg_variant_id

from .. import create_vcf

SAMPLE = "HG00733"
FLANK = 100
OPTIONS = ScoringOptions(temperature=1.0)
TWO_HAPLOTYPES = [(0, 0), (0, 1), (1, 1)]
THREE_HAPLOTYPES = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

CONTIG = "##contig=<ID=chr1,length=1000000>"
HEADER_LINES = f"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
{CONTIG}
##INFO=<ID=SOURCE,Number=1,Type=String,Description="Where the record came from">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=GQ,Number=1,Type=Float,Description="A conflicting definition of GQ">
##FORMAT=<ID=PS,Number=1,Type=String,Description="A conflicting definition of PS">
"""
DELETED = "ACGTACGTAC" * 6


def _deletion(position: int) -> tuple[int, str, str]:
    return position, "G" + DELETED, "G"


def _population_line(position, record_id, ref, alt, other_genotype="1|1"):
    return f"chr1\t{position}\t{record_id}\t{ref}\t{alt}\t10\tPASS\tSOURCE=population\tGT:GQ:PS\t0|1:5:a\t{other_genotype}:5:a"


def _inference_line(position, record_id, ref, alt, qual=33):
    return f"chr1\t{position}\t{record_id}\t{ref}\t{alt}\t{qual}\tPASS\tSOURCE=inference\tGT\t1/1"


def _write_vcfs(tmp_path):
    """A population VCF with a sample to exclude, and an inference VCF. Each of the deletions defines one region."""
    population = [
        _population_line(*_deletion(1000)[:1], "called_del", *_deletion(1000)[1:]),
        _population_line(1020, "pop_snv", "A", "T"),
        _population_line(1030, "pop_multi", "A", "C,T", other_genotype="1|2"),
        _population_line(*_deletion(5000)[:1], "single_del", *_deletion(5000)[1:]),
        _population_line(5020, "single_snv", "A", "T"),
        _population_line(*_deletion(10000)[:1], "timeout_del", *_deletion(10000)[1:]),
        _population_line(10020, "timeout_snv", "A", "T"),
        _population_line(10030, "timeout_multi", "A", "C,T", other_genotype="1|2"),
        _population_line(*_deletion(15000)[:1], "rejected_del", *_deletion(15000)[1:]),
        _population_line(*_deletion(20000)[:1], "insufficient_del", *_deletion(20000)[1:]),
        # Has no analysis variant with an inference allele, so isn't genotyped
        _population_line(*_deletion(25000)[:1], "unselected_del", *_deletion(25000)[1:]),
    ]
    inference = [
        # Extends across the start of the first region, [900, 1159)
        _inference_line(890, "straddle", "A" * 21, "A"),
        # In both VCFs, so merged with the population record
        _inference_line(*_deletion(1000)[:1], "inference_del", *_deletion(1000)[1:]),
        # Contained in the population's multi-allelic record
        _inference_line(1030, "inf_multi", "A", "T"),
        # Novel, in a scored region
        _inference_line(1040, "novel", "A", "G", qual="."),
        # Inside the graph's region, but not in the graph, since every ALT is '*'
        _inference_line(1050, "star", "A", "*"),
        # The analysis variants that select the other regions
        *(
            _inference_line(*_deletion(position)[:1], f"inf_{name}_del", *_deletion(position)[1:])
            for position, name in ((5000, "single"), (10000, "timeout"), (15000, "rejected"), (20000, "insufficient"))
        ),
        _inference_line(5040, "in_single", "A", "G"),
        # Partially overlaps the population's multi-allelic record, so is merged into a record with the union of the alleles
        _inference_line(10030, "in_timeout_multi", "A", "T,G"),
        _inference_line(10040, "in_timeout", "A", "G"),
        _inference_line(15040, "in_rejected", "A", "G"),
        _inference_line(20040, "in_insufficient", "A", "G"),
        # Small, so doesn't select the region it's in
        _inference_line(25040, "in_unselected", "A", "G"),
        _inference_line(30000, "no_region", "A", "G"),
    ]
    population_vcf = create_vcf(
        tmp_path,
        (HEADER_LINES + f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{SAMPLE}\tOTHER\n" + "\n".join(population) + "\n").encode(),
        name="population.vcf.gz",
    )
    inference_vcf = create_vcf(
        tmp_path,
        (HEADER_LINES + f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{SAMPLE}\n" + "\n".join(sorted(inference, key=lambda line: int(line.split("\t")[1]))) + "\n").encode(),
        name="inference.vcf.gz",
    )
    return population_vcf, inference_vcf


def _haplotype_record(vcf_path, region, alleles_by_id, num_haplotypes):
    """A haplotype record with `num_haplotypes` candidates, whose alleles of each record in `region` are in `alleles_by_id`
    by VCF ID (None, i.e., not in the graph, for the others)"""
    with VariantFileReader.open(vcf_path) as reader, pysam.VariantFile(vcf_path, drop_samples=True) as vcf_file:
        pairs = list(zip(vcf_file.fetch(**region.pysam_fetch), reader.fetch(region), strict=True))
    variant_alleles = []
    for haplotype in range(num_haplotypes):
        variant_alleles.append(
            [
                next(
                    (alleles[haplotype] for name, alleles in alleles_by_id.items() if name in (record.id or "").split(";")),
                    None,
                )
                for record, _variant in pairs
            ]
        )
    return {
        "paths": [[haplotype] for haplotype in range(num_haplotypes)],
        "variant_ids": [variant.variant_id for _record, variant in pairs],
        "variant_alleles": variant_alleles,
    }


def _write_images(images_dir, examples, *, height=2, width=2, channels=2):
    """Write an image dataset with `examples`, a list of (region, haplotype record)"""
    rng = np.random.default_rng(20261109)
    writer = EnergonExampleWriter(0, str(images_dir))
    try:
        for region, record in examples:
            num_haplotypes = len(record["paths"])
            num_genotypes = num_haplotypes * (num_haplotypes + 1) // 2
            writer.write(
                {
                    "region": str(region),
                    "image": rng.integers(0, 255, size=(height, width, channels), dtype=np.uint8),
                    "sim.images": rng.integers(0, 255, size=(1, num_genotypes, height, width, channels), dtype=np.uint8),
                    "num_genotypes": num_genotypes,
                    "genotype_idxs": list(range(num_genotypes)),
                    "num_haplotypes": num_haplotypes,
                    "haplotypes": record,
                },
                region=region,
                sample_name=SAMPLE,
            )
    finally:
        writer.close()
    return prepare_energon_dataset(str(images_dir))


def _prediction(region, metric):
    metric = np.asarray(metric, dtype=np.float32)
    return {
        "key": energon_key(SAMPLE, region),
        "region": str(region),
        "num_genotypes": metric.shape[1],
        "num_replicates": metric.shape[0],
        "metric": metric,
    }


class TestScoring:
    def test_mean_over_replicates(self):
        call = score_diplotypes(np.array([[0.1, 0.9, 0.2], [0.3, 0.1, 0.4]]), TWO_HAPLOTYPES, OPTIONS)
        np.testing.assert_allclose(call.similarities, [0.2, 0.5, 0.3])
        assert call.called == 1
        assert call.diplotype == (0, 1)

    def test_max_over_replicates(self):
        """The model's own predictor is the best support"""
        metric = np.array([[0.1, 0.5, 0.2], [0.6, 0.1, 0.1]])
        call = score_diplotypes(metric, TWO_HAPLOTYPES, ScoringOptions(temperature=1.0, aggregate="max"))
        np.testing.assert_allclose(call.similarities, [0.6, 0.5, 0.2])
        assert call.diplotype == (0, 0)

    def test_softmax_temperature(self):
        similarities = np.array([0.1, 0.2, 0.5])
        for temperature in (0.07, 1.0):
            call = score_diplotypes(similarities[np.newaxis], TWO_HAPLOTYPES, ScoringOptions(temperature=temperature))
            expected = np.exp(similarities / temperature) / np.exp(similarities / temperature).sum()
            np.testing.assert_allclose(np.exp(call.log_probabilities), expected)
            assert call.diplotype_quality == pytest.approx(-10 * math.log10(1 - expected[2]))

    def test_similarity_and_margin(self):
        call = score_diplotypes(np.array([[0.1, 0.8, 0.5]]), TWO_HAPLOTYPES, OPTIONS)
        assert call.similarity == pytest.approx(0.8)
        assert call.margin == pytest.approx(0.3)

    def test_rejects_invalid_shape(self):
        with pytest.raises(ValueError, match="similarities"):
            score_diplotypes(np.zeros((1, 4)), TWO_HAPLOTYPES, OPTIONS)

    def test_rejects_invalid_options(self):
        with pytest.raises(ValueError, match="aggregate"):
            ScoringOptions(temperature=1.0, aggregate="median")
        with pytest.raises(ValueError, match="temperature"):
            ScoringOptions(temperature=0.0)

    def test_very_low_temperature_is_finite(self):
        """Probabilities that underflow if computed directly"""
        call = score_diplotypes(np.array([[-1.0, 1.0, 0.0]]), TWO_HAPLOTYPES, ScoringOptions(temperature=1e-4))
        assert call.diplotype_quality == MAX_PHRED
        genotype = genotype_variant(call, [0, 1], 2)
        assert genotype.quality == MAX_PHRED
        assert genotype.likelihoods == [MAX_PHRED, 0, MAX_PHRED]

    @pytest.mark.usefixtures("hydra_setup")
    def test_temperature_from_the_loss(self):
        import hydra

        cfg = hydra.compose(config_name="config", overrides=["model=paired_packed_inception_infonce"])
        options = scoring_options(cfg)
        assert options == ScoringOptions(temperature=0.07, aggregate="mean", max_gq=99)

        cfg = hydra.compose(
            config_name="config",
            overrides=["model=paired_packed_inception_infonce", "scoring.temperature=0.5", "scoring.aggregate=max", "scoring.max_gq=60"],
        )
        assert scoring_options(cfg) == ScoringOptions(temperature=0.5, aggregate="max", max_gq=60)

    @pytest.mark.usefixtures("hydra_setup")
    def test_loss_without_scalar_temperature(self):
        import hydra

        cfg = hydra.compose(config_name="config", overrides=["model=paired_packed_inception_rince"])
        with pytest.raises(ValueError, match="scoring.temperature"):
            scoring_options(cfg)


def _probabilities(values):
    """A call with these diplotype probabilities"""
    return np.log(np.asarray(values, dtype=np.float64))[np.newaxis]


class TestGenotypeVariant:
    def test_marginalizes_over_diplotypes(self):
        """Haplotypes 0 and 1 both have the REF allele, so several diplotypes produce each genotype"""
        probabilities = [0.1, 0.05, 0.4, 0.05, 0.3, 0.1]
        call = score_diplotypes(_probabilities(probabilities), THREE_HAPLOTYPES, OPTIONS)
        assert call.diplotype == (0, 2)

        genotype = genotype_variant(call, [0, 0, 1], 2)
        assert genotype.alleles == (0, 1)
        # 0/0: (0,0), (0,1), (1,1). 0/1: (0,2), (1,2). 1/1: (2,2).
        p_hom_ref, p_het, p_hom_alt = 0.2, 0.7, 0.1
        assert genotype.quality == pytest.approx(-10 * math.log10(p_hom_ref))
        assert genotype.genotype_quality == round(-10 * math.log10(1 - p_het))
        phreds = [-10 * math.log10(p) for p in (p_hom_ref, p_het, p_hom_alt)]
        assert genotype.likelihoods == [round(phred - min(phreds)) for phred in phreds]
        assert genotype.likelihoods[1] == 0

    def test_keeps_phase(self):
        """The alleles are in the order of the called diplotype's haplotypes"""
        call = score_diplotypes(_probabilities([0.01, 0.97, 0.02]), TWO_HAPLOTYPES, OPTIONS)
        assert genotype_variant(call, [1, 0], 2).alleles == (1, 0)
        assert genotype_variant(call, [0, 1], 2).alleles == (0, 1)

    def test_multiallelic_likelihood_order(self):
        """PL follows the VCF's ordering: 0/0, 0/1, 1/1, 0/2, 1/2, 2/2"""
        probabilities = [0.01, 0.02, 0.5, 0.04, 0.4, 0.03]
        call = score_diplotypes(_probabilities(probabilities), THREE_HAPLOTYPES, OPTIONS)
        genotype = genotype_variant(call, [0, 1, 2], 3)
        assert genotype.alleles == (0, 2)
        # Haplotype i has allele i, so the diplotypes are, in order, 0/0, 0/1, 0/2, 1/1, 1/2, 2/2
        ordered = [0.01, 0.02, 0.04, 0.5, 0.4, 0.03]
        phreds = [-10 * math.log10(p) for p in ordered]
        assert genotype.likelihoods == [round(phred - min(phreds)) for phred in phreds]

    def test_gq_capped(self):
        call = score_diplotypes(_probabilities([1e-30, 1 - 2e-30, 1e-30]), TWO_HAPLOTYPES, OPTIONS)
        assert genotype_variant(call, [0, 1], 2).genotype_quality == 99
        assert genotype_variant(call, [0, 1], 2, max_gq=20).genotype_quality == 20

    def test_called_genotype_need_not_have_the_best_marginal(self):
        """The call is the best diplotype, but another genotype can have more probability over several diplotypes"""
        probabilities = [0.3, 0.25, 0.25, 0.05, 0.1, 0.05]
        call = score_diplotypes(_probabilities(probabilities), THREE_HAPLOTYPES, OPTIONS)
        assert call.diplotype == (0, 0)
        genotype = genotype_variant(call, [0, 1, 1], 2)
        assert genotype.alleles == (0, 0)
        assert genotype.likelihoods[1] == 0, "0/1 is the most probable genotype"
        assert genotype.likelihoods[0] > 0

    def test_unreachable_genotypes(self):
        """Every candidate haplotype has the ALT allele, so no diplotype is 0/0"""
        call = score_diplotypes(_probabilities([0.1, 0.1, 0.8]), TWO_HAPLOTYPES, OPTIONS)
        genotype = genotype_variant(call, [1, 1], 2)
        assert genotype.alleles == (1, 1)
        assert genotype.quality == MAX_PHRED
        assert genotype.likelihoods == [MAX_PHRED, MAX_PHRED, 0]
        assert genotype.genotype_quality == 99, "Every diplotype produces the called genotype"

    def test_star_allele(self):
        """A haplotype spanned by an overlapping deletion has the record's '*' allele"""
        call = score_diplotypes(_probabilities([0.1, 0.8, 0.1]), TWO_HAPLOTYPES, OPTIONS)
        assert genotype_variant(call, [1, 2], 3).alleles == (1, 2)


class TestRegionIndex:
    @pytest.fixture
    def index(self):
        region_a, region_b = Range("chr1", 100, 200), Range("chr1", 300, 400)
        return _RegionIndex([region_a, region_b], {str(region_a): None, str(region_b): "Timeout"})

    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [
            (150, 160, "NotInGraph"),  # Must not be in the graph, or it would have been written
            (350, 360, "Timeout"),
            (50, 60, "NoRegion"),
            (250, 260, "NoRegion"),
            (500, 510, "NoRegion"),
            (90, 110, "NotInGraph"),  # Straddles the start of a genotyped region
            (390, 410, "NotInGraph"),  # Straddles the end of an unscored region
            (100, 110, "NotInGraph"),  # Not strictly inside, as for graph construction
            (190, 200, "NotInGraph"),
            (150, 350, "NotInGraph"),  # Spans both regions
            (350, 350, "Timeout"),  # An insertion
            (300, 300, "NotInGraph"),  # An insertion at the region's boundary
        ],
    )
    def test_filter(self, index, start, end, expected):
        assert index.filter(Range("chr1", start, end)) == expected

    def test_other_contig(self, index):
        assert index.filter(Range("chr2", 150, 160)) == "NoRegion"


def test_variant_ids_agree(tmp_path):
    """The writer identifies the records it reads with pysam with the IDs from the native reader"""
    population_vcf, inference_vcf = _write_vcfs(tmp_path)
    for path in (population_vcf, inference_vcf):
        with VariantFileReader.open(path) as reader, pysam.VariantFile(path, drop_samples=True) as vcf_file:
            for record, variant in zip(vcf_file.fetch(), reader.fetch(), strict=True):
                assert vg_variant_id(record) == variant.variant_id


@pytest.fixture
def genotyped(tmp_path):
    """Write the genotyped VCF for a run with a region of each of the possible outcomes"""
    population_vcf, inference_vcf = _write_vcfs(tmp_path)
    inputs = prepare_genotyping_vcf(population_vcf, tmp_path / "inputs", sample_name=SAMPLE, inference_vcf=inference_vcf)
    regions = genotyping_regions(inputs.vcf_path, flank=FLANK, min_variant_size=50, inference_sites=inputs.inference_sites)
    assert len(regions) == 5
    called, single, timed_out, rejected, insufficient = regions

    # 'novel' and 'pop_multi' are carried by different candidates. 'star' and 'straddle' aren't in the graph.
    called_record = _haplotype_record(
        inputs.vcf_path,
        called,
        {
            "called_del": [0, 1, 0],
            "pop_snv": [0, 0, 1],
            "pop_multi": [0, 2, 1],
            "novel": [1, 0, 1],
        },
        3,
    )
    images_dir = tmp_path / "images"
    _write_images(images_dir, [(called, called_record)])

    # The call is (1, 2)
    metric = np.array([[0.0, 0.1, 0.1, 0.1, 0.9, 0.1], [0.0, 0.1, 0.1, 0.1, 0.7, 0.1]])
    predictions = pd.DataFrame([_prediction(called, metric)]).set_index("key", drop=False)
    single_record = _haplotype_record(
        inputs.vcf_path, single, {"single_del": [0], "single_snv": [1], "in_single": [0]}, 1
    )
    failures = [
        {"key": energon_key(SAMPLE, single), "region": str(single), "sample": SAMPLE, "reason": "InsufficientHaplotypes", "message": None, "haplotypes": single_record},
        {"key": energon_key(SAMPLE, timed_out), "region": str(timed_out), "sample": SAMPLE, "reason": "Timeout", "message": None},
        {"key": energon_key(SAMPLE, insufficient), "region": str(insufficient), "sample": SAMPLE, "reason": "InsufficientHaplotypes", "message": None, "haplotypes": None},
    ]
    rejections = [{"key": energon_key(SAMPLE, rejected), "region": str(rejected), "sample": SAMPLE, "reason": "TooManyGenotypes"}]

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "output" / "genotypes.vcf.gz"
    output.parent.mkdir()
    counts = write_genotypes_vcf(
        inputs.vcf_path,
        output,
        sample_name=SAMPLE,
        regions=regions,
        images_dir=images_dir,
        predictions=predictions,
        rejected=rejections,
        failures=failures,
        options=ScoringOptions(temperature=0.07),
        inference_sites=inputs.inference_sites,
        checkpoint=checkpoint,
    )
    return output, counts, regions


def _records(path):
    """The records by each of their IDs, e.g., both the population and inference IDs of a merged record"""
    with pysam.VariantFile(str(path)) as vcf_file:
        return {record_id: record.copy() for record in vcf_file for record_id in (record.id or "").split(";")}


def _inference_tags(record):
    return [None if tag in (None, ".") else tag for tag in record.info.get("INFERENCE_ALLELE", (None,) * len(record.alts))]


class TestWriteGenotypesVcf:
    def test_counts(self, genotyped):
        _output, counts, _regions = genotyped
        assert dict(counts) == {
            "called": 4,
            "SINGLE_HAPLOTYPE": 3,
            "NotInGraph": 2,
            "Timeout": 3,
            "Rejected": 2,
            "InsufficientHaplotypes": 2,
            "NoRegion": 2,
        }

    def test_header(self, genotyped):
        output, _counts, _regions = genotyped
        with pysam.VariantFile(str(output)) as vcf_file:
            header = vcf_file.header
            assert list(header.samples) == [SAMPLE]
            assert header.formats["GQ"].type == "Integer"
            assert header.formats["PS"].type == "Integer"
            assert header.formats["PL"].number == "G"
            assert "SOURCE" in header.info
            assert {"SINGLE_HAPLOTYPE", "INFERENCE_ALLELE", "INFERENCE_QUAL"} <= set(header.info)
            assert header.info["INFERENCE_ALLELE"].number == "A"
            assert {"NoRegion", "NotInGraph", "InsufficientHaplotypes", "Timeout", "Rejected"} <= set(header.filters)
            generic = {record.key: record.value for record in header.records if record.type == "GENERIC"}
        import hashlib

        assert generic["npsv3_checkpoint"].endswith("model.ckpt")
        assert generic["npsv3_checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()

    def test_sorted_and_indexed(self, genotyped):
        output, _counts, _regions = genotyped
        assert os.path.exists(f"{output}.tbi")
        with pysam.VariantFile(str(output)) as vcf_file:
            positions = [record.pos for record in vcf_file]
            assert positions == sorted(positions)
            assert len(list(vcf_file.fetch("chr1", 999, 1100))) == 5

    def test_called_records(self, genotyped):
        output, _counts, _regions = genotyped
        records = _records(output)

        record = records["inference_del"]
        assert record.id == "called_del;inference_del", "The merged record"
        sample = record.samples[SAMPLE]
        assert sample["GT"] == (1, 0), "Haplotypes 1 and 2 have the deletion's alleles 1 and 0"
        assert sample.phased
        assert sample["PS"] == 1000
        assert list(record.filter) == ["PASS"]
        assert record.info["SOURCE"] == "population", "The merged record's INFO"
        assert _inference_tags(record)[0] is not None
        assert record.info["INFERENCE_QUAL"] == pytest.approx((33,))
        assert "SINGLE_HAPLOTYPE" not in record.info
        assert record.qual is not None and record.qual > 0
        assert len(sample["PL"]) == 3
        assert sample["MT"] == pytest.approx(0.8, abs=1e-4)
        assert sample["MG"] == pytest.approx(0.7, abs=1e-4)
        assert sample["DQ"] > 0
        assert 0 <= sample["GQ"] <= 99

    def test_population_only_records_are_reported(self, genotyped):
        output, _counts, _regions = genotyped
        records = _records(output)
        record = records["pop_snv"]
        assert record.samples[SAMPLE]["GT"] == (0, 1)
        assert record.samples[SAMPLE]["PS"] == 1000, "The region's phase block"
        assert record.info["SOURCE"] == "population"
        assert "INFERENCE_ALLELE" not in record.info

    def test_multiallelic(self, genotyped):
        """The inference record is contained in the population's multi-allelic record, which is genotyped once"""
        output, _counts, _regions = genotyped
        records = _records(output)
        record = records["pop_multi"]
        assert record.id == "pop_multi;inf_multi"
        assert record.alts == ("C", "T")
        assert record.samples[SAMPLE]["GT"] == (2, 1)
        assert len(record.samples[SAMPLE]["PL"]) == 6
        assert _inference_tags(record)[0] is None
        assert _inference_tags(record)[1] is not None

    def test_novel_record(self, genotyped):
        output, _counts, _regions = genotyped
        record = _records(output)["novel"]
        assert record.samples[SAMPLE]["GT"] == (0, 1)
        assert _inference_tags(record)[0] is not None
        assert "INFERENCE_QUAL" not in record.info, "The original had no QUAL"

    def test_single_haplotype_region(self, genotyped):
        output, _counts, _regions = genotyped
        records = _records(output)
        for name in ("single_del", "single_snv", "in_single"):
            record = records[name]
            sample = record.samples[SAMPLE]
            assert sample["GT"] == (0, 0), "Even if the single haplotype has a small ALT allele"
            assert sample.phased
            assert sample["PS"] == 5000
            assert record.info["SINGLE_HAPLOTYPE"]
            assert record.qual is None
            assert list(record.filter) == ["PASS"]
            for field in ("GQ", "DQ", "PL"):
                assert field not in sample or sample[field] in (None, (None,))

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("straddle", "NotInGraph"),
            ("star", "NotInGraph"),
            ("in_timeout", "Timeout"),
            ("in_rejected", "Rejected"),
            ("in_insufficient", "InsufficientHaplotypes"),
            ("in_unselected", "NoRegion"),
            ("no_region", "NoRegion"),
        ],
    )
    def test_inference_records_not_genotyped(self, genotyped, name, expected):
        output, _counts, _regions = genotyped
        record = _records(output)[name]
        assert record.id == name
        assert record.samples[SAMPLE]["GT"] == (None, None)
        assert list(record.filter) == [expected]
        assert record.qual is None
        assert record.info["SOURCE"] == "inference"
        assert record.info["INFERENCE_QUAL"] == pytest.approx((33,) * len(record.alts))
        assert None not in _inference_tags(record)

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("inf_timeout_del", "Timeout"),
            ("inf_rejected_del", "Rejected"),
            ("inf_insufficient_del", "InsufficientHaplotypes"),
        ],
    )
    def test_merged_records_not_genotyped(self, genotyped, name, expected):
        """Records merged with a population record are reported as the merged record"""
        output, _counts, _regions = genotyped
        record = _records(output)[name]
        assert record.id == f"{name.removeprefix('inf_')};{name}"
        assert record.samples[SAMPLE]["GT"] == (None, None)
        assert list(record.filter) == [expected]
        assert record.info["SOURCE"] == "population"

    def test_partially_overlapping_record_not_genotyped(self, genotyped):
        """The merged record has the union of the population and inference alleles, and is reported once"""
        output, _counts, _regions = genotyped
        records = _records(output)
        record = records["in_timeout_multi"]
        assert record.id == "timeout_multi;in_timeout_multi"
        assert record.alleles == ("A", "C", "T", "G")
        tags = _inference_tags(record)
        assert tags[0] is None
        assert tags[1] is not None
        assert tags[2] is not None
        assert list(record.filter) == ["Timeout"]
        assert record.samples[SAMPLE]["GT"] == (None, None)

    def test_unscored_population_records_are_omitted(self, genotyped):
        """Records without an inference allele in a region that wasn't scored (or selected) are not reported"""
        output, _counts, _regions = genotyped
        records = _records(output)
        for name in ("timeout_snv", "unselected_del"):
            assert name not in records

    def test_every_inference_record_once(self, genotyped, tmp_path):
        output, _counts, _regions = genotyped
        with pysam.VariantFile(str(tmp_path / "inference.vcf.gz")) as vcf_file:
            expected = sorted(record.id for record in vcf_file)
        with pysam.VariantFile(str(output)) as vcf_file:
            reported = [record_id for record in vcf_file for record_id in record.id.split(";") if record_id in expected]
        assert sorted(reported) == expected

    def test_no_duplicate_records(self, genotyped):
        output, _counts, _regions = genotyped
        with pysam.VariantFile(str(output)) as vcf_file:
            keys = [(record.pos, record.alleles) for record in vcf_file]
        assert len(keys) == len(set(keys))

    def test_readable_by_bcftools(self, genotyped):
        output, _counts, _regions = genotyped
        stats = bcftools.stats(str(output))
        assert "number of records:\t18" in stats
        query = bcftools.query("-f", "%ID\t[%GT]\n", str(output))
        assert "pop_snv\t0|1" in query

    @pytest.mark.skipif(shutil.which("truvari") is None, reason="truvari isn't on the PATH")
    def test_readable_by_truvari(self, genotyped, tmp_path):
        """Truvari requires multi-allelic records be split, as for any input"""
        output, _counts, _regions = genotyped
        split = str(tmp_path / "split.vcf.gz")
        bcftools.norm("-m-any", "-Oz", "-o", split, "-Wtbi", str(output), catch_stdout=False)
        subprocess.check_call(
            ["truvari", "bench", "-b", split, "-c", split, "-o", str(tmp_path / "truvari"), "--passonly", "--pctseq", "0"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert (tmp_path / "truvari" / "summary.json").exists()


class TestAccounting:
    def test_every_region_is_accounted_for(self, tmp_path):
        population_vcf, _inference_vcf = _write_vcfs(tmp_path)
        region = Range("chr1", 900, 1159)
        with pytest.raises(ValueError, match="no prediction, failure or rejection"):
            write_genotypes_vcf(
                population_vcf, tmp_path / "genotypes.vcf.gz", sample_name=SAMPLE, regions=[region], images_dir=tmp_path,
                predictions=None, rejected=[], failures=[], options=OPTIONS,
            )

    def test_unexpected_failure(self, tmp_path):
        population_vcf, _inference_vcf = _write_vcfs(tmp_path)
        region = Range("chr1", 900, 1159)
        failures = [{"key": energon_key(SAMPLE, region), "region": str(region), "reason": "Downsampled"}]
        with pytest.raises(ValueError, match="Downsampled"):
            write_genotypes_vcf(
                population_vcf, tmp_path / "genotypes.vcf.gz", sample_name=SAMPLE, regions=[region], images_dir=tmp_path,
                predictions=None, rejected=[], failures=failures, options=OPTIONS,
            )

    def test_no_regions(self, tmp_path):
        """E.g., no inference record is in any analysis region, so they are all reported as such"""
        population_vcf, inference_vcf = _write_vcfs(tmp_path)
        inputs = prepare_genotyping_vcf(population_vcf, tmp_path / "inputs", sample_name=SAMPLE, inference_vcf=inference_vcf)
        output = tmp_path / "genotypes.vcf.gz"
        counts = write_genotypes_vcf(
            inputs.vcf_path, output, sample_name=SAMPLE, regions=[], images_dir=tmp_path, predictions=None, rejected=[],
            failures=[], options=OPTIONS, inference_sites=inputs.inference_sites,
        )
        assert dict(counts) == {"NoRegion": 16}, "Every inference record, each in its own merged record"
        assert {list(record.filter)[0] for record in _records(output).values()} == {"NoRegion"}

    def test_uncompressed_output(self, tmp_path):
        population_vcf, inference_vcf = _write_vcfs(tmp_path)
        inputs = prepare_genotyping_vcf(population_vcf, tmp_path / "inputs", sample_name=SAMPLE, inference_vcf=inference_vcf)
        output = tmp_path / "genotypes.vcf"
        write_genotypes_vcf(
            inputs.vcf_path, output, sample_name=SAMPLE, regions=[], images_dir=tmp_path, predictions=None, rejected=[],
            failures=[], options=OPTIONS, inference_sites=inputs.inference_sites,
        )
        assert output.read_text().startswith("##fileformat")
