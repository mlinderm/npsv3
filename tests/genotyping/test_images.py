import glob
import json
import os

import pysam
import pytest
import webdataset as wds
from omegaconf import OmegaConf

from npsv3.genotyping.images import (
    FORCED_SETTINGS,
    GENERATION_FILE,
    generate_genotyping_images,
    genotyping_config,
    read_generation,
)
from npsv3.genotyping.inputs import genotyping_regions, prepare_genotyping_vcf
from npsv3.graphs.graph import Graph
from npsv3.graphs.haplotype import (
    _create_graph_and_sampler,
    _filter_variants,
    _resolved_alleles,
    haplotype_alleles,
    serialize_graph_and_unique_kmers,
)
from npsv3.graphs.sharding import REGION_MEMBER
from npsv3.images.haplotypes import (
    HaplotypeStrategy,
    InsufficientHaplotypesError,
    decode_variant_alleles,
    generate_haplotypes,
    sample_path,
)
from npsv3.util.range import Range
from npsv3.util.sample import Sample
from npsv3.util.variant import VariantFileReader

from .. import HG38_BWA_MEM3_INDEX, HG00733_HG38_BAM, HG38_REF_FASTA, create_vcf

GRAPH_REGION = Range("chr1", 999900, 1000400)


def _reference(start: int, end: int) -> str:
    """Reference bases for 1-based, inclusive [start, end] on chr1"""
    with pysam.FastaFile(HG38_REF_FASTA) as fasta:
        return fasta.fetch("chr1", start - 1, end).upper()


def _alt_base(base: str) -> str:
    return "T" if base != "T" else "G"


def _region_vcf(tmp_path) -> str:
    """A VCF with SNVs, an SV deletion with an SNV (with a '*' allele) inside it, an SV insertion, and a deletion that
    straddles the graph region's start (and so isn't in the graph)"""
    snv, snv_in_del, ins = _reference(1000010, 1000010), _reference(1000080, 1000080), _reference(1000200, 1000200)
    deletion, straddle = _reference(1000050, 1000110), _reference(999890, 999950)
    records = [
        # S1: hap0 has the SNV and the SNV in the deletion; hap1 has the deletion (so '*' for the SNV in it) and insertion
        f"chr1\t999890\tstraddle\t{straddle}\t{straddle[0]}\t.\tPASS\t.\tGT\t0|0\t0|1",
        f"chr1\t1000010\tsnv\t{snv}\t{_alt_base(snv)}\t.\tPASS\t.\tGT\t1|0\t0|0",
        f"chr1\t1000050\tdel\t{deletion}\t{deletion[0]}\t.\tPASS\t.\tGT\t0|1\t0|0",
        f"chr1\t1000080\tsnv_in_del\t{snv_in_del}\t{_alt_base(snv_in_del)},*\t.\tPASS\t.\tGT\t1|2\t0|1",
        f"chr1\t1000200\tins\t{ins}\t{ins + 'C' * 60}\t.\tPASS\t.\tGT\t0|1\t1|1",
    ]
    header = (
        "##fileformat=VCFv4.2\n"
        '##FILTER=<ID=PASS,Description="All filters passed">\n'
        "##contig=<ID=chr1,length=248956422>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
    )
    return create_vcf(tmp_path, (header + "\n".join(records) + "\n").encode())


requires_hg38 = pytest.mark.skipif(not HG38_REF_FASTA, reason="HG38 reference FASTA not found")


@requires_hg38
class TestDecodeVariantAlleles:
    def test_sample_paths(self, tmp_path):
        vcf_path = _region_vcf(tmp_path)
        graph = Graph(HG38_REF_FASTA, vcf_path, GRAPH_REGION)
        with VariantFileReader.open(vcf_path) as vcf_file:
            variants = list(vcf_file.fetch(GRAPH_REGION))
        assert len(variants) == 5

        paths = {
            (sample, hap): sample_path(graph, sample, hap, "chr1") for sample in ("S1", "S2") for hap in range(2)
        }
        assert all(path is not None for path in paths.values()), "Every sample haplotype should be fully resolved"
        decoded = decode_variant_alleles(graph, [*paths.values(), graph.path_nodes("chr1")], variants)

        # The straddling deletion isn't in the graph. The SNV inside the deletion is '*' on the haplotype with the deletion.
        expected = [
            [None, 1, 0, 1, 0],  # S1 hap0
            [None, 0, 1, 2, 1],  # S1 hap1
            [None, 0, 0, 0, 1],  # S2 hap0
            [None, 0, 0, 1, 1],  # S2 hap1
            [None, 0, 0, 0, 0],  # Reference
        ]
        assert decoded == expected

    def test_consistent_with_sampler(self, tmp_path):
        """For the analysis variants, the decoding agrees with the sampler's own decoding"""
        vcf_path = _region_vcf(tmp_path)
        graph, _unique_kmers, sampler = _create_graph_and_sampler(HG38_REF_FASTA, vcf_path, GRAPH_REGION, k=31)
        with VariantFileReader.open(vcf_path) as vcf_file:
            analysis_variants = _filter_variants(list(vcf_file.fetch(GRAPH_REGION)), 50)
        assert len(analysis_variants) == 3, "The deletions (including the straddling deletion) and insertion"

        paths = [sample_path(graph, sample, hap, "chr1") for sample in ("S1", "S2") for hap in range(2)]
        decoded = decode_variant_alleles(graph, paths, analysis_variants)
        sampler_alleles = [_resolved_alleles(alleles) for alleles in haplotype_alleles(sampler, paths, analysis_variants)]
        for path_alleles, path_sampler_alleles in zip(decoded, sampler_alleles, strict=True):
            # The sampler doesn't report the variants that aren't in the graph
            assert {
                (variant.variant_id, allele)
                for variant, allele in zip(analysis_variants, path_alleles, strict=True)
                if allele is not None
            } == path_sampler_alleles
            assert path_alleles[0] is None


class _FixedStrategy(HaplotypeStrategy):
    """Return fixed haplotypes, with no "true" haplotypes (as for a sample that isn't in the VCF)"""

    def __init__(self, cfg, haplotypes):
        super().__init__(cfg)
        self.haplotypes = haplotypes

    def __call__(self, graph, sampler, region, sample, filtered_kmers, analysis_variants, *, ploidy=2, exact_reference=False):  # noqa: ARG002
        return list(self.haplotypes), [set() for _ in self.haplotypes], []


@requires_hg38
class TestGenerateHaplotypesWithoutSample:
    def _setup(self, tmp_path):
        vcf_path = _region_vcf(tmp_path)
        graph, _unique_kmers, sampler = _create_graph_and_sampler(HG38_REF_FASTA, vcf_path, GRAPH_REGION, k=31)
        with VariantFileReader.open(vcf_path) as vcf_file:
            variants = list(vcf_file.fetch(GRAPH_REGION))
        return graph, sampler, variants, Sample("ABSENT", mean_coverage=30, mean_insert_size=400, std_insert_size=100, sequencer="HSXn", read_length=150)

    def test_unlabeled(self, cfg, tmp_path):
        graph, sampler, variants, sample = self._setup(tmp_path)
        haplotypes = [graph.path_nodes("chr1"), sample_path(graph, "S1", 1, "chr1")]
        record = generate_haplotypes(
            cfg, graph, sampler, GRAPH_REGION, sample, None, None, _filter_variants(variants, 50),
            strategy=_FixedStrategy(cfg, haplotypes), variants=variants,
        )
        assert record["true"] == [None, None], "Examples for a sample not in the VCF are unlabeled"
        assert record["variant_ids"] == [variant.variant_id for variant in variants]
        assert record["variant_alleles"] == [[None, 0, 0, 0, 0], [None, 0, 1, 2, 1]]
        json.dumps(record)  # The record must be JSON-serializable

        without_variants = generate_haplotypes(
            cfg, graph, sampler, GRAPH_REGION, sample, None, None, [], strategy=_FixedStrategy(cfg, haplotypes),
        )
        assert "variant_alleles" not in without_variants, "Only recorded when requested"

    def test_single_haplotype(self, cfg, tmp_path):
        graph, sampler, variants, sample = self._setup(tmp_path)
        haplotypes = [sample_path(graph, "S2", 0, "chr1")]
        with pytest.raises(InsufficientHaplotypesError) as exc_info:
            generate_haplotypes(
                cfg, graph, sampler, GRAPH_REGION, sample, None, None, [], strategy=_FixedStrategy(cfg, haplotypes),
                variants=variants,
            )
        record = exc_info.value.haplotypes
        assert record is not None, "The single haplotype's record is attached to the error"
        assert len(record["paths"]) == 1
        assert record["variant_alleles"] == [[None, 0, 0, 0, 1]]


class TestGenotypingConfig:
    @pytest.mark.cfg_overrides("command=images", "graph.population_excluded_samples=[OTHER]")
    def test_forced_settings(self, cfg):
        OmegaConf.update(cfg, "hom_ref_downsample.enabled", True)
        genotyping_cfg = genotyping_config(cfg, "HG00733")
        for key, value in FORCED_SETTINGS.items():
            assert OmegaConf.select(genotyping_cfg, key) == value, key
        assert list(genotyping_cfg.graph.population_excluded_samples) == ["OTHER", "HG00733"]
        assert cfg.hom_ref_downsample.enabled, "The original config is unchanged"

        # Idempotent
        again = genotyping_config(genotyping_cfg, "HG00733")
        assert list(again.graph.population_excluded_samples) == ["OTHER", "HG00733"]

    @pytest.mark.cfg_overrides("haplotype.strategy._target_=npsv3.images.haplotypes.TrueHaplotypeStrategy")
    def test_rejects_true_haplotype_strategy(self, cfg):
        with pytest.raises(ValueError, match="SampledHaplotypeStrategy"):
            genotyping_config(cfg, "HG00733")


@requires_hg38
@pytest.mark.usefixtures("ray_setup")
class TestSerializeRegions:
    @pytest.mark.cfg_overrides(f"reference={HG38_REF_FASTA}")
    def test_exact_regions(self, cfg, tmp_path):
        vcf_path = _region_vcf(tmp_path)
        regions = [Range("chr1", 999995, 1000100), GRAPH_REGION]
        output_dir = tmp_path / "graphs"
        output_dir.mkdir()
        graph_shards, unique_kmer_path, region_count = serialize_graph_and_unique_kmers(
            cfg, vcf_path, str(output_dir), pool_kmers=False, regions=regions,
        )
        assert region_count == len(regions)
        assert unique_kmer_path is None
        built = sorted(record[REGION_MEMBER].decode() for record in wds.WebDataset(graph_shards, shardshuffle=False))  # type: ignore
        assert built == sorted(str(region) for region in regions), "Exactly the requested regions are built"

        with pytest.raises(ValueError, match="region"):
            serialize_graph_and_unique_kmers(cfg, vcf_path, str(output_dir), regions=regions, region=GRAPH_REGION)


# cSpell:disable
HG00733_VCF = b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=chr1,length=248956422>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00733	OTHER
chr1	1134739	.	A	ACATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGCACCGTGAGGGGTGTGGCTTCCTCGC	30	PASS	.	GT	1|0	0|1
chr1	1134748	.	T	G	30	PASS	.	GT	1|1	0|1
chr1	1134771	.	G	GCACCGTGAGGGGTGTGGCTTCCTCGCCATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGCACCGTGAGGGGTGTGGCTTCCTCGCCATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGA	30	PASS	.	GT	0|1	1|0
"""  # fmt: skip
# cSpell:enable


@pytest.mark.cfg_overrides(
    f"reference={HG38_REF_FASTA}",
    f"reads={HG00733_HG38_BAM}",
    "graph.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
    f"aligner_index={HG38_BWA_MEM3_INDEX}",
    "simulation.replicates=2",
)
class TestHG00733GenotypingImages:
    @pytest.mark.usefixtures("ray_setup")
    @pytest.mark.parametrize("replicates", [0, 2])
    def test_generate_genotyping_images(self, cfg, hg00733_sample, tmp_path, replicates):
        """Images for a sample excluded from the population are unlabeled, record every variant's alleles, and account
        for every region. Without simulated replicates, the aligner's index isn't needed."""
        from npsv3.simulation import bwa_index_loaded

        if not all(os.path.exists(f) for f in (cfg.reference, cfg.reads, f"{cfg.graph.ref_kmer_counts_kmc_prefix}.kmc_pre")):
            pytest.skip("Missing necessary inputs")
        if hg00733_sample.kmc_prefix is None:
            pytest.skip("Missing HG00733 KMC database")
        if replicates > 0 and not bwa_index_loaded(cfg.aligner_index, aligner=cfg.pileup.aligner):
            pytest.skip("BWA index not loaded")
        cfg.simulation.replicates = replicates

        population = create_vcf(tmp_path, HG00733_VCF, name="population.vcf.gz")
        inputs = prepare_genotyping_vcf(population, tmp_path / "work" / "inputs", sample_name=hg00733_sample.name)
        assert inputs.excluded_sample
        regions = genotyping_regions(inputs.vcf_path, flank=cfg.pileup.variant_padding, min_variant_size=50)
        assert len(regions) == 1

        images = generate_genotyping_images(cfg, cfg.reads, hg00733_sample, inputs, regions, tmp_path / "work")
        assert images.dataset_path is not None and os.path.exists(images.dataset_path)  # noqa: PT018
        assert images.counts.get("generated") == 1
        assert images.failures == []
        assert glob.glob(os.path.join(images.images_dir, "failures-*.jsonl")), "Every actor writes a failures file"

        generation = read_generation(images.images_dir)
        assert os.path.exists(os.path.join(images.images_dir, GENERATION_FILE))
        assert generation["image"]["channels"] == len(cfg.pileup.image_channels)
        assert generation["image"]["replicates"] == replicates
        assert generation["haplotype"]["population_haplotype_topup"] is False

        with open(os.path.join(images.images_dir, "meta_ds", "train.jsonl")) as meta_file:
            [row] = [json.loads(line) for line in meta_file if line.strip()]
        assert row["label"] is None and row["label_rank"] is None, "Examples are unlabeled"
        assert row["r"] == replicates and row["g"] == row["num_haplotypes"] * (row["num_haplotypes"] + 1) // 2  # noqa: PT018

        from megatron.energon.cache.file_store import WebdatasetFileStore
        from megatron.energon.epathlib import EPath

        from npsv3.constants import HAPLOTYPES_SUFFIX
        from npsv3.images.haplotypes import decode_haplotype_record

        store = WebdatasetFileStore(EPath(os.path.join(images.images_dir, "imgs_ds")))
        record = decode_haplotype_record(store[f"{row['key']}.{HAPLOTYPES_SUFFIX}"][0])
        assert record["true"] == [None, None]
        assert record["sample_paths"] == [None, None], "The excluded sample has no paths in the graph"
        assert len(record["variant_ids"]) == 3, "Every variant in the region, not just the analysis variants"
        assert len(record["variant_alleles"]) == len(record["paths"])
        # The decoded alleles agree with the analysis variants' alleles, e.g., the haplotype at index 0 has no SV ALT alleles
        for path_alleles, path_variant_alleles in zip(record["alleles"], record["variant_alleles"], strict=True):
            for variant_id, allele in path_alleles:
                assert path_variant_alleles[record["variant_ids"].index(variant_id)] == allele
        assert all(allele == 0 for _, allele in record["alleles"][0])


@requires_hg38
@pytest.mark.usefixtures("ray_setup")
class TestGenotypingImagesFailures:
    @pytest.mark.cfg_overrides(f"reference={HG38_REF_FASTA}", "simulation.replicates=0", "haplotype.max_haplotypes=1")
    def test_single_haplotype_regions(self, cfg, tmp_path):
        """When only the reference haplotype is sampled (the sample's k-mers are the reference's, and only one haplotype
        is sampled), the region is recorded as a failure (with its single haplotype) and, with no examples at all, there
        is no dataset. The reads are never needed."""
        import subprocess
        from shlex import quote

        population = _region_vcf(tmp_path)
        inputs = prepare_genotyping_vcf(population, tmp_path / "work" / "inputs", sample_name="S1")
        # Not overlapping the straddling deletion, since an analysis variant that isn't in the graph is never on any
        # haplotype, and so the reference haplotype (with every analysis variant's REF allele) would be added
        region = Range("chr1", 999995, 1000400)
        regions = [region]

        kmc_dir = tmp_path / "kmc"
        kmc_dir.mkdir()
        # Two copies of the region's reference sequence, so its k-mers are homozygous with a k-mer coverage of 1
        reference = _reference(region.start + 1, region.end)
        (kmc_dir / "kmers.fa").write_text(f">0\n{reference}\n>1\n{reference}\n")
        subprocess.check_call(
            f"kmc -t1 -k{cfg.graph.kmer_size} -b -ci1 -fa {quote(str(kmc_dir / 'kmers.fa'))} {quote(str(kmc_dir / 'kmers'))} {quote(str(kmc_dir))}",
            shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        sample = Sample("S1", mean_coverage=30, mean_insert_size=400, std_insert_size=100, sequencer="HSXn", read_length=150, kmer_coverage=1)
        images = generate_genotyping_images(
            cfg, "unused.bam", sample, inputs, regions, tmp_path / "work", filtered_kmer_path=str(kmc_dir / "kmers"),
            min_variant_size=50,
        )
        assert images.dataset_path is None
        assert images.counts == {"insufficient_haplotypes": 1}
        [failure] = images.failures
        assert failure["reason"] == "InsufficientHaplotypes"
        assert failure["region"] == str(region)
        assert failure["sample"] == "S1"
        assert len(failure["haplotypes"]["paths"]) == 1
        assert failure["haplotypes"]["variant_alleles"] == [[0, 0, 0, 0]], "The reference haplotype"
        assert all(
            allele == 0 for _, allele in failure["haplotypes"]["alleles"][0]
        ), "The single haplotype has none of the analysis variants' ALT alleles"
        assert os.path.exists(os.path.join(images.images_dir, GENERATION_FILE))
