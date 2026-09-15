import glob
import json
import os

import pytest
import torch

from npsv3.graphs.haplotype import KmerCounts, _create_graph_and_sampler, _filter_variants
from npsv3.images.example import (
    example_to_image,
    make_graph_example,
    vcf_to_graph_examples,
)
from npsv3.simulation import bwa_index_loaded
from npsv3.util.range import Range
from npsv3.util.variant import VariantFileReader

from .. import (
    HG00731_HG38_BAM,
    HG38_REF_FASTA,
    cache_filter_kmc_database,
    cache_graph_and_filter_kmc_database,
    create_vcf,
    result_path,
)


@pytest.mark.cfg_overrides(
    f"reference={HG38_REF_FASTA}",
    "input=/storage/mlinderman/projects/sv/npsv3-experiments/resources/HG00733.hgsvc3-hprc-2024-02-23.dipcall.passing.hg38.vcf.gz",
    f"reads={HG00731_HG38_BAM}",
    "graph.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
    "simulation.replicates=1",
)
class TestHG00733GraphExamples:
    def test_graph_to_example(self, cfg, hg00733_sample, tmp_path):
        # cSpell:disable
        vcf_path = create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=chr1,length=248956422>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00733
chr1	1134739	.	A	ACATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGCACCGTGAGGGGTGTGGCTTCCTCGC	30	PASS	.	GT	1|0
chr1	1134748	.	T	G	30	PASS	.	GT	1|1
chr1	1134771	.	G	GCACCGTGAGGGGTGTGGCTTCCTCGCCATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGCACCGTGAGGGGTGTGGCTTCCTCGCCATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGA	30	PASS	.	GT	0|1"""
        ) # fmt: skip
        # cSpell:enable

        region = Range("chr1:1134644-1134867")
        ref_kmer_counts = KmerCounts(cfg.graph.ref_kmer_counts_kmc_prefix)
        graph, unique_kmers, sampler = _create_graph_and_sampler(
            cfg.reference,
            vcf_path,
            region,
            k=cfg.graph.kmer_size,
            ref_kmer_counts=ref_kmer_counts,
        )

        filtered_kmer_path = cache_filter_kmc_database(cfg, hg00733_sample, vcf_path, region, unique_kmers, tmp_path=tmp_path)
        with VariantFileReader.open(vcf_path) as vcf_file:
            analysis_variants = _filter_variants(list(vcf_file.fetch(region)), min_variant_size=50)

        example = make_graph_example(
            cfg,
            graph,
            sampler,
            unique_kmers,
            region,
            vcf_path,
            hg00733_sample,
            cfg.reads,
            filtered_kmer_path,
            analysis_variants, # type: ignore
        )

        assert example["region"] == str(region)

        assert "image" in example
        assert example["image"].shape == (cfg.pileup.image_height, cfg.pileup.image_width, len(cfg.pileup.image_channels))

        assert example["label"] > 0

        assert "sim.images" in example
        # This region has 10 possible genotypes for the available haplotypes (4 haplotypes, so 4*(4+1)/2 genotypes)
        assert example["sim.images"].shape == (cfg.simulation.replicates, 10, *example["image"].shape)

        #png_path = tmp_path / "test.png"
        png_path = result_path("test.png")
        example_to_image(cfg, example, png_path, select_channels=[0, 1, 4], with_simulations=True) # ALIGNED, PAIRED, BASEQ
        assert os.path.exists(png_path)

    @pytest.mark.usefixtures("ray_setup")
    def test_vcf_to_energon_dataset(self, cfg, hg00733_sample, tmp_path):
        """`vcf_to_graph_examples` emits the polylithic energon trees directly from GraphWriter."""
        # cSpell:disable
        vcf_path = create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=chr1,length=248956422>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00733
chr1	1134739	.	A	ACATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGCACCGTGAGGGGTGTGGCTTCCTCGC	30	PASS	.	GT	1|0
chr1	1134748	.	T	G	30	PASS	.	GT	1|1
chr1	1134771	.	G	GCACCGTGAGGGGTGTGGCTTCCTCGCCATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGCACCGTGAGGGGTGTGGCTTCCTCGCCATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGA	30	PASS	.	GT	0|1"""
        ) # fmt: skip
        # cSpell:enable

        region = Range("chr1:1134644-1134867")
        graph_shards, filtered_kmer_path = cache_graph_and_filter_kmc_database(cfg, hg00733_sample, cfg.input, region)

        output_dir = str(tmp_path / "energon")
        vcf_to_graph_examples(
            cfg,
            cfg.reads,
            hg00733_sample,
            vcf_path,
            output_dir,
            graph_shards=graph_shards,
            filtered_kmer_path=filtered_kmer_path,
        )

        assert os.path.exists(os.path.join(output_dir, "meta_ds", "train.jsonl"))
        assert os.path.exists(os.path.join(output_dir, "meta_ds", "train.jsonl.idx"))
        imgs_shards = sorted(glob.glob(os.path.join(output_dir, "imgs_ds", "imgs-*.tar")))
        imgs_indexes = sorted(glob.glob(os.path.join(output_dir, "imgs_ds", "imgs-*.tar.idx")))
        assert len(imgs_shards) == len(imgs_indexes) > 0, "Every imgs_ds tar shard must have an index"

        from npsv3.models.energon import EnergonPackedImageDataModule

        dm = EnergonPackedImageDataModule(
            metadataset_path=os.path.join(output_dir, "dataset.yaml"),
            cache_dir=str(tmp_path / "cache"),
            batch_size=32,
            num_workers=0,
            repeat=False
        )
        [first_batch, *remaining_batches] = list(dm.train_dataloader())
        dm.teardown()

        assert len(remaining_batches) == 0, "This single region should be a single batch"
        assert first_batch.images.shape == torch.Size([32, len(cfg.pileup.image_channels), cfg.pileup.image_height, cfg.pileup.image_width])
        assert all(first_batch.labels[first_batch.genotype_label] == 1), "All 'true' genotypes should be labeled 1"
        assert all(first_batch.labels[10:] == -100), "Everyting after valid examples should be masked out with -100"
        assert torch.equal(first_batch.offsets, torch.tensor([0, 10]))
        assert len(first_batch.genotype_label) == 1 and all(first_batch.genotype_label > 0)
        assert first_batch.meta == [str(region)]



@pytest.mark.cfg_overrides(
    f"reference={HG38_REF_FASTA}",
    "input=/storage/mlinderman/projects/sv/npsv3-experiments/resources/hgsvc3-hprc-2024-02-23.dipcall.population.passing.training.hg38.vcf.gz",
    "reads=/storage/mlinderman/projects/sv/npsv3-experiments/resources/sequence/HG00096.final.cram",
    "graph.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
    "graph.population_prior=True",
    "simulation.replicates=1",
)
class TestPopulationGraphExamples:
    @pytest.mark.usefixtures("ray_setup")
    @pytest.mark.parametrize("region_str", [
        "chr7:136717247-136717522",
        "chr3:20967930-20972107",
        # "chr4:49101919-49110988", # Leads to timeout error
    ])
    def test_error_regions(self, cfg, hg00096_sample, tmp_path, region_str):
        if not all(os.path.exists(f) for f in (cfg.reference, cfg.input, cfg.reads, f"{cfg.graph.ref_kmer_counts_kmc_prefix}.kmc_pre")):
            pytest.skip("Missing necessary inputs")
        if not bwa_index_loaded(cfg.reference):
            pytest.skip("BWA index not loaded")

        region = Range(region_str)

        # Use cached files to speed up repeated runs of the test
        graph_shards, filtered_kmer_path = cache_graph_and_filter_kmc_database(cfg, hg00096_sample, cfg.input, region)

        output_dir = str(tmp_path / "shards")
        vcf_to_graph_examples(
            cfg,
            cfg.reads,
            hg00096_sample,
            cfg.input,
            output_dir,
            graph_shards=graph_shards,
            filtered_kmer_path=filtered_kmer_path,
        )
        assert os.path.exists(output_dir)

        # Confirm the region was actually written (not silently skipped by the InsufficientHaplotypesError
        # safety net) and that the population-panel top-up produced >=2 haplotypes to genotype against, not
        # just that nothing crashed.
        with open(os.path.join(output_dir, "meta_ds", "train.jsonl")) as meta_file:
            rows = [json.loads(line) for line in meta_file if line.strip()]
        matches = [row for row in rows if row["region"] == str(region)]
        assert matches, f"Region {region} was skipped rather than recovered"
        assert matches[0]["g"] >= 2, "Expected more than one possible genotype"
