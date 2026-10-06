import glob
import json
import os

import hydra
import pytest
import torch

from npsv3.graphs.haplotype import KmerCounts, _create_graph_and_sampler, _filter_variants
from npsv3.images import agreement
from npsv3.images.example import (
    example_to_image,
    make_graph_example,
    vcf_to_graph_examples,
)
from npsv3.models.loaders import group_to_image
from npsv3.simulation import bwa_index_loaded
from npsv3.util.range import Range
from npsv3.util.variant import VariantFileReader

from .. import (
    HG00733_HG38_BAM,
    HG38_REF_FASTA,
    cache_filter_kmc_database,
    cache_graph_and_filter_kmc_database,
    create_vcf,
    result_path,
)


@pytest.mark.cfg_overrides(
    f"reference={HG38_REF_FASTA}",
    "input=/storage/mlinderman/projects/sv/npsv3-experiments/resources/HG00733.hgsvc3-hprc-2024-02-23.dipcall.passing.hg38.vcf.gz",
    f"reads={HG00733_HG38_BAM}",
    "graph.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
    "simulation.replicates=1",
)
class TestHG00733GraphExamples:
    def test_graph_to_example(self, request, cfg, hg00733_sample, tmp_path):
        if not all(os.path.exists(f) for f in (cfg.reference, cfg.reads, f"{cfg.graph.ref_kmer_counts_kmc_prefix}.kmc_pre")):
            pytest.skip("Missing necessary inputs")
        if not bwa_index_loaded(cfg.reference):
            pytest.skip("BWA index not loaded")

        # cSpell:disable
        # Local VCF for testing that matches "chr1:1134644-1134867" in HG00733.hgsvc3-hprc-2024-02-23.dipcall.passing.hg38.vcf.gz
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
            sample_idx = vcf_file.samples().index(hg00733_sample.name)

        marker = request.node.get_closest_marker("cfg_overrides")
        local_cfg = hydra.compose(
            config_name="config",
            overrides=[*(marker.args if marker else []), "pileup=unphased_graph_both"],
        )
        example = make_graph_example(
            local_cfg,
            graph,
            sampler,
            unique_kmers,
            region,
            sample_idx, # type: ignore
            hg00733_sample,
            cfg.reads,
            filtered_kmer_path,
            analysis_variants, # type: ignore
        )

        assert example["region"] == str(region)
        assert "image" in example
        assert example["image"].shape == (local_cfg.pileup.image_height, local_cfg.pileup.image_width, len(local_cfg.pileup.image_channels))
        assert example["label"] > 0
        assert "sim.images" in example
        # This region has 10 possible genotypes for the available haplotypes (4 haplotypes, so 4*(4+1)/2 genotypes)
        assert example["sim.images"].shape == (local_cfg.simulation.replicates, 10, *example["image"].shape)

        all_png_path = tmp_path / "all.png" # result_path("all.png")
        example_to_image(local_cfg, example, all_png_path, render_channels=True, with_simulations=True)
        assert os.path.exists(all_png_path)

    @pytest.mark.parametrize(("missing_are_ref", "expected_label"), [
        (False, None),  # Default: a sample with no called genotype for any analysis variant is unlabeled
        (True, 0),  # Opt-in: missing genotypes are assumed homozygous REF
    ])
    def test_missing_genotype_label(self, request, cfg, hg00733_sample, tmp_path, missing_are_ref, expected_label):
        """A sample with no called variant in the region (i.e. every analysis variant's genotype for that
        sample is missing) should be unlabeled unless graph.missing_are_ref opts into treating missing
        genotypes as homozygous REF."""
        if not all(os.path.exists(f) for f in (cfg.reference, cfg.reads, f"{cfg.graph.ref_kmer_counts_kmc_prefix}.kmc_pre")):
            pytest.skip("Missing necessary inputs")
        if not bwa_index_loaded(cfg.reference):
            pytest.skip("BWA index not loaded")

        # Same region/variants as test_graph_to_example, but HG00733's genotype is missing at every
        # analysis variant, simulating a region with no called variant for this sample.
        # cSpell:disable
        vcf_path = create_vcf(tmp_path, b"""##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##contig=<ID=chr1,length=248956422>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	HG00733
chr1	1134739	.	A	ACATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGCACCGTGAGGGGTGTGGCTTCCTCGC	30	PASS	.	GT	.|.
chr1	1134748	.	T	G	30	PASS	.	GT	.|.
chr1	1134771	.	G	GCACCGTGAGGGGTGTGGCTTCCTCGCCATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGCACCGTGAGGGGTGTGGCTTCCTCGCCATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGA	30	PASS	.	GT	.|."""
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
            sample_idx = vcf_file.samples().index(hg00733_sample.name)

        marker = request.node.get_closest_marker("cfg_overrides")
        local_cfg = hydra.compose(
            config_name="config",
            overrides=[*(marker.args if marker else []), "pileup=unphased_graph_both", f"graph.missing_are_ref={missing_are_ref}"],
        )
        example = make_graph_example(
            local_cfg,
            graph,
            sampler,
            unique_kmers,
            region,
            sample_idx, # type: ignore
            hg00733_sample,
            cfg.reads,
            filtered_kmer_path,
            analysis_variants, # type: ignore
        )

        assert example.get("label") == expected_label

    @pytest.mark.parametrize("region_str", [
        "chr1:1134644-1134867",
        # Add additional regions here to broaden the BWA-vs-k-mer agreement comparison.
    ])
    def test_bwa_kmer_agreement(self, request, cfg, hg00733_sample, tmp_path, region_str):
        """Collect per-fragment BWA vs. k-mer realignment agreement for `region_str` against the real
        HG00733 dipcall VCF, using the dual-run agreement harness (npsv3.images.agreement), and write a
        human-readable report to the results dir for manual inspection."""
        if not all(os.path.exists(f) for f in (cfg.reference, cfg.input, cfg.reads, f"{cfg.graph.ref_kmer_counts_kmc_prefix}.kmc_pre")):
            pytest.skip("Missing necessary inputs")
        if not bwa_index_loaded(cfg.reference):
            pytest.skip("BWA index not loaded")

        region = Range(region_str)
        ref_kmer_counts = KmerCounts(cfg.graph.ref_kmer_counts_kmc_prefix)
        graph, unique_kmers, sampler = _create_graph_and_sampler(
            cfg.reference,
            cfg.input,
            region,
            k=cfg.graph.kmer_size,
            ref_kmer_counts=ref_kmer_counts,
        )

        filtered_kmer_path = cache_filter_kmc_database(cfg, hg00733_sample, cfg.input, region, unique_kmers, tmp_path=tmp_path)
        with VariantFileReader.open(cfg.input) as vcf_file:
            analysis_variants = _filter_variants(list(vcf_file.fetch(region)), min_variant_size=50)
            sample_idx = vcf_file.samples().index(hg00733_sample.name)

        marker = request.node.get_closest_marker("cfg_overrides")
        local_cfg = hydra.compose(
            config_name="config",
            # allele_evidence=both is required for both engines to run per fragment; the agreement harness only
            # inspects the "real" image, so skip the (otherwise unused) simulated-genotype images.
            overrides=[*(marker.args if marker else []), "pileup=unphased_graph_both", "simulation.replicates=0"],
        )

        sink = agreement.AgreementSink(assign_delta=local_cfg.pileup.assign_delta)
        make_graph_example(
            local_cfg,
            graph,
            sampler,
            unique_kmers,
            region,
            sample_idx, # type: ignore
            hg00733_sample,
            cfg.reads,
            filtered_kmer_path,
            analysis_variants, # type: ignore
            agreement_sink=sink,
        )

        assert sink.rows, f"No fragments collected for region {region}"
        df = agreement.to_dataframe(sink.rows)
        assert set(df["engine"]) == {"bwa", "kmer"}, "Both engines should have run for every fragment"

        report = agreement.format_report(df, assign_delta=local_cfg.pileup.assign_delta)
        with open(result_path(f"agreement.{region.slug}.txt"), "w") as report_file:
            report_file.write(report)

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

        # Read single variant in the dataset
        dm = EnergonPackedImageDataModule(
            train_metadataset_paths=os.path.join(output_dir, "dataset.yaml"),
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
        assert torch.equal(first_batch.offsets, torch.tensor([0, 10])), "This variant has 10 simulated images"
        assert all(first_batch.labels[10:] == -100), "Everyting after valid examples should be masked out with -100"
        assert len(first_batch.genotype_label) == 1 and all(first_batch.genotype_label > 0)
        assert first_batch.meta == [str(region)]

        # Read from repeating dataset, similar to intended use in a DDP setting
        dm = EnergonPackedImageDataModule(
            train_metadataset_paths=os.path.join(output_dir, "dataset.yaml"),
            cache_dir=str(tmp_path / "cache"),
            batch_size=32,
            num_workers=0,
        )

        # By default the dataset repeats indefinitely, so read just the first batch
        first_batch = next(iter(dm.train_dataloader()))
        dm.teardown()

        # This variant is 10 simulated images, so 11 images total (1 query + 10 support). We can fit 2 copies (22
        # images) total in the batch of 32.
        assert torch.equal(first_batch.offsets, torch.tensor([0, 10, 20]))
        assert all(first_batch.labels[20:] == -100)
        assert first_batch.meta == [str(region)]*2


@pytest.mark.cfg_overrides(
    f"reference={HG38_REF_FASTA}",
    "input=/storage/mlinderman/projects/sv/npsv3-experiments/resources/hgsvc3-hprc-2024-02-23.dipcall.population.passing.training.hg38.vcf.gz",
    "reads=/storage/mlinderman/projects/sv/npsv3-experiments/resources/sequence/HG00096.final.cram",
    "graph.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
    "graph.population_prior=True",
    "graph.missing_are_ref=True",
    "simulation.replicates=0", # Speed up tests by not generating simulated images
)
class TestHGSVC3HPRCDipcallPopulationGraphExamples:
    @pytest.mark.usefixtures("ray_setup")
    @pytest.mark.parametrize("region_str", [
        "chr7:136717247-136717522",
        "chr3:20967930-20972107",
        # "chr4:49101919-49110988", # Leads to timeout error
    ])
    def test_error_regions(self, cfg, hg00096_sample, tmp_path, region_str):
        if not all(os.path.exists(f) for f in (cfg.reference, cfg.input, cfg.reads, f"{cfg.graph.ref_kmer_counts_kmc_prefix}.kmc_pre")):
            pytest.skip("Missing necessary inputs")

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

    @pytest.mark.usefixtures("ray_setup")
    def test_uncertain_label(self, request, cfg, hg00096_sample, tmp_path):
        if not all(os.path.exists(f) for f in (cfg.reference, cfg.input, cfg.reads, f"{cfg.graph.ref_kmer_counts_kmc_prefix}.kmc_pre")):
            pytest.skip("Missing necessary inputs")

        region = Range("chr5:157044297-157044704")
        
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
        with open(os.path.join(output_dir, "meta_ds", "train.jsonl")) as meta_file:
            rows = [json.loads(line) for line in meta_file if line.strip()]
        assert len(rows) == 1, f"Expected exactly one example for region {region}"
        [example] = rows
        assert example["label"] == 0

@pytest.mark.cfg_overrides(
    f"reference={HG38_REF_FASTA}",
    "input=/storage/mlinderman/projects/sv/npsv3-experiments/resources/hgsvc3-hprc-2024-02-23-mc-chm13.GRCh38.vcfbub.a100k.wave.passing.training.hg38.vcf.gz",
    "reads=/storage/mlinderman/projects/sv/npsv3-experiments/resources/sequence/HG00096.final.cram",
    "graph.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
    "graph.population_prior=True",
    "simulation.replicates=0", # Speed up tests by not generating simulated images
)
class TestHGSVC3HPRCPopulationGraphExamples:
    @pytest.mark.usefixtures("ray_setup")
    @pytest.mark.parametrize("region_str", ["chr3:184555693-184555884"])
    def test_no_labels(self, cfg, hg00096_sample, region_str, tmp_path):
        """HG00096's genotype in this region is partially missing (one called allele, one missing) which is treated as
        unresolved, and thus unlabled genotype"""

        if not all(os.path.exists(f) for f in (cfg.reference, cfg.input, cfg.reads, f"{cfg.graph.ref_kmer_counts_kmc_prefix}.kmc_pre")):
            pytest.skip("Missing necessary inputs")

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

        with open(os.path.join(output_dir, "meta_ds", "train.jsonl")) as meta_file:
            rows = [json.loads(line) for line in meta_file if line.strip()]
        assert len(rows) == 1, f"Expected exactly one example for region {region}"
        
        [example] = rows
        assert example.get("label") is None
        assert example.get("label.rank") is None


@pytest.mark.cfg_overrides(
    f"reference={HG38_REF_FASTA}",
    "input=/storage/mlinderman/projects/sv/npsv3-experiments/resources/hgsvc3-hprc-2024-02-23-mc-chm13/hgsvc3-hprc-2024-02-23-mc-chm13.pav.population.passing.training.hg38.vcf.gz",
    "reads=/storage/mlinderman/projects/sv/npsv3-experiments/resources/sequence/HG00096.final.cram",
    "graph.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
    "graph.population_prior=True",
    "graph.missing_are_ref=True",
    "pileup=unphased_graph_kmer",
    "simulation.replicates=1",
)
class TestHGSVC3HPRCPAVPopulationGraphExamples:
    @pytest.mark.usefixtures("ray_setup")
    @pytest.mark.parametrize("region_str", [
        # "chr17:940789-941433",
        # "chr20:31640012-31640510",
        # "chr4:109850391-109850582",
        "chr7:125624057-125624248",
    ])
    def test_high_loss_regions(self, cfg, hg00096_sample, region_str, tmp_path):
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
            exact_reference=False, #True,
        )
        assert os.path.exists(output_dir)

        with open(os.path.join(output_dir, "meta_ds", "train.jsonl")) as meta_file:
            rows = [json.loads(line) for line in meta_file if line.strip()]
        assert len(rows) == 1, f"Expected exactly one example for region {region}"

        # Render the images as read back from the Energon dataset, i.e., the tensors the model actually sees. (The
        # metadata rows don't carry the image tensors.)
        from npsv3.models.energon import EnergonPackedImageDataModule

        dm = EnergonPackedImageDataModule(
            train_metadataset_paths=os.path.join(output_dir, "dataset.yaml"),
            cache_dir=str(tmp_path / "cache"),
            batch_size=32,
            num_workers=0,
            repeat=False,
        )
        batches = list(dm.train_dataloader())
        dm.teardown()

        assert len(batches) == 1, "This single region should be a single batch"
        [batch] = batches
        assert batch.num_groups == 1
        assert batch.meta == [str(region)]

        assert int(batch.offsets[1]) // int(batch.num_genotypes[0]) == cfg.simulation.replicates

        all_png_path = result_path("all.png")
        image = group_to_image(cfg, batch, 0, all_png_path, render_channels=True)
        assert os.path.exists(all_png_path)
        assert image.width > 0
        assert image.height > 0
