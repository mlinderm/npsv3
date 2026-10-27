import glob
import json
import logging
import os

import hydra
import pytest
import torch
from omegaconf import OmegaConf

from npsv3.graphs.haplotype import KmerCounts, _create_graph_and_sampler, _filter_variants
from npsv3.images import agreement
from npsv3.images.example import (
    example_to_image,
    make_graph_example,
    vcf_to_graph_examples,
    vcf_to_haplotypes,
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
    "simulation.replicates=2",
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

    def test_true_haplotype_strategy(self, request, cfg, hg00733_sample, tmp_path):
        """The "true" haplotype strategy generates the reference, true, and true +/- SV haplotypes"""
        if not all(os.path.exists(f) for f in (cfg.reference, cfg.reads, f"{cfg.graph.ref_kmer_counts_kmc_prefix}.kmc_pre")):
            pytest.skip("Missing necessary inputs")
       
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

        marker = request.node.get_closest_marker("cfg_overrides")
        local_cfg = hydra.compose(
            config_name="config",
            overrides=[
                *(marker.args if marker else []),
                f"input={vcf_path}",
                "pileup=unphased_graph",
                "simulation.replicates=0",
                "haplotype.strategy._target_=npsv3.images.haplotypes.TrueHaplotypeStrategy",
            ],
        )

        ref_kmer_counts = KmerCounts(local_cfg.graph.ref_kmer_counts_kmc_prefix)
        graph, unique_kmers, sampler = _create_graph_and_sampler(
            local_cfg.reference,
            vcf_path,
            region,
            k=local_cfg.graph.kmer_size,
            ref_kmer_counts=ref_kmer_counts,
        )

        filtered_kmer_path = cache_filter_kmc_database(local_cfg, hg00733_sample, vcf_path, region, unique_kmers, tmp_path=tmp_path)
        with VariantFileReader.open(vcf_path) as vcf_file:
            analysis_variants = _filter_variants(list(vcf_file.fetch(region)), min_variant_size=50)
            sample_idx = vcf_file.samples().index(hg00733_sample.name)

        example = make_graph_example(
            local_cfg,
            graph,
            sampler,
            unique_kmers,
            region,
            sample_idx, # type: ignore
            hg00733_sample,
            local_cfg.reads,
            filtered_kmer_path,
            analysis_variants, # type: ignore
        )

        # Reference, the two true haplotypes, and both insertions (4 haplotypes, so 4*(4+1)/2 genotypes) with the true
        # genotype being (1, 2)
        assert example["num_genotypes"] == 10  # noqa: PLR2004
        assert example["label"] == 5  # noqa: PLR2004

    @pytest.mark.parametrize(("missing_are_ref", "expected_label"), [
        (False, None),  # Default: a sample with no called genotype for any analysis variant is unlabeled
        (True, 0),  # Opt-in: missing genotypes are assumed homozygous REF
    ])
    def test_missing_genotype_label(self, request, cfg, hg00733_sample, tmp_path, missing_are_ref, expected_label):
        """A sample with no called variant in the region (i.e. every analysis variant's genotype for that
        sample is missing) should be unlabeled unless haplotype.missing_are_ref opts into treating missing
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
            overrides=[*(marker.args if marker else []), "pileup=unphased_graph_both", f"haplotype.missing_are_ref={missing_are_ref}"],
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
    @pytest.mark.parametrize("threads", [1, 3])
    def test_vcf_to_energon_dataset(self, cfg, hg00733_sample, tmp_path, threads):
        """`vcf_to_graph_examples` emits the polylithic energon trees directly from GraphWriter. With more threads than
        graph shards, the shard is split across actors (some of which receive no regions)."""
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
        assert len(graph_shards) < threads or threads == 1

        cfg.threads = threads
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
        _assert_graph_links(output_dir, hg00733_sample.name, expected_graphs_dir=os.path.dirname(graph_shards[0]))

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
        assert torch.equal(first_batch.offsets, torch.tensor([0, cfg.simulation.replicates * 10])), "This variant has 10 simulated images"
        assert all(first_batch.labels[cfg.simulation.replicates *10:] == -100), "Everyting after valid examples should be masked out with -100"
        assert len(first_batch.genotype_label) == 1 and all(first_batch.genotype_label > 0)
        assert first_batch.meta == [str(region)]

        # Read from repeating dataset, similar to intended use in a DDP setting
        dm = EnergonPackedImageDataModule(
            train_metadataset_paths=os.path.join(output_dir, "dataset.yaml"),
            cache_dir=str(tmp_path / "cache"),
            batch_size=2 * cfg.simulation.replicates * 10 + 2 + 1, # 2 copies of the 10 simulated images plus query images
            num_workers=0,
        )

        # By default the dataset repeats indefinitely, so read just the first batch
        first_batch = next(iter(dm.train_dataloader()))
        dm.teardown()

        # This variant is 10 simulated images, with 2 replicates it is 21 images (1 query + 20 support). We can fit 2 copies
        # total in the batch of 48.
        assert torch.equal(first_batch.offsets, torch.tensor([0, cfg.simulation.replicates * 10, 2 * cfg.simulation.replicates * 10]))
        assert all(first_batch.labels[2 * cfg.simulation.replicates * 10:] == -100)
        assert first_batch.meta == [str(region)]*2


    @pytest.mark.usefixtures("ray_setup")
    def test_vcf_to_energon_dataset_with_graphs_on_the_fly(self, cfg, hg00733_sample, tmp_path):
        """Graphs generated during image generation are kept as the dataset's `graphs` aux tree"""
        region = Range("chr1:1134644-1134867")
        _graph_shards, filtered_kmer_path = cache_graph_and_filter_kmc_database(cfg, hg00733_sample, cfg.input, region)

        output_dir = str(tmp_path / "energon")
        vcf_to_graph_examples(
            cfg, cfg.reads, hg00733_sample, cfg.input, output_dir, filtered_kmer_path=filtered_kmer_path, region=region,
        )
        assert not glob.glob(os.path.join(output_dir, "graphs_ds", "unique_kmers*")), "Pooled k-mers stay temporary"
        _assert_graph_links(output_dir, hg00733_sample.name, expected_graphs_dir=os.path.join(output_dir, "graphs_ds"))


    @pytest.mark.usefixtures("ray_setup")
    def test_vcf_to_energon_dataset_with_comparisons(self, cfg, hg00733_sample, tmp_path):
        """With `compare.enabled`, each labeled example's genotypes are compared with its label in the `comparisons` aux tree"""
        pytest.importorskip("edlib")
        pytest.importorskip("pyabpoa")
        import gzip

        import yaml
        from megatron.energon.cache.file_store import WebdatasetFileStore
        from megatron.energon.epathlib import EPath

        from npsv3.constants import COMPARISON_SUFFIX, COMPARISONS_AUX, COMPARISONS_DIR, HAPLOTYPES_SUFFIX
        from npsv3.images.sequence_compare import ComparisonOptions, comparison_record
        from npsv3.images.haplotypes import decode_haplotype_record

        region = Range("chr1:1134644-1134867")
        graph_shards, filtered_kmer_path = cache_graph_and_filter_kmc_database(cfg, hg00733_sample, cfg.input, region)
        cfg.compare.enabled = True
        output_dir = str(tmp_path / "energon")
        vcf_to_graph_examples(
            cfg, cfg.reads, hg00733_sample, cfg.input, output_dir, graph_shards=graph_shards,
            filtered_kmer_path=filtered_kmer_path,
        )

        with open(os.path.join(output_dir, "dataset.yaml")) as file:
            aux = yaml.safe_load(file)["splits"]["train"]["aux"]
        assert aux[COMPARISONS_AUX] == f"./{COMPARISONS_DIR}"

        imgs = WebdatasetFileStore(EPath(os.path.join(output_dir, "imgs_ds")))
        comparisons = WebdatasetFileStore(EPath(os.path.join(output_dir, COMPARISONS_DIR)))
        with open(os.path.join(output_dir, "meta_ds", "train.jsonl")) as meta_file:
            rows = [json.loads(line) for line in meta_file if line.strip()]
        labeled = [row for row in rows if row["label"] is not None]
        assert labeled
        for row in labeled:
            record = json.loads(gzip.decompress(comparisons[f"{row['key']}.{COMPARISON_SUFFIX}"][0]))
            assert len(record["genotype_differ"]) == row["g"]
            assert record["genotype_differ"][row["label"]] is False
            assert record["genotype_score"][row["label"]] == 1.0
            # The sample is in the VCF, so its own paths are in the graph and the label is compared with them
            assert record["label_differs"] is not None
            haplotypes = decode_haplotype_record(imgs[f"{row['key']}.{HAPLOTYPES_SUFFIX}"][0])
            assert record == comparison_record(haplotypes, ComparisonOptions.from_config(cfg.compare))


    @pytest.mark.usefixtures("ray_setup")
    def test_vcf_to_energon_dataset_with_precomputed_haplotypes(self, cfg, hg00733_sample, tmp_path, caplog):
        """Haplotypes (and comparisons) generated ahead of time are used by image generation"""
        pytest.importorskip("edlib")
        pytest.importorskip("pyabpoa")
        import gzip
        import logging

        from megatron.energon.cache.file_store import WebdatasetFileStore
        from megatron.energon.epathlib import EPath

        from npsv3.constants import COMPARISON_SUFFIX, COMPARISONS_DIR, HAPLOTYPES_DIR, HAPLOTYPES_SUFFIX
        from npsv3.images.haplotype_tree import HaplotypeTree
        from npsv3.images.haplotypes import decode_haplotype_record

        region = Range("chr1:1134644-1134867")
        graph_shards, filtered_kmer_path = cache_graph_and_filter_kmc_database(cfg, hg00733_sample, cfg.input, region)
        cfg.compare.enabled = True
        cfg.simulation.replicates = 0  # Simulated images don't depend on how the haplotypes were generated

        haplotypes_dir = str(tmp_path / "haplotypes")
        vcf_to_haplotypes(
            cfg, hg00733_sample, cfg.input, haplotypes_dir, graph_shards=graph_shards, filtered_kmer_path=filtered_kmer_path,
        )
        tree = HaplotypeTree(haplotypes_dir)
        assert tree.filtered_kmer_path == os.path.abspath(filtered_kmer_path), "Filtered k-mers used are recorded"

        # The filtered k-mers are found via the haplotypes
        output_dir = str(tmp_path / "energon")
        with caplog.at_level(logging.INFO):
            vcf_to_graph_examples(
                cfg, cfg.reads, hg00733_sample, cfg.input, output_dir, graph_shards=graph_shards, haplotypes_dir=haplotypes_dir,
            )
        assert "for 1 regions (and skipped 0), generated them for 0" in caplog.text
        _assert_graph_links(output_dir, hg00733_sample.name, expected_graphs_dir=os.path.dirname(graph_shards[0]))

        precomputed = WebdatasetFileStore(EPath(os.path.join(haplotypes_dir, HAPLOTYPES_DIR)))
        imgs = WebdatasetFileStore(EPath(os.path.join(output_dir, "imgs_ds")))
        comparisons = WebdatasetFileStore(EPath(os.path.join(output_dir, COMPARISONS_DIR)))
        with open(os.path.join(output_dir, "meta_ds", "train.jsonl")) as meta_file:
            rows = [json.loads(line) for line in meta_file if line.strip()]
        assert [row["label"] is not None for row in rows] == [True]
        for row in rows:
            expected = decode_haplotype_record(precomputed[f"{row['key']}.{HAPLOTYPES_SUFFIX}"][0])
            expected.pop("source_digest")
            assert decode_haplotype_record(imgs[f"{row['key']}.{HAPLOTYPES_SUFFIX}"][0]) == expected
            assert gzip.decompress(comparisons[f"{row['key']}.{COMPARISON_SUFFIX}"][0]) == gzip.decompress(tree.comparison(row["key"]))  # type: ignore[arg-type]

        # Haplotypes generated with different settings are an error
        cfg.haplotype.max_haplotypes += 1
        with pytest.raises(ValueError, match="different settings"):
            vcf_to_graph_examples(
                cfg, cfg.reads, hg00733_sample, cfg.input, str(tmp_path / "mismatch"), graph_shards=graph_shards,
                haplotypes_dir=haplotypes_dir,
            )


def _assert_graph_links(output_dir: str, sample_name: str, *, expected_graphs_dir: str):
    """Every row links to its graph in the `graphs` aux tree and carries a haplotypes member consistent with its genotypes"""
    import itertools

    import yaml
    from megatron.energon.cache.file_store import WebdatasetFileStore
    from megatron.energon.epathlib import EPath

    from npsv3.constants import HAPLOTYPES_SUFFIX
    from npsv3.graphs.graph import Graph
    from npsv3.graphs.sharding import GRAPH_MEMBER, decode_graph_record, graph_digest
    from npsv3.images.haplotypes import decode_haplotype_record

    with open(os.path.join(output_dir, "dataset.yaml")) as file:
        aux = yaml.safe_load(file)["splits"]["train"]["aux"]
    assert aux["graphs"] == os.path.realpath(expected_graphs_dir)

    imgs = WebdatasetFileStore(EPath(os.path.join(output_dir, "imgs_ds")))
    graphs = WebdatasetFileStore(EPath(aux["graphs"]))
    with open(os.path.join(output_dir, "meta_ds", "train.jsonl")) as meta_file:
        rows = [json.loads(line) for line in meta_file if line.strip()]
    assert rows
    for row in rows:
        payload, _ = graphs[f"{row['graph_key']}.{GRAPH_MEMBER}.gz"]
        graph_bytes = decode_graph_record({f"{GRAPH_MEMBER}.gz": payload})[GRAPH_MEMBER]
        assert graph_digest(graph_bytes) == row["graph_digest"]
        graph = Graph.load_bytes(graph_bytes)

        haplotypes = decode_haplotype_record(imgs[f"{row['key']}.{HAPLOTYPES_SUFFIX}"][0])
        paths = haplotypes["paths"]
        genotypes = list(itertools.combinations_with_replacement(range(len(paths)), 2))
        assert len(genotypes) == row["g"]
        assert haplotypes["sequences"] == [graph.path_sequence(path) for path in paths]
        assert len(haplotypes["alleles"]) == len(paths)
        if row["label"] is not None:
            assert genotypes[row["label"]] == tuple(sorted(haplotypes["true"]))
        # The sample is in the VCF, so its own phased paths are in the graph
        assert all(path is not None for path in haplotypes["sample_paths"])
        assert haplotypes["sample_sequences"] == [graph.path_sequence(path) for path in haplotypes["sample_paths"]]
        assert haplotypes["reference"] == graph.path_sequence(graph.path_nodes(Range(row["region"]).contig))
        assert sample_name == row["sample"]


@pytest.mark.cfg_overrides(
    f"reference={HG38_REF_FASTA}",
    "input=/storage/mlinderman/projects/sv/npsv3-experiments/resources/hgsvc3-hprc-2024-02-23-mc-chm13/hgsvc3-hprc-2024-02-23-mc-chm13.dipcall.population.passing.training.hg38.vcf.gz",
    "reads=/storage/mlinderman/projects/sv/npsv3-experiments/resources/sequence/HG00096.final.cram",
    "graph.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
    "graph.population_prior=True",
    "haplotype.missing_are_ref=True",
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
    def test_no_labels(self, cfg, hg00096_sample, region_str, tmp_path, caplog):
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

        # Unlabeled regions are skipped entirely with `skip_unlabeled` (the `images` command's default)
        OmegaConf.update(cfg, "skip_unlabeled", True, force_add=True)
        output_dir = str(tmp_path / "skipped")
        # This is the only region, so no examples are written at all (which is an error when preparing the dataset)
        with caplog.at_level(logging.INFO), pytest.raises(ValueError, match="No .* shards found"):
            vcf_to_graph_examples(
                cfg,
                cfg.reads,
                hg00096_sample,
                cfg.input,
                output_dir,
                graph_shards=graph_shards,
                filtered_kmer_path=filtered_kmer_path,
            )
        assert "Skipped 1 unlabeled regions" in caplog.text


@pytest.mark.cfg_overrides(
    f"reference={HG38_REF_FASTA}",
    "input=/storage/mlinderman/projects/sv/npsv3-experiments/resources/hgsvc3-hprc-2024-02-23-mc-chm13/hgsvc3-hprc-2024-02-23-mc-chm13.pav.population.passing.training.hg38.vcf.gz",
    "reads=/storage/mlinderman/projects/sv/npsv3-experiments/resources/sequence/HG00096.final.cram",
    "graph.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
    "graph.population_prior=True",
    "haplotype.missing_are_ref=True",
    "pileup=unphased_graph_kmer",
    "simulation.replicates=1",
)
class TestHGSVC3HPRCPAVPopulationGraphExamples:
    @pytest.mark.usefixtures("ray_setup")
    @pytest.mark.parametrize("region_str", [
        # "chr17:940789-941433",
        # "chr20:31640012-31640510",
        # "chr4:109850391-109850582",
        # "chr7:125624057-125624248",
        "chr6:140682234-140683310",
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
            batch_size=64,
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
