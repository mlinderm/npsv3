"""The end-to-end genotyper (`npsv3.genotype`)"""

import os
import shutil

import pysam
import pytest
import yaml

from npsv3 import genotype as genotype_module
from npsv3.genotype import INPUTS_DIR, genotype
from npsv3.genotyping.images import GENERATION_FILE, GRAPHS_DIR, GenotypingImages, generation_settings, write_generation
from npsv3.genotyping.inputs import PROVENANCE_FILE
from npsv3.images.example import FAILURES_GLOB, FailureWriter, energon_key, read_failures
from npsv3.simulation import bwa_index_loaded
from npsv3.util.sample import Sample

from . import HG00733_HG38_BAM, HG38_BWA_MEM3_INDEX, HG38_REF_FASTA, create_vcf
from .genotyping.test_images import HG00733_VCF
from .genotyping.test_predict import PREDICT_OVERRIDES, _compose, _train_checkpoint
from .genotyping.test_vcf import SAMPLE, _haplotype_record, _write_images, _write_vcfs

FAKE_SAMPLE = Sample(SAMPLE, mean_coverage=30, mean_insert_size=400, std_insert_size=100, sequencer="HSXn", read_length=150)


@pytest.fixture(scope="module")
def checkpoint(hydra_setup, tmp_path_factory):  # noqa: ARG001
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in ("SLURM_NTASKS", "SLURM_NPROCS"):
            monkeypatch.delenv(name, raising=False)
        return _train_checkpoint(tmp_path_factory)


@pytest.fixture
def genotype_cfg(checkpoint, tmp_path):
    reference = tmp_path / "reference.fasta"
    reference.touch()
    return _compose(
        f"+model.checkpoint='{checkpoint}'",
        f"reference={reference}",
        "pileup.variant_padding=100",
        *(f"data.{key}={value}" for key, value in PREDICT_OVERRIDES.items()),
        f"data.cache_dir={tmp_path / 'cache'}",
    )


class FakeImages:
    """Stands in for stages 2 and 3 (which need the reference, reads, aligner, ...), writing images for the first region and
    recording a failure for the second."""

    def __init__(self, num_regions=5):
        self.calls = []
        self.num_regions = num_regions

    def __call__(self, cfg, read_path, sample, inputs, regions, work_dir, *, graph_shards, unique_kmer_path, images_dir, ploidy, min_variant_size, progress_bar):  # noqa: ARG002
        self.calls.append({"regions": regions, "graph_shards": graph_shards, "unique_kmer_path": unique_kmer_path, "images_dir": images_dir})
        assert len(regions) == self.num_regions
        first = regions[0]
        record = _haplotype_record(
            inputs.vcf_path,
            first,
            {"called_del": [0, 1], "pop_snv": [0, 1], "pop_multi": [0, 2], "novel": [1, 0]},
            2,
        )
        dataset = _write_images(
            images_dir,
            [(first, record)],
            height=cfg.pileup.image_height,
            width=cfg.pileup.image_width,
            channels=len(cfg.pileup.image_channels),
        )
        writer = FailureWriter(0, images_dir)
        for region in regions[1:]:
            writer.write(energon_key(sample.name, region), region, sample.name, "Timeout", "Too slow")
        writer.close()
        write_generation(images_dir, generation_settings(cfg, ploidy=ploidy, min_variant_size=min_variant_size))
        graphs = os.path.join(work_dir, GRAPHS_DIR)
        os.makedirs(graphs, exist_ok=True)
        return GenotypingImages(
            dataset_path=dataset, images_dir=images_dir, graph_shards=[], unique_kmer_path=None, filtered_kmer_path=None,
            counts={}, failures=read_failures(images_dir),
        )


@pytest.fixture
def fake_images(monkeypatch):
    fake = FakeImages()
    monkeypatch.setattr(genotype_module, "generate_genotyping_images", fake)
    return fake


@pytest.fixture
def reads(tmp_path):
    path = tmp_path / "reads.bam"
    path.touch()
    return str(path)


def _genotypes(path):
    with pysam.VariantFile(str(path)) as vcf_file:
        # By each of the IDs of a merged record
        return {
            record_id: (record.samples[SAMPLE]["GT"], list(record.filter))
            for record in vcf_file
            for record_id in record.id.split(";")
        }


@pytest.mark.slow
class TestGenotypeOrchestration:
    def test_inference_and_population(self, genotype_cfg, fake_images, reads, tmp_path):
        """Real input preparation, inference and output, without generating the images"""
        population, inference = _write_vcfs(tmp_path)
        output = tmp_path / "out" / "genotypes.vcf.gz"
        counts = genotype(genotype_cfg, reads, FAKE_SAMPLE, population, output, inference_vcf=inference)
        assert len(fake_images.calls) == 1
        assert output.exists()
        assert os.path.exists(f"{output}.tbi")

        records = _genotypes(output)
        assert records["no_region"] == ((None, None), ["NoRegion"])
        assert records["in_timeout"] == ((None, None), ["Timeout"])
        assert records["star"] == ((None, None), ["NotInGraph"])
        assert records["straddle"] == ((None, None), ["NotInGraph"])
        for name in ("inference_del", "pop_snv", "pop_multi", "novel"):
            assert records[name][1] == ["PASS"]
            assert None not in records[name][0]
        assert counts["called"] == 4
        assert records["in_unselected"] == ((None, None), ["NoRegion"]), "Its region has no inference analysis variant"
        assert "unselected_del" not in records

    def test_genotype_all_regions(self, genotype_cfg, fake_images, reads, tmp_path):
        """With `region_selection=all`, the regions without an inference analysis variant are also genotyped"""
        population, inference = _write_vcfs(tmp_path)
        genotype_cfg.region_selection = "all"
        fake_images.num_regions = 6
        output = tmp_path / "out" / "genotypes.vcf.gz"
        genotype(genotype_cfg, reads, FAKE_SAMPLE, population, output, inference_vcf=inference)
        records = _genotypes(output)
        assert records["in_unselected"] == ((None, None), ["Timeout"]), "Its region is genotyped (and times out)"
        assert records["no_region"] == ((None, None), ["NoRegion"])

    def test_work_dir_keeps_intermediates(self, genotype_cfg, fake_images, reads, tmp_path):  # noqa: ARG002
        population, inference = _write_vcfs(tmp_path)
        output = tmp_path / "genotypes.vcf.gz"
        work = tmp_path / "work"
        genotype(genotype_cfg, reads, FAKE_SAMPLE, population, output, inference_vcf=inference, work_dir=work)
        assert (work / "images" / "dataset.yaml").exists()
        assert (work / INPUTS_DIR / PROVENANCE_FILE).exists()
        assert (work / "predictions" / "predictions-rank0.pkl").exists()

    def test_reuse_images_with_a_new_model(self, genotype_cfg, fake_images, reads, tmp_path):
        """Images specified for a run are reused by the next, without its working directory"""
        population, inference = _write_vcfs(tmp_path)
        images = tmp_path / "saved-images"
        genotype(genotype_cfg, reads, FAKE_SAMPLE, population, tmp_path / "first.vcf.gz", inference_vcf=inference, images_dir=images)
        assert len(fake_images.calls) == 1
        assert (images / GENERATION_FILE).exists()
        assert (images / INPUTS_DIR / "genotyping.vcf.gz").exists()

        genotype(genotype_cfg, reads, FAKE_SAMPLE, population, tmp_path / "second.vcf.gz", inference_vcf=inference, images_dir=images)
        assert len(fake_images.calls) == 1, "The images are reused"
        assert _genotypes(tmp_path / "first.vcf.gz") == _genotypes(tmp_path / "second.vcf.gz")

    def test_images_from_different_inputs(self, genotype_cfg, fake_images, reads, tmp_path):  # noqa: ARG002
        population, inference = _write_vcfs(tmp_path)
        images = tmp_path / "saved-images"
        genotype(genotype_cfg, reads, FAKE_SAMPLE, population, tmp_path / "first.vcf.gz", inference_vcf=inference, images_dir=images)

        other = tmp_path / "other.vcf.gz"
        shutil.copy(inference, other)
        shutil.copy(f"{inference}.tbi", f"{other}.tbi")
        with pytest.raises(ValueError, match="inference"):
            genotype(genotype_cfg, reads, FAKE_SAMPLE, population, tmp_path / "second.vcf.gz", inference_vcf=other, images_dir=images)

    def test_images_are_not_mixed_with_earlier_files(self, genotype_cfg, fake_images, reads, tmp_path):
        population, inference = _write_vcfs(tmp_path)
        work_dir = tmp_path / "work"
        genotype(genotype_cfg, reads, FAKE_SAMPLE, population, tmp_path / "first.vcf.gz", inference_vcf=inference, work_dir=work_dir)
        assert work_dir.exists(), "A specified working directory is kept"
        with pytest.raises(ValueError, match="already has files"):
            genotype(genotype_cfg, reads, FAKE_SAMPLE, population, tmp_path / "second.vcf.gz", inference_vcf=inference, work_dir=work_dir)
        assert len(fake_images.calls) == 1
        assert len(list((work_dir / "images").glob(FAILURES_GLOB))) == 1

    def test_reused_graphs_need_matching_inputs(self, genotype_cfg, fake_images, reads, tmp_path):
        population, inference = _write_vcfs(tmp_path)
        graphs = tmp_path / "graphs"
        graphs.mkdir()
        shard = graphs / "graphs-0000.tar"
        shard.touch()
        kmers = tmp_path / "kmers"
        with pytest.raises(ValueError, match=PROVENANCE_FILE):
            genotype(
                genotype_cfg, reads, FAKE_SAMPLE, population, tmp_path / "first.vcf.gz", inference_vcf=inference,
                graph_shards=[str(shard)], unique_kmer_path=str(kmers),
            )
        assert fake_images.calls == []

        # The graphs a previous run recorded
        work_dir = tmp_path / "work"
        genotype(genotype_cfg, reads, FAKE_SAMPLE, population, tmp_path / "first.vcf.gz", inference_vcf=inference, work_dir=work_dir)
        recorded = work_dir / GRAPHS_DIR
        assert (recorded / PROVENANCE_FILE).exists()
        with open(recorded / PROVENANCE_FILE) as file:
            assert "reads" in yaml.safe_load(file), "Everything is recorded, but the reads aren't compared"
        shard = recorded / "graphs-0000.tar"
        shard.touch()

        other_reads = tmp_path / "other.bam"
        other_reads.write_text("different")
        genotype(
            genotype_cfg, str(other_reads), FAKE_SAMPLE, population, tmp_path / "second.vcf.gz", inference_vcf=inference,
            images_dir=tmp_path / "images2", graph_shards=[str(shard)], unique_kmer_path=str(kmers),
        )
        assert fake_images.calls[-1]["graph_shards"] == [str(shard)]


class TestChecks:
    def test_no_replicates(self, genotype_cfg, reads, tmp_path):
        population, _inference = _write_vcfs(tmp_path)
        genotype_cfg.simulation.replicates = 0
        with pytest.raises(ValueError, match="replicate"):
            genotype(genotype_cfg, reads, FAKE_SAMPLE, population, tmp_path / "genotypes.vcf.gz")

    def test_invalid_devices(self, genotype_cfg, reads, tmp_path):
        population, _inference = _write_vcfs(tmp_path)
        genotype_cfg.devices = "gpu"
        with pytest.raises(ValueError, match="devices must be"):
            genotype(genotype_cfg, reads, FAKE_SAMPLE, population, tmp_path / "genotypes.vcf.gz")

    def test_invalid_region_selection(self, genotype_cfg, reads, tmp_path):
        population, _inference = _write_vcfs(tmp_path)
        genotype_cfg.region_selection = "some"
        with pytest.raises(ValueError, match="region_selection"):
            genotype(genotype_cfg, reads, FAKE_SAMPLE, population, tmp_path / "genotypes.vcf.gz")

    def test_no_graph_shards(self, genotype_cfg, reads, tmp_path):
        """A `graph_shards` pattern that doesn't match any files must not look like there is nothing to genotype"""
        population, _inference = _write_vcfs(tmp_path)
        with pytest.raises(ValueError, match="no graph shards"):
            genotype(genotype_cfg, reads, FAKE_SAMPLE, population, tmp_path / "genotypes.vcf.gz", graph_shards=[])

    def test_missing_checkpoint(self, genotype_cfg, reads, tmp_path):
        population, _inference = _write_vcfs(tmp_path)
        genotype_cfg.model.checkpoint = str(tmp_path / "missing.ckpt")
        with pytest.raises(ValueError, match="checkpoint"):
            genotype(genotype_cfg, reads, FAKE_SAMPLE, population, tmp_path / "genotypes.vcf.gz")

    def test_no_regions(self, genotype_cfg, fake_images, reads, tmp_path):
        """The inference VCF has no record in any analysis variant's region"""
        population, _inference = _write_vcfs(tmp_path)
        small = create_vcf(
            tmp_path,
            b"##fileformat=VCFv4.2\n##contig=<ID=chr1,length=1000000>\n##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
            b"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tOTHER\nchr1\t50000\tfar\tA\tG\t.\t.\t.\tGT\t0/1\n",
            name="small.vcf.gz",
        )
        output = tmp_path / "genotypes.vcf.gz"
        genotype(genotype_cfg, reads, FAKE_SAMPLE, population, output, inference_vcf=small)
        assert fake_images.calls == []
        assert _genotypes(output) == {"far": ((None, None), ["NoRegion"])}


class TestMain:
    def test_genotype_arguments(self, hydra_setup, monkeypatch, tmp_path):  # noqa: ARG002
        """The command line's relative paths are made absolute, and passed along"""
        # Importing the entry point registers the resolvers, which the tests have already done
        monkeypatch.setattr("npsv3.util.config.setup_resolvers", lambda: None)
        from npsv3 import main as main_module

        calls = []
        monkeypatch.setattr(genotype_module, "genotype", lambda *args, **kwargs: calls.append((args, kwargs)))
        monkeypatch.setattr(main_module, "_check_shared_reference", lambda cfg: None)  # noqa: ARG005
        monkeypatch.setattr(Sample, "from_json", classmethod(lambda cls, path: FAKE_SAMPLE))  # noqa: ARG005
        monkeypatch.chdir(tmp_path)
        (tmp_path / "shard-0.tar").touch()

        cfg = _compose(
            "command=genotype", "reference=ref.fa", "stats_path=stats.json", "reads=reads.bam", "population=population.vcf.gz",
            "inference=inference.vcf.gz", "output=out/genotypes.vcf.gz", "work_dir=work",
            "images_dir=images", "graph_shards=shard-*.tar", "unique_kmer_path=kmers", "+model.checkpoint=model.ckpt",
        )
        main_module.main.__wrapped__(cfg)

        [(args, kwargs)] = calls
        assert args[1] == str(tmp_path / "reads.bam")
        assert args[2] is FAKE_SAMPLE
        assert args[3] == str(tmp_path / "population.vcf.gz")
        assert args[4] == str(tmp_path / "out" / "genotypes.vcf.gz")
        assert kwargs == {
            "inference_vcf": str(tmp_path / "inference.vcf.gz"),
            "work_dir": str(tmp_path / "work"),
            "images_dir": str(tmp_path / "images"),
            "graph_shards": [str(tmp_path / "shard-0.tar")],
            "unique_kmer_path": str(tmp_path / "kmers"),
        }
        assert args[0].model.checkpoint == str(tmp_path / "model.ckpt")

    def test_defaults(self, hydra_setup):  # noqa: ARG002
        cfg = _compose("command=genotype")
        assert cfg.devices == "cpu"
        assert cfg.inference is None
        assert cfg.work_dir is None
        assert cfg.scoring.aggregate == "mean"
        assert cfg.scoring.max_gq == 99
        assert not cfg.skip_unlabeled


@pytest.mark.slow
@pytest.mark.skipif(
    not HG38_REF_FASTA or not HG38_BWA_MEM3_INDEX or not os.path.exists(HG00733_HG38_BAM or ""),
    reason="The HG38 reference, HG00733 reads and bwa-mem3 index are required",
)
@pytest.mark.usefixtures("ray_setup")
def test_genotype_small_region(checkpoint, hg00733_sample, tmp_path):
    """Genotype a small region of a sample excluded from its population, through every real stage"""
    if hg00733_sample.kmc_prefix is None:
        pytest.skip("Missing HG00733 KMC database")
    if shutil.which("bwa-mem3") is None or not bwa_index_loaded(HG38_BWA_MEM3_INDEX):
        pytest.skip("BWA index not loaded")

    cfg = _compose(
        f"+model.checkpoint='{checkpoint}'",
        f"reference={HG38_REF_FASTA}",
        f"reads={HG00733_HG38_BAM}",
        f"aligner_index={HG38_BWA_MEM3_INDEX}",
        "graph.ref_kmer_counts_kmc_prefix=/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
        *(f"data.{key}={value}" for key, value in PREDICT_OVERRIDES.items()),
        f"data.cache_dir={tmp_path / 'cache'}",
    )
    population = create_vcf(tmp_path, HG00733_VCF, name="population.vcf.gz")
    output = tmp_path / "genotypes.vcf.gz"
    counts = genotype(cfg, HG00733_HG38_BAM, hg00733_sample, population, output, work_dir=tmp_path / "work")

    assert sum(counts.values()) > 0
    with pysam.VariantFile(str(output)) as vcf_file:
        assert list(vcf_file.header.samples) == [hg00733_sample.name]
        records = list(vcf_file)
    assert records
    for record in records:
        sample = record.samples[hg00733_sample.name]
        assert sample.phased
        assert None not in sample["GT"]
        assert sample["PS"] == records[0].pos
        assert 0 <= sample["GQ"] <= 99
