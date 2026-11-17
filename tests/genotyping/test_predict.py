"""Stage 4 of the genotyper: scoring the genotyping images (`npsv3.genotyping.predict`)"""

import os

import hydra
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from npsv3.genotyping.images import generation_settings, write_generation
from npsv3.genotyping.predict import (
    PREDICTION_FILE,
    PREDICTIONS_DIR,
    PREDICTIONS_PATTERN,
    PredictionWriterCallback,
    _predict_rank,
    _spawned_rank,
    check_generation,
    predict_data_config,
    predict_genotypes,
    read_predictions,
    read_rejections,
    resolve_devices,
)
from npsv3.images.example import EnergonExampleWriter, energon_key
from npsv3.models import runners
from npsv3.models.energon import REJECT_TOO_MANY_GENOTYPES, prepare_energon_dataset
from npsv3.util.range import Range

from ..models.test_paired import _DATA_TEST_OVERRIDES, CFG_OVERRIDES, _build_split

SAMPLE = "HG00733"
OVERSIZED = {3}
"""Indices of the regions with more genotypes than `haplotype.max_diplotypes`, so the loader rejects them"""

# The largest region (1 + 1*36 image slots) must fit in a batch
PREDICT_OVERRIDES = {**_DATA_TEST_OVERRIDES, "batch_size": 40, "g_max": 36}


def _compose(*overrides):
    return hydra.compose(config_name="config", overrides=[*CFG_OVERRIDES, *overrides])


@pytest.fixture(scope="module")
def checkpoint(hydra_setup, tmp_path_factory):  # noqa: ARG001
    """A (barely) trained model checkpoint at the `CFG_OVERRIDES` geometry"""
    with pytest.MonkeyPatch.context() as monkeypatch:
        # Lightning's SLURM detection rejects the `--ntasks` of an interactive SLURM job (training isn't under test)
        for name in ("SLURM_NTASKS", "SLURM_NPROCS"):
            monkeypatch.delenv(name, raising=False)
        return _train_checkpoint(tmp_path_factory)


def _train_checkpoint(tmp_path_factory):
    cfg = _compose()
    root = tmp_path_factory.mktemp("checkpoint")
    geometry = {
        "height": cfg.pileup.image_height,
        "width": cfg.pileup.image_width,
        "channels": len(cfg.pileup.image_channels),
    }
    train = _build_split(str(root / "train"), "chr1", num_groups=8, seed=20261107, **geometry)
    train_cfg = _compose(
        f"data.metadataset_path={train}",
        *(f"data.{key}={value}" for key, value in {**_DATA_TEST_OVERRIDES, "limit_train_batches": 1}.items()),
        "+trainer.accelerator=cpu",
        "trainer.max_epochs=1",
    )
    return runners.train(train_cfg, output_dir=str(root / "model"))


@pytest.fixture
def genotype_cfg(checkpoint, tmp_path):
    return _compose(
        f"+model.checkpoint='{checkpoint}'",
        *(f"data.{key}={value}" for key, value in PREDICT_OVERRIDES.items()),
        f"data.cache_dir={tmp_path / 'cache'}",
    )


@pytest.fixture(scope="module")
def unlabeled_dataset(hydra_setup, tmp_path_factory):  # noqa: ARG001
    """An unlabeled image dataset like the genotyper writes (with a `generation.yaml`), and its regions by key"""
    cfg = _compose()
    output_dir = str(tmp_path_factory.mktemp("unlabeled"))
    rng = np.random.default_rng(20261108)
    height, width, channels = cfg.pileup.image_height, cfg.pileup.image_width, len(cfg.pileup.image_channels)

    regions = {}
    writers = [EnergonExampleWriter(shard, output_dir) for shard in range(2)]
    try:
        for index in range(12):
            num_genotypes = 40 if index in OVERSIZED else int(rng.integers(3, 7))
            region = Range(f"chr2:{index * 1000 + 1}-{index * 1000 + 500}")
            writers[index % 2].write(
                {
                    "region": str(region),
                    "image": rng.integers(0, 255, size=(height, width, channels), dtype=np.uint8),
                    "sim.images": rng.integers(0, 255, size=(1, num_genotypes, height, width, channels), dtype=np.uint8),
                    "num_genotypes": num_genotypes,
                },
                region=region,
                sample_name=SAMPLE,
            )
            regions[energon_key(SAMPLE, region)] = (str(region), num_genotypes)
    finally:
        for writer in writers:
            writer.close()
    dataset = prepare_energon_dataset(output_dir)
    write_generation(output_dir, generation_settings(cfg, ploidy=2, min_variant_size=50))
    return dataset, regions


def _check_partition(frame: pd.DataFrame, rejected: list[dict], regions: dict) -> None:
    """Every region is predicted exactly once, or rejected (and only the oversized ones are)"""
    assert not frame["key"].duplicated().any()
    assert {row["key"] for row in rejected} == {key for key, (_region, g) in regions.items() if g > 36}
    assert all(row["reason"] == REJECT_TOO_MANY_GENOTYPES for row in rejected)
    assert set(frame["key"]) | {row["key"] for row in rejected} == set(regions)
    assert not set(frame["key"]) & {row["key"] for row in rejected}
    for row in frame.itertuples():
        region, num_genotypes = regions[row.key]
        assert row.region == region
        assert row.num_genotypes == num_genotypes
        assert row.num_replicates == 1
        assert row.metric.shape == (1, num_genotypes)
        assert row.metric.dtype == np.float32
        # Cosine similarities
        assert np.all(np.isfinite(row.metric))
        assert np.all(np.abs(row.metric) <= 1 + 1e-5)


class TestResolveDevices:
    def test_cpu(self):
        assert resolve_devices("cpu") is None
        assert resolve_devices(None) is None

    def test_auto(self):
        expected = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else None
        assert resolve_devices("auto") == (expected or None)

    def test_too_many_gpus(self):
        available = torch.cuda.device_count() if torch.cuda.is_available() else 0
        with pytest.raises(ValueError, match="visible"):
            resolve_devices(available + 1)
        with pytest.raises(ValueError, match="visible"):
            resolve_devices([available])
        with pytest.raises(ValueError, match="devices must be"):
            resolve_devices("gpu")


class TestConfiguration:
    def test_check_generation(self, hydra_setup):  # noqa: ARG002
        cfg = _compose()
        generation = generation_settings(cfg, ploidy=2, min_variant_size=50)
        check_generation(cfg, generation)

        with pytest.raises(ValueError, match="pileup.image_width"):
            check_generation(_compose("pileup.image_width=144"), generation)
        with pytest.raises(ValueError, match="simulation.replicates"):
            check_generation(_compose("simulation.replicates=2"), generation)
        with pytest.raises(ValueError, match="pileup.image_channels"):
            check_generation(_compose("pileup=unphased_graph"), generation)
        no_replicates = _compose("simulation.replicates=0")
        with pytest.raises(ValueError, match="no simulated images"):
            check_generation(no_replicates, generation_settings(no_replicates, ploidy=2, min_variant_size=50))

    def test_predict_data_config(self, hydra_setup):  # noqa: ARG002
        cfg = _compose(
            "data.train_metadataset_paths=/other/dataset.yaml",
            "data.filters=[{_target_: npsv3.models.filters.LabelDiffersFilter}]",
            "data.batch_size=512",
        )
        generation = generation_settings(_compose("simulation.replicates=2"), ploidy=2, min_variant_size=50)
        data = predict_data_config(cfg, "/images/dataset.yaml", generation, num_workers=3)
        assert data.train_metadataset_paths == "/images/dataset.yaml"
        assert data.predict_unlabeled
        assert data.predict_split == "train"
        assert data.filters is None
        assert data.genotype_subset is None
        assert data.copositive is None
        assert (data.g_max, data.r_max, data.w_max) == (36, 2, cfg.pileup.image_width)
        assert data.num_workers == 3
        assert data.batch_size == 512, "Other data settings are kept"

    def test_batch_size_must_fit_largest_region(self, genotype_cfg, unlabeled_dataset, tmp_path):
        dataset, _regions = unlabeled_dataset
        cfg = genotype_cfg.copy()
        cfg.data.batch_size = 32
        with pytest.raises(ValueError, match="largest possible region"):
            predict_genotypes(cfg, dataset, tmp_path / PREDICTIONS_DIR)


@pytest.mark.slow
class TestPredictGenotypes:
    def test_cpu(self, genotype_cfg, unlabeled_dataset, tmp_path):
        dataset, regions = unlabeled_dataset
        predictions_dir = tmp_path / PREDICTIONS_DIR
        result = predict_genotypes(genotype_cfg, dataset, predictions_dir)

        assert result.paths == [str(predictions_dir / PREDICTIONS_PATTERN.format(rank=0))]
        frame = read_predictions(predictions_dir)
        _check_partition(frame, result.rejected, regions)
        assert read_rejections(predictions_dir) == result.rejected
        assert all(row["sample"] == SAMPLE for row in result.rejected)

        with open(predictions_dir / PREDICTION_FILE) as file:
            settings = yaml.safe_load(file)
        assert settings["checkpoint"]["path"] == os.path.abspath(genotype_cfg.model.checkpoint)
        assert settings["world_size"] == 1
        assert settings["devices"] == "cpu"

        # Rerunning replaces the predictions, rather than adding to them
        predict_genotypes(genotype_cfg, dataset, predictions_dir)
        pd.testing.assert_frame_equal(
            read_predictions(predictions_dir).drop(columns="metric"), frame.drop(columns="metric")
        )

    def test_ranks_partition_the_regions(self, genotype_cfg, unlabeled_dataset, tmp_path):
        """Two processes' slices (here, run one after the other on the CPU) together predict every region once, with the
        same similarities as a single process"""
        dataset, regions = unlabeled_dataset
        single = predict_genotypes(genotype_cfg, dataset, tmp_path / "single")
        expected = read_predictions(single.predictions_dir)

        from npsv3.genotyping.images import read_generation

        data_cfg = predict_data_config(genotype_cfg, dataset, read_generation(os.path.dirname(dataset)))
        predictions_dir = str(tmp_path / "ranks")
        os.makedirs(predictions_dir)
        paths = [_predict_rank(rank, 2, genotype_cfg, data_cfg, predictions_dir, None, None) for rank in range(2)]
        frames = [pd.read_pickle(path) for path in paths]
        assert all(len(frame) > 0 for frame in frames), "Each rank predicts some of the regions"

        frame = pd.concat(frames, ignore_index=True)
        _check_partition(frame, single.rejected, regions)
        for row in frame.itertuples():
            np.testing.assert_allclose(row.metric, expected.loc[row.key, "metric"], atol=1e-5)

    def test_spawned_ranks(self, genotype_cfg, unlabeled_dataset, tmp_path):
        """The multi-GPU path's `torch.multiprocessing.spawn` entry point, with CPU processes"""
        dataset, regions = unlabeled_dataset
        from npsv3.genotyping.images import read_generation

        data_cfg = predict_data_config(genotype_cfg, dataset, read_generation(os.path.dirname(dataset)), num_workers=0)
        predictions_dir = str(tmp_path / "spawned")
        os.makedirs(predictions_dir)
        # Single-threaded, so the two processes don't oversubscribe the CPUs
        genotype_cfg.threads = 1
        torch.multiprocessing.spawn(
            _spawned_rank, args=(2, genotype_cfg, data_cfg, predictions_dir, None, None), nprocs=2, join=True
        )
        frame = pd.concat(
            [pd.read_pickle(os.path.join(predictions_dir, PREDICTIONS_PATTERN.format(rank=rank))) for rank in range(2)],
            ignore_index=True,
        )
        # The rejections are found by the parent (`predict_genotypes`), not the ranks
        rejected = [{"key": key, "reason": REJECT_TOO_MANY_GENOTYPES} for key, (_region, g) in regions.items() if g > 36]
        _check_partition(frame, rejected, regions)


def test_writer_with_no_regions(tmp_path):
    """A process whose slice has no regions still writes (empty) predictions, so every rank's file exists"""
    callback = PredictionWriterCallback(tmp_path / "predictions-rank0.pkl")
    callback.on_predict_start(None, None)
    callback.on_predict_end(None, None)
    frame = pd.read_pickle(tmp_path / "predictions-rank0.pkl")
    assert len(frame) == 0
    assert list(frame.columns) == ["key", "region", "num_genotypes", "num_replicates", "metric"]
