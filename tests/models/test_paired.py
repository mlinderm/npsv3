import json
import os
import subprocess
import sys
import textwrap

import hydra
import lightning as L
import numpy as np
import pandas as pd
import pytest
import torch

from npsv3.images.example import EnergonExampleWriter
from npsv3.models import runners
from npsv3.models.energon import prepare_energon_dataset
from npsv3.models.paired import RecordLossCallback
from npsv3.util.range import Range

# Pins the Hydra config groups this module's `dm`/`model` are instantiated from (see `_make_dm`/
# `_make_model` below) -- real configs, not hand-built objects, so a change to
# conf/model/paired_packed_inception_infonce.yaml or conf/pileup/unphased_graph_kmer.yaml is caught
# here too.
CFG_OVERRIDES = (
    "model=paired_packed_inception_infonce",
    "pileup=unphased_graph_kmer",
    "simulation.replicates=1",
)

NUM_GENOTYPES = 4  # Small and fixed, well within graph.max_diplotypes, to keep packs tiny


def _write_group(writer, sample_name, chrom, index, rng, *, height, width, channels):
    real = rng.integers(0, 255, size=(height, width, channels), dtype=np.uint8)
    sims = rng.integers(0, 255, size=(1, NUM_GENOTYPES, height, width, channels), dtype=np.uint8)

    label = int(rng.integers(0, NUM_GENOTYPES))
    label_rank = np.zeros(NUM_GENOTYPES, dtype=np.int64)
    label_rank[label] = 1
    if label > 0:
        label_rank[(label_rank == 0) & (np.arange(NUM_GENOTYPES) > 0)] = 3

    region = f"{chrom}:{index * 1000 + 1}-{index * 1000 + 500}"
    writer.write(
        {"region": region, "image": real, "label": label, "label.rank": label_rank, "num_genotypes": NUM_GENOTYPES, "sim.images": sims},
        region=Range(region),
        sample_name=sample_name,
    )


def _build_split(output_dir, chrom, *, num_groups, seed, height, width, channels):
    """Write one energon split (its own imgs_ds/meta_ds tree) and prepare it as a single-split dataset."""
    rng = np.random.default_rng(seed)
    writer = EnergonExampleWriter(0, output_dir)
    try:
        for index in range(num_groups):
            _write_group(writer, "S0", chrom, index, rng, height=height, width=width, channels=channels)
    finally:
        writer.close()
    return prepare_energon_dataset(output_dir)


@pytest.fixture(scope="module")
def base_cfg(hydra_setup):  # noqa: ARG001
    """The config this module's synthetic data, and `dm`/`model`, are built against -- see `CFG_OVERRIDES`."""
    return hydra.compose(config_name="config", overrides=list(CFG_OVERRIDES))


@pytest.fixture(scope="module")
def paired_energon_datasets(base_cfg, tmp_path_factory):
    """Independently-prepared train/val/test `prepare_energon_dataset` trees, at the geometry
    `base_cfg.pileup` defines -- `EnergonPackedImageDataModule` combines these itself, in memory."""
    root = tmp_path_factory.mktemp("paired_energon")
    geometry = {
        "height": base_cfg.pileup.image_height,
        "width": base_cfg.pileup.image_width,
        "channels": len(base_cfg.pileup.image_channels),
    }
    return {
        "train": _build_split(str(root / "train"), "chr1", num_groups=40, seed=20260917, **geometry),
        "val": _build_split(str(root / "val"), "chr2", num_groups=20, seed=20260918, **geometry),
        "test": _build_split(str(root / "test"), "chr3", num_groups=20, seed=20260919, **geometry),
    }


# Real `data=energon_packed_images` defaults pack thousands of examples per virtual epoch; these keep a
# test batch small and finite.
_DATA_TEST_OVERRIDES = {
    "batch_size": 32,
    "g_max": NUM_GENOTYPES,
    "packing_buffer_size": 16,
    "shuffle_buffer_size": 32,
    "num_workers": 0,
    "cache_workers": 1,
    "max_cache_size_gbytes": 1,
    "limit_val_batches": 2,
    "limit_test_batches": 2,
}


def _data_overrides(datasets, **overrides):
    """Hydra CLI-style overrides pointing the real `data=energon_packed_images` config at `datasets`,
    plus `_DATA_TEST_OVERRIDES` (or `overrides`, in its place) to keep it small enough for a test."""
    options = {**_DATA_TEST_OVERRIDES, **overrides}
    return [
        f"data.train_metadataset_paths={datasets['train']}",
        f"data.val_metadataset_paths={datasets['val']}",
        f"data.test_metadataset_paths={datasets['test']}",
        *(f"data.{key}={value}" for key, value in options.items()),
    ]


def _standalone_compose(*overrides):
    """`hydra.compose` for a bare, non-pytest process (the DDP smoke test's subprocess) -- pytest's
    session-scoped `hydra_setup` fixture doesn't exist there, nor does the `tests/__init__.py` import
    that normally registers OmegaConf's custom resolvers (e.g. `${len:...}`, used by
    `model.encoder.num_channels`), since the subprocess imports this module directly rather than as
    part of the `tests` package -- so both are set up here instead.
    """
    from npsv3.util.config import setup_resolvers

    setup_resolvers()
    hydra.initialize(config_path="../../src/npsv3/conf", version_base=None)
    return hydra.compose(config_name="config", overrides=[*CFG_OVERRIDES, *overrides])


@pytest.mark.slow
@pytest.mark.cfg_overrides(*CFG_OVERRIDES)
class TestPackedVariantWithEnergonLoader:
    """End-to-end coverage of the InceptionEncoder-based Paired model against the energon loader's
    packed batches, instantiated from the real `model`/`data` Hydra configs -- the loader-only tests in
    test_energon_loaders.py never construct a model, and the model is otherwise only ever exercised
    through the full CLI/Hydra `train`/`test` commands."""

    def test_fit_validate_test(self, request, paired_energon_datasets):
        marker = request.node.get_closest_marker("cfg_overrides")
        local_cfg = hydra.compose(
            config_name="config",
            overrides=[*(marker.args if marker else []), *_data_overrides(paired_energon_datasets)],
        )
        dm = hydra.utils.instantiate(local_cfg.data)
        model = hydra.utils.instantiate(local_cfg.model)

        stem_weight_before = model.encoder.inception.Conv2d_1a_3x3.conv.weight.detach().clone()

        trainer = L.Trainer(
            accelerator="cpu",
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=2,
            num_sanity_val_steps=0,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, datamodule=dm)

        train_loss = trainer.callback_metrics["train_loss"]
        assert torch.isfinite(train_loss)
        stem_weight_after = model.encoder.inception.Conv2d_1a_3x3.conv.weight.detach()
        assert not torch.allclose(stem_weight_before, stem_weight_after), "A training step must update encoder weights"

        (val_results,) = trainer.validate(model, datamodule=dm)
        assert torch.isfinite(torch.tensor(val_results["val_loss"]))
        assert 0.0 <= val_results["val_concordance"] <= 1.0

        test_trainer = L.Trainer(
            accelerator="cpu",
            limit_test_batches=2,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        (test_results,) = test_trainer.test(model, datamodule=dm)
        assert torch.isfinite(torch.tensor(test_results["test_loss"]))
        assert 0.0 <= test_results["test_nonrefprecision"] <= 1.0
        assert 0.0 <= test_results["test_nonreff1"] <= 1.0

        dm.teardown()

    def test_train_without_val_split(self, paired_energon_datasets, tmp_path):
        """`runners.train` must skip validation entirely -- not just default to an empty run -- when the
        combined `data.metadataset_path` (as `prepare_energon_dataset` always writes it) defines only a
        "train" split. Regression test for a `KeyError: 'val'` out of
        `EnergonPackedImageDataModule.val_dataloader` when Lightning's sanity-check/validation loop
        requested a dataloader for a split the metadataset never defined (see `runners.train`'s
        `has_val_metadataset` gating, which must consult `available_splits` rather than assume every
        `metadataset_path` carries a "val" split)."""
        overrides = [
            f"data.metadataset_path={paired_energon_datasets['train']}",
            *(f"data.{key}={value}" for key, value in {
                "batch_size": 32,
                "g_max": NUM_GENOTYPES,
                "packing_buffer_size": 16,
                "shuffle_buffer_size": 32,
                "num_workers": 0,
                "cache_workers": 1,
                "max_cache_size_gbytes": 1,
                "limit_train_batches": 2,
            }.items()),
            "+trainer.accelerator=cpu",
            "trainer.max_epochs=1",
        ]
        cfg = hydra.compose(config_name="config", overrides=[*CFG_OVERRIDES, *overrides])

        checkpoint_path = runners.train(cfg, output_dir=str(tmp_path))

        assert checkpoint_path is not None and os.path.exists(checkpoint_path)

    def test_loss_records_region_losses(self, paired_energon_datasets, tmp_path):
        """`runners.predict` (the `loss` command) must run over the energon loader and record every region's loss.
        Regression test for Lightning's `MisconfigurationException` when it tried to re-instantiate energon's
        `SavableDataLoader` to inject a predict batch sampler."""
        train_overrides = [
            f"data.metadataset_path={paired_energon_datasets['train']}",
            *(f"data.{key}={value}" for key, value in {**_DATA_TEST_OVERRIDES, "limit_train_batches": 1}.items()),
            "+trainer.accelerator=cpu",
            "trainer.max_epochs=1",
        ]
        train_cfg = hydra.compose(config_name="config", overrides=[*CFG_OVERRIDES, *train_overrides])
        checkpoint_path = runners.train(train_cfg, output_dir=str(tmp_path))

        predict_cfg = hydra.compose(
            config_name="config",
            overrides=[
                *CFG_OVERRIDES,
                *train_overrides,
                "command=loss",
                f"model.checkpoint='{checkpoint_path}'",
            ],
        )
        losses_path = tmp_path / "losses.pkl.gz"
        runners.predict(predict_cfg, return_predictions=False, callbacks=[RecordLossCallback(str(losses_path))])

        df = pd.read_pickle(losses_path)
        assert len(df) > 0
        assert df["region"].str.startswith("chr1:").all(), "Rows are the train split's regions"
        assert np.isfinite(df["loss"]).all()
        assert (df["correct"] == (df["genotype"] == df["prediction"])).all()
        for row in df.itertuples():
            assert row.label_rank.shape == (row.num_genotypes,)
            assert row.metric.shape == (row.num_replicates, row.num_genotypes)
            np.testing.assert_allclose(row.genotype_metric, row.metric.mean(axis=0))
            assert row.label_metric == pytest.approx(row.genotype_metric[row.genotype])
            assert row.prediction_metric == pytest.approx(row.genotype_metric[row.prediction])

    def test_ddp_two_rank_training(self, paired_energon_datasets, tmp_path):
        """Two ranks train without hanging and finish with identical weights, confirming DDP's
        gradient all-reduce actually reaches the encoder through the packed/nested-tensor loss path
        (see test_energon_loaders.TestEnergonPackedImageDataModule.test_two_rank_ddp for the
        loader-only version of this check, which uses a trivial loss instead of the real model)."""
        data_overrides = _data_overrides(paired_energon_datasets, num_workers=1, limit_val_batches=0, limit_test_batches=0)
        script = textwrap.dedent(
            f"""
            import hydra, json, os, sys, torch, lightning as L
            sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})
            from models.test_paired import _standalone_compose

            cfg = _standalone_compose(*{data_overrides!r})
            dm = hydra.utils.instantiate(cfg.data)
            model = hydra.utils.instantiate(cfg.model)
            before = model.encoder.inception.Conv2d_1a_3x3.conv.weight.detach().clone()

            trainer = L.Trainer(
                accelerator="cpu", devices=2, strategy="ddp", max_epochs=1,
                limit_train_batches=3, limit_val_batches=0, num_sanity_val_steps=0,
                enable_checkpointing=False, logger=False, enable_progress_bar=False,
            )
            trainer.fit(model, datamodule=dm)

            after = model.encoder.inception.Conv2d_1a_3x3.conv.weight.detach()
            path = os.path.join({str(tmp_path)!r}, f"rank{{trainer.global_rank}}.json")
            with open(path, "w") as f:
                json.dump({{
                    "loss": trainer.callback_metrics["train_loss"].item(),
                    "weight_changed": not torch.allclose(before, after),
                    "weight_sample": after.flatten()[:16].tolist(),
                }}, f)
            """
        )
        script_path = tmp_path / "ddp_paired_smoke.py"
        script_path.write_text(script)

        subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ},
        )

        results = []
        for rank in (0, 1):
            with open(tmp_path / f"rank{rank}.json") as rank_file:
                results.append(json.load(rank_file))

        for result in results:
            assert result["weight_changed"], "DDP training must update encoder weights"
            assert np.isfinite(result["loss"])

        # DDP all-reduces gradients every step, so both ranks' replicas must land on the same weights
        # despite each rank only ever seeing its own (disjoint) shard of the packed batches.
        assert results[0]["weight_sample"] == pytest.approx(results[1]["weight_sample"]), (
            "Both DDP ranks must converge on the same weights after gradient sync"
        )
