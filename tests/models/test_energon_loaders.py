import copy
import hashlib
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch
import yaml

from npsv3.images.example import EnergonExampleWriter
from npsv3.models.energon import (
    EnergonMaskedImageDataModule,
    EnergonPackedImageDataModule,
    GroupMeta,
    PackedGroupTaskEncoder,
    _group_accepted,
    _group_cost,
    estimate_epoch_batches,
    prepare_energon_dataset,
)
from npsv3.models.loaders import select_packs
from npsv3.util.range import Range

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 8, 12, 3


def _write_group(writer, sample_name, index, num_replicates, num_genotypes, rng):
    """Write one synthetic group whose pixels are a deterministic function of (index, r, g)"""
    real = rng.integers(0, 255, size=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)
    sims = np.empty((num_replicates, num_genotypes, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)
    for r in range(num_replicates):
        for g in range(num_genotypes):
            sims[r, g] = (index + r * num_genotypes + g) % 251

    label = int(rng.integers(0, num_genotypes))
    label_rank = np.zeros(num_genotypes, dtype=np.int64)
    label_rank[label] = 1
    if label > 0:
        label_rank[(label_rank == 0) & (np.arange(num_genotypes) > 0)] = 3

    writer.write(
        {
            "region": f"chr1:{index * 1000 + 1}-{index * 1000 + 500}",
            "image": real,
            "label": label,
            "label.rank": label_rank,
            "sim.images": sims,
            "num_genotypes": num_genotypes,
        },
        region=Range(f"chr1:{index * 1000 + 1}-{index * 1000 + 500}"),
        sample_name=sample_name,
    )


def _build_dataset(output_dir, *, num_groups=2000, num_shards=4, seed=20260911):
    """Write the synthetic trees with the same structure as the example generator"""
    rng = np.random.default_rng(seed)
    writers = [EnergonExampleWriter(shard, output_dir) for shard in range(num_shards)]
    try:
        for index in range(num_groups):
            # G in [3, 36] is the range h(h+1)/2 spans for h in [2, 8] sampled haplotypes.
            num_genotypes = int(rng.integers(3, 37))
            num_replicates = 2 if index % 7 == 0 else 1
            while (1 + num_replicates * num_genotypes) > 256:
                num_genotypes //= 2
            _write_group(writers[index % num_shards], f"S{index % num_shards}", index, num_replicates, num_genotypes, rng)
    finally:
        for writer in writers:
            writer.close()

    return prepare_energon_dataset(output_dir)


@pytest.fixture(scope="module")
def energon_dataset(tmp_path_factory):
    return _build_dataset(str(tmp_path_factory.mktemp("energon")))


def _make_dm(metadataset_path, **kwargs):
    options = {
        "train_metadataset_paths": metadataset_path,
        "batch_size": 256,
        "w_bucket": IMAGE_WIDTH,
        "w_max": IMAGE_WIDTH,
        "g_max": 36,
        "r_max": 2,
        "packing_buffer_size": 128,
        "shuffle_buffer_size": 256,
        "num_workers": 0,
        "cache_workers": 2,
        "max_cache_size_gbytes": 1,
        "limit_val_batches": 4,
    }
    options.update(kwargs)
    return EnergonPackedImageDataModule(**options)


def batch_hash(batch) -> str:
    """Hash everything that defines a batch, so a reordering or relabeling shows up."""
    digest = hashlib.sha256()
    digest.update(batch.images.numpy().tobytes())
    for field in (batch.labels, batch.offsets, batch.widths, batch.genotype_label, batch.num_genotypes):
        digest.update(field.numpy().tobytes())
    digest.update("|".join(map(str, batch.meta)).encode())
    return digest.hexdigest()


class TestTaskEncoder:
    def _meta(self, **kwargs):
        options = {
            "__key__": "k",
            "__restore_key__": (),
            "r": 1,
            "g": 4,
            "h": IMAGE_HEIGHT,
            "w": IMAGE_WIDTH,
            "c": IMAGE_CHANNELS,
            "label": 1,
            "label_rank": [0, 1, 3, 3],
            "region": "chr1:1-100",
            "real": None,
        }
        options.update(kwargs)
        return GroupMeta(**options)

    def test_cost_and_rejects(self):
        encoder = PackedGroupTaskEncoder(budget=64, w_bucket=IMAGE_WIDTH, w_max=IMAGE_WIDTH, g_max=36, r_max=2)

        assert encoder.preencode_sample(self._meta(g=4, r=1)).cost == 1 + 4

        from megatron.energon import SkipSample

        for rejected in (
            self._meta(g=40),
            self._meta(r=3),
            self._meta(w=IMAGE_WIDTH + 1),
            self._meta(g=0),
            self._meta(label=None, label_rank=None),
        ):
            with pytest.raises(SkipSample):
                encoder.preencode_sample(rejected)

        # A group whose own cost exceeds the budget is rejected rather than given an oversized bin
        with pytest.raises(SkipSample):
            encoder.preencode_sample(self._meta(g=36, r=2))

    def test_select_samples_to_pack_respects_budget(self):
        encoder = PackedGroupTaskEncoder(budget=64, w_bucket=IMAGE_WIDTH)
        samples = [encoder.preencode_sample(self._meta(g=g, label=0, label_rank=[1] + [0] * (g - 1))) for g in range(3, 20)]

        packs = encoder.select_samples_to_pack(samples)

        assert sorted(sample.region for pack in packs for sample in pack) == sorted(s.region for s in samples)
        for pack in packs:
            assert sum(sample.cost for sample in pack) <= 64


def test_prepare_skips_empty_shards(tmp_path):
    """An actor that wrote no examples (e.g., more actors than graph shards) leaves empty shards that are dropped."""
    output_dir = str(tmp_path)
    rng = np.random.default_rng(0)
    writers = [EnergonExampleWriter(shard, output_dir) for shard in range(3)]
    try:
        for index in range(10):
            _write_group(writers[index % 2], "S0", index, 1, 3, rng)  # Writer 2 never receives an example
    finally:
        for writer in writers:
            writer.close()

    prepare_energon_dataset(output_dir)

    imgs_dir = tmp_path / "imgs_ds"
    assert sorted(p.name for p in imgs_dir.glob("*.tar")) == ["imgs-0000-000000.tar", "imgs-0001-000000.tar"]
    with open(tmp_path / "meta_ds" / "train.jsonl") as meta_file:
        assert sum(1 for _ in meta_file) == 10


@pytest.mark.slow
class TestEnergonPackedImageDataModule:
    def test_round_trip(self, energon_dataset):
        """Pixels read back through the loader match what the writer put in."""
        dm = _make_dm(energon_dataset)
        loader = dm.train_dataloader()

        seen = 0
        for batch in loader:
            offsets = batch.offsets
            for group in range(batch.num_groups):
                num_genotypes = int(batch.num_genotypes[group])
                start = int(offsets[group])
                num_replicates = int(offsets[group + 1] - offsets[group]) // num_genotypes

                # Pixels were written as (index + r*G + g) % 251, constant across the image
                base = int(batch.images[start, 0, 0, 0])
                for r in range(num_replicates):
                    for g in range(num_genotypes):
                        expected = (base + r * num_genotypes + g) % 251
                        actual = int(batch.images[start + r * num_genotypes + g, 0, 0, 0])
                        assert actual == expected, f"Support image {r},{g} is out of replicates-major order"
                seen += 1
            if seen >= 200:
                break

        assert seen >= 200
        dm.teardown()

    def test_occupancy(self, energon_dataset):
        dm = _make_dm(energon_dataset)
        loader = dm.train_dataloader()

        used = batches = 0
        for i, batch in enumerate(loader):
            used += batch.num_images
            batches += 1
            if i >= 49:
                break

        occupancy = used / (batches * 256)
        assert occupancy > 0.90, f"Mean pack occupancy {occupancy:.3f} is below the 0.90 sanity threshold"
        dm.teardown()

    def test_estimate_epoch_batches(self, energon_dataset):
        """The estimate should land close to the true one-pass pack count, without touching pixels."""
        budget, w_bucket, w_max, g_max, r_max = 256, IMAGE_WIDTH, IMAGE_WIDTH, 36, 2
        output_dir = os.path.dirname(energon_dataset)

        with open(energon_dataset) as metadataset_file:
            meta_path = yaml.safe_load(metadataset_file)["splits"]["train"]["path"]

        costs = []
        with open(os.path.join(output_dir, meta_path)) as jsonl_file:
            for line in jsonl_file:
                row = json.loads(line)
                r, g, w = int(row["r"]), int(row["g"]), int(row["w"])
                if _group_accepted(r, g, w, label=row.get("label"), g_max=g_max, r_max=r_max, w_max=w_max):
                    costs.append((_group_cost(r, g, w, w_bucket), w // w_bucket))

        true_packs = len(select_packs(costs, budget=budget))

        estimate = estimate_epoch_batches(
            energon_dataset, budget=budget, w_bucket=w_bucket, w_max=w_max, g_max=g_max, r_max=r_max
        )
        assert true_packs * 0.5 <= estimate <= true_packs * 1.5, (
            f"Estimate {estimate} too far from the true one-pass count {true_packs}"
        )

        # Splitting across ranks should roughly divide the estimate, and never go below 1
        halved = estimate_epoch_batches(
            energon_dataset, budget=budget, w_bucket=w_bucket, w_max=w_max, g_max=g_max, r_max=r_max, world_size=2
        )
        assert 1 <= halved <= estimate

    def test_uint8_and_gpu_normalization(self, energon_dataset):
        """Images leave the loader as uint8; normalization happens after transfer."""
        dm = _make_dm(energon_dataset)
        batch = next(iter(dm.train_dataloader()))

        assert batch.images.dtype == torch.uint8
        assert batch.images.shape == (256, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert torch.all(batch.widths == IMAGE_WIDTH), "Fixed geometry means a single width bucket"

        normalized = dm.on_after_batch_transfer(batch, 0)
        assert normalized.images.dtype == torch.float32
        dm.teardown()

    @pytest.mark.parametrize(
        ("num_workers", "with_val", "expected"),
        [
            # Without workers the main process uses the pool alone
            (0, False, {"train": 8.0, "test": 8.0}),
            (4, False, {"train": 2.0, "test": 2.0}),
            # The persistent train and val workers are alive together during `fit`, test runs alone
            (4, True, {"train": 1.0, "val": 1.0, "test": 2.0}),
        ],
    )
    def test_cache_pool_cap_is_per_rank(self, energon_dataset, num_workers, with_val, expected):
        """Energon's pool counts its size per (forked) worker process, so the per-rank cap is split across them"""
        dm = _make_dm(
            energon_dataset,
            num_workers=num_workers,
            max_cache_size_gbytes=8,
            val_metadataset_paths=energon_dataset if with_val else None,
        )
        for split, gbytes in expected.items():
            assert dm._cache_pool_gbytes(split) == pytest.approx(gbytes)
        assert dm._make_cache_pool("train").max_cache_size == int(expected["train"] * 1024**3)
        dm.teardown()

    def test_watchdog_settings_reach_loader(self, energon_dataset, monkeypatch):
        captured = {}

        def fake_get_savable_loader(dataset, **kwargs):  # noqa: ARG001
            captured.update(kwargs)
            return []

        monkeypatch.setattr("npsv3.models.energon.get_savable_loader", fake_get_savable_loader)
        dm = _make_dm(energon_dataset, watchdog_timeout_seconds=30, watchdog_initial_timeout_seconds=300)
        dm.train_dataloader()
        assert captured["watchdog_timeout_seconds"] == 30
        assert captured["watchdog_initial_timeout_seconds"] == 300
        dm.teardown()

    def test_positives_survive_packing(self, energon_dataset):
        """Every replicate of every labeled group lands on a rank-1 entry."""
        dm = _make_dm(energon_dataset)

        for i, batch in enumerate(dm.train_dataloader()):
            positives = batch.positive_indices()
            assert torch.all(batch.labels[positives] == 1)

            lengths = torch.diff(batch.offsets)
            replicates = lengths // batch.num_genotypes
            assert positives.numel() == int(replicates.sum()), "R positives per group, not one"
            if i >= 4:
                break
        dm.teardown()

    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_determinism_and_resume(self, energon_dataset, tmp_path, num_workers):
        """Save at step 20, restore in a NEW process, and reproduce hashes 20..40 exactly.

        A failure here almost always means something upstream is not actually `@stateless` --
        untracked RNG, dict iteration order, a mutable default, or a value cached on the task
        encoder. The seed is set explicitly rather than inherited, since `pytest-randomly` reseeds
        per test and this must not depend on execution order.
        """
        script = textwrap.dedent(
            f"""
            import hashlib, json, sys, torch
            torch.manual_seed(20260911)
            sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})
            from models.test_energon_loaders import _make_dm, batch_hash

            mode, state_path = sys.argv[1], sys.argv[2]
            dm = _make_dm({energon_dataset!r}, num_workers={num_workers})
            loader = dm.train_dataloader()

            hashes = []
            if mode == "record":
                it = iter(loader)
                for step in range(40):
                    hashes.append(batch_hash(next(it)))
                    if step == 19:
                        torch.save(dm.state_dict(), state_path)
            else:
                dm.load_state_dict(torch.load(state_path, weights_only=False))
                it = iter(loader)
                for _ in range(20):
                    hashes.append(batch_hash(next(it)))
            print(json.dumps(hashes))
            dm.teardown()
            """
        )
        script_path = tmp_path / "replay.py"
        script_path.write_text(script)
        state_path = str(tmp_path / "loader_state.pt")

        def run(mode):
            result = subprocess.run(
                [sys.executable, str(script_path), mode, state_path],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ},
            )
            return json.loads(result.stdout.strip().splitlines()[-1])

        recorded = run("record")
        assert len(set(recorded)) == len(recorded), "Distinct steps must produce distinct batches"

        restored = run("restore")
        assert restored == recorded[20:40], "Restored loader must reproduce the exact batch sequence"

    def test_two_rank_ddp(self, energon_dataset, tmp_path):
        """Two ranks cross an epoch boundary without hanging, in lockstep, on disjoint regions.

        This is the motivating failure: `wds.split_by_node` gives each rank a shard slice, greedy
        packing then makes the per-rank batch count data-dependent, and the ranks reach the
        all-reduce an unequal number of times. `limit_train_batches` over an indefinitely-repeating
        dataset removes the data dependence.
        """
        script = textwrap.dedent(
            f"""
            import json, os, sys, torch, lightning as L
            sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})
            from models.test_energon_loaders import _make_dm

            class KeyRecorder(L.LightningModule):
                def __init__(self):
                    super().__init__()
                    self.scale = torch.nn.Parameter(torch.ones(1))
                    self.steps = []

                def training_step(self, batch, batch_idx):
                    self.steps.append(list(batch.meta))
                    # A real (if trivial) loss, so DDP actually all-reduces gradients each step
                    return (self.scale * batch.images.float().mean()).squeeze()

                def configure_optimizers(self):
                    return torch.optim.SGD(self.parameters(), lr=0.0)

                def on_fit_end(self):
                    path = os.path.join({str(tmp_path)!r}, f"rank{{self.trainer.global_rank}}.json")
                    with open(path, "w") as f:
                        json.dump(self.steps, f)

            model = KeyRecorder()
            dm = _make_dm({energon_dataset!r}, num_workers=1, limit_val_batches=0)
            trainer = L.Trainer(
                accelerator="cpu", devices=2, strategy="ddp", max_epochs=2,
                limit_train_batches=100, limit_val_batches=0, num_sanity_val_steps=0,
                enable_checkpointing=False, logger=False, enable_progress_bar=False,
            )
            trainer.fit(model, datamodule=dm)
            """
        )
        script_path = tmp_path / "ddp_smoke.py"
        script_path.write_text(script)

        subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=1200,
            env={**os.environ},
        )

        steps = []
        for rank in (0, 1):
            with open(tmp_path / f"rank{rank}.json") as rank_file:
                steps.append(json.load(rank_file))

        assert len(steps[0]) == len(steps[1]) == 200, "Both ranks must take the same number of steps"
        for step, (rank0, rank1) in enumerate(zip(steps[0], steps[1], strict=True)):
            assert not set(rank0) & set(rank1), f"Step {step} gave the same region to both ranks"


def _combine_as_train_and_val(metadataset_path: str, tmp_path) -> str:
    """Build a pre-combined, multi-split `dataset.yaml` (see `EnergonPackedImageDataModule`'s
    `metadataset_path` argument) out of one `prepare_energon_dataset` tree, reused for both the "train"
    and "val" splits -- enough to check that a pre-combined file, not just per-split lists, is read
    correctly.
    """
    root = os.path.dirname(metadataset_path)
    with open(metadataset_path) as metadataset_file:
        single = yaml.safe_load(metadataset_file)
    (leaf,) = single["splits"].values()
    leaf_abs = {
        "path": os.path.join(root, leaf["path"]),
        "aux": {name: os.path.join(root, aux_path) for name, aux_path in leaf["aux"].items()},
    }
    combined = {
        "__module__": "megatron.energon",
        "__class__": "MetadatasetV2",
        # Deep copies, not the same dict twice: PyYAML would otherwise emit a YAML anchor/alias for the
        # repeated object, which energon's own loader does not resolve.
        "splits": {"train": leaf_abs, "val": copy.deepcopy(leaf_abs)},
    }
    combined_path = os.path.join(str(tmp_path), "combined_dataset.yaml")
    with open(combined_path, "w") as combined_file:
        yaml.safe_dump(combined, combined_file, sort_keys=False)
    return combined_path


class TestMetadatasetPathModes:
    """`EnergonPackedImageDataModule` accepts either a single, pre-combined `metadataset_path` (its own
    real train/val/test splits) or per-split `*_metadataset_paths` lists that it combines itself (see
    `_build_metadataset`) -- these check the former mode and the guard between the two."""

    def test_combined_metadataset_path_serves_train_and_val(self, energon_dataset, tmp_path):
        combined_path = _combine_as_train_and_val(energon_dataset, tmp_path)

        dm = EnergonPackedImageDataModule(
            metadataset_path=combined_path,
            batch_size=256,
            w_bucket=IMAGE_WIDTH,
            w_max=IMAGE_WIDTH,
            g_max=36,
            r_max=2,
            packing_buffer_size=128,
            shuffle_buffer_size=256,
            num_workers=0,
            cache_workers=2,
            max_cache_size_gbytes=1,
            limit_val_batches=2,
        )

        train_batch = next(iter(dm.train_dataloader()))
        val_batch = next(iter(dm.val_dataloader()))
        assert train_batch.images.shape == val_batch.images.shape == (256, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
        dm.teardown()

    def test_estimate_epoch_batches_reads_named_split_from_combined_metadataset(self, energon_dataset, tmp_path):
        """Unlike a plain `prepare_energon_dataset` tree (a single split, any name), a pre-combined file
        has more than one real split, so `split_part` must actually select by name."""
        combined_path = _combine_as_train_and_val(energon_dataset, tmp_path)

        train_estimate = estimate_epoch_batches(combined_path, budget=256, split_part="train")
        val_estimate = estimate_epoch_batches(combined_path, budget=256, split_part="val")

        assert train_estimate == val_estimate > 0, "Both splits point at the same leaf tree here"

    def test_metadataset_path_and_train_metadataset_paths_are_mutually_exclusive(self, energon_dataset):
        with pytest.raises(AssertionError, match="not both"):
            EnergonPackedImageDataModule(
                metadataset_path=energon_dataset,
                train_metadataset_paths=energon_dataset,
                batch_size=32,
            )

    def test_requires_one_of_metadataset_path_or_train_metadataset_paths(self):
        with pytest.raises(AssertionError, match="Must pass either"):
            EnergonPackedImageDataModule(batch_size=32)


def _make_masked_dm(metadataset_path, **kwargs):
    options = {
        "train_metadataset_paths": metadataset_path,
        "batch_size": 256,
        "w_bucket": IMAGE_WIDTH,
        "w_max": IMAGE_WIDTH,
        "g_max": 36,
        "r_max": 2,
        "packing_buffer_size": 128,
        "shuffle_buffer_size": 256,
        "num_workers": 0,
        "cache_workers": 2,
        "max_cache_size_gbytes": 1,
        "limit_val_batches": 4,
        "patch_size": 4,
        "mask_fraction": 0.5,
    }
    options.update(kwargs)
    return EnergonMaskedImageDataModule(**options)


@pytest.mark.slow
class TestEnergonMaskedImageDataModule:
    """Masked pretraining reuses `EnergonPackedImageDataModule`'s packing pipeline unchanged; only the
    `(pixel_values, bool_masked_pos)` reshaping in `on_after_batch_transfer` needs checking here."""

    def test_masked_batch_shape_and_mask_fraction(self, energon_dataset):
        dm = _make_masked_dm(energon_dataset)
        batch = next(iter(dm.train_dataloader()))

        images, mask = dm.on_after_batch_transfer(batch, 0)

        assert images.dtype == torch.float32, "Images are normalized to float by on_after_batch_transfer"
        b, c, h, w = images.shape
        assert (c, h, w) == (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)

        num_patches = (h // 4) * (w // 4)
        assert mask.shape == (b, num_patches)
        assert mask.dtype == torch.bool
        assert torch.all(mask.sum(dim=1) == int(0.5 * num_patches)), "Each image has exactly mask_fraction patches masked"
        dm.teardown()

    def test_masks_differ_per_image(self, energon_dataset):
        """The mask is drawn independently per image, not shared across the batch."""
        dm = _make_masked_dm(energon_dataset)
        batch = next(iter(dm.train_dataloader()))

        _images, mask = dm.on_after_batch_transfer(batch, 0)

        assert mask.shape[0] > 1
        assert not torch.all(mask == mask[0]), "Masks should vary across images in the batch"
        dm.teardown()

    def test_predict_dataloader_stands_in_on_train(self, energon_dataset):
        """Predict defaults to a single pass over the train split (see `predict_dataloader`)."""
        dm = _make_masked_dm(energon_dataset)

        batch = next(iter(dm.predict_dataloader()))
        images, mask = dm.on_after_batch_transfer(batch, 0)

        assert images.shape[1:] == (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert mask.dtype == torch.bool
        dm.teardown()
