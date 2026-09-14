"""Tests for the energon polylithic packing loader.

The fixture here is synthetic: groups are written through the Milestone 2 writer and prepared with
`energon prepare`, so nothing depends on BAM/simulation inputs or on /storage. That keeps the
highest-value test -- determinism across a process boundary -- cheap enough to run routinely.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from npsv3.images.example import EnergonExampleWriter
from npsv3.models.energon import (
    EnergonPackedImageDataModule,
    GroupMeta,
    PackedGroupTaskEncoder,
)
from npsv3.util.range import Range

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 8, 12, 3

METADATASET_YAML = """\
__module__: megatron.energon
__class__: MetadatasetV2
splits:
  train:
    path: ./meta_ds
    split_part: train
    aux:
      px: ./pixel_ds
  val:
    path: ./meta_ds
    split_part: val
    aux:
      px: ./pixel_ds
"""


def _write_group(writer, sample_name, index, num_replicates, num_genotypes, rng):
    """Write one synthetic group whose pixels are a deterministic function of (index, r, g)."""
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
        },
        region=Range(f"chr1:{index * 1000 + 1}-{index * 1000 + 500}"),
        sample_name=sample_name,
    )


def _build_dataset(output_dir, *, num_groups=2000, num_shards=4, seed=20260911):
    """Write the synthetic trees and run `energon prepare` on both."""
    rng = np.random.default_rng(seed)
    writers = [EnergonExampleWriter(shard, output_dir) for shard in range(num_shards)]
    try:
        for index in range(num_groups):
            # G in [3, 36] is the range h(h+1)/2 spans for h in [2, 8] sampled haplotypes. Include
            # some R=2 groups, which the generator cannot yet emit, so the tile-vs-interleave
            # expansion and the multi-positive path are exercised end to end.
            num_genotypes = int(rng.integers(3, 37))
            num_replicates = 2 if index % 7 == 0 else 1
            while (1 + num_replicates * num_genotypes) > 256:
                num_genotypes //= 2
            _write_group(writers[index % num_shards], f"S{index % num_shards}", index, num_replicates, num_genotypes, rng)
    finally:
        for writer in writers:
            writer.close()

    for tree in ("meta_ds", "pixel_ds"):
        subprocess.run(
            [
                "energon", "prepare", os.path.join(output_dir, tree),
                "--non-interactive", "--split-ratio", "0.75,0.25,0.0",
                "--sample-type", "CrudeWebdataset", "--num-workers", "2",
                "--force-overwrite", "--no-progress",
            ],
            check=True,
            capture_output=True,
        )  # fmt: skip

    metadataset_path = os.path.join(output_dir, "metadataset.yaml")
    with open(metadataset_path, "w") as metadataset_file:
        metadataset_file.write(METADATASET_YAML)
    return metadataset_path


@pytest.fixture(scope="module")
def energon_dataset(tmp_path_factory):
    if shutil.which("energon") is None:
        pytest.skip("energon CLI not available")
    return _build_dataset(str(tmp_path_factory.mktemp("energon")))


def _make_dm(metadataset_path, **kwargs):
    options = {
        "metadataset_path": metadataset_path,
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

        for rejected in (self._meta(g=40), self._meta(r=3), self._meta(w=IMAGE_WIDTH + 1), self._meta(g=0)):
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

    def test_genotype_subsampling_keeps_the_positive(self, energon_dataset):
        """`g_train` thins the G axis but must never drop the group's positive.

        Subsampling selects a subset of *genotype* indices; drop the positive and the group has
        nothing for the loss to pull towards, so the objective is undefined for it. The retained set
        also always includes genotype 0, the homozygous-reference candidate that anchors the
        third-rank rule.
        """
        dm = _make_dm(energon_dataset, g_train=3)

        for i, batch in enumerate(dm.train_dataloader()):
            assert torch.all(batch.num_genotypes == 3), "Every group is thinned to g_train genotypes"

            positives = batch.positive_indices()
            assert torch.all(batch.labels[positives] == 1), "The positive survives subsampling and stays rank 1"

            lengths = torch.diff(batch.offsets)
            assert torch.all(lengths % batch.num_genotypes == 0), "Replicates are never thinned"
            if i >= 4:
                break
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
