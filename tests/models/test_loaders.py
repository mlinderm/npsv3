import os
import random

import pytest
import torch
import webdataset as wds

from npsv3.images.example import (
    example_to_image,
)
from npsv3.models.loaders import (
    ImageGroup,
    MaskedImageDataModule,
    PackedImageDataModule,
    build_packed_batch,
    select_packs,
)

from .. import EXPERIMENTS_DIR, _first_existing, data_path, result_path

# A representative shard of the existing genotypes-major corpus (R=1), for which the legacy transpose
# is a no-op in flattened order and so must reproduce today's output exactly.
LEGACY_SHARD = _first_existing(
    os.path.join(
        EXPERIMENTS_DIR,
        "training/hgsvc3-hprc-2024-02-23-mc-chm13.GRCh38.vcfbub.a100k.wave.passing.training.hg38.images2",
        "HG01361/generator=coverage,pileup=unphased,simulation.replicates=1/images-0000.tar",
    )
)


def make_group(num_replicates, num_genotypes, *, genotype_label=0, width=8, height=4, channels=2, fill=0, meta=None):
    """Build an ImageGroup whose pixels encode (replicate, genotype) so packing order is checkable."""
    query = torch.full((channels, height, width), fill, dtype=torch.uint8)
    support = torch.empty((num_replicates, num_genotypes, channels, height, width), dtype=torch.uint8)
    for r in range(num_replicates):
        for g in range(num_genotypes):
            support[r, g] = 1 + r * num_genotypes + g

    # The same ranking rule as the generator: rank 1 at the true genotype, rank 3 on the other non-ref genotypes
    label_rank = torch.zeros(num_genotypes, dtype=torch.long)
    label_rank[genotype_label] = 1
    if genotype_label > 0:
        label_rank[(label_rank == 0) & (torch.arange(num_genotypes) > 0)] = 3

    return ImageGroup(query=query, support=support, label_rank=label_rank, genotype_label=genotype_label, meta=meta)


class TestSelectPacks:
    def test_no_pack_exceeds_budget(self):
        rng = random.Random(20260911)
        # G in [3, 36] is the range h(h+1)/2 spans for h in [2, 8] sampled haplotypes
        costs = [1 + rng.randint(3, 36) for _ in range(2000)]

        packs = select_packs(costs, budget=256)

        assert sum(len(pack) for pack in packs) == len(costs), "Every group must land in exactly one pack"
        assert sorted(i for pack in packs for i in pack) == list(range(len(costs))), "Packs must partition the input"
        for pack in packs:
            assert sum(costs[i] for i in pack) <= 256, "No pack may exceed the budget"

    def test_occupancy(self):
        rng = random.Random(20260911)
        costs = [1 + rng.randint(3, 36) for _ in range(2000)]

        packs = select_packs(costs, budget=256)

        occupancy = sum(sum(costs[i] for i in pack) for pack in packs) / (len(packs) * 256)
        assert occupancy > 0.90, f"First-fit-decreasing occupancy {occupancy:.3f} is below the 0.90 sanity threshold"

    def test_deterministic(self):
        rng = random.Random(7)
        costs = [1 + rng.randint(3, 36) for _ in range(500)]
        assert select_packs(costs, budget=64) == select_packs(costs, budget=64), "Packing must be deterministic"

    def test_buckets_are_not_mixed(self):
        costs = [(10, "narrow"), (10, "wide"), (10, "narrow"), (10, "wide")]

        packs = select_packs(costs, budget=256)

        assert len(packs) == 2, "Groups in different width buckets cannot share a pack"
        assert sorted(packs) == [[0, 2], [1, 3]]

    def test_oversized_group_rejected(self):
        with pytest.raises(ValueError, match="exceeds the budget"):
            select_packs([4, 400, 4], budget=256)


class TestBuildPackedBatch:
    def test_offsets_labels_and_widths(self):
        groups = [make_group(1, 3, genotype_label=1, meta="a"), make_group(1, 5, genotype_label=0, meta="b")]

        batch = build_packed_batch(groups, budget=32, pad=False)

        assert batch.images.shape == (3 + 5 + 2, 2, 4, 8), "Supports then one query per group"
        assert batch.images.dtype == torch.uint8, "Images stay uint8 until normalization on the GPU"
        assert batch.offsets.equal(torch.tensor([0, 3, 8], dtype=torch.long))
        # Group 0 is [0, 1, 3] (rank 1 at genotype 1, rank 3 on the other non-ref genotype), group 1 is
        # [1, 0, 0, 0, 0] (a reference call has no third-rank positives). Unpadded, labels stop at the supports.
        assert batch.labels.equal(torch.tensor([0, 1, 3, 1, 0, 0, 0, 0], dtype=torch.long))
        assert batch.num_genotypes.equal(torch.tensor([3, 5], dtype=torch.int32))
        assert batch.genotype_label.equal(torch.tensor([1, 0], dtype=torch.long))
        assert batch.widths.equal(torch.full((10,), 8, dtype=torch.int32))
        assert batch.meta == ["a", "b"]
        assert batch.num_groups == 2
        assert batch.num_images == 10

    def test_padding_rows_are_exactly_zero(self):
        batch = build_packed_batch([make_group(1, 3, fill=200)], budget=16, pad=True)

        assert batch.images.shape[0] == 16
        assert torch.all(batch.images[4:] == 0), "Padding rows must be the 0 'no data' sentinel, not uninitialized memory"
        assert torch.all(batch.images[3] == 200), "The query image follows the 3 support images"
        assert batch.labels.shape == (16,)
        assert torch.all(batch.labels[3:] == -100), "Labels beyond the support region are padded"

    def test_width_bucketing_pads_narrow_groups(self):
        batch = build_packed_batch([make_group(1, 2, width=5)], budget=16, width_bucket=8)

        assert batch.images.shape[-1] == 8, "Images are padded out to the bucket width"
        assert torch.all(batch.images[:, :, :, 5:] == 0), "Width padding is zero"
        assert batch.widths.equal(torch.tensor([5, 5, 5], dtype=torch.int32)), "widths records the true pre-bucket width"

    def test_unlabeled_group(self):
        group = make_group(1, 3)
        group.genotype_label = None

        batch = build_packed_batch([group], budget=16)

        assert batch.genotype_label.equal(torch.tensor([-100], dtype=torch.long))
        assert batch.positive_indices().numel() == 0, "An unlabeled group contributes no positives"

    def test_replicates_major_label_expansion(self):
        """R=2, G=3: the rank vector is tiled, not interleaved.

        This is the one transposed-axis mistake that yields a plausible-looking batch instead of an
        exception, since both expansions have length R*G.
        """
        group = make_group(2, 3, genotype_label=2)
        assert group.label_rank.equal(torch.tensor([0, 3, 1], dtype=torch.long))

        batch = build_packed_batch([group], budget=16)

        assert batch.labels.equal(torch.tensor([0, 3, 1, 0, 3, 1], dtype=torch.long)), "Tiled, not repeat_interleaved"
        assert not batch.labels.equal(torch.tensor([0, 0, 3, 3, 1, 1], dtype=torch.long)), "repeat_interleave is the genotypes-major expansion"

        # Pixels encode 1 + r*G + g, so the flattened support must run r0g0, r0g1, r0g2, r1g0, ...
        assert [int(batch.images[i, 0, 0, 0]) for i in range(6)] == [1, 2, 3, 4, 5, 6]

        # There are R positives per group, at offsets[b] + r*G_b + genotype_label[b] for each r
        positives = batch.positive_indices()
        assert positives.equal(torch.tensor([2, 5], dtype=torch.long))
        assert torch.all(batch.labels[positives] == 1), "genotype_label must resolve to a rank-1 entry for every replicate"

    def test_multi_group_replicates_major(self):
        groups = [make_group(2, 3, genotype_label=1), make_group(3, 2, genotype_label=0)]

        batch = build_packed_batch(groups, budget=32)

        assert batch.offsets.equal(torch.tensor([0, 6, 12], dtype=torch.long))
        assert batch.labels.equal(torch.tensor([0, 1, 3] * 2 + [1, 0] * 3, dtype=torch.long))
        positives = batch.positive_indices()
        assert positives.equal(torch.tensor([1, 4, 6, 8, 10], dtype=torch.long))
        assert torch.all(batch.labels[positives] == 1)

    def test_to_and_pin_memory_preserve_meta(self):
        batch = build_packed_batch([make_group(1, 3, meta="region")], budget=16)

        moved = batch.to("cpu")

        assert moved.meta == ["region"], "Non-tensor fields survive a device move"
        assert moved.images.equal(batch.images)



class TestPackedImageDataLoader:
    """Regression guard for the legacy genotypes-major corpus.

    `tests/data/images-0000.tar` is genotypes-major (4 genotypes, 2 replicates), so it is read with
    `legacy_axis_order=True`. That transposes it to replicates-major, which is what the tiled label
    order [0,3,3,1, 0,3,3,1] reflects: the per-genotype ranks [0,3,3,1] repeated once per replicate,
    rather than the genotypes-major interleave [0,0,3,3,3,3,1,1]. Same images, same labels on them,
    different flattening order.
    """

    def test_packed_loader(self, cfg):
        dm = PackedImageDataModule(train_urls=data_path("images-0000.tar"), batch_size=64, num_workers=cfg.threads, legacy_axis_order=True)
        dm.prepare_data()

        dm.setup(stage="fit")

        for _i, batch in enumerate(dm.train_dataloader()):
            b, c, h, w = batch.images.shape
            assert b == 2*4+1, "Batch size must be 2 replicates of 4 genotypes + 1 query image"
            assert batch.labels.shape == (2*4,), "Labels must be 1D tensor with 2 replicates of 4 genotypes"
            assert batch.labels.equal(torch.tensor([0,3,3,1, 0,3,3,1], dtype=torch.long)), "All replicates of correct genotype (3) should have 1 label"
            assert batch.offsets.equal(torch.tensor([0, 8], dtype=torch.long)), "Offsets should match the support images"
            assert batch.meta == ["12_22129565_22130387"], "Metadata should include key or region strings"
            assert batch.genotype_label.equal(torch.tensor([3], dtype=torch.long)), "The canonical positive is genotype 3"
            assert batch.num_genotypes.equal(torch.tensor([4], dtype=torch.int32)), "There are 4 genotypes in the group"
            assert batch.labels[batch.positive_indices()].equal(torch.tensor([1, 1], dtype=torch.long)), "Every replicate's positive is rank 1"

        assert _i == 0, "The 8 support images are packed into a single batch with the query image"

        dm.teardown(stage="fit")

    def test_packed_and_padded_loader(self, cfg):
        dm = PackedImageDataModule(train_urls=data_path("images-0000.tar"), batch_size=64, pad=True, num_workers=cfg.threads, legacy_axis_order=True)
        dm.prepare_data()

        dm.setup(stage="fit")

        for _i, batch in enumerate(dm.train_dataloader()):
            b, c, h, w = batch.images.shape
            assert b == 64, "Batch size must be padded to max batch size"
            assert batch.labels.shape == (b,), "Labels must be 1D tensor matching image batch size"
            assert batch.labels.equal(torch.tensor([0,3,3,1, 0,3,3,1] + [-100]*(64-8), dtype=torch.long)), "All replicates of correct genotype (3) should have 1 label"
            assert batch.offsets.equal(torch.tensor([0, 8], dtype=torch.long)), "Offsets should match the support images"
            assert batch.meta == ["12_22129565_22130387"], "Metadata should include key or region strings"

            # Padding rows are zero, i.e. an empty pileup, not uninitialized memory. `transforms` maps
            # the 0 sentinel to -1, exactly as it does for a real 0-valued pixel.
            assert torch.all(batch.images[9:] == -1.0), "Padding rows must be the zero sentinel"

        assert _i == 0, "The 8 support images are packed into a single batch with the query image"

        dm.teardown(stage="fit")

@pytest.mark.skipif(LEGACY_SHARD is None, reason="Existing genotypes-major image shards required")
class TestLegacyAxisOrder:
    def test_matches_genotypes_major_unpack(self, cfg):
        """The R=1 corpus must read identically through the legacy flag and through today's code.

        With R=1, `(1, G)` and `(G, 1)` hold the same pixel bytes in the same flattened order, so
        the transpose changes nothing that reaches the model — which is what makes shipping the flag
        cheaper than stranding 406 GB of shards.
        """
        dm = PackedImageDataModule(train_urls=LEGACY_SHARD, batch_size=256, num_workers=cfg.threads, legacy_axis_order=True)
        dm.setup(stage="fit")

        num_batches = 0
        for batch in dm.train_dataloader():
            num_batches += 1

            # Reproduce the pre-replicates-major unpack directly from the shard's own ranks
            expected = []
            for group in range(batch.num_groups):
                genotypes = int(batch.num_genotypes[group])
                replicates = int(batch.offsets[group + 1] - batch.offsets[group]) // genotypes
                assert replicates == 1, "The existing corpus is R=1; a legacy R>1 shard would reorder"
                start = int(batch.offsets[group])
                expected.append(batch.labels[start : start + genotypes].repeat_interleave(replicates))
            assert batch.labels.equal(torch.cat(expected)), "Legacy labels must match the genotypes-major expansion"

            # Every group's canonical positive resolves to a rank-1 entry
            positives = batch.positive_indices()
            labeled = batch.genotype_label >= 0
            assert positives.numel() == int(labeled.sum()), "R=1 gives exactly one positive per labeled group"
            assert torch.all(batch.labels[positives] == 1)

            if num_batches >= 3:
                break

        assert num_batches > 0, "Expected at least one batch from the legacy shard"
        dm.teardown(stage="fit")


class TestMaskedImageDataLoader:
    # `images-0000.tar` is genotypes-major, so it needs legacy_axis_order here too. Without it the
    # genotype axis is read as the replicate axis and the label no longer indexes a genotype, which
    # now fails loudly instead of silently mislabeling.
    def test_masked_loader(self, cfg):
        dm = MaskedImageDataModule(train_urls=data_path("images-0000.tar"), batch_size=2, num_workers=cfg.threads, patch_size=(16,16), mask_fraction=0.5, legacy_axis_order=True)
        dm.prepare_data()

        dm.setup(stage="fit")

        for _i, batch in enumerate(dm.train_dataloader()):
            images, masks = batch

            b, _c, h, w = images.shape
            assert b == 2

            num_patches = (h // 16) * (w // 16)
            assert masks.shape == (b, num_patches), f"Masks must be 2D tensor with {num_patches} patches per image"
            assert torch.equal(masks.sum(dim=1), torch.tensor([int(0.5 * num_patches)]*b)), "Each image should have 50% of patches masked"

        assert _i == 3, "The 9 images packed into 4 complete batches of 2 images each, with the remaining partial batch dropped"

        dm.teardown(stage="fit")

    def test_partial_masked_loader(self, cfg):
        dm = MaskedImageDataModule(test_urls=data_path("images-0000.tar"), batch_size=64, num_workers=cfg.threads, patch_size=(8,8), mask_fraction=0.8, legacy_axis_order=True)
        dm.prepare_data()

        # Ensure partial batches are included
        dm.setup(stage="test")

        for _i, batch in enumerate(dm.test_dataloader()):
            images, masks = batch

            b, _c, h, w = images.shape
            assert b == 2*4+1, "Batch size must be 2 replicates of 4 genotypes + 1 query image"

            num_patches = (h // 8) * (w // 8)
            assert masks.shape == (b, num_patches), f"Masks must be 2D tensor with {num_patches} patches per image"
            assert torch.equal(masks.sum(dim=1), torch.tensor([int(0.8 * num_patches)]*b)), "Each image should have 80% of patches masked"


        assert _i == 0, "All 9 images are packed into a single partial batch"

        dm.teardown(stage="test")

class TestNPSV2Examples:
    @pytest.mark.skipif(
        not os.path.exists(os.path.join(EXPERIMENTS_DIR, "training/freeze3.sv.alt.passing.training.hg38.DEL.images/HG00096/+pileup.snv_input=True,generator=single_depth_phaseread,pileup.discrete_mapq=True,pileup.render_snv=True,simulation.augment=True,simulation.chrom_norm_covg=True,simulation.replicates=5/images.tar")),
        reason="NPSV2 images dataset required"
    )
    def test_visualize_converted_images(self, tmp_path, cfg):
        urls = os.path.join(EXPERIMENTS_DIR, "training/freeze3.sv.alt.passing.training.hg38.DEL.images/HG00096/+pileup.snv_input=True,generator=single_depth_phaseread,pileup.discrete_mapq=True,pileup.render_snv=True,simulation.augment=True,simulation.chrom_norm_covg=True,simulation.replicates=5/images.tar")
        dataset = wds.WebDataset(urls, shardshuffle=False).decode()
        for _i, sample in enumerate(dataset):
            #png_path = str(tmp_path / "test.png")
            png_path = result_path("test.png")

            example_to_image(
                cfg,
                {"image": sample["image.npy.gz"], "sim.images": sample["sim.images.npy.gz"] },
                png_path, with_simulations=True, render_channels=True, select_channels=[0, 1, 5], # ALIGNED, PAIRED, ALLELE
            )
            assert os.path.exists(png_path)
            break

    def test_packed_loader(self, tmp_path, cfg):
        urls = os.path.join(EXPERIMENTS_DIR, "training/freeze3.sv.alt.passing.training.hg38.DEL.images/HG00096/+pileup.snv_input=True,generator=single_depth_phaseread,pileup.discrete_mapq=True,pileup.render_snv=True,simulation.augment=True,simulation.chrom_norm_covg=True,simulation.replicates=5/images.tar")
        dm = PackedImageDataModule(train_urls=urls, batch_size=64, num_workers=cfg.threads, legacy_axis_order=True)
        dm.prepare_data()

        dm.setup(stage="fit")

        for _i, batch in enumerate(dm.train_dataloader()):
            print(batch.images.shape, batch.labels.shape, batch.offsets, batch.meta)
            break

        dm.teardown(stage="fit")