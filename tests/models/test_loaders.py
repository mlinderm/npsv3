import random

import pytest
import torch

from npsv3.models.loaders import (
    ImageGroup,
    build_packed_batch,
    select_packs,
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

    def test_padding_images_are_exactly_zero(self):
        batch = build_packed_batch([make_group(1, 3, fill=200)], budget=16, pad=True)

        assert batch.images.shape[0] == 16
        assert torch.all(batch.images[4:] == 0), "Padding images must be the 0 'no data' sentinel, not uninitialized memory"
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

