import functools

import hydra
import lightning as L
import numpy as np
import pytest
import torch
from torch import nn

from npsv3.images.sequence_compare import load_comparison_records
from npsv3.models.copositive import VARIANTS, CopositiveTargets
from npsv3.models.energon import _read_split
from npsv3.models.loaders import ImageGroup, build_packed_batch
from npsv3.models.paired import CopositiveInfoNCE, InfoNCE, MaximizingPredictor, PackedVariant
from npsv3.models.transforms import write_annotated_dataset
from tests.models.test_energon_loaders import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, _build_dataset, _make_dm
from tests.models.test_filters import _add_comparisons, _rows

# A 5-genotype example labeled 0. Genotypes 1 and 2 are sequence-equivalent to the label (2 by score), genotype 1 is a
# look-alike in simulation and genotype 3 is a look-alike whose sequence differs.
CONF = [1.0, 0.8, 0.2, 0.9, 0.0]
COMPARISON = {"genotype_differ": [False, False, True, True, True], "genotype_score": [1.0, 1.0, 0.75, 0.1, 0.0]}
# The real query prefers the look-alike genotype 1, so the example is a "lookalike"
LOOKALIKE_PROBS = [0.3, 0.6, 0.05, 0.03, 0.02]
# The real query prefers the separable, equivalent genotype 2, so the example is "noisy_equiv"
NOISY_EQUIV_PROBS = [0.3, 0.05, 0.6, 0.03, 0.02]


def _row(probs, conf=CONF, **kwargs):
    return {"key": "example", "g": len(conf), "label": 0, "genotype_probs": probs, "genotype_weights": conf, **kwargs}


@pytest.mark.parametrize(
    ("variant", "probs", "positive", "denominator"),
    [
        ("baseline", LOOKALIKE_PROBS, [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]),
        ("mask", LOOKALIKE_PROBS, [1, 0, 0, 0, 0], [1, 0, 1, 1, 1]),
        ("weight", LOOKALIKE_PROBS, [1, 0, 0, 0, 0], [1, 0.2, 0.8, 1, 1]),
        ("multi", LOOKALIKE_PROBS, [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]),
        ("soft_multi", LOOKALIKE_PROBS, [1, 0.8, 0.2, 0, 0], [1, 1, 1, 1, 1]),
        ("mask+multi", LOOKALIKE_PROBS, [1, 0, 1, 0, 0], [1, 0, 1, 1, 1]),
        # Gated applies `weight` to "lookalike" examples and `multi` to "noisy_equiv" examples
        ("gated", LOOKALIKE_PROBS, [1, 0, 0, 0, 0], [1, 0.2, 0.8, 1, 1]),
        ("gated", NOISY_EQUIV_PROBS, [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]),
    ],
)
def test_variant_targets(variant, probs, positive, denominator):
    actual_positive, actual_denominator = CopositiveTargets(variant).targets(_row(probs), COMPARISON)
    np.testing.assert_allclose(actual_positive, positive, rtol=1e-6)
    np.testing.assert_allclose(actual_denominator, denominator, rtol=1e-6)


def test_gated_leaves_other_categories_unchanged():
    # The top alternative is separable and differs, and the real query prefers the label, so the example is "modeled"
    probs = [0.6, 0.05, 0.05, 0.05, 0.25]
    positive, denominator = CopositiveTargets("gated").targets(_row(probs), COMPARISON)
    np.testing.assert_array_equal(positive, [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(denominator, [1, 1, 1, 1, 1])


def test_missing_annotations_use_single_positive_targets():
    targets = CopositiveTargets("multi")
    baseline = ([1, 0, 0, 0, 0], [1, 1, 1, 1, 1])
    # Unscored, uncompared and skipped comparisons
    for row, comparison in [
        ({"key": "example", "g": 5, "label": 0}, COMPARISON),
        (_row(LOOKALIKE_PROBS), None),
        (_row(LOOKALIKE_PROBS), {"skipped": "length", "length": 20_000}),
    ]:
        for actual, expected in zip(targets.targets(row, comparison), baseline, strict=True):
            np.testing.assert_array_equal(actual, expected)


def test_targets_with_genotype_subset():
    # The example has (stored) genotypes 0, 2 and 3 of the 5 compared
    row = _row([0.3, 0.6, 0.1], conf=[1.0, 0.2, 0.9], genotype_idxs=[0, 2, 3])
    positive, denominator = CopositiveTargets("multi").targets(row, COMPARISON)
    np.testing.assert_array_equal(positive, [1, 1, 0])
    np.testing.assert_array_equal(denominator, [1, 1, 1])


def test_overrides_and_validation():
    targets = CopositiveTargets("gated", denominator="mask", positive_categories="all")
    assert targets.positives == "equivalent"
    assert targets.positive_categories is None
    assert targets.denominator_categories == ("lookalike",)
    positive, denominator = targets.targets(_row(LOOKALIKE_PROBS), COMPARISON)
    np.testing.assert_array_equal(positive, [1, 1, 1, 0, 0])  # Every category's equivalent alternatives are positives
    np.testing.assert_array_equal(denominator, [1, 1, 1, 1, 1])  # Positives are always fully in the denominator

    assert CopositiveTargets("gated") == CopositiveTargets(**CopositiveTargets("gated").params)
    for kwargs in [{"variant": "bogus"}, {"positives": "bogus"}, {"denominator": "bogus"}, {"positive_categories": ["bogus"]}]:
        with pytest.raises(ValueError, match="Unknown"):
            CopositiveTargets(**kwargs)
    with pytest.raises(ValueError, match="genotype subset"):
        CopositiveTargets().targets(_row(LOOKALIKE_PROBS, conf=CONF[:3]), COMPARISON)


# ---------------------------------------------------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------------------------------------------------


def _nested(*values):
    offsets = torch.tensor(np.cumsum([0, *(len(v) for v in values)]))
    return torch.nested.nested_tensor_from_jagged(torch.cat([torch.as_tensor(v) for v in values]), offsets=offsets)


def _tiled(values, num_replicates):
    return np.tile(np.asarray(values, dtype=np.float32), num_replicates)


R = 2
LABEL_RANKS = ([1, 3, 3, 3, 3], [0, 0, 1, 0])
RNG = np.random.default_rng(42)
METRICS = [RNG.uniform(-1, 1, size=R * len(ranks)).astype(np.float32) for ranks in LABEL_RANKS]


def _loss(loss_fn, metrics, ranks, positive=None, denominator=None):
    labels = _nested(*(torch.tensor(_tiled(r, R), dtype=torch.long) for r in ranks))
    weights = []
    if positive is not None:
        weights = [_nested(*(_tiled(p, R) for p in positive)), _nested(*(_tiled(d, R) for d in denominator))]
    return loss_fn(_nested(*metrics), labels, *weights)[0].item()


def test_single_positive_is_infonce():
    expected = _loss(InfoNCE(), METRICS, LABEL_RANKS)
    assert _loss(CopositiveInfoNCE(), METRICS, LABEL_RANKS) == pytest.approx(expected, rel=1e-5)
    baseline = ([[1, 0, 0, 0, 0], [0, 0, 1, 0]], [[1] * 5, [1] * 4])
    assert _loss(CopositiveInfoNCE(), METRICS, LABEL_RANKS, *baseline) == pytest.approx(expected, rel=1e-5)


def test_multiple_positives_are_infonce_positives():
    # Genotype 1 of the first group as a hard positive is a (first-rank) positive for InfoNCE
    expected = _loss(InfoNCE(), METRICS, ([1, 1, 3, 3, 3], LABEL_RANKS[1]))
    weights = ([[1, 1, 0, 0, 0], [0, 0, 1, 0]], [[1] * 5, [1] * 4])
    assert _loss(CopositiveInfoNCE(), METRICS, LABEL_RANKS, *weights) == pytest.approx(expected, rel=1e-5)


def test_masking_removes_supports():
    # Masking genotype 1 of the first group is the same as removing its supports
    keep = np.tile(np.arange(5) != 1, R)
    expected = _loss(InfoNCE(), [METRICS[0][keep], METRICS[1]], ([1, 3, 3, 3], LABEL_RANKS[1]))
    weights = ([[1, 0, 0, 0, 0], [0, 0, 1, 0]], [[1, 0, 1, 1, 1], [1] * 4])
    assert _loss(CopositiveInfoNCE(), METRICS, LABEL_RANKS, *weights) == pytest.approx(expected, rel=1e-5)


def test_weighted_targets():
    """A partial positive and down-weighted negative, against the loss computed directly from its definition"""
    positive, denominator = [1.0, 0.25, 0.0, 0.0], [1.0, 1.0, 0.5, 1.0]
    s = METRICS[1].astype(np.float64) / 0.07
    c, a = _tiled(positive, R), _tiled(denominator, R)
    anchor_losses = []
    for i in np.flatnonzero(c > 0):
        others = np.arange(len(s)) != i
        anchor_losses.append(np.log(np.exp(s[i]) + np.sum(((a - c) * np.exp(s))[others])) - s[i])
    expected = np.sum(c[c > 0] * np.array(anchor_losses)) / np.sum(c[c > 0])

    loss = _loss(CopositiveInfoNCE(), [METRICS[1]], [LABEL_RANKS[1]], [positive], [denominator])
    assert loss == pytest.approx(expected, rel=1e-4)


def test_loss_gradients_are_finite():
    values = torch.tensor(np.concatenate(METRICS), requires_grad=True)
    offsets = torch.tensor(np.cumsum([0, *(len(m) for m in METRICS)]))
    metric = torch.nested.nested_tensor_from_jagged(values, offsets=offsets)
    labels = _nested(*(torch.tensor(_tiled(r, R), dtype=torch.long) for r in LABEL_RANKS))
    weights = ([[1, 0.5, 0, 0, 0], [0, 0, 1, 0]], [[1, 1, 0, 0.5, 1], [1] * 4])
    positive, denominator = (_nested(*(_tiled(w, R) for w in ws)) for ws in weights)
    loss, batch_size = CopositiveInfoNCE()(metric, labels, positive, denominator)
    loss.backward()
    assert batch_size == 2
    assert torch.isfinite(loss)
    assert values.grad is not None
    assert torch.all(torch.isfinite(values.grad))


class _MeanEncoder(nn.Module):
    """Embed each image as its per-channel mean, so the model step runs without a real encoder"""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(IMAGE_CHANNELS, 4)

    def forward(self, images):
        return self.projection(images.float().mean(dim=(2, 3)))


def _model(loss):
    return PackedVariant(
        encoder=_MeanEncoder(), metric=nn.CosineSimilarity(), loss=loss, optimizer=None, predictor=MaximizingPredictor()  # type: ignore[arg-type]
    )


def _group(label, num_genotypes, positive=None, denominator=None, *, rng):
    label_rank = torch.zeros(num_genotypes, dtype=torch.long)
    label_rank[label] = 1
    return ImageGroup(
        query=torch.from_numpy(rng.integers(0, 255, size=(IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)),
        support=torch.from_numpy(
            rng.integers(0, 255, size=(R, num_genotypes, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        ),
        label_rank=label_rank,
        genotype_label=label,
        positive_weights=None if positive is None else torch.tensor(positive, dtype=torch.float32),
        denominator_weights=None if denominator is None else torch.tensor(denominator, dtype=torch.float32),
    )


def test_model_step_uses_targets():
    rng = np.random.default_rng(7)
    groups = [_group(0, 3, [1, 1, 0], [1, 1, 1], rng=rng), _group(1, 4, rng=rng)]
    batch = build_packed_batch(groups, budget=32, pad=True)
    # A group without targets gets the single-positive ones, and the targets are never padded
    assert batch.positive_weights is not None
    torch.testing.assert_close(batch.positive_weights, torch.tensor([1, 1, 0] * R + [0, 1, 0, 0] * R, dtype=torch.float32))
    assert batch.denominator_weights is not None
    assert batch.denominator_weights.shape == (R * 7,)

    model = _model(CopositiveInfoNCE())
    loss, *_ = model._model_step(batch, 0)
    single = build_packed_batch([_group(0, 3, rng=np.random.default_rng(7)), groups[1]], budget=32, pad=True)
    single.images = batch.images
    single_loss, *_ = model._model_step(single, 0)
    assert single.positive_weights is None
    assert loss.item() != pytest.approx(single_loss.item())

    with pytest.raises(TypeError, match="CopositiveInfoNCE"):
        _model(InfoNCE())._model_step(batch, 0)


# ---------------------------------------------------------------------------------------------------------------------
# Loader and configuration
# ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def copositive_dataset(tmp_path_factory):
    """A `command=copositive`-like dataset with a comparisons tree, where every other alternative is sequence-equivalent
    and a look-alike, and the real query prefers the label (so "modeled") or the next genotype. Every 5th example is
    uncompared and every 7th unscored."""
    root = tmp_path_factory.mktemp("copositive")
    source_path = _build_dataset(str(root / "source"), num_groups=100, num_shards=2)
    rows = _rows(source_path)
    _add_comparisons(
        source_path,
        {
            row["key"]: {
                "genotype_differ": [g % 2 == 1 and g != row["label"] for g in range(row["g"])],
                "genotype_score": [0.0] * row["g"],
            }
            for index, row in enumerate(rows)
            if index % 5 != 0
        },
    )
    annotations = {}
    for index, row in enumerate(rows):
        if index % 7 == 6:
            continue
        probs = np.full(row["g"], 0.01 / (row["g"] - 1))
        probs[row["label"]] = 0.99
        if index % 3 == 0:
            probs = np.roll(probs, 1)
        conf = np.where(np.arange(row["g"]) % 2 == 0, 0.6, 0.1)
        conf[row["label"]] = 1.0
        annotations[row["region"]] = {"genotype_probs": probs.tolist(), "genotype_weights": conf.tolist()}
    return write_annotated_dataset(source_path, root / "copositive", annotations)


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["gated", "soft_multi"])
def test_targets_in_loader(copositive_dataset, variant):
    targets = CopositiveTargets(variant)
    rows = {row["region"]: row for row in _rows(copositive_dataset)}
    root, split = _read_split(copositive_dataset)
    comparisons = load_comparison_records(root / split["aux"]["comparisons"])

    dm = _make_dm(copositive_dataset, repeat=False, copositive=targets)
    assert dm._task_encoder().copositive is None, "Only the train split has co-positive targets"
    changed = 0
    for batch in dm.train_dataloader():
        assert batch.positive_weights is not None
        assert batch.denominator_weights is not None
        offsets = batch.offsets.tolist()
        for group, region in enumerate(batch.meta[: batch.num_groups]):
            row = rows[region]
            positive, denominator = targets.targets(row, comparisons.get(row["key"]))
            num_replicates = (offsets[group + 1] - offsets[group]) // row["g"]
            np.testing.assert_allclose(batch.positive_weights[offsets[group] : offsets[group + 1]], np.tile(positive, num_replicates))
            np.testing.assert_allclose(batch.denominator_weights[offsets[group] : offsets[group + 1]], np.tile(denominator, num_replicates))
            changed += not (np.array_equal(positive, np.arange(row["g"]) == row["label"]) and np.all(denominator == 1))
    dm.teardown()
    assert changed > 0, "Some examples should have co-positive targets"


@pytest.mark.slow
def test_fit_with_targets(copositive_dataset, tmp_path):
    """Lightning moves the batches, including their targets, to the device and trains with them"""
    model = PackedVariant(
        encoder=_MeanEncoder(),
        metric=nn.CosineSimilarity(),
        loss=CopositiveInfoNCE(),
        optimizer=functools.partial(torch.optim.SGD, lr=0.01),  # type: ignore[arg-type]
        predictor=MaximizingPredictor(),
    )
    dm = _make_dm(copositive_dataset, copositive=CopositiveTargets())
    trainer = L.Trainer(
        default_root_dir=tmp_path,
        max_steps=3,
        limit_train_batches=3,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="cpu",
    )
    trainer.fit(model, datamodule=dm)
    dm.teardown()
    assert torch.isfinite(trainer.callback_metrics["train_loss"])


def test_targets_require_copositive_dataset(tmp_path):
    uncompared = _build_dataset(str(tmp_path / "uncompared"), num_groups=20, num_shards=1)
    with pytest.raises(ValueError, match="command=compare"):
        _make_dm(uncompared, copositive=CopositiveTargets())


def test_copositive_experiment_config(hydra_setup, copositive_dataset):  # noqa: ARG001
    """`+experiments=copositive` switches both the targets and the loss"""
    cfg = hydra.compose(
        config_name="config",
        overrides=[
            "model=paired_packed_inception_infonce",
            "data=energon_packed_images",
            f"data.train_metadataset_paths=[{copositive_dataset}]",
            "+experiments=copositive",
            "data.copositive.variant=mask",
        ],
    )
    dm = hydra.utils.instantiate(cfg.data)
    assert dm.train_copositive == CopositiveTargets("mask")
    assert type(hydra.utils.instantiate(cfg.model.loss)) is CopositiveInfoNCE
    assert set(VARIANTS) >= {"gated", "mask", "weight", "multi", "soft_multi", "mask+multi", "baseline"}
