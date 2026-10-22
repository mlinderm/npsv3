import gzip
import json
from pathlib import Path

import hydra
import numpy as np
import pytest
import webdataset as wds
from omegaconf import OmegaConf

from npsv3.constants import COMPARISON_SUFFIX, COMPARISONS_AUX, COMPARISONS_DIR
from npsv3.images.sequence_compare import ComparisonOptions, load_comparison_records, prepare_comparison_tree
from npsv3.models.energon import _read_split, estimate_epoch_batches, set_aux_tree
from npsv3.models.filters import (
    CopositiveFilter,
    LabelDiffersFilter,
    build_filters,
    classify_copositive,
)
from npsv3.models.transforms import write_annotated_dataset
from tests.models.test_energon_loaders import IMAGE_WIDTH, _build_dataset, _make_dm

# A 3-genotype example labeled 0, whose top alternative is genotype 1
SEPARABLE = [1.0, 0.1, 0.1]
LOOKALIKE = [1.0, 0.9, 0.1]
SAME = {"genotype_differ": [False, False, True], "genotype_score": [1.0, 1.0, 0.1]}
DIFFERENT = {"genotype_differ": [False, True, True], "genotype_score": [1.0, 0.2, 0.1]}
SIMILAR = {"genotype_differ": [False, True, True], "genotype_score": [1.0, 0.8, 0.1]}


@pytest.mark.parametrize(
    ("probs", "weights", "comparison", "category"),
    [
        ([0.3, 0.6, 0.1], SEPARABLE, None, "uncompared"),
        ([0.3, 0.6, 0.1], SEPARABLE, {"skipped": "length", "length": 20_000}, "uncompared"),
        ([0.6, 0.3, 0.1], SEPARABLE, None, "modeled"),
        ([0.3, 0.6, 0.1], LOOKALIKE, SAME, "lookalike"),
        ([0.3, 0.6, 0.1], LOOKALIKE, SIMILAR, "lookalike"),
        ([0.3, 0.6, 0.1], LOOKALIKE, DIFFERENT, "unresolved"),
        ([0.6, 0.3, 0.1], SEPARABLE, DIFFERENT, "modeled"),
        ([0.3, 0.6, 0.1], SEPARABLE, SAME, "noisy_equiv"),
        ([0.01, 0.9, 0.09], SEPARABLE, DIFFERENT, "suspect"),
        ([0.3, 0.6, 0.1], SEPARABLE, DIFFERENT, "suspect_soft"),
    ],
)
def test_classify_copositive(probs, weights, comparison, category):
    result = classify_copositive(0, probs, weights, comparison)
    assert result["category"] == category
    assert result["alt"] == 1


def test_label_differs_filter():
    assert LabelDiffersFilter().keep({}, {"label_differs": False})
    assert not LabelDiffersFilter().keep({}, {"label_differs": True})
    for missing in (None, {"label_differs": None}, {"skipped": "length", "length": 20_000}):
        assert LabelDiffersFilter().keep({}, missing)
        assert not LabelDiffersFilter(missing="exclude").keep({}, missing)


def test_copositive_filter():
    row = {"label": 0, "genotype_probs": [0.01, 0.9, 0.09], "genotype_weights": SEPARABLE}
    assert not CopositiveFilter().keep(row, DIFFERENT)
    assert CopositiveFilter(exclude=["noisy_equiv"]).keep(row, DIFFERENT)
    # Raising the threshold makes the alternative a look-alike
    assert CopositiveFilter(conf_threshold=0.05).keep(row, DIFFERENT)

    unscored = {"label": 0}
    assert CopositiveFilter().keep(unscored, DIFFERENT)
    assert not CopositiveFilter(missing="exclude").keep(unscored, DIFFERENT)


def test_invalid_filters():
    with pytest.raises(ValueError, match="Unknown co-positive categories"):
        CopositiveFilter(exclude=["bogus"])
    with pytest.raises(ValueError, match="missing-data policy"):
        LabelDiffersFilter(missing="drop")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="not an `ExampleFilter`"):
        build_filters([{"_target_": "builtins.dict"}])


def test_build_filters_from_config():
    cfg = OmegaConf.create([{"_target_": "npsv3.models.filters.CopositiveFilter", "exclude": ["suspect", "suspect_soft"]}])
    (built,) = build_filters(cfg)
    assert built == CopositiveFilter(exclude=("suspect", "suspect_soft"))
    assert build_filters(None) == ()


# ---------------------------------------------------------------------------------------------------------------------
# Applying filters in the data module
# ---------------------------------------------------------------------------------------------------------------------


def _rows(dataset_path) -> list[dict]:
    root, split = _read_split(dataset_path)
    with (root / split["path"]).open() as meta_file:
        return [json.loads(line) for line in meta_file if line.strip()]


def _add_comparisons(dataset_path, records: dict[str, dict]) -> None:
    """Write `records` (key -> comparison record) as `dataset_path`'s comparisons tree, without comparing anything"""
    root = Path(dataset_path).parent
    directory = root / COMPARISONS_DIR
    directory.mkdir()
    with wds.ShardWriter(str(directory / "compare-0000-%06d.tar"), encoder=False, verbose=0) as writer:
        for key, record in records.items():
            writer.write({"__key__": key, COMPARISON_SUFFIX: gzip.compress(json.dumps(record).encode())})
    assert prepare_comparison_tree(root, ComparisonOptions(), num_workers=1)
    set_aux_tree(dataset_path, COMPARISONS_AUX, f"./{COMPARISONS_DIR}")


@pytest.fixture(scope="module")
def compared_dataset(tmp_path_factory):
    """A dataset where every 4th example's label differs from the sample, and every 5th has no comparison record"""
    dataset_path = _build_dataset(str(tmp_path_factory.mktemp("filters")), num_groups=200, num_shards=2)
    records = {
        row["key"]: {"label_differs": index % 4 == 0, "label_score": 0.5}
        for index, row in enumerate(_rows(dataset_path))
        if index % 5 != 0
    }
    _add_comparisons(dataset_path, records)
    return dataset_path


def _train_regions(dm) -> list[str]:
    regions = []
    for batch in dm.train_dataloader():
        regions.extend(batch.meta[: batch.num_groups])
    dm.teardown()
    return regions


def _expected_regions(rows, keep, **limits) -> set[str]:
    g_max, r_max = limits.get("g_max", 36), limits.get("r_max", 2)
    return {row["region"] for index, row in enumerate(rows) if row["g"] <= g_max and row["r"] <= r_max and keep(index)}


def test_load_comparison_records(compared_dataset):
    rows = _rows(compared_dataset)
    records = load_comparison_records(Path(compared_dataset).parent / COMPARISONS_DIR)
    assert len(records) == sum(index % 5 != 0 for index in range(len(rows)))
    assert records[rows[1]["key"]] == {"label_differs": False, "label_score": 0.5}


@pytest.mark.slow
@pytest.mark.parametrize("missing", ["keep", "exclude"])
def test_label_differs_filter_in_loader(compared_dataset, missing):
    rows = _rows(compared_dataset)
    dm = _make_dm(compared_dataset, repeat=False, filters=[LabelDiffersFilter(missing=missing)])
    regions = _train_regions(dm)

    # Examples without a record (every 5th) are kept or excluded regardless of the label
    expected = _expected_regions(rows, lambda i: (missing == "keep") if i % 5 == 0 else i % 4 != 0)
    assert len(regions) == len(set(regions)), "A single pass should see each example once"
    assert set(regions) == expected

    # Validation isn't filtered
    assert dm._task_encoder().filters == ()


def test_estimate_epoch_batches_applies_filters(compared_dataset, caplog):
    rows = _rows(compared_dataset)
    options = {"budget": 64, "w_bucket": IMAGE_WIDTH, "w_max": IMAGE_WIDTH, "g_max": 36, "r_max": 2}
    unfiltered = estimate_epoch_batches(compared_dataset, **options)
    with caplog.at_level("INFO"):
        filtered = estimate_epoch_batches(compared_dataset, filters=[LabelDiffersFilter(missing="exclude")], **options)
    assert filtered < unfiltered
    excluded = len(_expected_regions(rows, lambda _: True)) - len(_expected_regions(rows, lambda i: i % 4 != 0 and i % 5 != 0))
    assert f"exclude {excluded} otherwise accepted groups" in caplog.text


@pytest.mark.slow
def test_copositive_filter_in_loader(tmp_path):
    """A copositive dataset inherits its source's comparisons tree, and its suspect (and here uncompared) examples are
    excluded"""
    source_path = _build_dataset(str(tmp_path / "source"), num_groups=200, num_shards=2)
    rows = _rows(source_path)
    # Every alternative genotype differs, so a confidently preferred alternative is "suspect" if compared...
    _add_comparisons(
        source_path,
        {
            row["key"]: {"genotype_differ": [g != row["label"] for g in range(row["g"])], "genotype_score": [0.0] * row["g"]}
            for index, row in enumerate(rows)
            if index % 5 != 0
        },
    )
    # ...as is every 3rd example's (the rest prefer the label), while every 7th is unscored
    preferred_alt = {index for index in range(len(rows)) if index % 3 == 0}
    annotations = {}
    for index, row in enumerate(rows):
        if index % 7 == 6:
            continue
        probs = np.full(row["g"], 0.01 / (row["g"] - 1))
        probs[row["label"]] = 0.99
        if index in preferred_alt:
            probs = np.roll(probs, 1)
        annotations[row["region"]] = {"genotype_probs": probs.tolist(), "genotype_weights": [0.0] * row["g"]}
    copositive_path = write_annotated_dataset(source_path, tmp_path / "copositive", annotations)

    dm = _make_dm(copositive_path, repeat=False, filters=[CopositiveFilter(exclude=["suspect", "uncompared"])])
    expected = _expected_regions(rows, lambda i: i not in preferred_alt or i % 7 == 6)
    assert set(_train_regions(dm)) == expected


def test_requirements_are_checked(compared_dataset, tmp_path):
    uncompared = _build_dataset(str(tmp_path / "uncompared"), num_groups=20, num_shards=1)
    with pytest.raises(ValueError, match="command=compare"):
        _make_dm(uncompared, filters=[LabelDiffersFilter()])
    with pytest.raises(ValueError, match="command=copositive"):
        _make_dm(compared_dataset, filters=[CopositiveFilter()])
    # Filters configured via Hydra are instantiated, and only the train split is checked
    dm = _make_dm(compared_dataset, val_metadataset_paths=uncompared, filters=[{"_target_": "npsv3.models.filters.LabelDiffersFilter"}])
    assert dm.train_filters == (LabelDiffersFilter(),)


def test_filters_from_data_config(hydra_setup, compared_dataset):  # noqa: ARG001
    """Filters configured with `data.filters` reach the data module as filter objects"""
    cfg = hydra.compose(
        config_name="config",
        overrides=[
            "data=energon_packed_images",
            f"data.train_metadataset_paths=[{compared_dataset}]",
            "data.filters=[{_target_: npsv3.models.filters.LabelDiffersFilter, missing: exclude}]",
        ],
    )
    dm = hydra.utils.instantiate(cfg.data)
    assert dm.train_filters == (LabelDiffersFilter(missing="exclude"),)
