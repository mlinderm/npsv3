import gzip
import itertools
import json
import random

import numpy as np
import pytest
import yaml
from megatron.energon.cache.file_store import WebdatasetFileStore
from megatron.energon.epathlib import EPath
from omegaconf import OmegaConf

pytest.importorskip("edlib")
pytest.importorskip("pyabpoa")

from npsv3.constants import COMPARISON_SUFFIX, COMPARISONS_AUX, COMPARISONS_DIR
from npsv3.graphs.sequence_compare import ComparisonParams
from npsv3.images.sequence_compare import PROVENANCE_FILE, ComparisonOptions, add_comparisons, comparison_record
from npsv3.images.example import EnergonExampleWriter, energon_key
from npsv3.models.energon import EnergonPackedImageDataModule, prepare_energon_dataset
from npsv3.util.range import Range

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 8, 12, 3


def _random_sequence(length: int, seed: int) -> str:
    return "".join(random.Random(seed).choices("ACGT", k=length))


def _haplotypes(seed: int, *, true=(0, 1), sample: bool = True) -> dict:
    """A haplotype record with the reference, a deletion, a near-identical deletion and an insertion as candidates"""
    reference = _random_sequence(2000, seed)
    sequences = [
        reference,
        reference[:900] + reference[1200:],
        reference[:905] + reference[1195:],
        reference[:1000] + _random_sequence(300, seed + 1) + reference[1000:],
    ]
    return {
        "true": list(true),
        "sequences": sequences,
        "reference": reference,
        "sample_sequences": [sequences[1], reference] if sample else [None, None],
    }


def _num_genotypes(haplotypes: dict) -> int:
    return len(list(itertools.combinations_with_replacement(range(len(haplotypes["sequences"])), 2)))


def _same_genotypes(record: dict, haplotypes: dict) -> set[tuple[int, int]]:
    genotypes = list(itertools.combinations_with_replacement(range(len(haplotypes["sequences"])), 2))
    return {genotype for genotype, differs in zip(genotypes, record["genotype_differ"], strict=True) if not differs}


def test_options_from_config(cfg):
    assert cfg.compare.enabled is False
    assert ComparisonOptions.from_config(cfg.compare) == ComparisonOptions()

    options = ComparisonOptions.from_config({"joint": False, "max_length": None, "params": {"sizemin": 50}})
    assert options == ComparisonOptions(params=ComparisonParams(sizemin=50), joint=False, max_length=None)


def test_comparison_record():
    haplotypes = _haplotypes(0)
    record = comparison_record(haplotypes, ComparisonOptions())
    assert record["skipped"] is None
    assert len(record["genotype_differ"]) == len(record["genotype_score"]) == _num_genotypes(haplotypes)
    assert _same_genotypes(record, haplotypes) == {(0, 1), (0, 2)}
    assert record["label_differs"] is False
    assert record["label_score"] == 1.0
    json.dumps(record)  # Serializable as is


def test_comparison_record_without_sample():
    record = comparison_record(_haplotypes(0, sample=False), ComparisonOptions())
    assert record["label_differs"] is None
    assert record["label_score"] is None
    assert record["genotype_differ"] is not None


@pytest.mark.parametrize("true", [(), (0, None), (None, None)])
def test_comparison_record_unlabeled(true):
    assert comparison_record(_haplotypes(0, true=true), ComparisonOptions()) is None


def test_comparison_record_too_long():
    haplotypes = _haplotypes(0)
    record = comparison_record(haplotypes, ComparisonOptions(max_length=2000))
    assert record == {"skipped": "length", "length": len(haplotypes["sequences"][3])}


def _write_dataset(output_dir, records: list[dict | None]) -> tuple[str, list[str]]:
    """An image dataset with one example per haplotype record (None for an example without one), returning its keys"""
    rng = np.random.default_rng(0)
    writer = EnergonExampleWriter(0, str(output_dir))
    keys = []
    try:
        for index, haplotypes in enumerate(records):
            region = Range("chr1", 1000 * (index + 1), 1000 * (index + 1) + 100)
            num_genotypes = 10
            example = {
                "region": str(region),
                "image": rng.integers(0, 255, size=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8),
                "label": None,
                "num_genotypes": num_genotypes,
                "sim.images": rng.integers(
                    0, 255, size=(2, num_genotypes, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8
                ),
            }
            if haplotypes is not None:
                example["haplotypes"] = haplotypes
                if haplotypes["true"] and None not in haplotypes["true"]:
                    genotypes = list(itertools.combinations_with_replacement(range(len(haplotypes["sequences"])), 2))
                    example["label"] = genotypes.index(tuple(sorted(haplotypes["true"])))
                    example["label.rank"] = [int(i == example["label"]) for i in range(num_genotypes)]
            writer.write(example, region=region, sample_name="S0")
            keys.append(energon_key("S0", region))
    finally:
        writer.close()
    return prepare_energon_dataset(str(output_dir)), keys


def _compare_cfg(threads: int = 1, **compare):
    return OmegaConf.create({"threads": threads, "compare": {"enabled": False, **compare}})


@pytest.mark.usefixtures("ray_setup")
@pytest.mark.parametrize("threads", [1, 2])
def test_add_comparisons(tmp_path, threads):
    records = [_haplotypes(0), _haplotypes(10, true=(1, 3)), _haplotypes(20, true=(None, None)), None]
    dataset_path, keys = _write_dataset(tmp_path / "images", records)

    assert add_comparisons(_compare_cfg(threads), dataset_path)

    with open(dataset_path) as file:
        aux = yaml.safe_load(file)["splits"]["train"]["aux"]
    assert aux == {"imgs": "./imgs_ds", COMPARISONS_AUX: f"./{COMPARISONS_DIR}"}

    comparisons = WebdatasetFileStore(EPath(tmp_path / "images" / COMPARISONS_DIR))
    first = json.loads(gzip.decompress(comparisons[f"{keys[0]}.{COMPARISON_SUFFIX}"][0]))
    assert first == comparison_record(records[0], ComparisonOptions())
    second = json.loads(gzip.decompress(comparisons[f"{keys[1]}.{COMPARISON_SUFFIX}"][0]))
    assert _same_genotypes(second, records[1]) == {(1, 3), (2, 3)}
    assert second["label_differs"] is True  # The sample's own haplotypes are the first label, not this one
    # Neither the unlabeled example nor the example without a haplotype record is compared
    for key in keys[2:]:
        with pytest.raises(KeyError):
            comparisons[f"{key}.{COMPARISON_SUFFIX}"]

    with open(tmp_path / "images" / COMPARISONS_DIR / PROVENANCE_FILE) as file:
        provenance = yaml.safe_load(file)
    assert provenance["joint"] is True
    assert provenance["params"]["sizemin"] == ComparisonParams().sizemin

    # The extra aux tree is passed to the cooker, which must accept (and ignore) it
    dm = EnergonPackedImageDataModule(
        train_metadataset_paths=dataset_path, batch_size=64, num_workers=0, cache_workers=1, max_cache_size_gbytes=1,
        cache_dir=str(tmp_path / "cache"),
    )
    try:
        groups = [region for batch in dm.predict_dataloader() for region in batch.meta]
    finally:
        dm.teardown()
    assert len(groups) == 2  # Only the labeled examples


def test_add_comparisons_replaces_existing(tmp_path):
    records = [_haplotypes(0)]
    dataset_path, keys = _write_dataset(tmp_path / "images", records)
    assert add_comparisons(_compare_cfg(), dataset_path)
    assert add_comparisons(_compare_cfg(max_length=1000), dataset_path)

    comparisons = WebdatasetFileStore(EPath(tmp_path / "images" / COMPARISONS_DIR))
    record = json.loads(gzip.decompress(comparisons[f"{keys[0]}.{COMPARISON_SUFFIX}"][0]))
    assert record["skipped"] == "length"
    with open(dataset_path) as file:
        assert yaml.safe_load(file)["splits"]["train"]["aux"][COMPARISONS_AUX] == f"./{COMPARISONS_DIR}"


def test_add_comparisons_without_labeled_examples(tmp_path):
    dataset_path, _keys = _write_dataset(tmp_path / "images", [_haplotypes(0, true=())])
    assert not add_comparisons(_compare_cfg(), dataset_path)
    with open(dataset_path) as file:
        assert COMPARISONS_AUX not in yaml.safe_load(file)["splits"]["train"]["aux"]


def test_add_comparisons_without_haplotype_records(tmp_path):
    dataset_path, _keys = _write_dataset(tmp_path / "images", [None])
    with pytest.raises(ValueError, match="haplotype record"):
        add_comparisons(_compare_cfg(), dataset_path)
