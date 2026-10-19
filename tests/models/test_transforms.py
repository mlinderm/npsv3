import json
from pathlib import Path

import hydra
import numpy as np
import pytest
import yaml

from npsv3.images.example import EnergonExampleWriter
from npsv3.models import runners
from npsv3.models.energon import prepare_energon_dataset
from npsv3.models.transforms import (
    copositive_annotations,
    copositive_conf,
    sim_confusion,
    write_annotated_dataset,
)
from npsv3.util.range import Range

CFG_OVERRIDES = (
    "model=paired_packed_inception_infonce",
    "pileup=unphased_graph_kmer",
    "simulation.replicates=2",
)

NUM_REPLICATES = 2
NUM_GENOTYPES = 6  # 3 haplotypes: (0,0) (0,1) (0,2) (1,1) (1,2) (2,2)


def _logits_from_embeddings(embeddings: np.ndarray, query: np.ndarray, tau: float = 0.1):
    """Cosine logits as the model computes them, from replicate-major (R*G, D) support embeddings"""
    support = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    query = query / np.linalg.norm(query)
    return (support @ query / tau).reshape(NUM_REPLICATES, -1), support @ support.T / tau


def _planted_embeddings(rng, lookalikes: dict[int, int], noise=0.01, dim=16):
    """Orthogonal genotype centers plus small replicate noise; genotype g in `lookalikes` shares lookalikes[g]'s center"""
    centers = np.eye(NUM_GENOTYPES, dim)
    for genotype, other in lookalikes.items():
        centers[genotype] = centers[other]
    return np.concatenate([centers + noise * rng.standard_normal(centers.shape) for _ in range(NUM_REPLICATES)])


class TestCopositiveMath:
    def test_planted_lookalike(self):
        rng = np.random.default_rng(0)
        label = 3  # (1,1)
        embeddings = _planted_embeddings(rng, {5: label})  # (2,2) is indistinguishable from (1,1)
        query_logits, support_logits = _logits_from_embeddings(embeddings, embeddings[label])

        conf = copositive_conf(sim_confusion(support_logits, NUM_REPLICATES, NUM_GENOTYPES), label)
        assert conf[5] == pytest.approx(1.0, abs=0.05)
        assert np.all(np.delete(conf, [label, 5]) < 0.01)

        annotations = copositive_annotations(query_logits, support_logits, NUM_REPLICATES, NUM_GENOTYPES, label)
        assert set(annotations) == {"genotype_weights", "genotype_probs"}
        np.testing.assert_allclose(annotations["genotype_weights"], conf, atol=1e-6)
        assert annotations["genotype_weights"][label] == 1.0
        assert sum(annotations["genotype_probs"]) == pytest.approx(1.0, abs=1e-5)
        # The real query matches both the label and its look-alike equally
        probs = annotations["genotype_probs"]
        assert probs[label] == pytest.approx(probs[5], abs=0.05)
        assert probs[label] + probs[5] > 0.99

    def test_presence_changing_lookalike_is_not_masked(self):
        """Whether SV-presence-changing look-alikes are masked is a downstream decision, so their weight is kept"""
        rng = np.random.default_rng(1)
        label = 0  # (0,0), no SV
        embeddings = _planted_embeddings(rng, {1: label})  # (0,1) looks like no SV
        query_logits, support_logits = _logits_from_embeddings(embeddings, embeddings[label])

        annotations = copositive_annotations(query_logits, support_logits, NUM_REPLICATES, NUM_GENOTYPES, label)
        assert annotations["genotype_weights"][1] == pytest.approx(1.0, abs=0.05)

    def test_separable_real_query(self):
        rng = np.random.default_rng(3)
        label = 3  # (1,1)
        embeddings = _planted_embeddings(rng, {})
        # The real image looks exactly like (0,0), which simulation separates from the label
        query_logits, support_logits = _logits_from_embeddings(embeddings, embeddings[0])

        annotations = copositive_annotations(query_logits, support_logits, NUM_REPLICATES, NUM_GENOTYPES, label)
        assert np.argmax(annotations["genotype_probs"]) == 0
        assert annotations["genotype_weights"][0] < 0.01

    def test_single_replicate_is_rejected(self):
        logits = np.zeros((1, NUM_GENOTYPES))
        with pytest.raises(ValueError, match=">= 2 simulated replicates"):
            copositive_annotations(logits, np.zeros((NUM_GENOTYPES, NUM_GENOTYPES)), 1, NUM_GENOTYPES, 0)


def _build_dataset(output_dir, *, num_groups, seed, height, width, channels, num_unlabeled=0):
    """The last `num_unlabeled` groups have no label, so the model does not score them"""
    rng = np.random.default_rng(seed)
    writer = EnergonExampleWriter(0, output_dir)
    try:
        for index in range(num_groups):
            if index < num_groups - num_unlabeled:
                label = int(rng.integers(0, NUM_GENOTYPES))
                label_rank = np.zeros(NUM_GENOTYPES, dtype=np.int64)
                label_rank[label] = 1
            else:
                label = label_rank = None
            region = f"chr1:{index * 1000 + 1}-{index * 1000 + 500}"
            writer.write(
                {
                    "region": region,
                    "image": rng.integers(0, 255, size=(height, width, channels), dtype=np.uint8),
                    "label": label,
                    "label.rank": label_rank,
                    "num_genotypes": NUM_GENOTYPES,
                    "sim.images": rng.integers(
                        0, 255, size=(NUM_REPLICATES, NUM_GENOTYPES, height, width, channels), dtype=np.uint8
                    ),
                },
                region=Range(region),
                sample_name="S0",
            )
    finally:
        writer.close()
    return prepare_energon_dataset(output_dir)


def _read_jsonl(path):
    with open(path) as file:
        return [json.loads(line) for line in file if line.strip()]


def test_write_annotated_dataset_reuses_images(tmp_path):
    source = _build_dataset(str(tmp_path / "source"), num_groups=3, seed=0, height=4, width=8, channels=2)
    source_rows = _read_jsonl(tmp_path / "source" / "meta_ds" / "train.jsonl")

    # The second region was not scored (e.g. rejected by the data module), so is copied unchanged
    annotations = {source_rows[0]["region"]: {"genotype_probs": [1.0]}, source_rows[2]["region"]: {"genotype_probs": [0.5]}}
    dataset_path = write_annotated_dataset(source, tmp_path / "out", annotations)

    with open(dataset_path) as file:
        split = yaml.safe_load(file)["splits"]["train"]
    assert split["path"] == "./meta_ds/train.jsonl"
    assert split["aux"]["imgs"] == str((tmp_path / "source" / "imgs_ds").resolve())
    assert not (tmp_path / "out" / "imgs_ds").exists()

    rows = _read_jsonl(tmp_path / "out" / "meta_ds" / "train.jsonl")
    assert rows == [{**source_rows[0], "genotype_probs": [1.0]}, source_rows[1], {**source_rows[2], "genotype_probs": [0.5]}]

    with pytest.raises(ValueError, match="overwrite"):
        write_annotated_dataset(source, tmp_path / "source", annotations)


@pytest.mark.slow
@pytest.mark.cfg_overrides(*CFG_OVERRIDES)
def test_copositive_transform(hydra_setup, tmp_path):  # noqa: ARG001
    from npsv3.models.energon import EnergonPackedImageDataModule
    from npsv3.models.transforms import copositive_transform

    base_cfg = hydra.compose(config_name="config", overrides=list(CFG_OVERRIDES))
    geometry = {
        "height": base_cfg.pileup.image_height,
        "width": base_cfg.pileup.image_width,
        "channels": len(base_cfg.pileup.image_channels),
    }
    num_groups, num_unlabeled = 12, 2
    source = _build_dataset(
        str(tmp_path / "source"), num_groups=num_groups, num_unlabeled=num_unlabeled, seed=20261010, **geometry
    )

    data_overrides = [
        "data.batch_size=32",
        f"data.g_max={NUM_GENOTYPES}",
        "data.packing_buffer_size=16",
        "data.shuffle_buffer_size=32",
        "data.num_workers=0",
        "data.cache_workers=1",
        "data.max_cache_size_gbytes=1",
        "+trainer.accelerator=cpu",
    ]
    train_cfg = hydra.compose(
        config_name="config",
        overrides=[
            *CFG_OVERRIDES, *data_overrides, f"data.metadataset_path={source}", "data.limit_train_batches=1",
            "trainer.max_epochs=1",
        ],
    )
    checkpoint_path = runners.train(train_cfg, output_dir=str(tmp_path / "model"))

    cfg = hydra.compose(
        config_name="config",
        overrides=[
            *CFG_OVERRIDES, *data_overrides, "command=copositive", f"model.checkpoint='{checkpoint_path}'",
            f"input={source}",
        ],
    )
    assert cfg.data.r_max is None, "The command must not reject groups by `simulation.replicates`"
    output = tmp_path / "copositive"
    dataset_path = copositive_transform(cfg, source, str(output))
    assert dataset_path == str(output / "dataset.yaml")
    assert yaml.safe_load((output / "copositive.yaml").read_text())["copositive"] == {"temperature": None}

    source_rows = {row["region"]: row for row in _read_jsonl(Path(source).parent / "meta_ds" / "train.jsonl")}
    rows = _read_jsonl(output / "meta_ds" / "train.jsonl")
    assert [row["region"] for row in rows] == list(source_rows), "Every source row, in the source's order"
    for row in rows:
        source_row = source_rows[row["region"]]
        if source_row["label"] is None:
            assert row == source_row, "Unscored rows are copied unchanged"
            continue
        assert set(row) - set(source_row) == {"genotype_weights", "genotype_probs"}
        assert {key: row[key] for key in source_row} == source_row
        assert len(row["genotype_weights"]) == len(row["genotype_probs"]) == NUM_GENOTYPES
        assert row["genotype_weights"][row["label"]] == 1.0
        assert all(0.0 <= weight <= 1.0 for weight in row["genotype_weights"])
        assert sum(row["genotype_probs"]) == pytest.approx(1.0, abs=1e-4)

    # The new dataset loads, with the source's images, through the training data module
    dm = EnergonPackedImageDataModule(
        train_metadataset_paths=dataset_path, batch_size=32, g_max=NUM_GENOTYPES, num_workers=0, cache_workers=1,
        max_cache_size_gbytes=1, packing_buffer_size=16, cache_dir=str(tmp_path / "cache"),
    )
    try:
        groups = sum(len(batch.meta) for batch in dm.predict_dataloader())
    finally:
        # Close the cache pool, whose fork hooks otherwise trip on later tests' forks
        dm.teardown()
    assert groups == num_groups - num_unlabeled
