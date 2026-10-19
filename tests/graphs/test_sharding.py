import gzip
import os

import numpy as np
import pytest
import webdataset as wds
import yaml
from megatron.energon.cache.file_store import WebdatasetFileStore
from megatron.energon.epathlib import EPath

from npsv3.constants import HAPLOTYPES_SUFFIX
from npsv3.graphs.sharding import (
    GRAPH_MEMBER,
    HAPLOTYPE_PRIOR_MEMBER,
    UNIQUE_KMERS_MEMBER,
    decode_graph_record,
    encode_graph_record,
    graph_aux_directory,
    graph_digest,
    graph_key,
    iter_graph_records,
    prepare_graph_tree,
    split_graph_shards,
)
from npsv3.images.example import EnergonExampleWriter
from npsv3.images.haplotypes import decode_haplotype_record
from npsv3.models.energon import EnergonPackedImageDataModule, prepare_energon_dataset
from npsv3.util.range import Range

NUM_GENOTYPES = 3
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 8, 12, 3


def _members(index: int) -> dict[str, bytes]:
    return {
        GRAPH_MEMBER: f"graph {index}".encode() * 50,
        UNIQUE_KMERS_MEMBER: f"kmers {index}".encode() * 50,
        HAPLOTYPE_PRIOR_MEMBER: f"prior {index}".encode() * 50,
    }


def _regions(num_regions: int) -> list[Range]:
    # Includes a contig with a dot, which must not truncate the key
    return [Range(f"{'chr1' if i % 2 else 'GL000207.1'}:{i * 1000 + 1}-{i * 1000 + 500}") for i in range(num_regions)]


def _write_graph_shards(directory, regions, *, num_shards=2) -> list[str]:
    writers = [wds.ShardWriter(os.path.join(directory, f"graphs-{i:05d}-%05d.tar"), verbose=False, encoder=False)
               for i in range(num_shards)]
    for index, region in enumerate(regions):
        writers[index % num_shards].write(encode_graph_record(region, _members(index)))
    for writer in writers:
        writer.close()
    return sorted(str(path) for path in directory.glob("graphs-*.tar"))


def test_graph_key():
    assert graph_key(Range("chr1:1001-1500")) == Range("chr1:1001-1500").slug
    assert "." not in graph_key(Range("GL000207.1:1001-1500"))


def test_record_round_trip_and_legacy_layout():
    region = Range("chr1:1001-1500")
    record = encode_graph_record(region, _members(0))
    assert record["__key__"] == graph_key(region)
    assert all(f"{name}.gz" in record for name in _members(0))
    assert decode_graph_record(record) == _members(0)

    # Legacy shards stored members uncompressed under their plain names
    assert decode_graph_record(_members(1)) == _members(1)
    # Readers may already have gunzipped `.gz` members
    assert decode_graph_record({f"{name}.gz": payload for name, payload in _members(2).items()}) == _members(2)

    assert graph_digest(_members(0)[GRAPH_MEMBER]) != graph_digest(_members(1)[GRAPH_MEMBER])


@pytest.mark.parametrize("num_workers", [1, 2, 3, 7])
def test_split_graph_shards_cover_every_record_once(tmp_path, num_workers):
    regions = _regions(9)
    shards = _write_graph_shards(tmp_path, regions)
    tasks = split_graph_shards(shards, num_workers)
    assert len(tasks) >= min(num_workers, len(regions))
    if num_workers <= len(shards):
        assert tasks == [(shard, 0, 1) for shard in shards]

    regions_read = [record["region.txt"].decode() for task in tasks for record in iter_graph_records(*task)]
    assert sorted(regions_read) == sorted(str(region) for region in regions)


def test_split_graph_shards_without_shards():
    assert split_graph_shards([], 4) == []


def test_prepared_graph_tree_is_addressable_by_key(tmp_path):
    regions = _regions(6)
    shards = _write_graph_shards(tmp_path, regions)
    # Not yet prepared, so not usable as an aux tree
    assert graph_aux_directory(shards) is None

    assert prepare_graph_tree(tmp_path)
    assert graph_aux_directory(shards) == str(tmp_path.resolve())

    store = WebdatasetFileStore(EPath(tmp_path))
    for index, region in enumerate(regions):
        payload, _ = store[f"{graph_key(region)}.{GRAPH_MEMBER}.gz"]
        assert decode_graph_record({f"{GRAPH_MEMBER}.gz": payload})[GRAPH_MEMBER] == _members(index)[GRAPH_MEMBER]

    # The same shards are still readable sequentially (as by `GraphWriter.from_shard`)
    regions_read = [record["region.txt"].decode() for record in wds.WebDataset(shards, shardshuffle=False)]
    assert sorted(regions_read) == sorted(str(region) for region in regions)


def test_graph_aux_directory_requires_one_tree(tmp_path):
    shards = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        shards.extend(_write_graph_shards(tmp_path / name, _regions(2)))
        prepare_graph_tree(tmp_path / name)
    assert graph_aux_directory(shards) is None


def test_prepare_graph_tree_without_current_shards(tmp_path):
    (tmp_path / "graphs-00000-00000.tar.gz").write_bytes(b"")
    assert not prepare_graph_tree(tmp_path)


@pytest.fixture
def linked_dataset(tmp_path):
    """An image dataset whose rows link to a prepared graph tree, with a haplotypes member per example"""
    (graphs_dir := tmp_path / "graphs").mkdir()
    regions = _regions(4)
    _write_graph_shards(graphs_dir, regions)
    prepare_graph_tree(graphs_dir)

    rng = np.random.default_rng(0)
    output_dir = tmp_path / "images"
    writer = EnergonExampleWriter(0, str(output_dir))
    try:
        for index, region in enumerate(regions):
            writer.write(
                {
                    "region": str(region),
                    "image": rng.integers(0, 255, size=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8),
                    "label": 1,
                    "label.rank": [0, 1, 0],
                    "num_genotypes": NUM_GENOTYPES,
                    "sim.images": rng.integers(
                        0, 255, size=(2, NUM_GENOTYPES, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8
                    ),
                    "haplotypes": {"paths": [[1, 2], [1, 3]], "true": [0, 1], "sequences": ["AC", f"AG{index}"]},
                    "graph_key": graph_key(region),
                    "graph_digest": graph_digest(_members(index)[GRAPH_MEMBER]),
                },
                region=region,
                sample_name="S0",
            )
    finally:
        writer.close()
    dataset_path = prepare_energon_dataset(str(output_dir), aux={"graphs": str(graphs_dir)})
    return dataset_path, regions


def test_linked_dataset(linked_dataset, tmp_path):
    dataset_path, regions = linked_dataset
    with open(dataset_path) as file:
        aux = yaml.safe_load(file)["splits"]["train"]["aux"]
    assert aux == {"imgs": "./imgs_ds", "graphs": str((tmp_path / "graphs").resolve())}

    imgs = WebdatasetFileStore(EPath(tmp_path / "images" / "imgs_ds"))
    graphs = WebdatasetFileStore(EPath(aux["graphs"]))
    with open(tmp_path / "images" / "meta_ds" / "train.jsonl") as file:
        rows = [yaml.safe_load(line) for line in file]
    assert len(rows) == len(regions)
    for index, row in enumerate(rows):
        haplotypes = decode_haplotype_record(imgs[f"{row['key']}.{HAPLOTYPES_SUFFIX}"][0])
        assert haplotypes["sequences"] == ["AC", f"AG{index}"]

        graph_bytes = gzip.decompress(graphs[f"{row['graph_key']}.{GRAPH_MEMBER}.gz"][0])
        assert graph_digest(graph_bytes) == row["graph_digest"]


def test_linked_dataset_loads_for_training(linked_dataset, tmp_path):
    """The `graphs` aux tree is passed to the cooker too, which must accept (and ignore) it"""
    dataset_path, regions = linked_dataset
    dm = EnergonPackedImageDataModule(
        train_metadataset_paths=dataset_path, batch_size=64, num_workers=0, cache_workers=1, max_cache_size_gbytes=1,
        cache_dir=str(tmp_path / "cache"),
    )
    try:
        groups = [region for batch in dm.predict_dataloader() for region in batch.meta]
    finally:
        dm.teardown()
    assert sorted(groups) == sorted(str(region) for region in regions)


def test_decode_haplotype_record():
    record = {"paths": [[1]], "true": [None, None]}
    raw = gzip.compress(b'{"paths": [[1]], "true": [null, null]}')
    assert decode_haplotype_record(raw) == record
    assert decode_haplotype_record(gzip.decompress(raw)) == record
    assert decode_haplotype_record(record) == record
