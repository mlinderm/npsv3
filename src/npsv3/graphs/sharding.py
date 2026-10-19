"""Graph shard layout shared by the `graphs` command (writer), image generation and other readers.

Graph shards are plain (uncompressed) tars with individually gzipped members, keyed by `graph_key(region)`, so that a
directory of shards can be prepared as an energon dataset and attached to image datasets as the `graphs` aux tree. An
image row's `graph_key` then addresses its region's members, e.g. `f"{graph_key}.{GRAPH_MEMBER}.gz"`. Legacy shards
(`graphs-*.tar.gz`, whole-file gzip with uncompressed members) are still readable with `decode_graph_record`, but cannot
be prepared as an aux tree.
"""

import gzip
import hashlib
import logging
import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

from npsv3.types import PathType
from npsv3.util.range import Range

GRAPH_MEMBER = "graph.bytes"
UNIQUE_KMERS_MEMBER = "unique_kmer_overlay.bytes"
HAPLOTYPE_PRIOR_MEMBER = "haplotype_prior_overlay.bytes"
REGION_MEMBER = "region.txt"
COMPRESSED_MEMBERS = (GRAPH_MEMBER, UNIQUE_KMERS_MEMBER, HAPLOTYPE_PRIOR_MEMBER)

SHARD_PATTERN = "graphs-{index:05d}-%05d.tar"
SHARD_GLOB = "graphs-*.tar"

# Graph bytes compress ~3x and the overlays ~7-9x at level 1, at a few ms per region
_COMPRESS_LEVEL = 1
_GZIP_MAGIC = b"\x1f\x8b"


def graph_key(region: Range) -> str:
    """Return the dot-free shard key for a region's graph.

    WebDataset (and energon) split a member's key from its extension at the *first* dot, so a dot in the key (hg38
    alt/decoy contigs like `GL000207.1`) would silently truncate it. Dots become underscores.
    """
    key = region.slug.replace(".", "_")
    assert "." not in key, f"Graph key {key!r} must not contain a dot"
    return key


def graph_digest(graph_bytes: bytes | bytearray | memoryview) -> str:
    """Return the digest of a serialized (uncompressed) graph"""
    return hashlib.blake2b(graph_bytes, digest_size=16).hexdigest()


def encode_graph_record(region: Range, members: Mapping[str, bytes | bytearray | memoryview]) -> dict:
    """Return a ShardWriter record (`encoder=False`) for `region` with each of `members` gzipped as `{name}.gz`"""
    record = {"__key__": graph_key(region), REGION_MEMBER: str(region).encode("utf-8")}
    for name, payload in members.items():
        assert name in COMPRESSED_MEMBERS, f"Unexpected graph shard member {name}"
        record[f"{name}.gz"] = gzip.compress(payload, compresslevel=_COMPRESS_LEVEL, mtime=0)
    return record


def decode_graph_record(record: Mapping) -> dict[str, bytes]:
    """Return {member: uncompressed bytes} for a graph shard record in either the current or the legacy layout.

    Also accepts members that a reader already decompressed (e.g. energon's default decoder gunzips `.gz` members).
    """
    members = {}
    for name in COMPRESSED_MEMBERS:
        if (payload := record.get(f"{name}.gz")) is not None:
            members[name] = gzip.decompress(payload) if bytes(payload[:2]) == _GZIP_MAGIC else payload
        elif (payload := record.get(name)) is not None:
            members[name] = payload
    return members


def split_graph_shards(graph_shards: Sequence[PathType], num_workers: int) -> list[tuple[str, int, int]]:
    """Return `(shard_path, split, num_splits)` tasks covering every record in `graph_shards` across at least
    `num_workers` tasks.
     
    Ensure parallelism isn't limited by the shard count even when there are fewer shards than workers).
    """
    if not graph_shards:
        return []
    num_splits = max(1, math.ceil(num_workers / len(graph_shards)))
    return [(str(shard), split, num_splits) for shard in graph_shards for split in range(num_splits)]


def iter_graph_records(shard_path: PathType, split: int = 0, num_splits: int = 1) -> Iterator[dict]:
    """Yield the raw (undecoded) records of a graph shard at `split` stride (i.e., every `num_splits`-th record).

    Every split reads the whole shard sequentially, but only its own records are yielded to be decoded and processed.
    """
    import webdataset as wds

    assert 0 <= split < num_splits, f"Invalid split {split} of {num_splits}"
    for index, record in enumerate(wds.WebDataset([str(shard_path)], shardshuffle=False)):  # type: ignore
        if index % num_splits == split:
            yield record


def prepare_graph_tree(directory: PathType, *, num_workers: int = 2) -> bool:
    """Prepare `directory`'s `graphs-*.tar` shards as an energon dataset so it can be an aux tree. Returns False if there
    are no shards in the current layout (e.g. only legacy `graphs-*.tar.gz` shards)."""
    from npsv3.models.energon import prepare_aux_tree

    if not any(Path(directory).glob(SHARD_GLOB)):
        return False
    prepare_aux_tree(directory, glob=SHARD_GLOB, num_workers=num_workers)
    return True


def graph_aux_directory(graph_shards: Sequence[PathType]) -> str | None:
    """Return the prepared graph tree containing all of `graph_shards`, or None if they are not all in one prepared tree."""
    from megatron.energon.flavors.webdataset.config import INDEX_SQLITE_FILENAME, MAIN_FOLDER_NAME

    directories = {Path(shard).resolve().parent for shard in graph_shards}
    if len(directories) != 1:
        logging.warning("Graph shards span %d directories, so no graphs aux tree is attached", len(directories))
        return None
    (directory,) = directories
    if not (directory / MAIN_FOLDER_NAME / INDEX_SQLITE_FILENAME).is_file():
        logging.warning(
            "Graph shards in %s are not a prepared energon tree (e.g. legacy graphs-*.tar.gz), so no graphs aux tree "
            "is attached", directory,
        )
        return None
    return str(directory)
