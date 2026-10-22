"""Precompute haplotypes (and comparisons) for a sample to be reused by every image generation run for that 
sample (`haplotypes_dir`).

A sample's haplotypes deterministically depend only on its graphs, its k-mer counts and the `haplotype.*` (and some
`graph.*`) settings, not on how images are rendered, and so they can be generated ahead of time. The output directory
contains:

    HAPLOTYPES_DIR/haplotypes-*.tar: One `{key}.{HAPLOTYPES_SUFFIX}` member per region, keyed like the region's example
        (`images.example.energon_key`), containing its `images.haplotypes.haplotype_record` plus `source_digest`, the
        digest of the graph shard members it was generated from. Regions without enough haplotypes have a
        `{"skipped": ...}` record instead, so image generation skips them without regenerating.
    HAPLOTYPES_DIR/haplotypes.yaml: The settings the haplotypes were generated with (see `haplotype_options`)
    COMPARISONS_DIR: The `comparisons` tree for the haplotypes (if `compare.enabled`), as in an image dataset
    filtered_kmers.kmc_*: The sample's filtered k-mers (if computed rather than provided)
    graphs_ds: Graphs generated on-the-fly (if graphs weren't provided)

Image generation uses a region's record if its `source_digest` matches the region's graph, otherwise it generates the
haplotypes (and comparison) inline, as without precomputed haplotypes.
"""

import gzip
import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path

import webdataset as wds
import yaml
from omegaconf import OmegaConf

from npsv3.constants import COMPARISON_SUFFIX, COMPARISONS_DIR, HAPLOTYPES_DIR, HAPLOTYPES_SUFFIX
from npsv3.graphs.sharding import COMPRESSED_MEMBERS
from npsv3.types import PathType
from npsv3.util.sample import Sample

HAPLOTYPE_SHARD_GLOB = "haplotypes-*.tar"
PROVENANCE_FILE = "haplotypes.yaml"
FILTERED_KMERS = "filtered_kmers"

_GRAPH_KEYS = ("kmer_size", "population_excluded_samples", "filtered_as_missing")
"""The `graph.*` settings, used both in graph construction and haplotype generation, that determine a sample's haplotypes,
given its graphs"""

_HAPLOTYPE_KEYS = ("strategy", "max_haplotypes", "population_haplotype_topup", "sampler_params", "missing_are_ref")
"""The `haplotype.*` settings that determine a sample's haplotypes, given its graphs (i.e., all but `max_diplotypes`)"""

_GZIP_MAGIC = b"\x1f\x8b"


def haplotype_options(cfg, sample: Sample, *, ploidy: int, exact_reference: bool, min_variant_size: int) -> dict:
    """The settings that determine a sample's haplotypes, given its graphs (whose identity is checked per-region with the
    record's `source_digest`), as plain YAML-serializable data"""
    graph = {key: OmegaConf.select(cfg, f"graph.{key}", default=None) for key in _GRAPH_KEYS}
    haplotype = {key: OmegaConf.select(cfg, f"haplotype.{key}", default=None) for key in _HAPLOTYPE_KEYS}
    options = {
        "sample": sample.name,
        "kmer_coverage": sample.kmer_coverage,
        "ploidy": ploidy,
        "exact_reference": exact_reference,
        "min_variant_size": min_variant_size,
        "graph": OmegaConf.to_container(OmegaConf.create(graph), resolve=True),
        "haplotype": OmegaConf.to_container(OmegaConf.create(haplotype), resolve=True),
    }
    # Round-trip so options are compared exactly as they would be after being recorded
    return yaml.safe_load(yaml.safe_dump(options))


def source_digest(members: Mapping[str, bytes]) -> str:
    """The digest of every (uncompressed) graph shard member that haplotype generation depends on, i.e., the graph and
    its unique k-mer and (if present) population prior overlays"""
    digest = hashlib.blake2b(digest_size=16)
    for name in COMPRESSED_MEMBERS:
        if (payload := members.get(name)) is not None:
            digest.update(name.encode("utf-8"))
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
    return digest.hexdigest()


def _encode(record: dict) -> bytes:
    return gzip.compress(json.dumps(record).encode("utf-8"), mtime=0)


def _gzipped(payload: str | bytes | bytearray) -> bytes:
    """`payload` as gzipped bytes, whether or not the reader already decompressed (or decoded) it"""
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return bytes(payload) if bytes(payload[:2]) == _GZIP_MAGIC else gzip.compress(bytes(payload), mtime=0)


class HaplotypeTreeWriter:
    """Writes one actor's `HAPLOTYPES_DIR` shards under `output_dir`"""

    def __init__(self, index: int, output_dir: PathType):
        directory = Path(output_dir) / HAPLOTYPES_DIR
        directory.mkdir(parents=True, exist_ok=True)
        self._writer = wds.ShardWriter( # type: ignore
            str(directory / f"haplotypes-{index:04d}-%06d.tar"),
            mtime=0,  # Deterministic shards, like the pixel shards
            encoder=False,  # Members are already-encoded, gzipped bytes
            verbose=0,
        )

    def write(self, key: str, haplotypes: dict, *, digest: str) -> None:
        """Write the region's haplotype record, generated from graph members with `source_digest` `digest`"""
        self._writer.write({"__key__": key, HAPLOTYPES_SUFFIX: _encode({**haplotypes, "source_digest": digest})})

    def write_skipped(self, key: str, reason: str, *, digest: str) -> None:
        """Record that the region has no haplotype record (e.g., too few haplotypes), so image generation skips it too"""
        self._writer.write({"__key__": key, HAPLOTYPES_SUFFIX: _encode({"skipped": reason, "source_digest": digest})})

    def close(self) -> None:
        self._writer.close()


def prepare_haplotype_tree(
    output_dir: PathType, options: dict, *, filtered_kmer_path: PathType | None = None, num_workers: int = 2
) -> None:
    """Prepare `output_dir`'s `HAPLOTYPES_DIR` shards for random access by key and record `options` (and the filtered
    k-mer database, relative to `output_dir` if within it) beside them"""
    from npsv3.models.energon import prepare_aux_tree

    directory = Path(output_dir) / HAPLOTYPES_DIR
    prepare_aux_tree(directory, glob=HAPLOTYPE_SHARD_GLOB, num_workers=num_workers)

    provenance = dict(options)
    if filtered_kmer_path is not None:
        filtered_kmer_path = Path(filtered_kmer_path).resolve()
        root = Path(output_dir).resolve()
        provenance["filtered_kmers"] = str(
            filtered_kmer_path.relative_to(root) if filtered_kmer_path.is_relative_to(root) else filtered_kmer_path
        )
    with (directory / PROVENANCE_FILE).open("w") as file:
        yaml.safe_dump(provenance, file, sort_keys=False)


class HaplotypeTree:
    """Random access, by example key, to a `command=haplotypes` output directory's records (and comparisons)"""

    def __init__(self, directory: PathType):
        self.directory = Path(directory)
        provenance_path = self.directory / HAPLOTYPES_DIR / PROVENANCE_FILE
        if not provenance_path.is_file():
            msg = f"{directory} is not a prepared haplotypes directory (no {HAPLOTYPES_DIR}/{PROVENANCE_FILE})"
            raise ValueError(msg)
        with provenance_path.open() as file:
            self.provenance: dict = yaml.safe_load(file)
        self._haplotypes = None
        self._comparisons = None

    @property
    def options(self) -> dict:
        return {key: value for key, value in self.provenance.items() if key != "filtered_kmers"}

    @property
    def filtered_kmer_path(self) -> str | None:
        """The sample's filtered k-mer database the haplotypes were generated with, if it still exists"""
        if (path := self.provenance.get("filtered_kmers")) is None:
            return None
        path = self.directory / path  # Absolute paths are unchanged
        return str(path) if path.with_name(f"{path.name}.kmc_pre").is_file() else None

    def check(self, options: dict) -> None:
        """Raise ValueError if the haplotypes were generated with different `options` (see `haplotype_options`)"""
        recorded = self.options
        if recorded != options:
            differ = sorted(
                key for key in recorded.keys() | options.keys() if recorded.get(key) != options.get(key)
            )
            msg = (
                f"Haplotypes in {self.directory} were generated with different settings ({', '.join(differ)}): "
                f"{ {key: recorded.get(key) for key in differ} } vs. the current { {key: options.get(key) for key in differ} }"
            )
            raise ValueError(msg)

    def comparison_options(self):
        """The options the tree's comparisons were made with, None if there are none"""
        from npsv3.images.sequence_compare import PROVENANCE_FILE as COMPARISON_PROVENANCE_FILE
        from npsv3.images.sequence_compare import ComparisonOptions

        path = self.directory / COMPARISONS_DIR / COMPARISON_PROVENANCE_FILE
        if not path.is_file():
            return None
        with path.open() as file:
            return ComparisonOptions.from_config(yaml.safe_load(file))

    def _store(self, directory: Path):
        from megatron.energon.cache.file_store import WebdatasetFileStore
        from megatron.energon.epathlib import EPath

        return WebdatasetFileStore(EPath(str(directory)))

    def lookup(self, key: str, digest: str) -> dict | None:
        """The region's haplotype record (or `{"skipped": reason}` record), None if there is none or it was generated
        from different graph members (`digest` is the region's current `source_digest`)"""
        from npsv3.images.haplotypes import decode_haplotype_record

        if self._haplotypes is None:
            self._haplotypes = self._store(self.directory / HAPLOTYPES_DIR)
        try:
            payload, _ = self._haplotypes[f"{key}.{HAPLOTYPES_SUFFIX}"]
        except KeyError:
            return None
        record = decode_haplotype_record(payload)
        if record.pop("source_digest", None) != digest:
            logging.debug("Precomputed haplotypes for %s were generated from a different graph", key)
            return None
        return record

    def comparison(self, key: str) -> bytes | None:
        """The region's gzipped comparison member, None if there is none (e.g., unlabeled or no comparisons tree)"""
        if self._comparisons is None:
            directory = self.directory / COMPARISONS_DIR
            if not directory.is_dir():
                return None
            self._comparisons = self._store(directory)
        try:
            payload, _ = self._comparisons[f"{key}.{COMPARISON_SUFFIX}"]
        except KeyError:
            return None
        return _gzipped(payload)
