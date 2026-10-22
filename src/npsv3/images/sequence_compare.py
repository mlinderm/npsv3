"""The optional `comparisons` aux tree: each labeled example's candidate genotypes compared with its label by Truvari-style
sequence matching (`graphs.sequence_compare.compare_genotypes`), e.g., to identify genotypes that are look-alikes of the label.

A dataset's tree is its `COMPARISONS_DIR`, beside `imgs_ds`, with one `{key}.{COMPARISON_SUFFIX}` member (gzipped JSON) per
labeled example, keyed like the example's pixels, and is referenced as the `COMPARISONS_AUX` aux tree in `dataset.yaml`. It is
written during image generation (with `compare.enabled`) or added to an existing dataset by `add_comparisons`
(`command=compare`), in both cases from the example's haplotype record (`images.haplotypes.haplotype_record`). The same
tree is also made ahead of image generation with precomputed haplotypes (see `images.haplotype_tree`), and its members are
then copied into the dataset's tree for the examples generated from those haplotypes. Unlabeled
examples have no member. The options used, including every comparison parameter, are recorded in
`{COMPARISONS_DIR}/comparison.yaml`. Needs the optional `harmonize` extra.

Member fields:
    genotype_differ: (G,) whether each genotype differs from the label, i.e., its haplotypes can't be paired with the
        label's without a Truvari-unmatched variant (False for the label)
    genotype_score: (G,) similarity of each genotype to the label (1 for the label)
    label_differs: whether the label differs from the sample's own (assembled) haplotypes, null if those are not resolved
    label_score: similarity of the label to the sample's own haplotypes, null if those are not resolved
    skipped: null, or why no comparison was made, in which case the fields above are absent: "length" if any sequence is
        longer than `compare.max_length` (alignment memory grows roughly quadratically with length)
"""

import dataclasses
import gzip
import json
import logging
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import webdataset as wds
import yaml
from omegaconf import DictConfig, OmegaConf

from npsv3.constants import COMPARISON_SUFFIX, COMPARISONS_AUX, COMPARISONS_DIR, HAPLOTYPES_DIR, HAPLOTYPES_SUFFIX
from npsv3.graphs.sequence_compare import DEFAULT_PARAMS, ComparisonParams, compare_genotypes
from npsv3.types import PathType

COMPARISON_SHARD_GLOB = "compare-*.tar"
PROVENANCE_FILE = "comparison.yaml"


@dataclass(frozen=True)
class ComparisonOptions:
    """How examples are compared (see the `compare` config)"""

    params: ComparisonParams = field(default_factory=lambda: DEFAULT_PARAMS)
    joint: bool = True
    aligner: str = "poa"
    max_length: int | None = 10_000

    @classmethod
    def from_config(cls, cfg: DictConfig | dict) -> "ComparisonOptions":
        options = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
        options.pop("enabled", None)
        params = ComparisonParams(**(options.pop("params", None) or {}))
        return cls(params=params, **options)


def comparison_record(haplotypes: dict, options: ComparisonOptions) -> dict | None:
    """The `{key}.{COMPARISON_SUFFIX}` member for an example's haplotype record, None if the example is unlabeled"""
    label = haplotypes.get("true") or []
    if not label or any(index is None for index in label):
        return None
    sample = haplotypes.get("sample_sequences")
    if sample is None or any(sequence is None for sequence in sample):
        sample = None

    length = max(len(sequence) for sequence in (haplotypes["reference"], *haplotypes["sequences"], *(sample or ())))
    if options.max_length is not None and length > options.max_length:
        return {"skipped": "length", "length": length}

    result = compare_genotypes(
        haplotypes["reference"],
        haplotypes["sequences"],
        label,
        sample,
        options.params,
        aligner=options.aligner,
        joint=options.joint,
    )
    return {
        "genotype_differ": [bool(differs) for differs in result.differs],
        "genotype_score": [round(float(score), 6) for score in result.score],
        "label_differs": result.label_differs,
        "label_score": None if result.label_score is None else round(float(result.label_score), 6),
        "skipped": None,
    }


class ComparisonWriter:
    """Writes one actor's (or task's) `comparisons` shards under `output_dir`'s `COMPARISONS_DIR`"""

    def __init__(self, index: int, output_dir: PathType, options: ComparisonOptions):
        self.options = options
        directory = Path(output_dir) / COMPARISONS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        self._writer = wds.ShardWriter(
            str(directory / f"compare-{index:04d}-%06d.tar"),
            mtime=0,  # Deterministic shards, like the pixel shards
            encoder=False,  # Members are already-encoded, gzipped bytes
            verbose=0,
        )

    def write(self, key: str, haplotypes: dict) -> bool:
        """Compare the example's genotypes and write its member, returning False if it was unlabeled (and not written)"""
        record = comparison_record(haplotypes, self.options)
        if record is None:
            return False
        self.write_member(key, gzip.compress(json.dumps(record).encode("utf-8"), mtime=0))
        return True

    def write_member(self, key: str, payload: bytes) -> None:
        """Write an already-made (gzipped) comparison member, e.g., precomputed with the same haplotypes and options"""
        self._writer.write({"__key__": key, COMPARISON_SUFFIX: payload})

    def close(self) -> None:
        self._writer.close()


def make_comparison_writer(cfg, index: int, output_dir: PathType) -> ComparisonWriter | None:
    """Create a per-actor comparison writer if `compare.enabled`, otherwise None"""
    if not OmegaConf.select(cfg, "compare.enabled", default=False):
        return None
    return ComparisonWriter(index, output_dir, ComparisonOptions.from_config(cfg.compare))


def prepare_comparison_tree(output_dir: PathType, options: ComparisonOptions, *, num_workers: int = 2) -> bool:
    """Prepare `output_dir`'s `COMPARISONS_DIR` shards as an aux tree and record `options` beside them. Returns False if no
    member was written (e.g., every example was unlabeled), in which case there is no tree to reference."""
    from npsv3.models.energon import prepare_aux_tree

    directory = Path(output_dir) / COMPARISONS_DIR
    shards = sorted(directory.glob(COMPARISON_SHARD_GLOB))
    if not any(_has_members(shard) for shard in shards):
        logging.warning("No comparisons were written to %s (no labeled examples?), so there is no comparisons tree", directory)
        return False
    prepare_aux_tree(directory, glob=COMPARISON_SHARD_GLOB, num_workers=num_workers)
    with (directory / PROVENANCE_FILE).open("w") as file:
        yaml.safe_dump(dataclasses.asdict(options), file, sort_keys=False)
    return True


def decode_comparison_record(payload: bytes | bytearray | str | dict) -> dict:
    """Decode a `{key}.{COMPARISON_SUFFIX}` member, whether raw (gzipped JSON) or already decompressed or decoded by the
    reader (energon's default decoder gunzips `.gz` members and may parse `.json`)"""
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, (bytes, bytearray)) and bytes(payload[:2]) == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    return json.loads(payload)


def load_comparison_records(directory: PathType) -> dict[str, dict]:
    """Every member of the comparisons tree in `directory`, keyed by example key, read sequentially from the shards (much
    faster than reading each key from a `FileStore` when the whole tree is needed)"""
    records = {}
    suffix = f".{COMPARISON_SUFFIX}"
    for shard in sorted(Path(directory).glob(COMPARISON_SHARD_GLOB)):
        with tarfile.open(shard) as tar:
            for member in tar:
                if member.isfile() and member.name.endswith(suffix):
                    file = tar.extractfile(member)
                    assert file is not None
                    records[member.name.removesuffix(suffix)] = decode_comparison_record(file.read())
    return records


def _has_members(shard: Path) -> bool:
    with tarfile.open(shard) as tar:
        return tar.next() is not None


def _compare_keys(
    index: int, keys: list[str], haplotypes_dir: str, output_dir: str, options: ComparisonOptions
) -> tuple[int, int]:
    """Write the comparisons for `keys` from their haplotype records in the prepared `haplotypes_dir` tree (e.g.,
    `imgs_ds`), returning (written, without a record)"""
    from megatron.energon.cache.file_store import WebdatasetFileStore
    from megatron.energon.epathlib import EPath

    from npsv3.images.haplotypes import decode_haplotype_record

    imgs = WebdatasetFileStore(EPath(haplotypes_dir))
    writer = ComparisonWriter(index, output_dir, options)
    written, missing = 0, 0
    try:
        for key in keys:
            try:
                payload, _ = imgs[f"{key}.{HAPLOTYPES_SUFFIX}"]
            except KeyError:
                missing += 1
                continue
            written += writer.write(key, decode_haplotype_record(payload))
    finally:
        writer.close()
    return written, missing


def _list_keys(directory: Path, glob: str, suffix: str) -> list[str]:
    """The keys of every `{key}.{suffix}` member of the `glob` shards in `directory`, in shard order"""
    keys = []
    for shard in sorted(directory.glob(glob)):
        with tarfile.open(shard) as tar:
            keys.extend(member.name.removesuffix(f".{suffix}") for member in tar if member.name.endswith(f".{suffix}"))
    return keys


def _write_comparisons(cfg, keys: list[str], haplotypes_dir: Path, root: Path, options: ComparisonOptions) -> bool:
    """Compare the examples with `keys` from their haplotype records in the prepared `haplotypes_dir` tree, across
    `cfg.threads` workers, into `root`'s `COMPARISONS_DIR` tree (replacing any existing tree). Returns False if there
    is no tree, i.e., there were no labeled examples."""
    import ray

    directory = root / COMPARISONS_DIR
    if directory.exists():
        logging.info("Replacing the existing comparisons in %s", directory)
        shutil.rmtree(directory)

    num_tasks = max(1, min(cfg.threads, len(keys)))
    tasks = [(index, keys[index::num_tasks], str(haplotypes_dir), str(root), options) for index in range(num_tasks)]
    logging.info("Comparing the genotypes of %d examples in %s across %d tasks", len(keys), root, num_tasks)
    if num_tasks == 1:
        counts = [_compare_keys(*task) for task in tasks]
    else:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
            shutdown = not ray.is_initialized()
            if shutdown:
                ray.init(num_cpus=cfg.threads, num_gpus=0, _temp_dir=tmp_dir, include_dashboard=False)
            try:
                compare_keys = ray.remote(_compare_keys)
                counts = ray.get([compare_keys.remote(*task) for task in tasks])
            finally:
                if shutdown:
                    ray.shutdown()

    written, missing = (sum(column) for column in zip(*counts, strict=True))
    if keys and missing == len(keys):
        msg = f"No example in {root} has a haplotype record ({HAPLOTYPES_SUFFIX}); was it generated before they were?"
        raise ValueError(msg)
    if missing:
        logging.warning("%d examples in %s have no haplotype record and were not compared", missing, root)
    logging.info("Wrote comparisons for %d of %d examples (the rest unlabeled or without a record)", written, len(keys))

    return prepare_comparison_tree(root, options, num_workers=cfg.threads)


def add_comparisons(cfg, path: PathType) -> bool:
    """Compare every labeled example with `cfg.compare` options (regardless of `compare.enabled`) across `cfg.threads`
    workers, replacing any existing tree. `path` is either:

    - The `dataset.yaml` of an energon image dataset (a `prepare_energon_dataset` tree), which is rewritten in place to
      reference the tree as the `comparisons` aux tree
    - A `command=haplotypes` output directory (see `images.haplotype_tree`), so its comparisons are reused by subsequent
      image generation with these options

    Returns:
        True if the tree was written (and referenced), False if there were no labeled examples.
    """
    from npsv3.images.haplotype_tree import HAPLOTYPE_SHARD_GLOB
    from npsv3.models.energon import _read_split, set_aux_tree

    options = ComparisonOptions.from_config(cfg.compare)
    if Path(path).is_dir():
        root = Path(path).resolve()
        haplotypes_dir = root / HAPLOTYPES_DIR
        if not haplotypes_dir.is_dir():
            msg = f"{path} is neither an image dataset's dataset.yaml nor a haplotypes directory (no {HAPLOTYPES_DIR})"
            raise ValueError(msg)
        keys = _list_keys(haplotypes_dir, HAPLOTYPE_SHARD_GLOB, HAPLOTYPES_SUFFIX)
        return _write_comparisons(cfg, keys, haplotypes_dir, root, options)

    root, split = _read_split(path)
    root = root.resolve()
    with (root / split["path"]).open() as meta_file:
        keys = [json.loads(line)["key"] for line in meta_file if line.strip()]
    if not _write_comparisons(cfg, keys, root / split["aux"]["imgs"], root, options):
        return False
    set_aux_tree(path, COMPARISONS_AUX, f"./{COMPARISONS_DIR}")
    return True
