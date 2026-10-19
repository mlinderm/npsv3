"""The optional `comparisons` aux tree: each labeled example's candidate genotypes compared with its label by Truvari-style
sequence matching (`graphs.sequence_compare.compare_genotypes`), e.g., to identify genotypes that are look-alikes of the label.

A dataset's tree is its `COMPARISONS_DIR`, beside `imgs_ds`, with one `{key}.{COMPARISON_SUFFIX}` member (gzipped JSON) per
labeled example, keyed like the example's pixels, and is referenced as the `COMPARISONS_AUX` aux tree in `dataset.yaml`. It is
written during image generation (with `compare.enabled`) or added to an existing dataset by `add_comparisons`
(`command=compare`), in both cases from the example's haplotype record (`images.haplotypes.haplotype_record`). Unlabeled
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

from npsv3.constants import COMPARISON_SUFFIX, COMPARISONS_AUX, COMPARISONS_DIR, HAPLOTYPES_SUFFIX
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
        self._writer.write({"__key__": key, COMPARISON_SUFFIX: gzip.compress(json.dumps(record).encode("utf-8"), mtime=0)})
        return True

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


def _has_members(shard: Path) -> bool:
    with tarfile.open(shard) as tar:
        return tar.next() is not None


def _compare_keys(
    index: int, keys: list[str], imgs_dir: str, output_dir: str, options: ComparisonOptions
) -> tuple[int, int]:
    """Write the comparisons for `keys` from their haplotype records in `imgs_dir`, returning (written, without a record)"""
    from megatron.energon.cache.file_store import WebdatasetFileStore
    from megatron.energon.epathlib import EPath

    from npsv3.images.haplotypes import decode_haplotype_record

    imgs = WebdatasetFileStore(EPath(imgs_dir))
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


def add_comparisons(cfg, dataset_path: PathType) -> bool:
    """Compare every labeled example in the energon image dataset `dataset_path` (a `prepare_energon_dataset` tree) with
    `cfg.compare` options (regardless of `compare.enabled`) across `cfg.threads` workers, replacing any existing tree, and
    reference it as the `comparisons` aux tree by rewriting `dataset_path` in place.

    Returns:
        True if the tree was written and referenced, False if there were no labeled examples.
    """
    import ray

    from npsv3.models.energon import _read_split, set_aux_tree

    options = ComparisonOptions.from_config(cfg.compare)
    root, split = _read_split(dataset_path)
    root = root.resolve()
    imgs_dir = str(root / split["aux"]["imgs"])
    with (root / split["path"]).open() as meta_file:
        keys = [json.loads(line)["key"] for line in meta_file if line.strip()]

    directory = root / COMPARISONS_DIR
    if directory.exists():
        logging.info("Replacing the existing comparisons in %s", directory)
        shutil.rmtree(directory)

    num_tasks = max(1, min(cfg.threads, len(keys)))
    tasks = [(index, keys[index::num_tasks], imgs_dir, str(root), options) for index in range(num_tasks)]
    logging.info("Comparing the genotypes of %d examples in %s across %d tasks", len(keys), dataset_path, num_tasks)
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
        msg = f"No example in {dataset_path} has a haplotype record ({HAPLOTYPES_SUFFIX}); was it generated before they were?"
        raise ValueError(msg)
    if missing:
        logging.warning("%d examples in %s have no haplotype record and were not compared", missing, dataset_path)
    logging.info("Wrote comparisons for %d of %d examples (the rest unlabeled or without a record)", written, len(keys))

    if not prepare_comparison_tree(root, options, num_workers=cfg.threads):
        return False
    set_aux_tree(dataset_path, COMPARISONS_AUX, f"./{COMPARISONS_DIR}")
    return True
