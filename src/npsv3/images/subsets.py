"""Reducing the training data footprint with subsets of the examples and of each example's genotypes.

Two independent, optional reductions, applied either when generating images (`command=images`) or when training on a
dataset generated without them (to compare reductions without regenerating images):

- Hom. ref. downsampling: keep every non-reference example, but only about `max(min_keep, ceil(ratio * carriers))` of a
  region's hom. ref. examples, where `carriers` is the number of samples with an analysis-variant ALT allele in the
  region. Most examples are hom. ref. examples in regions with few carriers, and those are the "easiest" examples.
- Genotype subsets: simulate (or train with) only the genotypes that share a haplotype with the label, plus hom. ref.
  and, optionally, every homozygous genotype. These are most of the "hard" negatives.

Hom. ref. downsampling needs cohort-wide counts, which a single sample's image generation can't compute, so they are
computed once with `command=region_stats` (see `write_region_stats`), from the samples' `command=haplotypes` output or
image datasets. Each example's keep/drop decision is then a deterministic function of the counts, the region, the
sample and a seed (see `keep_hom_ref`), so every sample's (independent) image generation, and training, make the same
decisions without any coordination. A larger `ratio` (or `min_keep`) keeps a superset of the examples a smaller one does,
so a dataset generated with a loose cutoff can be filtered to a tighter cutoff during training, exactly as if it had been
generated with the tighter cutoff.

Genotype indices are always into the "full" genotype space, i.e., the i-th pair of
`combinations_with_replacement(range(num_haplotypes), ploidy)`, and each example records which of those genotypes it
has (`genotype_idxs`).
"""

import csv
import dataclasses
import glob
import hashlib
import itertools
import json
import logging
import math
import os
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from npsv3.types import PathType

REGION_STATS_COLUMNS = ("region", "carriers", "hom_ref", "labeled")


@dataclasses.dataclass(frozen=True)
class RegionCounts:
    """Labeled samples in a region, the number of those that are hom. ref., and the number that are carriers"""

    carriers: int
    hom_ref: int
    labeled: int


def region_key(key: str, sample: str) -> str:
    """The region part of an example's key (`images.example.energon_key`), identifying the region across samples"""
    prefix = f"{sample}_"
    assert key.startswith(prefix), f"Example key {key!r} doesn't start with its sample {sample!r}"
    return key[len(prefix) :]


def num_genotypes(num_haplotypes: int, ploidy: int = 2) -> int:
    return math.comb(num_haplotypes + ploidy - 1, ploidy)


def num_haplotypes_from_genotypes(g: int) -> int:
    """The number of (diploid) candidate haplotypes with `g` genotypes in total"""
    n = (math.isqrt(8 * g + 1) - 1) // 2
    assert n * (n + 1) // 2 == g, f"{g} is not the total number of genotypes for any number of haplotypes"
    return n


def genotypes(num_haplotypes: int, ploidy: int = 2) -> list[tuple[int, ...]]:
    """Every genotype, as sorted haplotype indices, in genotype index order"""
    return list(itertools.combinations_with_replacement(range(num_haplotypes), ploidy))


def example_genotype_idxs(row: Mapping) -> list[int]:
    """The (full) genotype index of each of an example's genotypes from its metadata `row`. Examples written before
    `genotype_idxs` was recorded have every genotype."""
    if (idxs := row.get("genotype_idxs")) is not None:
        return [int(idx) for idx in idxs]
    return list(range(int(row["g"])))


def example_num_haplotypes(row: Mapping) -> int:
    """The number of candidate haplotypes from an example's metadata `row`"""
    if (n := row.get("num_haplotypes")) is not None:
        return int(n)
    assert row.get("genotype_idxs") is None, "Examples with `genotype_idxs` should record `num_haplotypes`"
    return num_haplotypes_from_genotypes(int(row["g"]))


def is_hom_ref(true: Sequence[int | None]) -> bool:
    """Whether the true haplotype indices (see `images.haplotypes.haplotype_record`) are hom. ref."""
    return all(idx == 0 for idx in true)


def is_hom_ref_label(row: Mapping) -> bool:
    """Whether an example's label, from its metadata `row`, is hom. ref. (False if unlabeled)"""
    if (label := row.get("label")) is None:
        return False
    idxs = row.get("genotype_idxs")
    # Genotype 0 is always hom. ref.
    return (int(label) if idxs is None else int(idxs[int(label)])) == 0


# ---------------------------------------------------------------------------------------------------------------------
# Genotype subsets
# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GenotypeSubset:
    """Genotypes that share a haplotype with the label (one haplotype change away), plus hom. ref. and, if
    `include_homozygous`, every homozygous genotype

    Args:
        include_homozygous: Include every homozygous genotype. These are the most confusable genotypes for hom. alt.
            labels, which otherwise share a haplotype with few genotypes.
    """

    include_homozygous: bool = True

    def genotype_idxs(self, num_haplotypes: int, true: Sequence[int], *, ploidy: int = 2) -> list[int]:
        """The (full) genotype indices, in order, in the subset for an example with `num_haplotypes` candidates and the
        true haplotype indices `true`"""
        shared = set(true)
        return [
            idx
            for idx, genotype in enumerate(genotypes(num_haplotypes, ploidy))
            if not shared.isdisjoint(genotype)
            or all(h == 0 for h in genotype)
            or (self.include_homozygous and len(set(genotype)) == 1)
        ]

    def select(self, row: Mapping, *, ploidy: int = 2) -> list[int] | None:
        """Positions, along the example's genotype axis, of its genotypes in the subset from its metadata `row`, or None
        if the example is unlabeled (and so is left unchanged)"""
        label = row.get("label")
        if label is None:
            return None
        stored = example_genotype_idxs(row)
        n = example_num_haplotypes(row)
        true = genotypes(n, ploidy)[stored[int(label)]]
        subset = set(self.genotype_idxs(n, true, ploidy=ploidy))
        return [position for position, idx in enumerate(stored) if idx in subset]


def genotype_subset_from_config(cfg: Mapping | None) -> GenotypeSubset | None:
    """The `GenotypeSubset` for a `genotype_subset` config (with `enabled` and the `GenotypeSubset` fields), None if
    disabled. A `GenotypeSubset` is returned as-is."""
    if cfg is None or isinstance(cfg, GenotypeSubset):
        return cfg
    params = dict(cfg)
    if not params.pop("enabled", True):
        return None
    return GenotypeSubset(**params)


# ---------------------------------------------------------------------------------------------------------------------
# Hom. ref. downsampling
# ---------------------------------------------------------------------------------------------------------------------


def _uniform(seed: int, region: str, sample: str) -> float:
    """A deterministic pseudo-random number in [0, 1) for an example"""
    digest = hashlib.blake2b(f"{seed}\0{region}\0{sample}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") / 2**64


@dataclasses.dataclass(frozen=True)
class HomRefDownsample:
    """Keep each hom. ref. example with probability `k / hom_ref`, where `k = max(min_keep, ceil(ratio * carriers))`,
    so about `k` of a region's hom. ref. examples are kept. The decision for an example is deterministic given `seed`,
    and nested: a larger `ratio` (or `min_keep`) keeps a superset of the examples.

    Args:
        ratio: Hom. ref. examples to keep per carrier
        min_keep: Minimum hom. ref. examples to keep per region
        seed: Seed for the per-example decision
    """

    ratio: float = 2.0
    min_keep: int = 5
    seed: int = 0

    def keep(self, region: str, sample: str, counts: RegionCounts) -> bool:
        """Whether to keep `sample`'s hom. ref. example for `region` (the region part of its key, see `region_key`)"""
        k = max(self.min_keep, math.ceil(self.ratio * counts.carriers))
        if counts.hom_ref <= k:
            return True
        return _uniform(self.seed, region, sample) < k / counts.hom_ref


# ---------------------------------------------------------------------------------------------------------------------
# Region statistics
# ---------------------------------------------------------------------------------------------------------------------


def read_region_stats(path: PathType) -> dict[str, RegionCounts]:
    """Read a `write_region_stats` file, as counts by region (see `region_key`)"""
    with open(path, newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        return {
            row["region"]: RegionCounts(carriers=int(row["carriers"]), hom_ref=int(row["hom_ref"]), labeled=int(row["labeled"]))
            for row in reader
        }


def _haplotype_tree_classes(directory: Path) -> Iterator[tuple[str, str, bool]]:
    """Yield (region, sample, hom. ref.) for every labeled example in a `command=haplotypes` output directory"""
    import gzip
    import tarfile

    import yaml

    from npsv3.constants import HAPLOTYPES_DIR, HAPLOTYPES_SUFFIX
    from npsv3.images.haplotype_tree import HAPLOTYPE_SHARD_GLOB, PROVENANCE_FILE

    with (directory / HAPLOTYPES_DIR / PROVENANCE_FILE).open() as file:
        sample = yaml.safe_load(file)["sample"]
    suffix = f".{HAPLOTYPES_SUFFIX}"
    for shard in sorted((directory / HAPLOTYPES_DIR).glob(HAPLOTYPE_SHARD_GLOB)):
        with tarfile.open(shard) as tar:
            for member in tar:
                if not member.name.endswith(suffix):
                    continue
                record = json.loads(gzip.decompress(tar.extractfile(member).read()))  # type: ignore[union-attr]
                true = record.get("true")
                if true is None or None in true:  # Skipped or unlabeled
                    continue
                yield region_key(member.name[: -len(suffix)], sample), sample, is_hom_ref(true)


def _dataset_classes(dataset_yaml: Path) -> Iterator[tuple[str, str, bool]]:
    """Yield (region, sample, hom. ref.) for every labeled example in an image dataset"""
    from npsv3.models.energon import _read_split

    root, split = _read_split(dataset_yaml)
    with (root / split["path"]).open() as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") is not None:
                yield region_key(row["key"], row["sample"]), row["sample"], is_hom_ref_label(row)


def _source_classes(path: str) -> list[tuple[str, str, bool]]:
    from npsv3.constants import HAPLOTYPES_DIR

    source = Path(path)
    if (source / HAPLOTYPES_DIR).is_dir():
        return list(_haplotype_tree_classes(source))
    if source.is_dir():
        source = source / "dataset.yaml"
    if source.is_file():
        return list(_dataset_classes(source))
    msg = f"{path} is neither a `command=haplotypes` output directory nor an image dataset"
    raise ValueError(msg)


def _is_source(path: str) -> bool:
    """Whether `path` is a `command=haplotypes` output directory or an image dataset (`dataset.yaml` or its directory)"""
    from npsv3.constants import HAPLOTYPES_DIR

    source = Path(path)
    if source.is_dir():
        return (source / HAPLOTYPES_DIR).is_dir() or (source / "dataset.yaml").is_file()
    return source.name == "dataset.yaml" and source.is_file()


def expand_sources(sources: str | Iterable[str]) -> list[str]:
    """Expand each (glob) pattern in `sources`. Glob matches that aren't sources (e.g., Hydra's logs and `multirun.yaml`
    beside the per-sample directories) are skipped, while paths without wildcards are kept as-is."""
    patterns = [sources] if isinstance(sources, str) else list(sources)
    paths = []
    for pattern in patterns:
        pattern = os.path.expanduser(pattern)  # noqa: PLW2901
        if not any(char in pattern for char in "*?["):
            paths.append(pattern)
            continue
        matches = sorted(glob.glob(pattern))
        sources_matched = [path for path in matches if _is_source(path)]
        if skipped := len(matches) - len(sources_matched):
            logging.warning("Skipping %d paths matching %s that aren't haplotypes directories or image datasets", skipped, pattern)
        if not sources_matched:
            msg = f"No inputs match {pattern}"
            raise ValueError(msg)
        paths.extend(sources_matched)
    return paths


def region_stats(sources: Iterable[str], *, max_workers: int = 1) -> dict[str, RegionCounts]:
    """Count each region's labeled, hom. ref. and carrier samples across `command=haplotypes` output directories and/or
    image datasets (`dataset.yaml` or its directory). An example in more than one source is counted once."""
    sources = list(sources)
    classes: dict[tuple[str, str], bool] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for source, source_classes in zip(sources, pool.map(_source_classes, sources), strict=True):
            logging.info("Read %d labeled examples from %s", len(source_classes), source)
            for region, sample, hom_ref in source_classes:
                classes[(region, sample)] = hom_ref

    labeled, hom_ref = Counter(), Counter()
    for (region, _), example_hom_ref in classes.items():
        labeled[region] += 1
        hom_ref[region] += example_hom_ref
    return {
        region: RegionCounts(carriers=labeled[region] - hom_ref[region], hom_ref=hom_ref[region], labeled=labeled[region])
        for region in sorted(labeled)
    }


def write_region_stats(stats: Mapping[str, RegionCounts], path: PathType) -> None:
    """Write `region_stats` to a TSV file with `REGION_STATS_COLUMNS`"""
    with open(path, "w", newline="") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        writer.writerow(REGION_STATS_COLUMNS)
        for region, counts in stats.items():
            writer.writerow([region, counts.carriers, counts.hom_ref, counts.labeled])
