"""Configurable exclusion of training examples, applied by `EnergonPackedImageDataModule` to the train split.

A filter decides from an example's metadata row and any auxiliary records whether to keep the example. Filters are
applied when the crude sample is cooked, before any pixels are fetched, and by `estimate_epoch_batches`, so the virtual
epoch length reflects them. Only the train split is filtered; validation, test and prediction (including
`command=copositive`, which must annotate every example) see every example.

Filters are configured as a list of Hydra `_target_` configs in `data.filters`, e.g.,

    data.filters='[{_target_: npsv3.models.filters.LabelDiffersFilter}]'
    data.filters='[{_target_: npsv3.models.filters.CopositiveFilter, exclude: [suspect, suspect_soft]}]'
    data.filters='[{_target_: npsv3.models.filters.HomRefDownsampleFilter, region_stats: region_stats.tsv}]'

Each filter declares what data it needs (`requires_comparisons`, `required_fields`), which
`check_filter_requirements` verifies against the train datasets before training, so a missing aux tree or annotation
fails with an actionable error instead of silently keeping (or dropping) everything.
"""

import abc
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from npsv3.constants import COMPARISONS_AUX
from npsv3.images.subsets import HomRefDownsample, is_hom_ref_label, read_region_stats, region_key

MissingPolicy = Literal["keep", "exclude"]


class ExampleFilter(abc.ABC):
    """Base class for filters: `keep` returns False to exclude an example from training.

    Filters are copied into every DataLoader worker and must not change after construction, so that every worker (and a
    resumed run) filters identically. Two filters are equal, and print the same, if they have the same type and `params`.
    """

    #: Whether `keep` needs the example's `comparisons` aux tree record
    requires_comparisons: ClassVar[bool] = False
    #: Metadata row fields `keep` needs. At least one row of each train dataset must have them.
    required_fields: ClassVar[tuple[str, ...]] = ()

    @abc.abstractmethod
    def keep(self, row: Mapping, comparison: Mapping | None) -> bool:
        """Return True to keep the example with metadata `row` and comparison record `comparison` (None if the example has no
        record, or the filter doesn't need one)"""

    @property
    @abc.abstractmethod
    def params(self) -> dict[str, Any]:
        """The filter's constructor arguments, e.g., for logging which filters were applied"""

    def __call__(self, row: Mapping, comparison: Mapping | None) -> bool:
        return self.keep(row, comparison)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(f'{name}={value!r}' for name, value in self.params.items())})"

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        assert isinstance(other, ExampleFilter)
        return self.params == other.params

    def __hash__(self) -> int:
        return hash((type(self), tuple(self.params.items())))


def _resolve(missing: MissingPolicy) -> bool:
    if missing not in ("keep", "exclude"):
        msg = f"Unknown missing-data policy {missing!r}, expected 'keep' or 'exclude'"
        raise ValueError(msg)
    return missing == "keep"


class LabelDiffersFilter(ExampleFilter):
    """Exclude examples whose labeled genotype differs from the sample's own (assembled) haplotypes, according to
    sequence comparison (e.g. Truvari) analysis.

    Args:
        missing: How to handle examples whose label wasn't compared with the sample (typically a result of missing
        sample haplotypes, variants larger than `compare.max_length`, etc.)
    """

    requires_comparisons: ClassVar[bool] = True

    missing: MissingPolicy

    def __init__(self, missing: MissingPolicy = "keep"):
        _resolve(missing)
        self.missing = missing

    @property
    def params(self) -> dict[str, Any]:
        return {"missing": self.missing}

    def keep(self, row: Mapping, comparison: Mapping | None) -> bool:  # noqa: ARG002
        differs = None if comparison is None else comparison.get("label_differs")
        if differs is None:
            return _resolve(self.missing)
        return not differs


COPOSITIVE_CATEGORIES = (
    "uncompared",
    "lookalike",
    "unresolved",
    "modeled",
    "noisy_equiv",
    "suspect",
    "suspect_soft",
)


@dataclass(frozen=True)
class CopositiveThresholds:
    """Thresholds for `classify_copositive`"""

    #: conf[g'] at or above which g' is a look-alike of the label in simulation
    conf_threshold: float = 0.5
    #: genotype_score at or above which a differing g' is treated as sequence-equivalent, e.g., tandem repeat length variants
    score_equiv: float = 0.7
    #: A separable, differing g' is "suspect" if p[label] is below `filter_p_label`...
    filter_p_label: float = 0.05
    #: ...and p[g'] is at or above `filter_p_alt` (otherwise "suspect_soft")
    filter_p_alt: float = 0.5


def classify_copositive(
    label: int,
    genotype_probs: Sequence[float],
    genotype_weights: Sequence[float],
    comparison: Mapping | None,
    thresholds: CopositiveThresholds = CopositiveThresholds(),  # noqa: B008
    genotype_idxs: Sequence[int] | None = None,
) -> dict:
    """Classify one example from its real query's top alternative genotype g', by whether g' is a look-alike of the label
    in simulation (`genotype_weights`, the co-positive `conf`), sequence-equivalent to the label (the comparison record's
    `genotype_differ`/`genotype_score`) and preferred by the real data (`genotype_probs`):

    - "uncompared": No sequence comparison (no record or a skipped one) and the real data prefers g', so look-alikes can't
      be told apart from noisy or suspect labels
    - "lookalike": Not separable in simulation and sequence-equivalent, so likely a good label with an indistinguishable g'
    - "unresolved": Not separable in simulation, but the sequences differ, i.e., a model limitation rather than a look-alike
    - "modeled": Separable (or uncompared) and the real data prefers the label, i.e., the model learned the label
    - "noisy_equiv": Separable and the real data prefers a sequence-equivalent g', possibly a noisy label
    - "suspect": Separable and the real data confidently prefers a differing g', likely an incorrect label
    - "suspect_soft": Separable and the real data weakly prefers a differing g', possibly a noisy label

    `label`, `genotype_probs` and `genotype_weights` are along the example's genotypes, while the comparison record has
    every possible genotype. `genotype_idxs` maps the former to the latter for an example with a subset of the genotypes
    (see `images.subsets`), None if the example has every genotype.

    Returns:
        A dict with the `category` and the values it was derived from (`prediction`, `alt`, `conf_alt`, `p_label`,
        `p_alt`, `differ_alt` and `score_alt`). `prediction` and `alt` are along the example's genotypes.
    """
    p = np.asarray(genotype_probs, dtype=float)
    conf = np.asarray(genotype_weights, dtype=float)

    alternatives = np.delete(np.arange(len(p)), label)
    best_alt = int(alternatives[np.argmax(p[alternatives])])

    # Skipped comparisons (sequences > `max_length`) have no per-genotype `genotype_differ`/`genotype_score`
    differ = None if comparison is None else comparison.get("genotype_differ")
    score = None if comparison is None else comparison.get("genotype_score")
    compared_alt = best_alt if genotype_idxs is None else int(genotype_idxs[best_alt])
    differ_alt = bool(differ[compared_alt]) if differ is not None else None
    score_alt = float(score[compared_alt]) if score is not None else float("nan")

    look = conf[best_alt] >= thresholds.conf_threshold
    equiv = differ_alt is False or score_alt >= thresholds.score_equiv
    prefers_alt = p[best_alt] >= p[label]

    if differ_alt is None:
        category = "uncompared" if prefers_alt else "modeled"
    elif look and equiv:
        category = "lookalike"
    elif look:
        category = "unresolved"
    elif not prefers_alt:
        category = "modeled"
    elif equiv:
        category = "noisy_equiv"
    elif p[label] < thresholds.filter_p_label and p[best_alt] >= thresholds.filter_p_alt:
        category = "suspect"
    else:
        category = "suspect_soft"

    return {
        "prediction": int(np.argmax(p)),
        "alt": best_alt,
        "conf_alt": float(conf[best_alt]),
        "p_label": float(p[label]),
        "p_alt": float(p[best_alt]),
        "differ_alt": differ_alt,
        "score_alt": score_alt,
        "category": category,
    }


class CopositiveFilter(ExampleFilter):
    """Exclude examples by their `classify_copositive` category, e.g., "suspect" labels that a previous model (M0)
    confidently and separably contradicts. Needs a `command=copositive` dataset (for `genotype_probs` and
    `genotype_weights`) with a `comparisons` aux tree.

    Args:
        exclude: Categories to exclude (see `classify_copositive`)
        missing: What to do with examples M0 didn't score (no `genotype_probs`/`genotype_weights`)
        conf_threshold, score_equiv, filter_p_label, filter_p_alt: See `CopositiveThresholds`
    """

    requires_comparisons: ClassVar[bool] = True
    required_fields: ClassVar[tuple[str, ...]] = ("genotype_probs", "genotype_weights")

    exclude: tuple[str, ...]
    missing: MissingPolicy
    thresholds: CopositiveThresholds

    def __init__(
        self,
        exclude: Iterable[str] = ("suspect",),
        missing: MissingPolicy = "keep",
        conf_threshold: float = CopositiveThresholds.conf_threshold,
        score_equiv: float = CopositiveThresholds.score_equiv,
        filter_p_label: float = CopositiveThresholds.filter_p_label,
        filter_p_alt: float = CopositiveThresholds.filter_p_alt,
    ):
        # A tuple, since Hydra may pass a (mutable, unhashable) ListConfig
        exclude = tuple(exclude)
        unknown = set(exclude) - set(COPOSITIVE_CATEGORIES)
        if unknown:
            msg = f"Unknown co-positive categories {sorted(unknown)}, expected some of {COPOSITIVE_CATEGORIES}"
            raise ValueError(msg)
        _resolve(missing)
        self.exclude = exclude
        self.missing = missing
        self.thresholds = CopositiveThresholds(conf_threshold, score_equiv, filter_p_label, filter_p_alt)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "exclude": self.exclude,
            "missing": self.missing,
            "conf_threshold": self.thresholds.conf_threshold,
            "score_equiv": self.thresholds.score_equiv,
            "filter_p_label": self.thresholds.filter_p_label,
            "filter_p_alt": self.thresholds.filter_p_alt,
        }

    def keep(self, row: Mapping, comparison: Mapping | None) -> bool:
        probs, weights, label = row.get("genotype_probs"), row.get("genotype_weights"), row.get("label")
        if probs is None or weights is None or label is None:
            return _resolve(self.missing)
        category = classify_copositive(
            int(label), probs, weights, comparison, self.thresholds, genotype_idxs=row.get("genotype_idxs")
        )["category"]
        return category not in self.exclude


class HomRefDownsampleFilter(ExampleFilter):
    """Exclude some hom. ref. examples: keep every non-reference example and about
    max(`min_keep`, ceil(`ratio` x carriers)) of each region's hom. ref. examples, exactly as `hom_ref_downsample` would
    during image generation with the same `region_stats` and `seed` (see `images.subsets.HomRefDownsample`).

    Since a larger `ratio` (or `min_keep`) keeps a superset of the examples, this can also further reduce a dataset that
    was already downsampled with a smaller `ratio` (and the same statistics and seed).

    Args:
        region_stats: The cohort's region statistics (the output of `command=region_stats`)
        ratio: Hom. ref. examples to keep per carrier
        min_keep: Minimum hom. ref. examples to keep per region
        seed: Seed for the per-example decision
        missing: What to do with hom. ref. examples whose region isn't in `region_stats`
    """

    required_fields: ClassVar[tuple[str, ...]] = ("sample",)

    region_stats: str
    missing: MissingPolicy
    downsample: HomRefDownsample

    def __init__(
        self,
        region_stats: str,
        ratio: float = HomRefDownsample.ratio,
        min_keep: int = HomRefDownsample.min_keep,
        seed: int = HomRefDownsample.seed,
        missing: MissingPolicy = "keep",
    ):
        _resolve(missing)
        self.region_stats = str(region_stats)
        self.missing = missing
        self.downsample = HomRefDownsample(ratio=float(ratio), min_keep=int(min_keep), seed=int(seed))
        self._counts = read_region_stats(self.region_stats)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "region_stats": self.region_stats,
            "ratio": self.downsample.ratio,
            "min_keep": self.downsample.min_keep,
            "seed": self.downsample.seed,
            "missing": self.missing,
        }

    def keep(self, row: Mapping, comparison: Mapping | None) -> bool:  # noqa: ARG002
        if not is_hom_ref_label(row):
            return True
        region = region_key(row["key"], row["sample"])
        if (counts := self._counts.get(region)) is None:
            return _resolve(self.missing)
        return self.downsample.keep(region, row["sample"], counts)


# ---------------------------------------------------------------------------------------------------------------------
# Applying filters
# ---------------------------------------------------------------------------------------------------------------------


def build_filters(filters: Iterable[ExampleFilter | Mapping] | None) -> tuple[ExampleFilter, ...]:
    """Filters from filter objects or (Hydra `_target_`) configs, e.g., if `data` was instantiated non-recursively"""
    if not filters:
        return ()
    import hydra

    built = tuple(f if isinstance(f, ExampleFilter) else hydra.utils.instantiate(f) for f in filters)
    for f in built:
        if not isinstance(f, ExampleFilter):
            msg = f"{f!r} is not an `ExampleFilter`"
            raise TypeError(msg)
    return built


def requires_comparisons(filters: Iterable[ExampleFilter]) -> bool:
    return any(f.requires_comparisons for f in filters)


def keep_example(filters: Iterable[ExampleFilter], row: Mapping, comparison: Mapping | None) -> bool:
    """Whether every filter keeps the example"""
    return all(f.keep(row, comparison) for f in filters)


def describe_filters(filters: Iterable[ExampleFilter]) -> str:
    return ", ".join(map(repr, filters))


def check_filter_requirements(leaves: Iterable[tuple[Path, Mapping[str, str]]], filters: Sequence[ExampleFilter]) -> None:
    """Verify that every train dataset, given as (metadata JSONL path, absolute aux paths) leaves, has the data `filters`
    need: a `comparisons` aux tree and at least one metadata row with each required field.

    Raises:
        ValueError: Naming the dataset and how to add the missing data
    """
    if not filters:
        return
    needs_comparisons = requires_comparisons(filters)
    fields = sorted({name for f in filters for name in f.required_fields})
    for meta_path, aux in leaves:
        if needs_comparisons and COMPARISONS_AUX not in aux:
            msg = (
                f"Filters {describe_filters(filters)} need the {COMPARISONS_AUX!r} aux tree, which the dataset with "
                f"metadata {meta_path} doesn't reference. Add it with `command=compare input=<its dataset.yaml>` (a "
                "`command=copositive` dataset only inherits the trees its source had when it was written)."
            )
            raise ValueError(msg)
        if fields:
            missing = set(fields)
            with Path(meta_path).open() as meta_file:
                for line in meta_file:
                    if line.strip():
                        row = json.loads(line)
                        missing -= {name for name in missing if row.get(name) is not None}
                        if not missing:
                            break
            if missing:
                msg = (
                    f"Filters {describe_filters(filters)} need the {sorted(missing)} metadata fields, which no row of "
                    f"{meta_path} has. Was it written by `command=copositive`?"
                )
                raise ValueError(msg)
