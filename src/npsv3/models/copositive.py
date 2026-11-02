"""Co-positive aware training targets for `paired.CopositiveInfoNCE`, derived from a `command=copositive` dataset.

Some candidate genotypes can't be told apart from the label in the short-read data (they are "co-positive"), e.g., a
same-site INS a few bases longer, and the label itself may be a sequence-equivalent stand-in for the sample's true
genotype. The single-positive InfoNCE keeps pushing such genotypes away from the query, which it can only do by learning
artifacts. `CopositiveTargets` instead assigns each genotype g of a (labeled) example a numerator (positive) weight c_g
and a denominator weight a_g >= c_g, generalizing the loss to

    L = -log sum_g c_g p_g + log sum_g a_g p_g

(see `CopositiveInfoNCE` for the replicate-level form used in training). The single-positive loss has c = 1[g = label]
and a = 1. The weights are derived from three per-genotype annotations (see `filters.classify_copositive`):

- conf: The co-positive weight (`genotype_weights`), i.e., the previous model's (M0's) symmetrized sim-vs-sim confusion
  between the label and g
- equivalent: Whether g is sequence-equivalent to the label, i.e., doesn't differ (`genotype_differ`) or is similar
  enough (`genotype_score` >= `score_equiv`) according to the `comparisons` aux tree
- category: The example's category from its real query's top alternative (needs M0's `genotype_probs`)

Only equivalent alternatives are ever made (partial) positives, masked or down-weighted, so look-alikes whose sequences
differ, i.e., a model limitation, remain negatives. The named `VARIANTS` are those evaluated as counterfactual losses
in the `loss_distribution` notebook:

| variant    | positives (c_g = 1, or conf for "soft")      | denominator a_g of the other genotypes                |
|------------|----------------------------------------------|-------------------------------------------------------|
| baseline   | label                                        | 1                                                     |
| mask       | label                                        | 0 for equivalent look-alikes (conf >= threshold)      |
| weight     | label                                        | 1 - conf for equivalent alternatives                  |
| multi      | label and equivalent alternatives            | 1                                                     |
| soft_multi | label, equivalent alternatives with c = conf | 1                                                     |
| mask+multi | label and separable equivalent alternatives  | as `mask`                                             |
| gated      | `multi` for "noisy_equiv" examples           | `weight` for "lookalike" examples                     |

A variant is a preset of the `positives` and `denominator` rules and the example categories they apply to, any of which
can be overridden, e.g., `gated` with `mask` in place of `weight` for "lookalike" examples.

Examples without the annotations (unscored by M0) or without a comparison (no record, or a skipped one) keep the
single-positive targets.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, ClassVar, Literal

import numpy as np

from npsv3.models.filters import COPOSITIVE_CATEGORIES, CopositiveThresholds, classify_copositive

PositiveRule = Literal["label", "equivalent", "separable", "soft"]
DenominatorRule = Literal["full", "mask", "weight"]

POSITIVE_RULES = ("label", "equivalent", "separable", "soft")
DENOMINATOR_RULES = ("full", "mask", "weight")

ALL_CATEGORIES = "all"
"""`positive_categories`/`denominator_categories` value applying the rule to every example"""

VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "mask": {"denominator": "mask"},
    "weight": {"denominator": "weight"},
    "multi": {"positives": "equivalent"},
    "soft_multi": {"positives": "soft"},
    "mask+multi": {"positives": "separable", "denominator": "mask"},
    "gated": {
        "positives": "equivalent",
        "positive_categories": ("noisy_equiv",),
        "denominator": "weight",
        "denominator_categories": ("lookalike",),
    },
}


def _categories(categories: str | Iterable[str] | None) -> tuple[str, ...] | None:
    """Categories as a tuple, or None for every category"""
    if categories is None or categories == ALL_CATEGORIES:
        return None
    # A tuple, since Hydra may pass a (mutable, unhashable) ListConfig
    categories = (categories,) if isinstance(categories, str) else tuple(categories)
    unknown = set(categories) - set(COPOSITIVE_CATEGORIES)
    if unknown:
        msg = f"Unknown co-positive categories {sorted(unknown)}, expected some of {COPOSITIVE_CATEGORIES} or {ALL_CATEGORIES!r}"
        raise ValueError(msg)
    return categories


def equivalent_genotypes(
    comparison: Mapping | None, num_genotypes: int, score_equiv: float, genotype_idxs: Sequence[int] | None = None
) -> np.ndarray | None:
    """(G,) whether each of the example's genotypes is sequence-equivalent to the label according to its comparison
    record, i.e., doesn't differ or scores at least `score_equiv`. None if there is no comparison (no record or a skipped
    one). The record has every possible genotype, which `genotype_idxs` maps the example's genotypes to (None if the
    example has every genotype, see `images.subsets`)."""
    differ = None if comparison is None else comparison.get("genotype_differ")
    score = None if comparison is None else comparison.get("genotype_score")
    if differ is None or score is None:
        return None
    idxs = np.arange(num_genotypes) if genotype_idxs is None else np.asarray(genotype_idxs, dtype=int)
    return ~np.asarray(differ, dtype=bool)[idxs] | (np.asarray(score, dtype=float)[idxs] >= score_equiv)


class CopositiveTargets:
    """Per-genotype numerator (positive) and denominator weights for `CopositiveInfoNCE` (see the module docstring).

    Configured in `data.copositive` and applied by `EnergonPackedImageDataModule` to the train split only, so the
    validation and test losses remain the single-positive InfoNCE (and comparable between experiments). Needs
    `command=copositive` train datasets with a `comparisons` aux tree.

    Args:
        variant: The preset (see `VARIANTS`) for any of the following left as None
        positives: Which alternatives become positives: none ("label"), the equivalent alternatives ("equivalent"), the
            equivalent alternatives that are separable in simulation, i.e., conf < `conf_threshold` ("separable"), or
            the equivalent alternatives as partial positives with c = conf ("soft")
        positive_categories: Example categories `positives` applies to ("all" for every example)
        denominator: How alternatives are weighted in the denominator: not at all ("full"), removing the equivalent
            look-alikes, i.e., conf >= `conf_threshold` ("mask"), or down-weighting equivalent alternatives to 1 - conf
            ("weight")
        denominator_categories: Example categories `denominator` applies to ("all" for every example)
        conf_threshold, score_equiv, filter_p_label, filter_p_alt: See `filters.CopositiveThresholds`
    """

    #: The data needed, checked against the train datasets as for `filters.ExampleFilter`
    requires_comparisons: ClassVar[bool] = True
    required_fields: ClassVar[tuple[str, ...]] = ("genotype_probs", "genotype_weights")

    def __init__(
        self,
        variant: str = "gated",
        *,
        positives: PositiveRule | None = None,
        positive_categories: str | Iterable[str] | None = None,
        denominator: DenominatorRule | None = None,
        denominator_categories: str | Iterable[str] | None = None,
        conf_threshold: float = CopositiveThresholds.conf_threshold,
        score_equiv: float = CopositiveThresholds.score_equiv,
        filter_p_label: float = CopositiveThresholds.filter_p_label,
        filter_p_alt: float = CopositiveThresholds.filter_p_alt,
    ):
        if variant not in VARIANTS:
            msg = f"Unknown co-positive variant {variant!r}, expected one of {list(VARIANTS)}"
            raise ValueError(msg)
        preset = VARIANTS[variant]
        self.variant = variant
        self.positives = positives if positives is not None else preset.get("positives", "label")
        self.denominator = denominator if denominator is not None else preset.get("denominator", "full")
        if self.positives not in POSITIVE_RULES:
            msg = f"Unknown positives rule {self.positives!r}, expected one of {POSITIVE_RULES}"
            raise ValueError(msg)
        if self.denominator not in DENOMINATOR_RULES:
            msg = f"Unknown denominator rule {self.denominator!r}, expected one of {DENOMINATOR_RULES}"
            raise ValueError(msg)
        self.positive_categories = _categories(
            positive_categories if positive_categories is not None else preset.get("positive_categories")
        )
        self.denominator_categories = _categories(
            denominator_categories if denominator_categories is not None else preset.get("denominator_categories")
        )
        self.thresholds = CopositiveThresholds(conf_threshold, score_equiv, filter_p_label, filter_p_alt)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "positives": self.positives,
            "positive_categories": self.positive_categories or ALL_CATEGORIES,
            "denominator": self.denominator,
            "denominator_categories": self.denominator_categories or ALL_CATEGORIES,
            "conf_threshold": self.thresholds.conf_threshold,
            "score_equiv": self.thresholds.score_equiv,
            "filter_p_label": self.thresholds.filter_p_label,
            "filter_p_alt": self.thresholds.filter_p_alt,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(f'{name}={value!r}' for name, value in self.params.items())})"

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        assert isinstance(other, CopositiveTargets)
        return self.params == other.params

    def __hash__(self) -> int:
        return hash((type(self), tuple(self.params.items())))

    @property
    def needs_category(self) -> bool:
        return self.positive_categories is not None or self.denominator_categories is not None

    def targets(self, row: Mapping, comparison: Mapping | None) -> tuple[np.ndarray, np.ndarray]:
        """(G,) numerator and denominator weights along the example's (stored) genotypes for the labeled example with
        metadata `row` and comparison record `comparison` (None if it has none)"""
        num_genotypes, label = int(row["g"]), int(row["label"])
        is_label = np.arange(num_genotypes) == label
        positive, denominator = is_label.astype(np.float32), np.ones(num_genotypes, dtype=np.float32)

        probs, conf = row.get("genotype_probs"), row.get("genotype_weights")
        if probs is None or conf is None:
            return positive, denominator  # Unscored by M0
        if len(probs) != num_genotypes or len(conf) != num_genotypes:
            msg = (
                f"{row.get('key')} has {num_genotypes} genotypes, but {len(conf)} co-positive weights and {len(probs)} "
                "probabilities. Was it annotated with a genotype subset?"
            )
            raise ValueError(msg)

        genotype_idxs = row.get("genotype_idxs")
        equivalent = equivalent_genotypes(comparison, num_genotypes, self.thresholds.score_equiv, genotype_idxs)
        if equivalent is None:
            return positive, denominator  # Uncompared, so no alternatives are known to be equivalent
        equivalent_alt = equivalent & ~is_label
        if not np.any(equivalent_alt):
            return positive, denominator

        conf = np.asarray(conf, dtype=np.float32)
        lookalike_alt = equivalent_alt & (conf >= self.thresholds.conf_threshold)

        category = None
        if self.needs_category:
            category = classify_copositive(label, probs, conf, comparison, self.thresholds, genotype_idxs)["category"]

        if self.positive_categories is None or category in self.positive_categories:
            if self.positives == "equivalent":
                positive[equivalent_alt] = 1.0
            elif self.positives == "separable":
                positive[equivalent_alt & ~lookalike_alt] = 1.0
            elif self.positives == "soft":
                positive[equivalent_alt] = conf[equivalent_alt]

        if self.denominator_categories is None or category in self.denominator_categories:
            if self.denominator == "mask":
                denominator[lookalike_alt] = 0.0
            elif self.denominator == "weight":
                denominator[equivalent_alt] = 1.0 - conf[equivalent_alt]

        # A positive is always fully in the denominator, so the loss is non-negative
        return positive, np.maximum(denominator, positive)


def copositive_targets_from_config(config: "CopositiveTargets | Mapping | None") -> CopositiveTargets | None:
    """`CopositiveTargets` from an object or (Hydra `_target_`) config, e.g., if `data` was instantiated non-recursively"""
    if config is None or isinstance(config, CopositiveTargets):
        return config
    import hydra

    targets = hydra.utils.instantiate(config)
    if not isinstance(targets, CopositiveTargets):
        msg = f"{targets!r} is not a `CopositiveTargets`"
        raise TypeError(msg)
    return targets
