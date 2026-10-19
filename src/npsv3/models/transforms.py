"""Dataset transforms driven by a trained model's predictions.

`copositive_transform` scores an energon image dataset with a trained (single-positive) model M0 and writes the metadata
half of a new polylithic dataset (`meta_ds/train.jsonl` and `dataset.yaml`) whose `imgs` aux tree is the *source's*
`imgs_ds`, i.e., the images are reused rather than copied. Each scored row is the source row plus the model outputs that
"Bootstrapping co-positives" in REFERENCE_HAPLOTYPE_NOTES.md needs: per-genotype co-positive weights and the real image's
genotype probabilities. Diagnostics and classifications (look-alike categories, SV presence changes, flags, etc.) are left
to the downstream analyses that consume the dataset, as are policy decisions such as masking presence-changing
look-alikes. The output is intended both for statistics/evaluation and as the input to a second round of training with a
weighted InfoNCE.

Everything is per group (region) and per genotype, an index into `combinations_with_replacement(range(H), 2)`. Supports
are replicate-major (support index = r*G + g), as everywhere else in the codebase. The source must have >= 2 simulated
replicates, so that each replicate of a genotype can be held out as a simulated query against the others.

Fields added to each scored row (rows the model did not score, e.g. unlabeled or rejected by the data module's
`g_max`/`w_max`/budget, are copied unchanged):
    genotype_weights: (G,) co-positive weight in [0, 1] (1 for the label): the symmetrized sim-vs-sim confusion `conf`
        between the label and each genotype, i.e., how far M0 cannot separate them even with a simulated query
    genotype_probs: (G,) M0's posterior on the real image (softmax over genotypes, replicates pooled by log-mean-exp)
"""

import json
import logging
import os
from collections import Counter
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml
from megatron.energon import CrudeJsonlDatasetFactory
from megatron.energon.epathlib import EPath
from omegaconf import OmegaConf

from npsv3.models.energon import _read_split
from npsv3.models.paired import MinimizingPredictor


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    """Return log(sum(exp(x))) along an axis without overflow or underflow reducing slice shtat are entirely -inf to -inf (not NaN).
    """
    # numpy has no axis-wise logsumexp: np.logaddexp is a binary ufunc, and np.logaddexp.reduce folds it pairwise (N-1
    # exp/log1p evaluations with accumulating rounding error) instead of a single shifted sum. This does the same as
    # scipy.special.logsumexp, but keeps this per-group math numpy-only. A slice that is all -inf has max -inf, and
    # shifting by it would give -inf - (-inf) = NaN, so a non-finite max is replaced by 0, leaving log(0) = -inf (with
    # the divide-by-zero warning suppressed).
    peak = np.max(x, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    with np.errstate(divide="ignore"):
        return np.squeeze(peak, axis=axis) + np.log(np.sum(np.exp(x - peak), axis=axis))


def pooled_softmax(logits: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Softmax over genotypes of (R, G) logits, pooling replicates by log-mean-exp. `mask` True excludes a support.

    Log-*mean*-exp rather than logsumexp so a genotype that lost a replicate to the mask is not penalized by log(R/(R-1)).
    """
    if mask is None:
        mask = np.zeros_like(logits, dtype=bool)
    counts = np.sum(~mask, axis=0)
    with np.errstate(divide="ignore"):
        pooled = _logsumexp(np.where(mask, -np.inf, logits), axis=0) - np.log(counts)
    pooled = np.where(counts > 0, pooled, -np.inf)
    return np.exp(pooled - _logsumexp(pooled, axis=0))


def sim_confusion(support_logits: np.ndarray, num_replicates: int, num_genotypes: int) -> np.ndarray:
    """C[h, g]: mean over replicates r of h of the pooled softmax probability of g, querying with simulated support (r, h),
    which is excluded from its own supports."""
    R, G = num_replicates, num_genotypes  # noqa: N806
    C = np.zeros((G, G))  # noqa: N806
    for h in range(G):
        for r in range(R):
            mask = np.zeros((R, G), dtype=bool)
            mask[r, h] = True
            C[h] += pooled_softmax(support_logits[r * G + h].reshape(R, G), mask)
    return C / R


def copositive_conf(C: np.ndarray, label: int) -> np.ndarray:  # noqa: N803
    """conf[g'] in [0, 1]: how often simulated g* and g' are mistaken for each other, relative to recognizing themselves.

    Symmetrized: P(g' | sim g*) / P(g* | sim g*) and P(g* | sim g') / P(g' | sim g'), averaged.
    """
    diag = np.diag(C)
    with np.errstate(divide="ignore", invalid="ignore"):
        forward = C[label] / diag[label]
        reverse = C[:, label] / diag
    conf = np.clip(0.5 * (np.nan_to_num(forward) + np.nan_to_num(reverse)), 0.0, 1.0)
    conf[label] = 1.0
    return conf


def copositive_annotations(
    query_logits: np.ndarray, support_logits: np.ndarray, num_replicates: int, num_genotypes: int, label: int
) -> dict:
    """The per-group fields added to each row (see module docstring) from scaled logits: `query_logits` (R, G) of the
    real query against every support, `support_logits` (R*G, R*G) of every support against every support."""
    R, G = num_replicates, num_genotypes  # noqa: N806
    if R < 2:  # noqa: PLR2004
        msg = "Co-positive weights need >= 2 simulated replicates per genotype to hold one out as a simulated query"
        raise ValueError(msg)

    def to_list(values: np.ndarray) -> list[float]:
        return [round(float(value), 6) for value in values]

    return {
        "genotype_weights": to_list(copositive_conf(sim_confusion(support_logits, R, G), label)),
        "genotype_probs": to_list(pooled_softmax(query_logits)),
    }


# ---------------------------------------------------------------------------------------------------------------------
# Lightning prediction and dataset writing
# ---------------------------------------------------------------------------------------------------------------------


def write_annotated_dataset(source: str | os.PathLike, output_dir: str | os.PathLike, annotations: dict[str, dict]) -> str:
    """Write `output_dir`/{meta_ds/train.jsonl, dataset.yaml}: `source`'s rows, in the source's order, with
    `annotations[region]` merged into those that have them, and an `imgs` aux tree pointing at the source's (absolute)
    `imgs_ds`. Rows without annotations are copied unchanged.

    Returns:
        The path to the `dataset.yaml` written.
    """
    root, split = _read_split(source)
    source_meta = (root / split["path"]).resolve()
    output = Path(output_dir).resolve()
    meta_path = output / "meta_ds" / "train.jsonl"
    if meta_path == source_meta:
        msg = f"Output {output} would overwrite the source metadata {source_meta}"
        raise ValueError(msg)

    with source_meta.open() as source_file:
        rows = [json.loads(line) for line in source_file if line.strip()]
    duplicates = [region for region, count in Counter(row.get("region", row["key"]) for row in rows).items() if count > 1]
    if duplicates:
        # Predictions are identified only by region, so they cannot be attributed to a row if regions repeat
        msg = f"Regions repeat in {source_meta} (e.g. {duplicates[0]}); transform one sample's dataset at a time"
        raise ValueError(msg)

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = 0
    with meta_path.open("w") as meta_file:
        for row in rows:
            annotation = annotations.get(row.get("region", row["key"]))
            if annotation is not None:
                row = {**row, **annotation}  # noqa: PLW2901
                annotated += 1
            meta_file.write(json.dumps(row))
            meta_file.write("\n")
    logging.info("Wrote %d source rows, %d annotated, to %s", len(rows), annotated, meta_path)

    unmatched = len(annotations) - annotated
    if unmatched:
        logging.warning("%d scored regions are not in the source metadata %s", unmatched, source_meta)

    CrudeJsonlDatasetFactory.prepare_dataset(EPath(meta_path))
    dataset = {
        "__module__": "megatron.energon",
        "__class__": "MetadatasetV2",
        "splits": {
            "train": {
                "path": f"./meta_ds/{meta_path.name}",
                "aux": {name: str((root / aux_path).resolve()) for name, aux_path in split["aux"].items()},
            }
        },
    }
    dataset_yaml_path = output / "dataset.yaml"
    with dataset_yaml_path.open("w") as dataset_yaml_file:
        yaml.safe_dump(dataset, dataset_yaml_file, sort_keys=False)
    return str(dataset_yaml_path)


class CopositiveTransformCallback(L.pytorch.callbacks.Callback):
    """Score every group seen during prediction (see `copositive_annotations`) and, once prediction ends, write the
    annotated dataset (see `write_annotated_dataset`). Under DDP the annotations are gathered and rank 0 writes.

    The model's forward pass only exposes query-vs-support metrics, so the encoder's output is captured with a forward hook
    to compute support-vs-support metrics without a second forward pass.
    """

    def __init__(self, source: str, output_dir: str, *, temperature: float | None = None):
        """
        Args:
            source: The source dataset's `dataset.yaml` (as written by `prepare_energon_dataset`)
            output_dir: Directory for the new dataset's `meta_ds/` and `dataset.yaml`
            temperature: Softmax temperature applied to the metric. Defaults to the model loss's `temperature`.
        """
        super().__init__()
        self.source = source
        self.output_dir = output_dir
        self.temperature = temperature
        self.annotations: dict[str, dict] = {}
        self.dataset_path: str | None = None
        self._embeddings = None
        self._handle = None

    def on_predict_start(self, trainer, model):  # noqa: ARG002
        self.annotations = {}
        self._handle = model.encoder.register_forward_hook(self._capture)
        # Distances (minimized) become logits by negation, matching how the predictor ranks them
        self._sign = -1.0 if isinstance(model.predictor, MinimizingPredictor) else 1.0
        self._tau = self.temperature if self.temperature is not None else getattr(model.loss, "temperature", None)
        if not isinstance(self._tau, (int, float)):
            msg = f"Loss {type(model.loss).__name__} has no scalar temperature; specify one explicitly"
            raise TypeError(msg)

    def _capture(self, module, inputs, output):  # noqa: ARG002
        self._embeddings = output.detach()

    def _to_logits(self, metric: torch.Tensor) -> np.ndarray:
        return (self._sign * metric.float() / self._tau).cpu().numpy()

    def on_predict_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):  # noqa: ARG002
        metric, _preds, _labels, genotype_label, num_genotypes, regions = outputs
        offsets = metric.offsets().tolist()
        num_support = offsets[-1]
        embeddings = self._embeddings

        with torch.inference_mode():
            for group, region in enumerate(regions):
                start, end = offsets[group], offsets[group + 1]
                G = int(num_genotypes[group])  # noqa: N806
                R = (end - start) // G  # noqa: N806
                label = int(genotype_label[group])

                # Queries follow all the supports in the pack, one per group (see `PackedVariant.forward`)
                support = embeddings[start:end]
                query = embeddings[num_support + group]
                N = end - start  # noqa: N806
                query_metric = model.metric(query.expand(N, -1), support)
                support_metric = model.metric(support.repeat_interleave(N, dim=0), support.repeat(N, 1)).reshape(N, N)
                self.annotations[region] = copositive_annotations(
                    self._to_logits(query_metric).reshape(R, G), self._to_logits(support_metric), R, G, label
                )

    def on_predict_end(self, trainer, model):  # noqa: ARG002
        self._handle.remove()
        annotations = self.annotations
        if trainer.world_size > 1:
            gathered = [None] * trainer.world_size
            torch.distributed.all_gather_object(gathered, annotations)
            annotations = {region: row for rank in gathered for region, row in rank.items()}
        if trainer.is_global_zero:
            self.dataset_path = write_annotated_dataset(self.source, self.output_dir, annotations)


def copositive_transform(cfg, source: str, output_dir: str) -> str | None:
    """Score `source` (an energon image dataset with >= 2 simulated replicates) with the model in `cfg.model.checkpoint`
    and write the annotated metadata dataset, reusing `source`'s images, to `output_dir`.

    Prediction-time options (`cfg.data`, `cfg.trainer`) are used as is, except the data module is pointed at `source`'s
    train split. Annotation options come from `cfg.copositive`.

    Returns:
        The path to the new `dataset.yaml` (None on DDP ranks other than 0).
    """
    from npsv3.models.runners import predict

    cfg = cfg.copy()
    OmegaConf.update(cfg, "data.metadataset_path", None)
    OmegaConf.update(cfg, "data.train_metadataset_paths", [source], merge=False)
    OmegaConf.update(cfg, "data.val_metadataset_paths", None)
    OmegaConf.update(cfg, "data.test_metadataset_paths", None)
    OmegaConf.update(cfg, "data.predict_split", "train")

    params = OmegaConf.to_container(cfg.copositive, resolve=True)
    callback = CopositiveTransformCallback(source, output_dir, **params)
    predict(cfg, return_predictions=False, callbacks=[callback])

    # Record how the annotations were made beside the dataset
    if callback.dataset_path is not None:
        provenance = {"source": str(Path(source).resolve()), "checkpoint": cfg.model.checkpoint, "copositive": params}
        with (Path(output_dir) / "copositive.yaml").open("w") as file:
            yaml.safe_dump(provenance, file, sort_keys=False)
    return callback.dataset_path
