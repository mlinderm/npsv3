"""Co-positive diagnostic: how much of the high-loss mass comes from genotypes the model cannot separate even in simulation?

Implements the per-group quantities from "Bootstrapping co-positives" in REFERENCE_HAPLOTYPE_NOTES.md for a trained
(single-positive) model M0, and splits the high-loss groups into:

  a_lookalike  the real query's top alternative g' is confused with the label g* even when the query is a *simulated*
               replicate of g* (conf[g'] >= threshold). Co-positive weights / masking would help.
  a_semantic   g' is separable in simulation but Truvari-equivalent to the sample's true assembled paths (needs `semantic`).
               The label, not the loss, is the problem; g' is arguably correct.
  b_flag       g' is separable in simulation, not semantically equivalent, and the reads prefer it over g*. A label or
               simulation problem (the chr4 type); bootstrapping should NOT absorb it.
  c_model      g' is separable and the reads do not prefer it; the model is just uncertain on the real image.

Subcommands:

  record    Run M0 over a data split (same Hydra overrides as `npsv3 command=loss`) and write a per-group DataFrame:
              python3 copositive.py record --output cop.pkl.gz -- model=paired_packed_inception_infonce \\
                  model.checkpoint=best.ckpt data.train_metadataset_paths=[...] data.limit_predict_batches=200
            Use data generated with simulation.replicates >= 2: with one replicate there is no held-out replicate of g*,
            so `conf` falls back to a noise-free lower bound (query == its own support) and `sim_loss` is undefined.
  semantic  Compute the Truvari term from `capture.py` pickles (one per region; needs the `harmonize` extras):
              python3 copositive.py semantic caps/*.pkl --output sem.pkl.gz
  classify  Summarize and split the high-loss groups:
              python3 copositive.py classify cop.pkl.gz [--semantic sem.pkl.gz] [--regions-out todo.tsv]

Supports are replicate-major (support index = r*G + g), as everywhere else in the codebase.
"""

import argparse
import itertools
import logging
import math
import pickle
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------------------------------
# Per-group math (numpy only, so it can be checked without a model)
# ---------------------------------------------------------------------------------------------------------------------


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
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


def infonce(logits: np.ndarray, label: int, mask: np.ndarray | None = None) -> float:
    """`paired.InfoNCE` for one query against (R, G) logits: mean over positives of -log(e^p / (e^p + sum(e^negatives)))."""
    if mask is None:
        mask = np.zeros_like(logits, dtype=bool)
    is_positive = np.zeros_like(mask)
    is_positive[:, label] = True
    positives = logits[is_positive & ~mask]
    negatives = logits[~is_positive & ~mask]
    if positives.size == 0:
        return math.nan
    log_negatives = _logsumexp(negatives, axis=0) if negatives.size else -np.inf
    return float(np.mean(np.logaddexp(positives, log_negatives) - positives))


def sim_confusion(support_logits: np.ndarray, num_replicates: int, num_genotypes: int) -> tuple[np.ndarray, bool]:
    """C[h, g]: mean over replicates r of h of the pooled softmax probability of g, querying with simulated support (r, h).

    The query is excluded from its own supports. With a single replicate that would leave h with no supports, so the
    query is kept instead (query == support), a noise-free lower bound on confusion; the second return value flags it.
    """
    R, G = num_replicates, num_genotypes
    lower_bound = R < 2  # noqa: PLR2004
    C = np.zeros((G, G))
    for h in range(G):
        for r in range(R):
            mask = np.zeros((R, G), dtype=bool)
            if not lower_bound:
                mask[r, h] = True
            C[h] += pooled_softmax(support_logits[r * G + h].reshape(R, G), mask)
        C[h] /= R
    return C, lower_bound


def copositive_conf(C: np.ndarray, label: int) -> np.ndarray:
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


def sim_loss(support_logits: np.ndarray, num_replicates: int, num_genotypes: int, label: int) -> float:
    """InfoNCE when each replicate of g* in turn is the (simulated) query: the loss the evidence allows without a sim/real gap."""
    R, G = num_replicates, num_genotypes
    if R < 2:  # noqa: PLR2004
        return math.nan
    losses = []
    for r in range(R):
        mask = np.zeros((R, G), dtype=bool)
        mask[r, label] = True
        losses.append(infonce(support_logits[r * G + label].reshape(R, G), label, mask))
    return float(np.mean(losses))


def analyze_group(query_logits: np.ndarray, support_logits: np.ndarray, num_replicates: int, num_genotypes: int,
                  label: int, conf_threshold: float = 0.5) -> dict:
    """All per-group quantities from scaled logits: `query_logits` (R, G), `support_logits` (R*G, R*G)."""
    R, G = num_replicates, num_genotypes
    C, lower_bound = sim_confusion(support_logits, R, G)
    conf = copositive_conf(C, label)
    k = int(np.sum(np.delete(conf, label) >= conf_threshold))
    return {
        "p_real": pooled_softmax(query_logits),
        "real_infonce": infonce(query_logits, label),
        "sim_loss": sim_loss(support_logits, R, G, label),
        "confusion": C,
        "conf": conf,
        "conf_lower_bound": lower_bound,
        "k_lookalike": k,
        # Lowest achievable `real_infonce` if the k look-alikes were truly identical to g*
        "floor": math.log1p(k * R),
    }


# ---------------------------------------------------------------------------------------------------------------------
# record: run M0 and capture embeddings
# ---------------------------------------------------------------------------------------------------------------------


def _make_callback(path: str, temperature: float | None, conf_threshold: float):
    import lightning as L
    import torch

    from npsv3.models.paired import MinimizingPredictor

    class RecordCopositiveCallback(L.pytorch.callbacks.Callback):
        """Capture the encoder's embeddings for each packed batch and compute the per-group co-positive quantities."""

        def __init__(self):
            super().__init__()
            self.rows = []
            self._embeddings = None
            self._handle = None

        def on_predict_start(self, trainer, model):  # noqa: ARG002
            self.rows = []
            # Hook the encoder rather than re-encoding, so support-vs-support metrics cost no extra forward pass
            self._handle = model.encoder.register_forward_hook(self._capture)
            # Distances (minimized) become logits by negation, matching how the predictor ranks them
            self.sign = -1.0 if isinstance(model.predictor, MinimizingPredictor) else 1.0
            self.tau = temperature if temperature is not None else getattr(model.loss, "temperature", None)
            if not isinstance(self.tau, (int, float)):
                msg = f"Loss {type(model.loss).__name__} has no scalar temperature; pass --temperature"
                raise ValueError(msg)

        def _capture(self, module, inputs, output):  # noqa: ARG002
            self._embeddings = output.detach()

        def on_predict_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):  # noqa: ARG002
            metric, preds, labels, genotype_label, num_genotypes, regions = outputs
            embeddings = self._embeddings
            offsets = metric.offsets().tolist()
            metric_values, label_values = metric.values(), labels.values()
            num_support = offsets[-1]

            with torch.inference_mode():
                for group, region in enumerate(regions):
                    start, end = offsets[group], offsets[group + 1]
                    G = int(num_genotypes[group])
                    R = (end - start) // G
                    label = int(genotype_label[group])
                    if label < 0:
                        continue

                    # The model's own loss for this group, exactly as RecordLossCallback computes it
                    group_offsets = torch.tensor([0, end - start], device=metric_values.device)
                    loss = model.loss(
                        torch.nested.nested_tensor_from_jagged(metric_values[start:end], offsets=group_offsets),
                        torch.nested.nested_tensor_from_jagged(label_values[start:end], offsets=group_offsets),
                    )[0].item()

                    support = embeddings[start:end]
                    query = embeddings[num_support + group]
                    N = end - start
                    query_metric = model.metric(query.expand(N, -1), support)
                    support_metric = model.metric(support.repeat_interleave(N, dim=0), support.repeat(N, 1)).reshape(N, N)
                    to_logits = lambda m: (self.sign * m.float() / self.tau).cpu().numpy()  # noqa: E731

                    row = analyze_group(to_logits(query_metric).reshape(R, G), to_logits(support_metric), R, G, label,
                                        conf_threshold)
                    self.rows.append({
                        "region": region,
                        "loss": loss,
                        "genotype": label,
                        "prediction": int(preds[group]) % G,
                        "num_genotypes": G,
                        "num_replicates": R,
                        "label_rank": label_values[start : start + G].cpu().numpy().copy(),
                        **row,
                    })

        def on_predict_end(self, trainer, model):  # noqa: ARG002
            self._handle.remove()
            pd.DataFrame(self.rows).to_pickle(path)
            logging.info("Wrote co-positive diagnostics for %d groups to %s", len(self.rows), path)

    return RecordCopositiveCallback()


def record(args, overrides):
    import torch
    from hydra import compose, initialize_config_module

    from npsv3.models.runners import predict
    from npsv3.util.config import setup_resolvers

    setup_resolvers()
    with initialize_config_module("npsv3.conf", version_base=None):
        cfg = compose("config", overrides=["command=loss", *overrides])
    torch.set_num_threads(cfg.threads)
    callback = _make_callback(args.output, args.temperature, args.conf_threshold)
    predict(cfg, return_predictions=False, callbacks=[callback])


# ---------------------------------------------------------------------------------------------------------------------
# semantic: Truvari equivalence of each candidate genotype to the sample's true assembled paths
# ---------------------------------------------------------------------------------------------------------------------


def semantic_vector(reference: str, haplotypes: list[str], true_paths: list[str], hap_equiv=None) -> np.ndarray:
    """sem[g] = 1 if genotype g's two haplotypes can be paired with the true paths such that both pairs are equivalent."""
    if hap_equiv is None:
        from npsv3.graphs.sequence_compare import haplotypes_differ

        def hap_equiv(a, b):
            return not haplotypes_differ(reference, a, b)

    equiv = np.array([[hap_equiv(hap, true) for true in true_paths] for hap in haplotypes], dtype=bool)
    genotypes = list(itertools.combinations_with_replacement(range(len(haplotypes)), 2))
    return np.array(
        [float((equiv[a, 0] and equiv[b, 1]) or (equiv[a, 1] and equiv[b, 0])) for a, b in genotypes]
    )


def semantic(args):
    rows = []
    for path in args.captures:
        with open(path, "rb") as file:
            cap = pickle.load(file)  # noqa: S301 -- our own capture.py output
        if cap.get("true_paths") is None or any(p is None for p in cap["true_paths"]):
            logging.warning("%s: no assembled true paths for %s, skipping", path, cap["region"])
            continue
        sem = semantic_vector(cap["graph_ref_path"], cap["haps"], cap["true_paths"])
        true = cap.get("true")
        genotypes = list(itertools.combinations_with_replacement(range(len(cap["haps"])), 2))
        label = genotypes.index(tuple(sorted(true))) if true is not None and None not in true else None
        rows.append({"region": cap["region"], "sample": cap["sample"], "sem": sem, "sem_label": label})
        print(f"{cap['region']}\t{cap['sample']}\tlabel={label}\tsem={sem.astype(int).tolist()}")
    pd.DataFrame(rows).to_pickle(args.output)


# ---------------------------------------------------------------------------------------------------------------------
# classify: split the high-loss mass
# ---------------------------------------------------------------------------------------------------------------------


CATEGORIES = ("a_lookalike", "a_semantic", "b_flag", "c_model")


def classify_row(row, conf_threshold: float, sem: np.ndarray | None = None) -> pd.Series:
    """Category of one group from its real query's top alternative g' (see module docstring)."""
    label, p = row["genotype"], row["p_real"]
    alternatives = np.delete(np.arange(len(p)), label)
    alt = int(alternatives[np.argmax(p[alternatives])])
    conf_alt = float(row["conf"][alt])
    sem_alt = float(sem[alt]) if sem is not None else math.nan
    if conf_alt >= conf_threshold:
        category = "a_lookalike"
    elif sem_alt == 1.0:
        category = "a_semantic"
    elif p[alt] >= p[label]:
        category = "b_flag"
    else:
        category = "c_model"
    return pd.Series({
        "alt": alt,
        "conf_alt": conf_alt,
        "sem_alt": sem_alt,
        "p_label": float(p[label]),
        "p_alt": float(p[alt]),
        # Genotype 0 is homozygous reference haplotype 0, so this is a cheap proxy for "SV presence differs"
        "presence_differs": (alt == 0) != (label == 0),
        "category": category,
    })


def _summary(df: pd.DataFrame, title: str) -> None:
    total_loss = df["loss"].sum()
    print(f"\n{title}: {len(df)} groups, total loss {total_loss:.1f}")
    print(f"{'category':<12} {'n':>7} {'%groups':>8} {'%loss':>7} {'med loss':>9} {'med sim':>8} {'%pres':>6}")
    for category in CATEGORIES:
        sub = df[df["category"] == category]
        if sub.empty:
            continue
        print(
            f"{category:<12} {len(sub):>7} {100 * len(sub) / len(df):>7.1f}% {100 * sub['loss'].sum() / total_loss:>6.1f}% "
            f"{sub['loss'].median():>9.3f} {sub['sim_loss'].median():>8.3f} {100 * sub['presence_differs'].mean():>5.1f}%"
        )


def classify(args):
    df = pd.read_pickle(args.diagnostics)  # noqa: S301
    sem_by_region = {}
    if args.semantic:
        for _, row in pd.read_pickle(args.semantic).iterrows():  # noqa: S301
            sem_by_region[row["region"]] = row["sem"]

    def sem_for(row):
        sem = sem_by_region.get(row["region"])
        if sem is not None and len(sem) != row["num_genotypes"]:
            logging.warning("%s: capture has %d genotypes, example has %d; ignoring semantic term",
                            row["region"], len(sem), row["num_genotypes"])
            return None
        return sem

    df = df.join(df.apply(lambda row: classify_row(row, args.conf_threshold, sem_for(row)), axis=1))
    threshold = df["loss"].quantile(args.loss_quantile)
    high = df[df["loss"] >= threshold]

    replicates = df["num_replicates"].value_counts()
    print(f"Groups: {len(df)}; replicates per group: {dict(replicates)}")
    if df["conf_lower_bound"].any():
        print(f"WARNING: {int(df['conf_lower_bound'].sum())} groups have a single replicate, so conf is a noise-free "
              "lower bound and sim_loss is undefined; regenerate with simulation.replicates >= 2")
    if args.semantic:
        print(f"Semantic term available for {int(df['region'].isin(sem_by_region).sum())} groups")

    _summary(df, "All groups")
    _summary(high, f"High-loss groups (loss >= {threshold:.3f}, top {100 * (1 - args.loss_quantile):.0f}%)")

    # If look-alikes explain the loss, real InfoNCE should sit near (not far above) the log(1 + kR) floor
    print("\nLoss vs. look-alike floor (all groups), by number of look-alike negatives k:")
    print(f"{'k':>3} {'n':>7} {'floor log(1+kR)':>16} {'med real':>9} {'med sim':>8} {'med real-floor':>15}")
    for k, sub in df.groupby("k_lookalike"):
        print(f"{k:>3} {len(sub):>7} {sub['floor'].median():>16.3f} {sub['real_infonce'].median():>9.3f} "
              f"{sub['sim_loss'].median():>8.3f} {(sub['real_infonce'] - sub['floor']).median():>15.3f}")

    if args.regions_out:
        columns = ["region", "category", "loss", "sim_loss", "genotype", "alt", "conf_alt", "p_label", "p_alt"]
        todo = high[high["category"].isin(args.export)].sort_values("loss", ascending=False)
        todo[columns].to_csv(args.regions_out, sep="\t", index=False, float_format="%.4g")
        print(f"\nWrote {len(todo)} high-loss {'/'.join(args.export)} regions to {args.regions_out} (inputs for capture.py)")
    if args.output:
        df.to_pickle(args.output)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    argv = sys.argv[1:] if argv is None else argv
    # Everything after `--` is Hydra overrides for `record`
    overrides = []
    if "--" in argv:
        split = argv.index("--")
        argv, overrides = argv[:split], argv[split + 1 :]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    threshold = argparse.ArgumentParser(add_help=False)
    threshold.add_argument("--conf-threshold", type=float, default=0.5, help="conf[g'] at or above which g' is a look-alike")
    commands = parser.add_subparsers(dest="command", required=True)

    record_parser = commands.add_parser("record", parents=[threshold], help="Run the model and record per-group diagnostics")
    record_parser.add_argument("--output", required=True, help="Output DataFrame (.pkl.gz)")
    record_parser.add_argument("--temperature", type=float, help="Softmax temperature (default: the loss's)")

    semantic_parser = commands.add_parser("semantic", help="Truvari equivalence from capture.py pickles")
    semantic_parser.add_argument("captures", nargs="+")
    semantic_parser.add_argument("--output", required=True)

    classify_parser = commands.add_parser("classify", parents=[threshold], help="Split the high-loss mass into categories")
    classify_parser.add_argument("diagnostics", help="Output of `record`")
    classify_parser.add_argument("--semantic", help="Output of `semantic`")
    classify_parser.add_argument("--loss-quantile", type=float, default=0.9, help="High-loss cutoff quantile")
    classify_parser.add_argument("--regions-out", help="TSV of high-loss regions to capture next")
    classify_parser.add_argument("--export", nargs="+", default=["a_lookalike", "b_flag"], choices=CATEGORIES)
    classify_parser.add_argument("--output", help="Write the classified DataFrame")

    args = parser.parse_args(argv)
    if args.command == "record":
        record(args, overrides)
    elif args.command == "semantic":
        semantic(args)
    else:
        classify(args)


if __name__ == "__main__":
    main()
