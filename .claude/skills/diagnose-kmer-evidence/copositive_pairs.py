"""Record co-positive diagnostics with R=2 by pairing two independently simulated image sets of the same regions.

`copositive.py record` needs `simulation.replicates >= 2` for a held-out simulated query (`conf`, `sim_loss`). With a
deterministic haplotype strategy (e.g. `TrueHaplotypeStrategy`) two image sets generated with different `+repeat=` values
share each region's candidate haplotypes, genotypes and label, and differ only in the simulated reads (and in the real
image's random read subsampling where depth exceeds the image height). Stacking their sims gives R=2 replicates without
regenerating images.

    python3 copositive_pairs.py --ckpt model.ckpt --rep0 <images dir A> --rep1 <images dir B> --output cop.pkl.gz

Replicate 0 is from `--rep0` (typically the set the model was trained on), replicate 1 from `--rep1` (held out); the real
query is `--rep1`'s. Besides `copositive.analyze_group`'s columns, each row has `loss` (InfoNCE against `--rep1` sims only,
which reproduces `command=loss` on `--rep1`), `loss_rep0`, the per-replicate posteriors and the raw `query_logits`.
Regions whose genotype count, label, label ranks or width differ between the sets are skipped (and counted).

The model architecture comes from the training run's Hydra config (default: `.hydra/config.yaml` beside the checkpoint),
since the checkpoint does not store the encoder/metric/loss/predictor objects.
"""

import argparse
import contextlib
import gzip
import json
import logging
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Same directory as this script, so on sys.path when run as `python3 $S/copositive_pairs.py`
from copositive import analyze_group, infonce, pooled_softmax

REAL, SIMS = "real.image.bin.gz", "sims.image.bin.gz"


def read_meta(directory) -> dict[str, dict]:
    """key -> JSON metadata record of an energon image dataset"""
    with open(Path(directory) / "meta_ds" / "train.jsonl") as file:
        return {(m := json.loads(line))["key"]: m for line in file}


def tar_index(directory) -> dict[str, tuple[str, int, int]]:
    """member name -> (tar path, data offset, size) for every pixel member of an energon image dataset"""
    index = {}
    for path in sorted((Path(directory) / "imgs_ds").glob("*.tar")):
        with tarfile.open(path) as tar:
            for member in tar.getmembers():
                index[member.name] = (str(path), member.offset_data, member.size)
    return index


def paired_keys(metas: list[dict], g_max: int | None = 36) -> tuple[list[str], int]:
    """Keys present and consistent (labeled, R=1, same G/label/ranks/width) in both datasets, and the number skipped"""
    keys, skipped = [], 0
    for key, m1 in metas[1].items():
        m0 = metas[0].get(key)
        ok = (
            m0 is not None and m1.get("label") is not None and m0["r"] == m1["r"] == 1
            and m0["g"] == m1["g"] and m0["label"] == m1["label"] and m0["label_rank"] == m1["label_rank"]
            and m0["w"] == m1["w"] and (g_max is None or m0["g"] <= g_max)
        )
        if ok:
            keys.append(key)
        else:
            skipped += 1
    return keys, skipped


class Pairs(torch.utils.data.Dataset):
    """Item i: (key, uint8 images (2G+1, C, H, W) as [rep0 sims, rep1 sims, rep1 real], whether the real images match)"""

    def __init__(self, keys, metas, indexes):
        self.keys, self.metas, self.indexes = keys, metas, indexes

    def _array(self, which, name, shape):
        path, offset, size = self.indexes[which][name]
        with open(path, "rb") as file:
            file.seek(offset)
            raw = file.read(size)
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return np.frombuffer(bytearray(raw), dtype=np.uint8).reshape(shape)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        key = self.keys[i]
        m = self.metas[0][key]
        h, w, c, g = m["h"], m["w"], m["c"], m["g"]
        reals = [self._array(j, f"{key}.{REAL}", (h, w, c)) for j in (0, 1)]
        sims = [self._array(j, f"{key}.{SIMS}", (1, g, h, w, c)) for j in (0, 1)]
        images = np.concatenate([np.concatenate(sims, axis=0).reshape(2 * g, h, w, c), reals[1][None]], axis=0)
        return key, torch.from_numpy(images).movedim(-1, -3).contiguous(), bool(np.array_equal(reals[0], reals[1]))


def load_model(ckpt: str, config: str | None, device: str):
    import hydra
    from omegaconf import OmegaConf

    from npsv3.util.config import setup_resolvers

    with contextlib.suppress(ValueError):  # already registered
        setup_resolvers()
    cfg = OmegaConf.load(config or Path(ckpt).parent / ".hydra" / "config.yaml")
    model_cls = hydra.utils.get_class(cfg.model._target_)
    parts = {key: hydra.utils.instantiate(cfg.model[key]) for key in model_cls.ignored_hyperparameters if key in cfg.model}
    model = model_cls.load_from_checkpoint(ckpt, **parts, weights_only=False, map_location=device)
    return model.eval(), cfg


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", help="Training run's Hydra config (default: <ckpt dir>/.hydra/config.yaml)")
    parser.add_argument("--rep0", required=True, help="Image dataset directory (containing meta_ds/ and imgs_ds/)")
    parser.add_argument("--rep1", required=True, help="Second image dataset of the same regions; supplies the real query")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, help="Only the first N paired regions (smoke test)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=512, help="Max images per encoder call")
    args = parser.parse_args()

    metas = [read_meta(args.rep0), read_meta(args.rep1)]
    keys, skipped = paired_keys(metas)
    logging.info("%d paired regions, %d unpaired/mismatched/unlabeled skipped", len(keys), skipped)
    if args.limit:
        keys = keys[: args.limit]
    indexes = [tar_index(args.rep0), tar_index(args.rep1)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.ckpt, args.config, device)
    if not isinstance(model.metric, torch.nn.CosineSimilarity):
        msg = "Only cosine-similarity models are supported"
        raise NotImplementedError(msg)
    tau = model.loss.temperature
    mean, std = cfg.data.mean[0], cfg.data.std[0]

    loader = torch.utils.data.DataLoader(
        Pairs(keys, metas, indexes), batch_size=None, shuffle=False, num_workers=args.workers,
        prefetch_factor=8 if args.workers else None,
    )
    rows, real_mismatch = [], 0
    # Matches training/`command=loss` (16-mixed precision, ToDtype(scale=True) + Normalize(mean, std))
    with torch.inference_mode(), torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        for n, (key, images, real_same) in enumerate(loader):
            real_mismatch += not real_same
            m = metas[1][key]
            G, R, label = m["g"], 2, m["label"]  # noqa: N806
            x = images.to(device, non_blocking=True).float().div_(255).sub_(mean).div_(std)
            emb = torch.cat([model.encoder(chunk) for chunk in x.split(args.chunk)]).float()
            emb = torch.nn.functional.normalize(emb, dim=-1)
            support, query = emb[:-1], emb[-1]
            query_logits = (support @ query / tau).cpu().numpy().reshape(R, G)
            support_logits = (support @ support.T / tau).cpu().numpy()
            rows.append({
                "region": m["region"], "key": key, "genotype": label, "num_genotypes": G, "num_replicates": R,
                "real_same": real_same,
                "loss": infonce(query_logits[1:], label), "loss_rep0": infonce(query_logits[:1], label),
                "p_real_rep0": pooled_softmax(query_logits[:1]), "p_real_rep1": pooled_softmax(query_logits[1:]),
                "query_logits": query_logits.astype(np.float32),
                **analyze_group(query_logits, support_logits, R, G, label),
            })
            if n % 2000 == 0:
                logging.info("%d/%d groups", n, len(keys))
    df = pd.DataFrame(rows)
    df["prediction"] = df["p_real"].map(np.argmax)
    df.to_pickle(args.output)
    logging.info("Wrote %d groups (%d with differing real images) to %s", len(df), real_mismatch, args.output)


if __name__ == "__main__":
    main()
