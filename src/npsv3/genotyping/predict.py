"""Score the genotyping images with the model (stage 4 of the genotyper).

The images are read with the same loader and packing as training (`EnergonPackedImageDataModule`), in its predict mode
(`predict_unlabeled`, see `npsv3.models.energon.PackedGroupTaskEncoder`). Each region's raw (R, G) similarity of its real
image to each of its simulated images is recorded, rather than a call, so scoring (stage 5) can change without rerunning
inference.

- On the CPU, a single process runs the prediction with `torch.set_num_threads(cfg.threads)`.
- On one or more GPUs, there is one process per device, each with its own single-device `Trainer` and its own slice of the
  shards (the data module's `rank`/`world_size`). These are spawned with `torch.multiprocessing.spawn` rather than run with
  Lightning DDP, which would re-launch the whole genotyping entry point, and prediction doesn't need a process group.

Every process writes its own `predictions-rank{i}.pkl`, a pickled DataFrame with one row per region (see
`PredictionWriterCallback`). The regions the loader can't pack are listed up front in `rejected.jsonl` (see
`npsv3.models.energon.predict_rejections`), so every region in the image dataset is in exactly one of those files.
"""

import copy
import dataclasses
import glob
import json
import logging
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import yaml
from lightning.pytorch.plugins.environments import LightningEnvironment
from omegaconf import DictConfig, OmegaConf

from npsv3.genotyping.images import read_generation
from npsv3.genotyping.inputs import file_provenance
from npsv3.models.energon import EnergonPackedImageDataModule, max_group_cost
from npsv3.types import PathType

PREDICTIONS_DIR = "predictions"
"""The predictions, in the working directory"""

PREDICTIONS_PATTERN = "predictions-rank{rank}.pkl"
"""Each process's predictions, in the predictions directory"""

REJECTED_FILE = "rejected.jsonl"
"""The regions the loader can't pack (and so aren't predicted), in the predictions directory"""

PREDICTION_FILE = "prediction.yaml"
"""The settings the predictions were made with, in the predictions directory"""

PREDICTION_COLUMNS = ("key", "region", "num_genotypes", "num_replicates", "metric")


class PredictionWriterCallback(L.pytorch.callbacks.Callback):
    """Record every region's raw similarities during prediction as a pickled DataFrame at `path`.

    Columns:
        key: The example's key (`energon_key`), which identifies it in the image dataset and failures sidecar files
        region: The region (string)
        num_genotypes, num_replicates: G and R
        metric: (R, G) float32 similarities of the real image to each replicate's simulated image of each genotype, i.e.,
            the same quantities `RecordLossCallback` records

    The file is written (atomically) at the end of prediction, even if there were no regions.
    """

    def __init__(self, path: PathType):
        super().__init__()
        self.path = Path(path)
        self.rows = []

    def on_predict_start(self, trainer, model):  # noqa: ARG002
        self.rows = []

    def on_predict_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):  # noqa: ARG002
        metric, _preds, _labels, _genotype_label, num_genotypes, meta = outputs
        offsets = metric.offsets().tolist()
        values = metric.values().float().cpu().numpy()
        for group, (key, region) in enumerate(meta):
            num_genotype = int(num_genotypes[group])
            # Supports are replicate-major, i.e., support index = r*G + g
            group_metric = values[offsets[group] : offsets[group + 1]].reshape(-1, num_genotype)
            self.rows.append({
                "key": key,
                "region": region,
                "num_genotypes": num_genotype,
                "num_replicates": group_metric.shape[0],
                "metric": group_metric.astype(np.float32, copy=True),
            })

    def on_predict_end(self, trainer, model):  # noqa: ARG002
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(self.rows, columns=list(PREDICTION_COLUMNS))
        partial = self.path.with_name(f".{self.path.name}.partial")
        frame.to_pickle(partial)
        os.replace(partial, self.path)
        logging.info("Wrote predictions for %d regions to %s", len(frame), self.path)


@dataclasses.dataclass
class GenotypingPredictions:
    """The outputs of inference for genotyping"""

    predictions_dir: str
    """The directory with the predictions, rejections and settings"""
    paths: list[str]
    """Each process's predictions (see `PredictionWriterCallback`)"""
    rejected: list[dict]
    """The regions the loader rejected, `{key, region, sample, reason}` (see `npsv3.models.energon.predict_rejections`)"""


def resolve_devices(devices: str | int | Sequence[int] | None) -> list[int] | None:
    """The GPUs to predict on, one process per GPU, or None for the CPU.

    Args:
        devices: "cpu" (or None) for the CPU; "auto" for every visible GPU (or the CPU if there are none); a number of
            GPUs to use, starting with the first; or a list of GPU indices

    Raises:
        ValueError: If more GPUs are requested than are visible, or `devices` isn't one of the above
    """
    if devices is None or devices == "cpu":
        return None
    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if devices == "auto":
        return list(range(available)) or None
    if isinstance(devices, str):
        try:
            devices = int(devices)
        except ValueError:
            msg = f"devices must be 'cpu', 'auto', a number of GPUs or a list of GPU indices, not {devices!r}"
            raise ValueError(msg) from None
    gpus = list(range(devices)) if isinstance(devices, int) else [int(device) for device in devices]
    if not gpus or len(set(gpus)) != len(gpus) or any(not 0 <= gpu < available for gpu in gpus):
        msg = f"devices={devices!r} requests GPUs {gpus}, but {available} are visible"
        raise ValueError(msg)
    return gpus


def check_generation(cfg: DictConfig, generation: dict) -> None:
    """Raise ValueError if the images, as recorded in `generation.yaml`, don't match what `cfg`'s model expects: the
    channels (`pileup.image_channels` and `model.encoder.num_channels`), geometry (`pileup.image_{height,width}`) and
    replicates (`simulation.replicates`)."""
    image = generation["image"]
    expected = {
        "pileup.image_channels": (list(generation["pileup"]["image_channels"]), list(cfg.pileup.image_channels)),
        "pileup.image_height": (image["height"], cfg.pileup.image_height),
        "pileup.image_width": (image["width"], cfg.pileup.image_width),
        "simulation.replicates": (image["replicates"], cfg.simulation.replicates),
    }
    if (num_channels := OmegaConf.select(cfg, "model.encoder.num_channels", default=None)) is not None:
        expected["model.encoder.num_channels"] = (image["channels"], num_channels)
    mismatches = [
        f"{name}: images {actual}, configuration {value}" for name, (actual, value) in expected.items() if actual != value
    ]
    if image["replicates"] < 1:
        mismatches.append("replicates: there are no simulated images to score against")
    if mismatches:
        msg = "The images don't match the model's configuration (" + "; ".join(mismatches) + ")"
        raise ValueError(msg)


def predict_data_config(cfg: DictConfig, dataset_path: str, generation: dict, *, num_workers: int | None = None) -> DictConfig:
    """The data module's configuration for predicting over the image dataset `dataset_path`: `cfg.data` (the batch size,
    caching, buffering, ...) in predict mode, over just `dataset_path`, with the group limits from `generation`.

    `g_max` is `haplotype.max_diplotypes` and `r_max` the replicates, so the data module checks the batch size fits the
    largest possible region. Any training-only settings (filters, genotype subsets, co-positive targets) are cleared.
    """
    data = copy.deepcopy(cfg.data)
    width = generation["image"]["width"]
    overrides = {
        "_target_": "npsv3.models.energon.EnergonPackedImageDataModule",
        "metadataset_path": None,
        "train_metadataset_paths": dataset_path,
        "val_metadataset_paths": None,
        "test_metadataset_paths": None,
        "predict_split": "train",
        "limit_predict_batches": None,
        "predict_unlabeled": True,
        "w_bucket": width,
        "w_max": width,
        "g_max": generation["haplotype"]["max_diplotypes"],
        "r_max": generation["image"]["replicates"],
        "filters": None,
        "genotype_subset": None,
        "copositive": None,
    }
    if num_workers is not None:
        overrides["num_workers"] = num_workers
    for key, value in overrides.items():
        OmegaConf.update(data, key, value, force_add=True)
    return data


def _ensure_resolvers() -> None:
    """Register the custom resolvers (e.g. `${len:...}`) in a spawned process, which hasn't imported `npsv3.main`"""
    from npsv3.util.config import setup_resolvers

    if not OmegaConf.has_resolver("len"):
        setup_resolvers()


def _predict_rank(
    rank: int,
    world_size: int,
    cfg: DictConfig,
    data_cfg: DictConfig,
    predictions_dir: str,
    gpus: list[int] | None,
    precision: str | None,
) -> str:
    """Predict this rank's slice (of `world_size`) of the image dataset, on GPU `gpus[rank]` (or the CPU if `gpus` is
    None), and return the path of its predictions"""
    from npsv3.models.runners import load_model_from_checkpoint

    _ensure_resolvers()
    if gpus is None:
        torch.set_num_threads(cfg.threads)
        trainer_args = {"accelerator": "cpu", "devices": 1, "precision": precision or "32-true"}
    else:
        # Reduce precision to enable use of GPU tensor cores
        torch.set_float32_matmul_precision("high")
        default_precision = OmegaConf.select(cfg, "trainer.precision", default="32-true")
        trainer_args = {"accelerator": "gpu", "devices": [gpus[rank]], "precision": precision or default_precision}

    # Each process gets its own cache pool directory, since the pools don't coordinate
    cache_root = data_cfg.get("cache_dir") or tempfile.gettempdir()
    data_cfg = copy.deepcopy(data_cfg)
    OmegaConf.update(data_cfg, "cache_dir", os.path.join(cache_root, f"npsv3-predict-{os.getpid()}-rank{rank}"))
    OmegaConf.update(data_cfg, "rank", rank, force_add=True)
    OmegaConf.update(data_cfg, "world_size", world_size, force_add=True)
    dm = hydra.utils.instantiate(data_cfg)
    assert isinstance(dm, EnergonPackedImageDataModule)

    # Our own checkpoints include non-tensor state (e.g., the energon loader position) that `weights_only` rejects
    model = load_model_from_checkpoint(cfg, weights_only=False)

    path = os.path.join(predictions_dir, PREDICTIONS_PATTERN.format(rank=rank))
    trainer = L.Trainer(
        **trainer_args,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[PredictionWriterCallback(path)],
        # Each process is a standalone, single-device run, so don't let Lightning infer a cluster from the SLURM job's
        # environment (which, e.g., rejects `--ntasks` and would assign every process the job's rank 0 anyway)
        plugins=[LightningEnvironment()],
    )
    trainer.predict(model=model, datamodule=dm, return_predictions=False)
    return path


def _spawned_rank(rank: int, *args) -> None:
    """`torch.multiprocessing.spawn` entry point (which passes the rank first)"""
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [rank {rank}] %(levelname)s %(message)s")
    _predict_rank(rank, *args)


def predict_genotypes(
    cfg: DictConfig,
    dataset_path: PathType,
    predictions_dir: PathType,
    *,
    devices: str | int | Sequence[int] | None = "cpu",
    num_workers: int | None = None,
    precision: str | None = None,
) -> GenotypingPredictions:
    """Score every region in the genotyping image dataset `dataset_path` with the model `cfg.model.checkpoint`.

    Args:
        cfg: The genotyping configuration, for the model, the data module's settings (`cfg.data`), and the image settings
            to check the images against (see `check_generation`)
        dataset_path: The image dataset's `dataset.yaml` (`GenotypingImages.dataset_path`), whose directory also has the
            `generation.yaml`
        predictions_dir: Directory for the predictions, rejections and settings
        devices: Where to predict (see `resolve_devices`). Defaults to "cpu".
        num_workers: Data loader workers per process. Defaults to None, i.e., `cfg.data.num_workers`.
        precision: Lightning precision. Defaults to None, i.e., "32-true" on the CPU and `cfg.trainer.precision` on GPUs.

    Returns:
        The predictions, with every region in the dataset either predicted (in exactly one process's predictions) or
        rejected
    """
    dataset_path = os.path.abspath(dataset_path)
    predictions_dir = os.path.abspath(predictions_dir)
    os.makedirs(predictions_dir, exist_ok=True)
    # Remove any earlier predictions, e.g., from a run with more processes, so they can't be read with these
    for stale in glob.glob(os.path.join(predictions_dir, PREDICTIONS_PATTERN.format(rank="*"))):
        os.remove(stale)

    generation = read_generation(os.path.dirname(dataset_path))
    check_generation(cfg, generation)
    gpus = resolve_devices(devices)
    data_cfg = predict_data_config(cfg, dataset_path, generation, num_workers=num_workers)

    # Check the batch size fits the largest possible region before any process is started (the data module checks too,
    # but its error is wrapped by Hydra)
    largest = max_group_cost(
        budget=data_cfg.batch_size, g_max=data_cfg.g_max, r_max=data_cfg.r_max, w_max=data_cfg.w_max,
        w_bucket=data_cfg.w_bucket,
    )
    if largest > data_cfg.batch_size:
        msg = (
            f"data.batch_size ({data_cfg.batch_size}) must be at least the largest possible region's cost ({largest} image "
            f"slots for {data_cfg.r_max} replicates of {data_cfg.g_max} genotypes), so no region is dropped"
        )
        raise ValueError(msg)

    # The rejections are found from the metadata, once, rather than per process
    rejected = hydra.utils.instantiate(data_cfg).predict_rejections()
    with open(os.path.join(predictions_dir, REJECTED_FILE), "w") as file:
        file.writelines(json.dumps(row) + "\n" for row in rejected)

    with open(os.path.join(predictions_dir, PREDICTION_FILE), "w") as file:
        yaml.safe_dump(
            {
                "checkpoint": file_provenance(cfg.model.checkpoint),
                "dataset": dataset_path,
                "devices": "cpu" if gpus is None else gpus,
                "world_size": 1 if gpus is None else len(gpus),
            },
            file,
            sort_keys=True,
        )

    world_size = 1 if gpus is None else len(gpus)
    logging.info(
        "Predicting %s on %s", dataset_path, "the CPU" if gpus is None else f"{world_size} GPU(s) {gpus}"
    )
    args = (world_size, cfg, data_cfg, predictions_dir, gpus, precision)
    if world_size == 1:
        paths = [_predict_rank(0, *args)]
    else:
        torch.multiprocessing.spawn(_spawned_rank, args=args, nprocs=world_size, join=True)
        paths = [os.path.join(predictions_dir, PREDICTIONS_PATTERN.format(rank=rank)) for rank in range(world_size)]
    return GenotypingPredictions(predictions_dir=predictions_dir, paths=paths, rejected=rejected)


def read_predictions(predictions_dir: PathType) -> pd.DataFrame:
    """Every process's predictions in `predictions_dir`, as one DataFrame indexed by `key` (see
    `PredictionWriterCallback` for the columns)

    Raises:
        ValueError: If a region was predicted more than once
    """
    with open(os.path.join(predictions_dir, PREDICTION_FILE)) as file:
        world_size = yaml.safe_load(file)["world_size"]
    frames = [
        pd.read_pickle(os.path.join(predictions_dir, PREDICTIONS_PATTERN.format(rank=rank)))  # noqa: S301 (our own files)
        for rank in range(world_size)
    ]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(PREDICTION_COLUMNS))
    if frame["key"].duplicated().any():
        duplicates = frame.loc[frame["key"].duplicated(), "key"].tolist()
        msg = f"Regions predicted more than once: {duplicates[:5]}"
        raise ValueError(msg)
    return frame.set_index("key", drop=False)


def read_rejections(predictions_dir: PathType) -> list[dict]:
    """The regions the loader rejected, as recorded in `predictions_dir`"""
    with open(os.path.join(predictions_dir, REJECTED_FILE)) as file:
        return [json.loads(line) for line in file if line.strip()]
