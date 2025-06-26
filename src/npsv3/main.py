import json
import logging
import os
import typing

import hydra
from omegaconf import DictConfig, OmegaConf

from npsv3.simulation import bwa_index_loaded


def _check_shared_reference(cfg: DictConfig):
    """Check if BWA shared index is loaded, loading it if specified configuration"""
    if cfg.simulation.replicates > 0:
        cfg.shared_reference = bwa_index_loaded(hydra.utils.to_absolute_path(cfg.reference), load=cfg.load_reference)
        if not cfg.shared_reference:
            logging.warning(
                "Consider loading BWA indices into shared memory before generating examples with 'bwa shm %s'",
                cfg.reference,
            )


def _make_paths_absolute(cfg: DictConfig, keys: typing.Iterable[str]):
    """Make list of hydra configuration keys, e.g. 'pileup.snv_vcf_input' absolute paths"""
    for key in keys:
        if not OmegaConf.is_missing(cfg, key) and OmegaConf.select(cfg, key) is not None:
            OmegaConf.update(cfg, key, hydra.utils.to_absolute_path(OmegaConf.select(cfg, key)))


def _to_webdataset_urls(urls: str | typing.Iterable[str]) -> str:
    # Join multiple URLs with "::" separator expected by WebDataset
    list_of_urls = urls.split("::") if isinstance(urls, str) else urls
    list_of_urls = [hydra.utils.to_absolute_path(url) for url in list_of_urls]
    return "::".join(list_of_urls)


OmegaConf.register_new_resolver("strip_ext", lambda path: os.path.splitext(path)[0])
OmegaConf.register_new_resolver("len", lambda arg: len(arg))

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # print("\ncommand:",cfg.command)
    if cfg.command == "preprocess":
        from npsv3.util.sample import compute_read_stats

        # If no output file is specified, create a fixed file in the Hydra output directory
        if OmegaConf.is_missing(cfg, "output"):
            output = "stats.json"
        else:
            output = hydra.utils.to_absolute_path(cfg.output)

        _make_paths_absolute(cfg, ["reference"])

        stats = compute_read_stats(cfg, hydra.utils.to_absolute_path(cfg.reads))
        with open(output, "w") as file:
            json.dump(stats, file)

    elif cfg.command == "images":
        from npsv3.util.sample import Sample

        _make_paths_absolute(cfg, ["reference", "stats_path"])
        _check_shared_reference(cfg)

        sample = Sample.from_json(hydra.utils.to_absolute_path(cfg.stats_path))

        # If no output directory is specified, use the Hydra output directory (the current working directory)
        if OmegaConf.is_missing(cfg, "output"):
            output = os.getcwd()
        else:
            output = hydra.utils.to_absolute_path(cfg.output)

        vcf_to_examples = hydra.utils.get_method(cfg.pileup.example_fn)
        vcf_to_examples(
            cfg,
            hydra.utils.to_absolute_path(cfg.reads),
            sample,
            hydra.utils.to_absolute_path(cfg.input),
            output,
            background_vcf=hydra.utils.to_absolute_path(cfg.background),
            progress_bar=True,
        )
    elif cfg.command == "train":
        import torch
        torch.set_float32_matmul_precision("high")

        from npsv3.models.runners import train

        torch.set_num_threads(cfg.threads)

        # If no output directory is specified, use the Hydra output directory (the current working directory)
        if OmegaConf.is_missing(cfg, "output"):
            output = os.getcwd()
        else:
            output = hydra.utils.to_absolute_path(cfg.output)

        OmegaConf.update(cfg, "data.train_urls", _to_webdataset_urls(cfg.data.train_urls), merge=False)
        if not OmegaConf.is_missing(cfg, "data.validate_urls") and OmegaConf.select(cfg, "data.validate_urls") is not None:
            OmegaConf.update(cfg, "data.validate_urls", _to_webdataset_urls(cfg.data.validate_urls), merge=False)

        train(cfg, output_dir=output, limit_train_batches=1.0)
        # TODO: Create link to the best model to serve as the final model

    elif cfg.command == "full_train":
        import torch
        torch.set_float32_matmul_precision("high")

        from npsv3.models.runners import train, assess_accuracy

        torch.set_num_threads(cfg.threads)

        # If no output directory is specified, use the Hydra output directory (the current working directory)
        if OmegaConf.is_missing(cfg, "output"):
            output = os.getcwd()
        else:
            output = hydra.utils.to_absolute_path(cfg.output)

        OmegaConf.update(cfg, "data.train_urls", _to_webdataset_urls(cfg.data.train_urls), merge=False)
        OmegaConf.update(cfg, "data.predict_urls", _to_webdataset_urls(cfg.data.predict_urls), merge=False)
        if not OmegaConf.is_missing(cfg, "data.validate_urls") and OmegaConf.select(cfg, "data.validate_urls") is not None:
            OmegaConf.update(cfg, "data.validate_urls", _to_webdataset_urls(cfg.data.validate_urls), merge=False)

        pretraining_model = cfg.model
        # print(cfg.data.train_urls)
        ckpt_path = train(cfg, output_dir=output, limit_train_batches=10)
        OmegaConf.update(cfg, "model._target_", "npsv3.models.transformer.Classifier", merge=False)
        # print("\ncheckpoint path:",ckpt_path)
        OmegaConf.update(cfg, "pretrained.path", ckpt_path, merge=False)
        OmegaConf.update(cfg, "checkpoint.name", "full_train-{step}", merge=False)
        ckpt_path = train(cfg, output_dir=output, limit_train_batches=10)
        assess_accuracy(cfg, ckpt_path, limit_predict_batches=1.0)
        # print(cfg.data._target_, cfg.data.batch_size, cfg.data, pretraining_model, cfg.pileup, cfg.trainer.max_epochs)

    elif cfg.command == "assess_accuracy":
        # print("\nassessing accuracy")
        import torch
        torch.set_float32_matmul_precision("high")
        from npsv3.models.runners import assess_accuracy

        OmegaConf.update(cfg, "data.predict_urls", _to_webdataset_urls(cfg.data.predict_urls), merge=False)

        ckpt_path = cfg.pretrained.path
        assess_accuracy(cfg, ckpt_path, limit_predict_batches=1.0)

    elif cfg.command == "test":
        import torch

        from npsv3.models.paired import test

        torch.set_num_threads(cfg.threads)

        _make_paths_absolute(cfg, ["model.checkpoint"])
        OmegaConf.update(cfg, "data.test_urls", _to_webdataset_urls(cfg.data.test_urls), merge=False)

        test(cfg)

    elif cfg.command == "predict":
        import torch

        from npsv3.models.paired import predict

        torch.set_num_threads(cfg.threads)

        _make_paths_absolute(cfg, ["model.checkpoint"])
        OmegaConf.update(cfg, "data.prediction_urls", _to_webdataset_urls(cfg.data.prediction_urls), merge=False)

        predict(cfg)

    elif cfg.command == "encode":
        import torch

        from npsv3.models.dvae import encode

        torch.set_num_threads(cfg.threads)

         # If no output directory is specified, use the Hydra output directory (the current working directory)
        if OmegaConf.is_missing(cfg, "output"):
            output = os.getcwd()
        else:
            output = hydra.utils.to_absolute_path(cfg.output)

        _make_paths_absolute(cfg, ["model.checkpoint"])
        OmegaConf.update(cfg, "data.predict_urls", _to_webdataset_urls(cfg.data.predict_urls), merge=False)

        encode(cfg, output_dir=output)
    else:
        msg = f"Command {cfg.command} not implemented"
        raise NotImplementedError(msg)


if __name__ == "__main__":
    main()
