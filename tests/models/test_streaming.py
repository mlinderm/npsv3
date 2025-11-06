import json
import math
from typing import Optional

import pytest
import lightning as L
import torch
import webdataset as wds
from streaming import Stream, StreamingDataLoader

from npsv3.models.streaming import PackableDataModule, PackableMDSWriter, PackableStreamingDataset

from .. import data_path

pytest.skip(allow_module_level=True)

# Adapted from: https://github.com/Lightning-AI/pytorch-lightning/blob/851e66f475eda49a1e1ce3e716a5c487172fea1c/src/lightning/pytorch/demos/boring_classes.py#L96
class DummyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

    def loss(self, preds: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        if labels is None:
            labels = torch.ones_like(preds)
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(preds, labels)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        return self.loss(output)

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LRScheduler]]:
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


class TestPackedStreaming:
    def test_customized_index(self, tmp_path):
        shard_dir = tmp_path / "shards"
        columns = {
            "image": "ndarray:uint8",
            "sim.images": "ndarray:uint8",
            "label": "int",
        }
        with PackableMDSWriter(out=str(shard_dir), columns=columns, compression="zstd") as writer:
            dataset = wds.WebDataset(data_path("images-0000.tar"), shardshuffle=False).decode()
            for _i, sample in enumerate(dataset):
                mds_sample = {
                    "image": sample["image.npy.gz"],
                    "sim.images": sample["sim.images.npy.gz"],
                    "label": sample["label.cls"],
                }
                # Sample has 1 real image + (G *R) simulated images
                writer.write(mds_sample, sample_length=math.prod(sample["sim.images.npy.gz"].shape[:-3]) + 1)

        assert shard_dir.exists()
        index_file = shard_dir / "index.json"
        assert index_file.exists()

        with open(index_file) as f:
            index_obj = json.load(f)
        for shard in index_obj["shards"]:
            assert "sample_lengths" in shard, "Custom sample lengths not found in shard index"
            assert len(shard["sample_lengths"]) == shard["samples"], (
                "Number of sample lengths does not match number of samples"
            )

        stream = Stream(local=str(shard_dir), repeat=4)

        dataset = PackableStreamingDataset(
            streams=[stream], packing_method="packed_samples", batch_size=32, shuffle=True
        )
        # breakpoint()
        for i, sample in enumerate(dataset):
            print(i, sample)
        # dataloader = StreamingDataLoader(dataset, batch_size=32, drop_last=False)
        # for i, batch in enumerate(dataloader):
        #     print(i, batch)

    def test_ldm(self, tmp_path):
        model = DummyModel()
        dm = PackableDataModule(train_urls=data_path("images-0000.tar"))

        trainer = L.Trainer(devices=2, accelerator="cpu", strategy="ddp", fast_dev_run=True)
        trainer.fit(model=model, datamodule=dm)
