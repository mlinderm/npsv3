import io
import os
from collections.abc import Sequence
from typing import Any

import hydra
import lightning as L
import torch
import torch.nn.functional as F
import webdataset as wds
from torch import nn
from torchvision.transforms import v2 as transforms
from transformers import AutoImageProcessor, ViTForMaskedImageModeling, ViTConfig
from npsv3.models.dvae import Denormalize
    
class RealImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_urls=None,
        validate_urls=None,
        predict_urls=None,
        test_urls=None,
        num_channels=3,
        batch_size=16,
        num_workers=1,
        shuffle_size=1000,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_urls", "validate_urls", "predict_urls", "test_urls"])

        self.train_urls = train_urls
        self.validate_urls = validate_urls
        self.predict_urls = predict_urls
        self.test_urls = test_urls


        self.transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.Resize(size=(224, 224)),
                transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input
                transforms.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels),
            ]
        )



        self.configuration = ViTConfig(num_channels=num_channels)
        #self.model = ViTForMaskedImageModeling(configuration)

        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    def make_loader(self, urls, mode="train"):
        # Adapted from: https://github.com/webdataset/webdataset/blob/main/examples/out/train-resnet50-multiray-wds.ipynb

        dataset = wds.WebDataset(urls, shardshuffle=100 if mode == "train" else False)
        if mode == "train":
            dataset = dataset.shuffle(self.hparams.shuffle_size)

        def to_tuple(data):
            # Handle missing fields (https://webdataset.github.io/webdataset/FAQ/, issue #246)
            image = data["image.npy.gz"]
            
            # input_data_format="channels_first", do_normalize=False, do_rescale=False, do_resize=False
            pixel_values = torch.squeeze(self.image_processor(images=self.transforms(image), return_tensors="pt", input_data_format="channels_first", do_normalize=False, do_rescale=False, do_resize=False).pixel_values, 0)

            # random masking
            num_patches = (self.configuration.image_size // self.configuration.patch_size) ** 2
            bool_masked_pos = torch.randint(low=0, high=2, size=(num_patches,)).bool()
            
            return pixel_values, bool_masked_pos, data["__key__"], data.get("region.txt", data["__key__"])

        dataset = (
            dataset.decode()
            .map(to_tuple)
            #.map_tuple(wds.utils.identity, self.transforms)
            .batched(self.hparams.batch_size, partial=mode != "train")
        )

        # We unbatch, shuffle, and rebatch to mix samples from different workers as shown in webdataset examples
        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        ).unbatched()
        if mode == "train":
            loader = loader.shuffle(self.hparams.shuffle_size)
        loader = loader.batched(self.hparams.batch_size, partial=mode != "train")

        return loader

    def train_dataloader(self):
        return self.make_loader(self.train_urls, mode="train")

    def predict_dataloader(self):
        return self.make_loader(self.predict_urls, mode="predict")



class Transformer(L.LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model_name="google/vit-base-patch16-224-in21k",
        num_channels = 7,
    ):
        super().__init__()
        self.save_hyperparameters()
        configuration = ViTConfig(num_channels=num_channels)

        
        self.model = ViTForMaskedImageModeling(configuration)

    def forward(self, pixel_values, bool_masked_pos):
        outputs = self.model(pixel_values, bool_masked_pos=bool_masked_pos)
        return outputs
    
    def training_step(self, batch):
        pixel_values, bool_masked_pos, *_ = batch
        out = self(pixel_values, bool_masked_pos)
        loss = out.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return { "optimizer": optimizer }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pixel_values, bool_masked_pos, *_ = batch
        return self(pixel_values, bool_masked_pos)
    


class ReconstructionToWebDatasetCallback(L.pytorch.callbacks.Callback):
    def __init__(self, output_dir: str, num_channels=3):
        pattern = os.path.join(output_dir, "reconstructions-%04d.tar.gz")
        self._writer = wds.ShardWriter(pattern, maxsize=500e6)
        self.denormalize = transforms.Compose(
            [
                Denormalize(mean=[0.5] * num_channels, std=[0.5] * num_channels),
                transforms.ToDtype(torch.uint8, scale=True),
            ]
        )

    def on_predict_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
        images, _, keys, regions = batch
        #print(outputs)
        #encodings, recon_images = outputs
        for key, real_image, recon_image, region in zip(keys, images, outputs.reconstruction, regions, strict=False):
            sample = {
                "__key__": key,
                "image.npy": self.denormalize(real_image).permute(1, 2, 0).cpu().numpy(),
                "recon_image.npy": self.denormalize(recon_image).permute(1, 2, 0).cpu().numpy(),
                "region.txt": region,
            }
            self._writer.write(sample)

    def on_predict_end(self, trainer, model):
        self._writer.close()


def reconstruct(cfg, output_dir, **kw_args):
    dm = hydra.utils.instantiate(cfg.data)

    model_cls = hydra.utils.get_class(cfg.model._target_)
    model = model_cls.load_from_checkpoint(
        cfg.model.checkpoint,
    )

    trainer = L.Trainer(
        callbacks=[ReconstructionToWebDatasetCallback(output_dir, len(cfg.pileup.image_channels))], **kw_args
    )
    trainer.predict(model, dm)

    