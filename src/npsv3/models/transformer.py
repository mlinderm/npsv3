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
from transformers import ViTConfig
from PIL import Image
import random
# from transformers import ViTForMaskedImageModeling
from npsv3.models.dvae import Denormalize

from dataclasses import dataclass

import math
from typing import Optional, Union, Tuple

from transformers import ViTConfig, ViTModel, ViTPreTrainedModel

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
                # transforms.Resize(size=(224, 224)),
                transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input
                transforms.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels),
            ]
        )



        self.configuration = ViTConfig(num_channels=num_channels)
        # self.model = ViTForMaskedImageModeling(configuration)


        #self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k") #can delete

    def make_loader(self, urls, mode="train"):
        # Adapted from: https://github.com/webdataset/webdataset/blob/main/examples/out/train-resnet50-multiray-wds.ipynb

        dataset = wds.WebDataset(urls, shardshuffle=100 if mode == "train" else False)
        if mode == "train":
            dataset = dataset.shuffle(self.hparams.shuffle_size)

        def to_tuple(data):
            # Handle missing fields (https://webdataset.github.io/webdataset/FAQ/, issue #246)
            image = data["image.npy.gz"]
            # input_data_format="channels_first", do_normalize=False, do_rescale=False, do_resize=False
            # pixel_values = torch.squeeze(self.image_processor(images=self.transforms(image), return_tensors="pt", input_data_format="channels_first", do_normalize=False, do_rescale=False, do_resize=False).pixel_values, 0) # can delete
            pixel_values = self.transforms(image)

            # random masking
            num_patches = (pixel_values.shape[1] // self.configuration.patch_size) * (pixel_values.shape[2] // self.configuration.patch_size)
            print("Image shape: ", pixel_values.shape)
            # bool_masked_pos = torch.randint(low=0, high=2, size=(num_patches,)).bool()
            vals = [True, False]
            top_weights = [3, 1]
            bottom_weights = [1, 2]
            top_masked_pos = random.choices(vals, weights=top_weights, k=num_patches//2)
            bottom_masked_pos = random.choices(vals, weights=bottom_weights, k=num_patches//2)
            bool_masked_pos = torch.tensor(top_masked_pos+bottom_masked_pos).bool()
            
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
        image_size=(96, 288)
    ):
        super().__init__()
        self.save_hyperparameters()
        # configuration = ViTConfig(num_channels=num_channels)
        configuration = ViTConfig(num_channels=num_channels, image_size=image_size)

        
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










# From Hugging Face https://github.com/huggingface/transformers/blob/main/examples/pytorch/instance-segmentation/run_instance_segmentation.py#L155
class ModelOutput:
    class_queries_logits: torch.Tensor
    masks_queries_logits: torch.Tensor

@dataclass
class MaskedImageModelingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    reconstruction: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

    @property
    def logits(self):
        return self.reconstruction

class ViTForMaskedImageModeling(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        self.vit = ViTModel(config, add_pooling_layer=False, use_mask_token=True)

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.encoder_stride**2 * config.num_channels,
                kernel_size=1,
            ),
            nn.PixelShuffle(config.encoder_stride),
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, MaskedImageModelingOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if bool_masked_pos is not None and (self.config.patch_size != self.config.encoder_stride):
            raise ValueError(
                "When `bool_masked_pos` is provided, `patch_size` must be equal to `encoder_stride` to ensure that "
                "the reconstructed image has the same dimensions as the input. "
                f"Got `patch_size` = {self.config.patch_size} and `encoder_stride` = {self.config.encoder_stride}."
            )

        outputs = self.vit(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # Reshape to (batch_size, num_channels, height, width)
        sequence_output = sequence_output[:, 1:]
        batch_size, sequence_length, num_channels = sequence_output.shape
        # height = math.floor(sequence_length**0.5)
        # width = math.floor(sequence_length**0.5)
        height = pixel_values.shape[2] // self.config.patch_size
        width = pixel_values.shape[3] // self.config.patch_size
        sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)

        # Reconstruct pixel values
        reconstructed_pixel_values = self.decoder(sequence_output)

        masked_im_loss = None
        if bool_masked_pos is not None:
            # size = self.config.image_size // self.config.patch_size
            bool_masked_pos = bool_masked_pos.reshape(-1, height, width)
            mask = (
                bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
                .repeat_interleave(self.config.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
            )
            reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
            masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels

        if not return_dict:
            output = (reconstructed_pixel_values,) + outputs[1:]
            return ((masked_im_loss,) + output) if masked_im_loss is not None else output

        return MaskedImageModelingOutput(
            loss=masked_im_loss,
            reconstruction=reconstructed_pixel_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    