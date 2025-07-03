import io
import os

import hydra
import random
import lightning as L
import torch
import time
import webdataset as wds
from dataclasses import dataclass
from typing import Union, Optional, Tuple
from PIL import Image
import numpy as np
from torch import nn
from torchvision.transforms import v2 as transforms
from transformers import ViTConfig, ViTPreTrainedModel, ViTModel, ViTForImageClassification
from npsv3.models.dvae import Denormalize

class RealImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_urls=None,
        validate_urls=None,
        predict_urls=None,
        test_urls=None,
        batch_size=16,
        num_workers=1,
        patch_size=32,
        shuffle_size=1000,
        num_channels=3,
        mask_scheme=["random", 20]
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_urls", "validate_urls", "predict_urls", "test_urls"])

        self.train_urls = train_urls
        self.validate_urls = validate_urls
        self.predict_urls = predict_urls
        self.test_urls = test_urls
        self.mask_scheme=mask_scheme

        self.transforms = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input
                transforms.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels),
            ]
        )

        self.configuration = ViTConfig(num_channels=num_channels, patch_size=self.hparams.patch_size)

    def make_loader(self, urls, mode="train"):
        # Adapted from: https://github.com/webdataset/webdataset/blob/main/examples/out/train-resnet50-multiray-wds.ipynb

        dataset = wds.WebDataset(urls, shardshuffle=100 if mode == "train" else False)
        if mode == "train":
            dataset = dataset.shuffle(self.hparams.shuffle_size)

        def to_tuple(data):
            # Handle missing fields (https://webdataset.github.io/webdataset/FAQ/, issue #246)
            image = data["image.npy.gz"]
            pixel_values = self.transforms(image)

            num_patches = (pixel_values.shape[1] // self.configuration.patch_size) * (pixel_values.shape[2] // self.configuration.patch_size)

            # number of patches we want to mask
            num_masked = int(self.mask_scheme[1] / 100 * num_patches)
            selected_indices = torch.randperm(num_patches)[:num_masked]

            init_bool_mask_pos = torch.zeros(num_patches, dtype=torch.bool)
            init_bool_mask_pos[selected_indices] = True

            if self.mask_scheme[0] == "random":
                bool_masked_pos = init_bool_mask_pos
            # performance?
            if self.mask_scheme[0] == "data_driven":
                bool_masked_pos = data_driven_masking(pixel_values, num_patches, self.configuration.patch_size, init_bool_mask_pos)
            
            label = data["label.cls"]
            # print("\nloaded label: ",label)

            return pixel_values, bool_masked_pos, data["__key__"], data.get("region.txt", data["__key__"]), 0 if label == 0 else 1

        dataset = (
            dataset.decode()
            .map(to_tuple)
            .batched(self.hparams.batch_size, partial=mode != "train")
        )

        pin_memory = False
        worker_init_fn = None
        if torch.cuda.is_available():
            pin_memory = True
            worker_init_fn = torch.set_num_threads(1)

        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            persistent_workers=True
        ).unbatched()
        if mode == "train":
            loader = loader.shuffle(self.hparams.shuffle_size)
        loader = loader.batched(self.hparams.batch_size, partial=mode != "train")

        return loader

    def train_dataloader(self):
        return self.make_loader(self.train_urls, mode="train")

    def predict_dataloader(self):
        return self.make_loader(self.predict_urls, mode="predict")



class MiM(L.LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_channels = 7,
        image_size=(96, 288),
        patch_size=32,
    ):
        super().__init__()
        self.save_hyperparameters()
        configuration = ViTConfig(num_channels=num_channels, image_size=image_size, patch_size=patch_size, encoder_stride=patch_size)
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
        outputs = self(pixel_values, bool_masked_pos)

        reconstructed = outputs.logits  # (B, C, H, W)

        print("\nLoss on reconstruct: ", outputs.loss)

        B, C, H, W = pixel_values.shape
        patch_size = self.hparams.patch_size
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size

        # Convert patch-wise mask to pixel-wise mask
        bool_masked_pos_reshaped = bool_masked_pos.view(B, num_patches_h, num_patches_w)
        mask_upsampled = bool_masked_pos_reshaped.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
        mask_upsampled = mask_upsampled.unsqueeze(1)  # (B, 1, H, W)

        # Apply mask: reconstruct only masked pixels
        final_reconstruction = torch.where(mask_upsampled.bool(), reconstructed, pixel_values)

        return {
            "original": pixel_values,
            "masked_pos": bool_masked_pos,
            "reconstructed_only": reconstructed,
            "final_reconstruction": final_reconstruction
        }

#Adapted from: https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/models/vit/modeling_vit.py#L592
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
        height = pixel_values.shape[2]//self.config.patch_size
        width = pixel_values.shape[3]//self.config.patch_size
        sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)

        # Reconstruct pixel values
        reconstructed_pixel_values = self.decoder(sequence_output)

        masked_im_loss = None
        if bool_masked_pos is not None:
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

class Classifier(L.LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_channels = 7,
        image_size = (96, 288),
        num_labels = 2,
        patch_size=32
    ):
        super().__init__()
        self.save_hyperparameters()
        configuration = ViTConfig(num_channels=num_channels, image_size=image_size, num_labels=num_labels, patch_size=patch_size, encoder_stride=patch_size)
        self.model = ViTForImageClassification(configuration)

    def forward(self, pixel_values, labels):
        outputs = self.model(pixel_values, labels=labels)
        return outputs
    
    def training_step(self, batch):
        pixel_values, _, _, _, labels = batch
        out = self(pixel_values, labels)
        loss = out.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return { "optimizer": optimizer }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pixel_values, _, _, _, labels = batch
        return self(pixel_values, labels)


def torch_decode(key, data):
    # Use custom decoder to eliminate a warning with torch.load about weights_only
    if key.endswith((".pth", ".pt")):
        stream = io.BytesIO(data)
        return torch.load(stream, weights_only=True, map_location='cpu')

class ModelAssessmentCallback(L.pytorch.callbacks.Callback):
    def __init__(self):
        self.results = []

    #I believe this is an override for the default function that allows us to execute some code each prediction
    def on_predict_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
        images, bool_masked_pos, keys, regions, label = batch
        for i, logits in enumerate(outputs.logits):
            if label[i].item() == torch.argmax(logits).item():
                self.results.append(1)
            else: 
                self.results.append(0)
  
    def on_predict_end(self, trainer, model):
        print(f"\nAssessing accuracy of {len(self.results)} predictions")
        correct = sum(self.results)
        print("\nAccuracy:",correct/len(self.results))


class ReconstructionToWebDatasetCallback(L.pytorch.callbacks.Callback):
    def __init__(self, output_dir: str, mask_path, num_channels=3,):
        pattern = os.path.join(output_dir, "reconstructions-%04d.tar.gz")
        self._writer = wds.ShardWriter(pattern, maxsize=500e6)
        self.denormalize = transforms.Compose(
            [
                Denormalize(mean=[0.5] * num_channels, std=[0.5] * num_channels),
                transforms.ToDtype(torch.uint8, scale=True),
            ]
        )
        self.mask_path=mask_path

    def on_predict_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
        images, bool_masked_pos, keys, regions, label = batch

        # comment out to make more efficient
        generate_mask_visual(bool_masked_pos, 32, self.mask_path)

        recon_images = outputs["final_reconstruction"]

        # encodings, recon_images = outputs

        for key, real_image, recon_image, region, in zip(keys, images, recon_images, regions, strict=False):
            sample = {
                "__key__": key,
                "image.npy": self.denormalize(real_image).permute(1, 2, 0).cpu().numpy(),
                "recon_image.npy": self.denormalize(recon_image).permute(1, 2, 0).cpu().numpy(),
                "region.txt": region,
            }
            self._writer.write(sample)

    def on_predict_end(self, trainer, model):
        self._writer.close()


def reconstruct(cfg, output_dir, mask_path, **kw_args):
    dm = hydra.utils.instantiate(cfg.data)

    model_cls = hydra.utils.get_class(cfg.model._target_)
    model = model_cls.load_from_checkpoint(
        cfg.model.checkpoint,
    )

    trainer = L.Trainer(
        callbacks=[ReconstructionToWebDatasetCallback(output_dir, mask_path, len(cfg.pileup.image_channels))], **kw_args,
    )
    trainer.predict(model, dm)

def display_image(urls):
    dataset = wds.WebDataset(urls, shardshuffle=False)
    for sample in enumerate(dataset):
        image = sample["image.npy"]
    return image

# Need to either remove this because it's only for testing or change the png path to a non-user path
def generate_mask_visual(bool_masked_pos, patch_size, mask_path):
    
    mask = Image.new("RGB", (288, 96))
    pixel_array = np.array(mask)

    for i in range (len(pixel_array)):
        for j in range (len(pixel_array[0])):
            bool_index = (i // patch_size) * (len(pixel_array[0]) // patch_size) + (j // patch_size)
            # case where pixel should be masked
            if (bool_masked_pos[0][bool_index] == True):
                mask.putpixel((j, i), (255, 255, 255))
            else:
                mask.putpixel((j, i), (0, 0, 0))
                
    mask.save(mask_path)

def data_driven_masking(pixel_values, num_patches, patch_size, i_bool_mask_pos):

    patches_per_row = len(pixel_values[0][0]) // patch_size
    patches_per_col = len(pixel_values[0]) // patch_size

    for i in range (len(i_bool_mask_pos)):
        if (i_bool_mask_pos[i] == False): continue

        patch_coord = (i % patches_per_row, i // patches_per_row)
        patch_col, patch_row = patch_coord

        pixel_col = patch_col * patch_size
        pixel_row = patch_row * patch_size

        if (pixel_values[0][pixel_row][pixel_col] == -1.0):
            i_bool_mask_pos[i] = False

    return i_bool_mask_pos