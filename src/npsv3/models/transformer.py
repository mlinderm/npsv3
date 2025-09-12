import io
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generator, Iterable, Optional, Tuple, Union

import hydra
import lightning as L
import numpy as np
import timm
import torch
import webdataset as wds
from PIL import Image
from torch import nn
from torchvision.transforms import v2 as transforms
from transformers import ViTConfig, ViTForImageClassification, ViTModel, ViTPreTrainedModel

from npsv3.models.dvae import Denormalize
from npsv3.models.loaders import ntuple

# https://github.com/huggingface/pytorch-image-models/issues/1477

class Identity(nn.Module):
    """A more flexible indentity module that allows keyword arguments in forward"""
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        super().__init__()

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: ARG002
        return input


class MaskableVisionTransformer(timm.models.VisionTransformer):
    def __init__(self, *args, use_mask_token: bool = False, **kwargs):
        """Extension to timm VisionTransformer to support masked image modeling.

        Supports all of timm's VisionTransformer arguments.

        Args:
            use_mask_token (bool, optional): Implement masking. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        assert self.global_pool in ('token',)

        self.num_channels = kwargs.get("in_chans", 3)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) if use_mask_token else None

        # Apply mask (adapted from https://github.com/huggingface/transformers/blob/68a13cd4a65d0624a5b87827c6e0709a882613f0/src/transformers/models/vit/modeling_vit.py#L109C8-L114C72)
        def _apply_mask(embeddings: torch.Tensor, *, bool_masked_pos: torch.Tensor) -> torch.Tensor:
            batch_size, seq_length, *_ = embeddings.shape
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # Replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            return embeddings * (1.0 - mask) + mask_tokens * mask
        self.apply_mask = _apply_mask if use_mask_token else Identity()


    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        """Set of parameters that should not use weight decay."""
        no_decay = super().no_weight_decay()
        if self.mask_token is not None:
            no_decay.add("mask_token")
        return no_decay

    def forward_features(self, pixel_values: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, bool_masked_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through feature layers (embeddings, transformer blocks, post-transformer norm)"""
        # This is a copy of timm's forward_features with the addition of the mask application
        embeddings = self.patch_embed(pixel_values)
        embeddings = self.apply_mask(embeddings, bool_masked_pos=bool_masked_pos)
        embeddings = self._pos_embed(embeddings)
        embeddings = self.patch_drop(embeddings)

        output = self.norm_pre(embeddings)
        if attn_mask is not None:
            # If attn mask provided, we need to apply blocks one by one
            for blk in self.blocks:
                output = blk(output, attn_mask=attn_mask)
        elif self.grad_checkpointing and not torch.jit.is_scripting():
            output = timm.models._manipulate.checkpoint_seq(self.blocks, output)
        else:
            output = self.blocks(output)
        return self.norm(output)

class MaskedL1Loss(nn.Module):
     def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

     def forward(self, pixel_values: torch.Tensor, reconstructed_pixel_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
        # Normalize by the number of masked pixels and the number of channels
        return (reconstruction_loss * mask).sum() / (mask.sum() + self.eps) / pixel_values.size(1)

# We tried to implement an equivalent to timm's Cosine scheduler using combinations of Pytorch LR schedulers but could not implement
# the decay feature due to limitations in the ChainedScheduler (which doesn't pass the step through to the underlying schedulers). W
# would likely need to implement a custom scheduler to achieve the same functionality.

# Adapted from: https://github.com/huggingface/transformers/blob/41980ce93e775f6c88500c51c8db7946fc6a2add/src/transformers/models/vit/modeling_vit.py#L610
class MaskedImageModeling(L.LightningModule):
    ignored_hyperparameters = ("encoder", "loss")

    def __init__(self, encoder: nn.Module, loss: nn.Module, optimizer: Callable[...,torch.optim.Optimizer], scheduler: Callable):
        super().__init__()

        self.image_size = encoder.patch_embed.img_size
        self.num_channels = encoder.num_channels
        self.patch_size = encoder.patch_embed.patch_size
        self.grid_size = encoder.patch_embed.grid_size

        # Limitation introduced by PixelShuffle in the decoder
        assert self.patch_size[0] == self.patch_size[1], "Currently only square patches supported"
        self.save_hyperparameters(ignore=type(self).ignored_hyperparameters)

        self.encoder = encoder
        self.loss = loss
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=encoder.embed_dim,
                out_channels=self.patch_size[0] * self.patch_size[1] * self.num_channels,
                kernel_size=1,
            ),
            nn.PixelShuffle(self.patch_size[0]),
        )

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        """Set of parameters that should not use weight decay."""
        # This is used by timm's optimizers to determine which parameters should not use weight decay
        return { f"encoder.{name}" for name in self.encoder.no_weight_decay() }

    def forward(self, batch: tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        pixel_values, bool_masked_pos = batch
        sequence_output = self.encoder.forward_features(pixel_values, bool_masked_pos=bool_masked_pos)

        # Reshape to BCHW
        sequence_output = sequence_output[:, self.encoder.num_prefix_tokens:]
        b, _s, c = sequence_output.shape
        sequence_output = sequence_output.permute(0, 2, 1).reshape(b, c, self.grid_size[0], self.grid_size[1])

        # Reconstruct pixel values
        return self.decoder(sequence_output)

    def _model_step(self, batch, batch_idx):
        pixel_values, bool_masked_pos = batch
        reconstructed_pixel_values = self(batch)

        bool_masked_pos = bool_masked_pos.reshape(-1, self.grid_size[0], self.grid_size[1])
        mask = (bool_masked_pos
            .repeat_interleave(self.patch_size[0], 1)
            .repeat_interleave(self.patch_size[1], 2)
            .unsqueeze(1)
            .contiguous()
        )
        loss = self.loss(pixel_values, reconstructed_pixel_values, mask)
        return loss, reconstructed_pixel_values

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss, *_ = self._model_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, *_ = self._model_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        # Based on https://lightning.ai/docs/pytorch/stable/common/optimization.html#bring-your-own-custom-learning-rate-schedulers
        optimizer = self.hparams.optimizer(self.trainer.model)
        scheduler= self.hparams.scheduler(optimizer=optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, metric):
        # timm's schedulers appear to need the "next" epoch at each step
        # https://github.com/huggingface/pytorch-image-models/blob/954613a470652e4a113ff45b62dbd15c4e229218/train.py#L1067
        scheduler.step(epoch=self.current_epoch + 1)


class VisionTransformerEncoder(timm.models.VisionTransformer):
    def __init__(self, *args, checkpoint_path: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # Load pre-training checkpoint (from masked image modeling) if provided. We assume the relevant weights are prefixed with "encoder."
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            encoder_weights = { k.removeprefix("encoder."): v for k, v in checkpoint["state_dict"].items() if k.startswith("encoder.")}
            self.load_state_dict(encoder_weights, strict=False)

    def forward(self, pixel_values: torch.Tensor):
        sequence_output = self.forward_features(pixel_values)
        return sequence_output[:, 0] # Use CLS token
        # An alternate approach that incorporates FC norm and dropout layers
        return self.forward_head(self.forward_features(pixel_values), pre_logits=True)


class MaskedImageReconstructionToWebDatasetCallback(L.pytorch.callbacks.Callback):
    def __init__(self, output_dir: str, patch_size: tuple[int, int]|int = 16, mean=(0.5,), std=(0.5,)):
        """Callback to save masked image reconstructions to a WebDataset during prediction.

        Args:
            output_dir (str): Directory to save output images.
            patch_size (tuple[int, int]|int, optional): Size of the image patches. Defaults to 16.
            mean (tuple, optional): Mean values for denormalization. Defaults to (0.5,).
            std (tuple, optional): Standard deviation values for denormalization. Defaults to (0.5,).
        """
        self.patch_size = ntuple(patch_size, 2)

        pattern = os.path.join(output_dir, "reconstructions-%04d.tar.gz")
        self._writer = wds.ShardWriter(pattern, maxsize=500e6, verbose=0)

        self.denormalize = transforms.Compose(  # Reverse the normalization applied to the images
            [
                Denormalize(mean=mean, std=std),
                transforms.ToDtype(torch.uint8, scale=True),
            ]
        )

    def on_predict_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):  # noqa: ARG002
        images, bool_masks_pos = batch
        reconstructed_images = outputs

        grid_size = (images.shape[2] // self.patch_size[0], images.shape[3] // self.patch_size[1])

        for i, (real_image, bool_masked_pos, recon_image), in enumerate(zip(images, bool_masks_pos, reconstructed_images, strict=True)):
            bool_masked_pos = bool_masked_pos.reshape(grid_size[0], grid_size[1])
            mask = (bool_masked_pos
                .repeat_interleave(self.patch_size[0], 0)
                .repeat_interleave(self.patch_size[1], 1)
                .contiguous()
            )
            # We only reconstruct the masked pixels, so make the reconstruction the composite of the original and the reconstructed patches
            recon_image = recon_image * mask + real_image * (~mask)
            sample = {
                "__key__": f"{batch_idx:04d}-{i:04d}",
                "image.npy": self.denormalize(real_image).permute(1, 2, 0).cpu().numpy(),
                "recon_image.npy": self.denormalize(recon_image).permute(1, 2, 0).cpu().numpy(),
                "mask.npy": mask.cpu().numpy(),
            }
            self._writer.write(sample)

    def on_predict_end(self, trainer, model):  # noqa: ARG002
        self._writer.close()

class RealImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_urls=None,
        validate_urls=None,
        predict_urls=None,
        test_urls=None,
        batch_size=16,
        num_workers=1,
        patch_size=16,
        shuffle_size=1000,
        num_channels=7,
        mask_scheme=["random", 80]
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
        patch_size=16,
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

            # Use this line to reconstruct the entire image
            # "final_reconstruction": outputs.logits

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
        patch_size=16
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
        self.missed = []

    #I believe this is an override for the default function that allows us to execute some code each prediction
    def on_predict_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
        images, bool_masked_pos, keys, regions, label = batch
        for i, logits in enumerate(outputs.logits):
            real_label = label[i].item()
            predicted_label = torch.argmax(logits).item()
            if real_label == predicted_label:
                self.results.append(1)
            else: 
                self.results.append(0)
                self.missed.append((regions[i], "prediction:", predicted_label, "logits:", logits))
  
    def on_predict_end(self, trainer, model):
        print(f"\nAssessing accuracy of {len(self.results)} predictions")
        correct = sum(self.results)
        print("\nAccuracy:",correct/len(self.results))
        for i in range(25):
            print(random.choice(self.missed))


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
        generate_mask_visual(bool_masked_pos, 16, self.mask_path)

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

    for i in range (len(i_bool_mask_pos)):
        if (i_bool_mask_pos[i] == False): continue

        patch_coord = (i % patches_per_row, i // patches_per_row)
        patch_col, patch_row = patch_coord

        pixel_col = patch_col * patch_size
        pixel_row = patch_row * patch_size

        if (pixel_values[0][pixel_row][pixel_col] == -1.0):
            i_bool_mask_pos[i] = False

    return i_bool_mask_pos