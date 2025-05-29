import io, itertools, math, os, random
from typing import Any, Dict, Sequence
from functools import partial

import hydra

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from torch import nn
from torchvision.transforms import v2 as transforms
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint


# Check out: https://github.com/SerezD/vqvae-vqgan-pytorch-lightning/tree/master
# https://github.com/lucidrains/vector-quantize-pytorch

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        #dimensionality of each codebook vector
        self._embedding_dim = embedding_dim
        #number of distinct codebook vectors
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach() #quantized replaces channels with embedding_dim
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
    

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        #dimensionality of each codebook vector
        self._embedding_dim = embedding_dim
        #number of distinct codebook vectors
        self._num_embeddings = num_embeddings

        # Note: Other example disables gradients for embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        # Do these need to be parameters, or just weights?
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # Ensure inputs are in the format (B, C, H, W) for compatibility with the expected input shape
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        # Equivalent to: self._embedding.weight.index_select(dim=1, index=encoding_indices) 

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                    (1 - self._decay) * torch.sum(encodings, 0)

            # Note: Other example using batch size instead of n here
            # https://github.com/SerezD/vqvae-vqgan-pytorch-lightning/blob/master/vqvae/modules/vector_quantizers.py
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = ((self._ema_cluster_size + self._epsilon) /
                                    (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            # Note do these need to be parameters are can they be buffers?
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings, encoding_indices.view(-1, input_shape[1], input_shape[2])

    # TODO Add method to just get indices for prediction
    @torch.no_grad()
    def indices(self, inputs: torch.Tensor) -> torch.IntTensor:
        # Ensure inputs are in the format (B, C, H, W) for compatibility with the expected input shape
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input (in dimension row-based format)
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Token indices, converting to (B, H, W)
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices.view(-1, input_shape[1], input_shape[2])

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels, out_channels=num_residual_hiddens, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens, out_channels=num_hiddens, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens) for _ in range(num_residual_layers)])

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim, strides=[4, 4], padding=[0,0]):
        super(Encoder, self).__init__()
        # Here is the 16x downsampling on each dimension. Alternate approaches seem to interleave downsampling with residual blocks
        # Could we use different downsampling factors for height and width?
        self._conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens//2, kernel_size=4, stride=strides[0], padding=padding[0])
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2, out_channels=num_hiddens, kernel_size=4, stride=strides[1], padding=padding[1])
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens, num_hiddens=num_hiddens, num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens)
        
        self._pre_quant_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

    def forward(self, inputs):
        x = F.relu(self._conv_1(inputs))
        x = F.relu(self._conv_2(x))
        x = self._conv_3(x)
        x = self._residual_stack(x)

        return self._pre_quant_conv(x)
        

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels, strides=[4, 4], padding=[0,0]):
        super(Decoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens, num_hiddens=num_hiddens, num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens)
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, out_channels=num_hiddens//2, kernel_size=4, stride=strides[1], padding=padding[1])
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, out_channels=out_channels, kernel_size=4, stride=strides[0], padding=padding[0])  

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = F.relu(self._conv_trans_1(x))
        return self._conv_trans_2(x)
    

class DVAE(L.LightningModule):
    def __init__(self, encoder, quantizer, decoder, optimizer: torch.optim.Optimizer):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "quantizer", "decoder"])

        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
    
    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, perplexity, _, encoding_indices = self.quantizer(z)
       
        return loss, quantized, perplexity, encoding_indices

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        _, data, *_ = batch
        loss, quantized, perplexity, _ = self(data)
        
        # Only perform reconstruction as part of training
        x_recon = self.decoder(quantized)
        recon_error = F.mse_loss(x_recon, data)
        total_loss = recon_error + loss

        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_recon_error', recon_error, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_perplexity', perplexity, on_step=True, on_epoch=False, prog_bar=True)

        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        _, data, *_ = batch
        if self.decoder:
            _, quantized, _, encoding_indices = self(data)
            return encoding_indices, self.decoder(quantized)
        else:
            return self.quantizer.indices(self.encoder(data))

    def configure_optimizers(self):
        # Complete construction of optimizer from Hydra-provided partial stored in hparams
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return { "optimizer": optimizer }


class EmptyDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(EmptyDataset).__init__()
    
    def __iter__(self):
        return iter(())
    

class Denormalize(transforms.Transform):
    def __init__(self, mean: Sequence[float], std: Sequence[float], inplace: bool = False):
        super().__init__()
        self.mean = list(mean)
        self.std = list(std)
        self.inplace = inplace

    def _transform(self, image: Any, params: Dict[str, Any]) -> Any:
        if type(image) == torch.Tensor:
            mean = torch.as_tensor(self.mean, dtype=image.dtype, device=image.device)
            std = torch.as_tensor(self.std, dtype=image.dtype, device=image.device)
            if mean.ndim == 1:
                mean = mean.view(-1, 1, 1)
            if std.ndim == 1:
                std = std.view(-1, 1, 1)

            if self.inplace:
                image = image.mul_(std)
            else:
                image = image.mul(std)

            return image.add_(mean).clamp_(0, 1) # Make sure in the range [0, 1]

class RealImageDataModule(L.LightningDataModule):
    def __init__(self, train_urls=None, validate_urls=None, predict_urls=None, test_urls=None, num_channels=3, batch_size=16, num_workers=1, shuffle_size=1000):
        super().__init__()
        self.save_hyperparameters(ignore=["train_urls", "validate_urls", "predict_urls", "test_urls"])

        self.train_urls = train_urls
        self.validate_urls = validate_urls
        self.predict_urls = predict_urls
        self.test_urls = test_urls
        
        self.transforms = transforms.Compose([
            transforms.ToImage(),
            #transforms.Resize(size=(224, 224)),
            transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            transforms.Normalize(mean=[0.5]*num_channels, std=[0.5]*num_channels),
        ])

    def make_loader(self, urls, mode="train"):
        # Adapted from: https://github.com/webdataset/webdataset/blob/main/examples/out/train-resnet50-multiray-wds.ipynb

        dataset = wds.WebDataset(urls, shardshuffle=100 if mode == "train" else False)
        if mode == "train":
            dataset = dataset.shuffle(self.hparams.shuffle_size)
        
        def to_tuple(data):
            # Handle missing fields (https://webdataset.github.io/webdataset/FAQ/, issue #246)
            return data["__key__"], data["image.npy.gz"], data.get("region.txt", data["__key__"])
        
        dataset = (
            dataset
            .decode()
            .map(to_tuple)
            .map_tuple(wds.utils.identity, self.transforms)
            .batched(self.hparams.batch_size, partial=mode != "train")
        )

        # We unbatch, shuffle, and rebatch to mix samples from different workers as shown in webdataset examples
        loader = (
            wds.WebLoader(
                dataset,
                batch_size=None,
                shuffle=False,
                num_workers=self.hparams.num_workers,
            )
            .unbatched()
        )
        if mode == "train":
            loader = loader.shuffle(self.hparams.shuffle_size)
        loader = loader.batched(self.hparams.batch_size, partial=mode != "train")

        return loader

 
    def train_dataloader(self):
        return self.make_loader(self.train_urls, mode="train")

    def predict_dataloader(self):
        return self.make_loader(self.predict_urls, mode="predict")

class EncodingToWebDatasetCallback(L.pytorch.callbacks.Callback):
    def __init__(self, output_dir: str):
        pattern = os.path.join(output_dir, f"encodings-%04d.tar.gz")
        self._writer = wds.ShardWriter(pattern,  maxsize=500e6)

    def on_predict_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
        keys, _, regions = batch
        for key, encoding, region in zip(keys, outputs, regions):
            # Per https://webdataset.github.io/webdataset/FAQ/ Issue #261
            buffer = io.BytesIO()
            torch.save(encoding.clone(), buffer)
            sample = {
                "__key__": key,
                "image.encoded.pth": buffer.getvalue(),
                "region.txt": region,
            }
            self._writer.write(sample)


    def on_predict_end(self, trainer, model):
        self._writer.close()


def encode(cfg, output_dir, **kw_args):
    dm = hydra.utils.instantiate(cfg.data)
    
    model_cls = hydra.utils.get_class(cfg.model._target_)
    # We need to instantiate any of ignored components in the model
    model = model_cls.load_from_checkpoint(
        cfg.model.checkpoint,
        strict=False,  # Allow for loading weights with missing decoder
        encoder=hydra.utils.instantiate(cfg.model.encoder),
        quantizer=hydra.utils.instantiate(cfg.model.quantizer),
        decoder=None,
    )
    
    trainer = L.Trainer(callbacks=[EncodingToWebDatasetCallback(output_dir)], **kw_args)
    trainer.predict(model, dm)


class ReconstructionToWebDatasetCallback(L.pytorch.callbacks.Callback):
    def __init__(self, output_dir: str, num_channels=3):
        pattern = os.path.join(output_dir, f"reconstructions-%04d.tar.gz")
        self._writer = wds.ShardWriter(pattern,  maxsize=500e6)
        self.denormalize = transforms.Compose([
            Denormalize(mean=[0.5]*num_channels, std=[0.5]*num_channels),
            transforms.ToDtype(torch.uint8, scale=True),
        ])

    def on_predict_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
        keys, images, regions = batch
        encodings, recon_images = outputs
        for key, real_image, encoding, recon_image, region in zip(keys, images, encodings, recon_images, regions):
            # Per https://webdataset.github.io/webdataset/FAQ/ Issue #261
            buffer = io.BytesIO()
            torch.save(encoding.clone(), buffer)
            sample = {
                "__key__": key,
                "image.npy": self.denormalize(real_image).permute(1, 2, 0).cpu().numpy(),
                "recon_image.npy": self.denormalize(recon_image).permute(1, 2, 0).cpu().numpy(),
                "image.encoded.pth": buffer.getvalue(),
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
        encoder=hydra.utils.instantiate(cfg.model.encoder),
        quantizer=hydra.utils.instantiate(cfg.model.quantizer),
        decoder=hydra.utils.instantiate(cfg.model.decoder),
    )
    
    trainer = L.Trainer(callbacks=[ReconstructionToWebDatasetCallback(output_dir, len(cfg.pileup.image_channels))], **kw_args)
    trainer.predict(model, dm)



