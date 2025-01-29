import itertools
import random
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from PIL import Image
from torch import nn, optim
import torch.utils.data as data
from torchvision.transforms import v2 as transforms
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import hydra

# Check out: https://github.com/SerezD/vqvae-vqgan-pytorch-lightning/tree/master

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

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
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

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                    (1 - self._decay) * torch.sum(encodings, 0)

            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = ((self._ema_cluster_size + self._epsilon) /
                                    (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings, encoding_indices


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
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens//2, kernel_size=4, stride=4, padding=0)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2, out_channels=num_hiddens, kernel_size=4, stride=4, padding=0)
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
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels):
        super(Decoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens, num_hiddens=num_hiddens, num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens)
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, out_channels=num_hiddens//2, kernel_size=4, stride=4, padding=0)
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, out_channels=out_channels, kernel_size=4, stride=4, padding=0)  

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
        data, *_ = batch
        loss, quantized, perplexity, _ = self(data)
        
        # Only perform reconstruction as part of training
        x_recon = self.decoder(quantized)
        recon_error = F.mse_loss(x_recon, data)
        total_loss = recon_error + loss

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_recon_error', recon_error, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_perplexity', perplexity, on_step=True, on_epoch=False, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return { "optimizer": optimizer }


class EmptyDataset(data.Dataset):
    def __init__(self):
        super(EmptyDataset).__init__()
    
    def __iter__(self):
        return iter(())
    

class RealImageDataModule(L.LightningDataModule):
    def __init__(self, training_urls=None, validation_urls=None, prediction_urls=None, test_urls=None, num_channels=3, batch_size=16, num_workers=1, shuffle_size=1000):
        super().__init__()
        self.save_hyperparameters(ignore=["training_urls", "validation_urls", "prediction_urls", "test_urls"])

        self.training_urls = training_urls
        self.validation_urls = validation_urls
        self.prediction_urls = prediction_urls
        self.test_urls = test_urls
        
        self.transforms = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize(size=(224, 224)),
            transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            transforms.Normalize(mean=[0.5]*num_channels, std=[0.5]*num_channels),
        ])

    def make_loader(self, urls, mode="train"):
        # Adapted from: https://github.com/webdataset/webdataset/blob/main/examples/out/train-resnet50-multiray-wds.ipynb

        dataset = wds.WebDataset(urls, shardshuffle=100 if mode == "train" else False)
        if mode == "train":
            dataset = dataset.shuffle(self.hparams.shuffle_size)
        dataset = (
            dataset
            .decode()
            .to_tuple("image.npy.gz")
            .map_tuple(self.transforms)
            .batched(self.hparams.batch_size, partial=False)
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
        loader = loader.batched(self.hparams.batch_size)

        return loader


    def train_dataloader(self):
        return self.make_loader(self.training_urls, mode="train")

# def train(cfg, output_dir=None, **kw_args):
#     # dm = RealImageDataModule(
#     #     training_urls=training_urls,
#     #     num_channels=num_channels,
#     #     batch_size=2,
#     # )
    
#     # model = DVAE(
#     #     Encoder(in_channels=num_channels, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32, embedding_dim=64),
#     #     VectorQuantizerEMA(num_embeddings=512, embedding_dim=64, commitment_cost=0.25, decay=0.99),
#     #     Decoder(in_channels=64, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32, out_channels=num_channels),
#     #     partial(optim.Adam, lr=0.001, amsgrad=False),
#     # )

#     dm = hydra.utils.instantiate(cfg.data)
#     model = hydra.utils.instantiate(cfg.model)
    
#     # Overwrite existing checkpoints, instead of creating new versions
#     checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath=output_dir, enable_version_counter=False)

#     trainer =  hydra.utils.instantiate(cfg.trainer, callbacks=[checkpoint_callback], **kw_args)

#     #trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[checkpoint_callback], limit_val_batches=limit_val_batches, num_sanity_val_steps=num_sanity_val_steps, limit_test_batches=limit_test_batches, **kw_args)
    
#     # TODO: Check if we have reached the final, if not, continue training by setting ckpt_path
#     # https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#resume-training-state
#     trainer.fit(model=model, datamodule=dm)

#     return checkpoint_callback.best_model_path

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define transformations: resize to 100x300, convert to tensor, and normalize
# TRANSFORMATIONS = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((100, 300)),  # Resize image
#     transforms.Normalize(mean=[0.5]*6, std=[0.5]*6)  # Adjust for 6 channels
# ])

# def make_sample(sample):
#     return (TRANSFORMATIONS(sample["image.pyd.gz"]),)

# def make_model(
#     num_hiddens=128,
#     num_residual_layers=2,
#     num_residual_hiddens=32,
#     num_embeddings=512,
#     embedding_dim=64,
#     commitment_cost=0.25,
#     decay=0.99
# ):
#     model = Model(
#         num_hiddens,
#         num_residual_layers,
#         num_residual_hiddens,
#         num_embeddings,
#         embedding_dim,
#         commitment_cost,
#         decay
#     ).to(device)
#     return model

# def train():
#     local_dir = "/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze4.sv.alt.passing.training.hg38.images/HG00731/generator=coverage,simulation.replicates=1/images-0000.tar"

#     batch_size = 32

#     num_training_updates = 100000
#     learning_rate = 1e-3

#     dataset = wds.WebDataset(local_dir).decode().map(make_sample).batched(batch_size)
#     dataloader = DataLoader(dataset, batch_size=None)

#     model = make_model()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

#     model.train()
#     train_res_recon_error = []
#     train_res_perplexity = []

#     for i in range(num_training_updates):
#         (data,) = next(iter(dataloader))
#         data = data.to(device)
#         optimizer.zero_grad()

#         vq_loss, data_recon, perplexity = model(data)
#         recon_error = F.mse_loss(data_recon, data)
#         loss = recon_error + vq_loss
#         loss.backward()

#         optimizer.step()

#         train_res_recon_error.append(recon_error.item())
#         train_res_perplexity.append(perplexity.item())

#         if (i+1) % 50 == 0:
#             print("%d iterations" % (i+1))
#             print("recon_error: %.3f" % np.mean(train_res_recon_error[-100:]))
#             print("perplexity: %.3f" % np.mean(train_res_perplexity[-100:]))
#             print(flush=True)

#     # Save the model state
#     torch.save({
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#       }, "dVAE_model.tar")

#     return model


# def reconstruct(model):
#     local_dir = "/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze4.sv.alt.passing.training.hg38.images/HG00731/generator=coverage,simulation.replicates=1/images-0000.tar"
#     dataset = wds.WebDataset(local_dir).decode().map(make_sample).batched(1)
#     dataloader = DataLoader(dataset, batch_size=None)

#     original_images = []
#     reconstructed_images = []
#     for (original_image,) in itertools.islice(dataloader, 6):
#         original_image = original_image.to(device)
#         _, reconstructed_image, _ = model(original_image)

#         # Convert tensors to numpy arrays for visualization, reversing normalization
#         # to bring the pixel values back to [0, 1]
#         original_image_np = original_image.squeeze(0)[:3].detach().mul(0.5).add(0.5).permute(1, 2, 0).cpu().numpy()
#         reconstructed_image_np = reconstructed_image.squeeze(0)[:3].detach().mul(0.5).add(0.5).permute(1, 2, 0).cpu().numpy()

#         # Store processed images
#         original_images.append(original_image_np)
#         reconstructed_images.append(reconstructed_image_np)

#     # Plotting
#     fig, axes = plt.subplots(len(original_images), 2, figsize=(10, 30))

#     for idx in range(len(original_images)):
#         axes[idx, 0].imshow(np.clip(original_images[idx], 0, 1))
#         axes[idx, 0].set_title(f"Original Image {idx+1}")
#         axes[idx, 0].axis("off")

#         axes[idx, 1].imshow(np.clip(reconstructed_images[idx], 0, 1))
#         axes[idx, 1].set_title(f"Reconstructed Image {idx+1}")
#         axes[idx, 1].axis("off")

#     plt.tight_layout()
#     plt.savefig("test.png")

# if __name__ == "__main__":
#     _ = train()

#     model = make_model()
#     # optimizer = optim.Adam(model.parameters(), lr=1e-3, amsgrad=False)

#     checkpoint = torch.load("dVAE_model.tar")
#     model.load_state_dict(checkpoint["model_state_dict"])
#     # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     model.eval()

#     reconstruct(model)


