import lightning as L
import torch
import torch.nn.functional as F
import webdataset as wds
from torch import nn
from torchvision.transforms import v2 as transforms
from transformers import AutoImageProcessor, ViTForMaskedImageModeling, AdamW

# class Residual(nn.Module):
#     def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
#         super(Residual, self).__init__()
#         self._block = nn.Sequential(
#             nn.ReLU(True),
#             nn.Conv2d(
#                 in_channels=in_channels,
#                 out_channels=num_residual_hiddens,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 bias=False,
#             ),
#             nn.ReLU(True),
#             nn.Conv2d(in_channels=num_residual_hiddens, out_channels=num_hiddens, kernel_size=1, stride=1, bias=False),
#         )

#     def forward(self, x):
#         return x + self._block(x)

# class ResidualStack(nn.Module):
#     def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
#         super(ResidualStack, self).__init__()
#         self._layers = nn.ModuleList(
#             [Residual(in_channels, num_hiddens, num_residual_hiddens) for _ in range(num_residual_layers)]
#         )

#     def forward(self, x):
#         for layer in self._layers:
#             x = layer(x)
#         return F.relu(x)

# class Encoder(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         num_hiddens,
#         num_residual_layers,
#         num_residual_hiddens,
#         embedding_dim,
#         strides=[4, 4],
#         padding=[0, 0],
#     ):
#         super(Encoder, self).__init__()
#         # Here is the 16x downsampling on each dimension. Alternate approaches seem to interleave downsampling with residual blocks
#         # Could we use different downsampling factors for height and width?
#         self._conv_1 = nn.Conv2d(
#             in_channels=in_channels, out_channels=num_hiddens // 2, kernel_size=4, stride=strides[0], padding=padding[0]
#         )
#         self._conv_2 = nn.Conv2d(
#             in_channels=num_hiddens // 2, out_channels=num_hiddens, kernel_size=4, stride=strides[1], padding=padding[1]
#         )
#         self._conv_3 = nn.Conv2d(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)
#         self._residual_stack = ResidualStack(
#             in_channels=num_hiddens,
#             num_hiddens=num_hiddens,
#             num_residual_layers=num_residual_layers,
#             num_residual_hiddens=num_residual_hiddens,
#         )

#         self._pre_quant_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

#     def forward(self, inputs):
#         x = F.relu(self._conv_1(inputs))
#         x = F.relu(self._conv_2(x))
#         x = self._conv_3(x)
#         x = self._residual_stack(x)

#         return self._pre_quant_conv(x)


# class Decoder(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         num_hiddens,
#         num_residual_layers,
#         num_residual_hiddens,
#         out_channels,
#         strides=[4, 4],
#         padding=[0, 0],
#     ):
#         super(Decoder, self).__init__()
#         self._conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)
#         self._residual_stack = ResidualStack(
#             in_channels=num_hiddens,
#             num_hiddens=num_hiddens,
#             num_residual_layers=num_residual_layers,
#             num_residual_hiddens=num_residual_hiddens,
#         )
#         self._conv_trans_1 = nn.ConvTranspose2d(
#             in_channels=num_hiddens, out_channels=num_hiddens // 2, kernel_size=4, stride=strides[1], padding=padding[1]
#         )
#         self._conv_trans_2 = nn.ConvTranspose2d(
#             in_channels=num_hiddens // 2,
#             out_channels=out_channels,
#             kernel_size=4,
#             stride=strides[0],
#             padding=padding[0],
#         )

#     def forward(self, inputs):
#         x = self._conv_1(inputs)
#         x = self._residual_stack(x)
#         x = F.relu(self._conv_trans_1(x))
#         return self._conv_trans_2(x)

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

    def make_loader(self, urls, mode="train"):
        # Adapted from: https://github.com/webdataset/webdataset/blob/main/examples/out/train-resnet50-multiray-wds.ipynb

        dataset = wds.WebDataset(urls, shardshuffle=100 if mode == "train" else False)
        if mode == "train":
            dataset = dataset.shuffle(self.hparams.shuffle_size)

        def to_tuple(data):
            # Handle missing fields (https://webdataset.github.io/webdataset/FAQ/, issue #246)
            return data["__key__"], data["image.npy.gz"], data.get("region.txt", data["__key__"])

        dataset = (
            dataset.decode()
            .map(to_tuple)
            .map_tuple(wds.utils.identity, self.transforms)
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
    
class MiM(L.LightningModule):
    def __init__(self, model_name: str = "google/vit-base-patch16-224-in21k") -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k")
        self.learning_rate = self.hparams.learning_rate

    def forward(self, image, num_patches):
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values
        return self.model(pixel_values)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, logits = self(input_ids, attention_mask, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss