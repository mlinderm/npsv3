
from collections.abc import Generator, Iterable

import hydra
import lightning as L
import numpy as np
import torch
import webdataset as wds
from torch import nn
from torch.utils import data
from torchmetrics.classification.accuracy import Accuracy
from torchvision import models
from torchvision.transforms import v2 as transforms


def transform_images(images: np.ndarray) -> torch.Tensor:
    """Preprocess images from numpy array to torch tensor

    Args:
        images (np.ndarray): *HWC numpy array.

    Returns:
        torch.Tensor: Normalized *CHW torch tensor.
    """
    return transforms.functional.to_dtype(
        torch.from_numpy(images).movedim(-1, -3).contiguous(),
        torch.float32,
        scale=True,
    )

def pack_yield_support(images, labels, variants, remaining_space, padding_value=0):
    """Helper function to yield a batch of images with padding. 
    Helps to avoid code duplication in the main packing function."""

    # If the current sample doesn't fit in the batch, yield the current batch
    #Cat the images list into a tensor and add padding
    images_tensor = torch.cat(images, dim=0)
    padding = (0,) * (2 * len(images_tensor.shape) - 1) + (remaining_space,)
    images_tensor  =  torch.nn.functional.pad(images_tensor, padding, mode='constant', value=padding_value)
    
    #Add padding to the variants and labels lists
    variants += [-100] * (remaining_space)
    labels += [-100] * (remaining_space)

    return (images_tensor, torch.tensor(variants), torch.tensor(labels))
    

def _pack_and_pad_support(data: Iterable[tuple], batch_size = 256, padding_value=0) -> Generator[tuple, None, None]:
    """packs real and support images into a batch with padding.

    Args:
        data (Iterable[tuple]): Iterable of (query, support, label) tuples.
        batch_size (int, optional): Maximum images in output batch. Defaults to 256.
        padding_value (int, optional): Padding value. Defaults to 0.

    Yields:
        Generator[tuple, None, None]: (Images list, variant_groups, labels ) tuples.
    """
    images = [] #Images list to hold the images in the current batch

    #Variants and labels lists to hold the metadata in the current batch
    variants = []
    labels = [] 
    
    num_images = 0 #Keeps track of how many images are in images[]. This does not equal len(images)
    remaining_space = batch_size
    for sample in data:
        query, support, label, *_ = sample
        num_genotypes, num_replicates, *image_size = support.shape
        if num_replicates > 1:
            print("\nNUM replicates", num_replicates)
        assert label < num_genotypes and len(image_size) == 3, "Unexpected data shape"

        remaining_space = batch_size - num_images
        if num_genotypes * num_replicates + 1  > remaining_space and len(images) > 0:
            # If the current sample doesn't fit in the batch, yield the current batch.
            assert num_images == len(variants), "Missmatch of images and variants"
            final = pack_yield_support(images, labels, variants, remaining_space, padding_value)

            #Clear the images list and the META data lists for the next batch
            images.clear()
            variants.clear()
            labels.clear()
            num_images = 0

            #Yield the current batch with padding
            yield final

             
        #Append the current sample to the images list
        #Append the unsqueezed query image to the images list
        images.append(query.unsqueeze(0))

        #Append the respahed support images to the images list
        images.append(support.reshape(-1, *image_size))
        

        #META DATA
        variants += [num_images] * ((num_genotypes * num_replicates) + 1)

        #labels
        #l = [0] #Query image
        labels.append(0) #Query label
        genotype_list = list(range(0, num_genotypes))
        for genotype in genotype_list:
            if genotype == label:
                labels += [1] * (num_replicates)
            else:
                labels += [0] * (num_replicates)
        
        #labels.append(l) #Padding the labels to be batch_size to work with stacking  
        num_images += ((num_genotypes * num_replicates) + 1)

    
    remaining_space = batch_size - num_images
    #YIELD
    #If images is not empty after the loop and there's enough space, yield the last batch
    if remaining_space>= num_images and len(images)>0:  
        assert num_images == len(variants), "Missmatch of images and variants"
        final = pack_yield_support(images, labels, variants, remaining_space, padding_value)

        #Clear the images list and the META data lists for the next batch
        images.clear()
        variants.clear()
        labels.clear()
        num_images = 0

        #Yield the current batch with padding
        yield final

pack_and_pad_support= wds.pipelinefilter(_pack_and_pad_support)



def _split_and_pad_support(data: Iterable[tuple], max_genotypes=6, padding_value=0) -> Generator[tuple, None, None]:
    """Split support images into groups of at most `max_genotypes` images, padding with `padding_value`.

    Transforms support images from GRCHW with variable size G to GCHW with fixed
    and padded G. A positive support image is guaranteed to be present in each group.

    Args:
        data (Iterable[tuple]): Iterable of (query, support, label) tuples.
        max_genotypes (int, optional): Maximum genotypes in output groups. Defaults to 6.
        padding_value (int, optional): Padding value. Defaults to 0.

    Yields:
        Generator[tuple, None, None]: (query, support, num_support, label) tuples.
    """
    for sample in data:
        query, support, label, *_ = sample
        genotypes, replicates, *image_size = support.shape
        assert label < genotypes and len(image_size) == 3, "Unexpected data shape"

        i = 0
        while i < genotypes:
            # Make sure there is a positive support image in each yielded example
            indices = list(range(i, min(i + max_genotypes, genotypes)))
            if i <= label < i + max_genotypes:
                # positive support image is already present in this group
                group_label = label - i
                i += len(indices)
            elif len(indices) < max_genotypes:
                # Space in current group to append positive
                i += len(indices)
                indices.append(label)
                group_label = len(indices) - 1
            else:
                # Swap positive support image into group
                group_label = len(indices) - 1
                i, indices[group_label] = indices[group_label], label

            # yield a separate example for each replicate
            num_support = len(indices)
            padding = (0, 0) * len(image_size) + (0, max_genotypes - num_support)
            for j in range(replicates):
                yield query, torch.nn.functional.pad(
                    support[indices, j],
                    padding,
                    mode='constant',
                    value=padding_value,
                ), num_support, group_label


split_and_pad_support = wds.pipelinefilter(_split_and_pad_support)


def _wrap_and_pad_support(data: Iterable[tuple], max_genotypes=6, padding_value=0) -> Generator[tuple, None, None]:
    for sample in data:
        query, support, label, key, *_ = sample
        genotypes, replicates, *image_size = support.shape
        assert label < genotypes and len(image_size) == 3, "Unexpected data shape"

        for i in range(0, genotypes, max_genotypes):
            indices = range(i, min(i + max_genotypes, genotypes))

            # yield a separate example for each replicate
            num_support = len(indices)
            padding = (0, 0) * len(image_size) + (0, max_genotypes - num_support)
            for j in range(replicates):
                yield query, torch.nn.functional.pad(
                    support[indices, j],
                    padding,
                    mode='constant',
                    value=padding_value,
                ), num_support, label, genotypes, key


wrap_and_pad_support = wds.pipelinefilter(_wrap_and_pad_support)


class EmptyDataset(data.Dataset):
    def __init__(self):
        super(EmptyDataset).__init__()

    def __iter__(self):
        return iter(())
    
    def __len__(self):
        return 0


class GroupedImageDataModule(L.LightningDataModule):
    def __init__(self, train_urls=None, validate_urls=None, predict_urls=None, test_urls=None, batch_size=256, num_workers=1, max_group_size=6, shuffle_size=1000):
        super().__init__()
        self.save_hyperparameters(ignore=["training_urls", "validation_urls", "prediction_urls", "test_urls"])

        self.train_urls = train_urls
        self.validate_urls = validate_urls
        self.predict_urls = predict_urls
        self.test_urls = test_urls

    def make_loader(self, urls, mode="train"):
        # Adapted from: https://github.com/webdataset/webdataset/blob/main/examples/out/train-resnet50-multiray-wds.ipynb

        dataset = wds.WebDataset(urls, shardshuffle=100 if mode == "train" else False)
        if mode == "train":
            dataset = dataset.shuffle(self.hparams.shuffle_size)
        dataset = (
            dataset
            .decode()
            .to_tuple("image.npy.gz", "sim.images.npy.gz", "label.cls")
            .map_tuple(transform_images, transform_images, wds.utils.identity)
            .compose(split_and_pad_support(max_genotypes=self.hparams.max_group_size, padding_value=0))
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
        return self.make_loader(self.train_urls, mode="train")

    def val_dataloader(self):
        # Make sure to return a valid dataloader, even if validation data is not available since
        # Lightning still calls this method with zero validation steps
        if self.validate_urls:
            return self.make_loader(self.validate_urls or [], mode="val")
        return data.DataLoader(EmptyDataset())

    def test_dataloader(self):
        # Make sure to return a valid dataloader, even if validation data is not available since
        # Lightning still calls this method with zero validation steps
        if self.test_urls:
            return self.make_loader(self.test_urls or [], mode="test")
        return data.DataLoader(EmptyDataset())

    def predict_dataloader(self):
        dataset = (
            wds.WebDataset(self.predict_urls, shardshuffle=False)
            .decode()
            .to_tuple("image.npy.gz", "sim.images.npy.gz", "label.cls", "__key__")
            .map_tuple(transform_images, transform_images, wds.utils.identity, wds.utils.identity)
            .compose(wrap_and_pad_support(max_genotypes=self.hparams.max_group_size, padding_value=0))
            .batched(self.hparams.batch_size, partial=True)
        )

        return wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=self.hparams.num_workers)
    

class PackedImageDataModule(L.LightningDataModule):
    def __init__(self, train_urls=None, validate_urls=None, predict_urls=None, test_urls=None, batch_size=256, num_workers=1, max_group_size=6, shuffle_size=1000):
        super().__init__()
        self.save_hyperparameters(ignore=["training_urls", "validation_urls", "prediction_urls", "test_urls"])

        self.train_urls = train_urls
        self.validate_urls = validate_urls
        self.predict_urls = predict_urls
        self.test_urls = test_urls
        self.batch_size = batch_size

    def make_loader(self, urls, mode="train"):
        # Adapted from: https://github.com/webdataset/webdataset/blob/main/examples/out/train-resnet50-multiray-wds.ipynb

        dataset = wds.WebDataset(urls, shardshuffle=100 if mode == "train" else False)
        if mode == "train":
            dataset = dataset.shuffle(self.hparams.shuffle_size)
        dataset = (
            dataset
            .decode()
            .to_tuple("image.npy.gz", "sim.images.npy.gz", "label.cls")
            .map_tuple(transform_images, transform_images, wds.utils.identity)
            .compose(pack_and_pad_support(batch_size = self.batch_size, padding_value=0))
            #.batched(self.hparams.batch_size, partial=False)
        )

        # We unbatch, shuffle, and rebatch to mix samples from different workers as shown in webdataset examples
        loader = (
            wds.WebLoader(
                dataset,
                batch_size=None,
                shuffle=False,
                num_workers=self.hparams.num_workers,
            )
            #.unbatched()
        )
        # if mode == "train":
        #     loader = loader.shuffle(self.hparams.shuffle_size)
        # loader = loader.batched(self.hparams.batch_size)

        return loader

    def train_dataloader(self):
        return self.make_loader(self.train_urls, mode="train")

    def val_dataloader(self):
        # Make sure to return a valid dataloader, even if validation data is not available since
        # Lightning still calls this method with zero validation steps
        if self.validate_urls:
            return self.make_loader(self.validate_urls or [], mode="val")
        return data.DataLoader(EmptyDataset())

    def test_dataloader(self):
        # Make sure to return a valid dataloader, even if validation data is not available since
        # Lightning still calls this method with zero validation steps
        print(self.test_urls)
        if self.test_urls:
            return self.make_loader(self.test_urls or [], mode="test")
        return data.DataLoader(EmptyDataset())

    def predict_dataloader(self):
        dataset = (
            wds.WebDataset(self.predict_urls, shardshuffle=False)
            .decode()
            .to_tuple("image.npy.gz", "sim.images.npy.gz", "label.cls", "__key__")
            .map_tuple(transform_images, transform_images, wds.utils.identity, wds.utils.identity)
            .compose(pack_and_pad_support(batch_size=self.batch_size, padding_value=0))
            #.batched(self.hparams.batch_size, partial=True)
        )

        return wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=self.hparams.num_workers)

class InceptionEncoder(nn.Module):
    def __init__(self, num_channels=8, projection_size=512):
        super(InceptionEncoder, self).__init__()
        self.num_channels = num_channels
        self.projection_size = projection_size

        self.inception = models.inception_v3(weights=None, aux_logits=False)

        # Replace the first layer for our number of channels
        self.inception.Conv2d_1a_3x3.conv = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

        # Replace the final layer with our projection head
        self.inception.fc = nn.Linear(self.inception.fc.in_features, projection_size, bias=False)
        self.bn = nn.BatchNorm1d(projection_size)

    def forward(self, x):
        embeddings = self.inception(x)
        projection = self.bn(embeddings)
        return projection

class EuclideanDistanceMetric(nn.Module):
    def __init__(self):
        super(EuclideanDistanceMetric, self).__init__()
        self.batched_distance = torch.cdist
    def forward(self, query_embeddings, support_embeddings):
        query = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
        support = torch.nn.functional.normalize(support_embeddings, p=2, dim=-1)
        
        # Compute distance between corresponding rows
        distances = torch.norm(query - support, dim=-1)  # shape: (B,)
        return distances

class DotProductSimilarityMetric(nn.Module):
    def __init__(self):
        super(DotProductSimilarityMetric, self).__init__()

    def forward(self, query_embeddings, support_embeddings):
        # Ensure inputs are the same shape
        assert query_embeddings.shape == support_embeddings.shape, \
            f"Shape mismatch: {query_embeddings.shape} vs {support_embeddings.shape}"
        
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
        support_embeddings = torch.nn.functional.normalize(support_embeddings, p=2, dim=-1)

        similarity = torch.sum(query_embeddings * support_embeddings, dim=-1)  # shape: (B,)
        return similarity

class ContrastiveLoss(nn.Module):
    """
    Computes a contrastive loss based on distances between query and support embeddings.

    Args:
        margin (float): The margin enforced between dissimilar pairs. Default is 1.0.

    Inputs:
        distances (torch.Tensor): A 1D tensor containing the distances between each query and each support sample.
        label (torch.Tensor): A 1D tensor of shape (batch_size,) containing the index of the
            correct (positive) support sample for each query.
        mask (torch.Tensor): A boolean tensor of shape (batch_size,) indicating which examples
            in the batch should contribute to the final loss.
        query_embeddings (torch.Tensor): The embeddings for the query samples. (Unused in loss but
            may be passed for logging or future use.)
        support_embeddings (torch.Tensor): The embeddings for the support samples. (Unused in loss but
            may be passed for logging or future use.)

    Returns:
        torch.Tensor: A scalar tensor representing the mean contrastive loss over the masked batch.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, distances: torch.Tensor, label: torch.Tensor, mask: torch.Tensor,  query_embeddings: torch.Tensor, support_embeddings: torch.Tensor, variants):
        label_pair = label[mask]
        loss = label_pair * torch.square(distances[mask]) + (1.0 - label_pair) * torch.square(
            torch.clamp(self.margin - distances[mask], min=0)
        )

        return torch.mean(loss)

class NPairsLoss(nn.Module):
    def __init__(self, l2_reg=0.002):
        super(NPairsLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, metric, label, mask, query_embeddings, support_embeddings, variants):
        # Apply mask to filter valid entries
        metric = metric[mask]                  # (B_valid, S)
        label = label[mask]                    # (B_valid,)
        query_embeddings = query_embeddings[mask]        # (B_valid, D)
        support_embeddings = support_embeddings[mask]    # (B_valid, S, D)
        variants = variants[mask] 

        # Cross-entropy loss on similarity scores
        loss = []
        uniques, counts = variants.unique(return_counts=True)
        for i, (v, c) in enumerate(zip(uniques, counts)):
            v_mask = (variants == v)
            loss.append( torch.nn.functional.cross_entropy(metric[v_mask][1:], label[v_mask][1:].float(), reduction="mean"))
        

        return torch.mean(torch.stack(loss)) #+ reg_term
    
class InfoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, metric, label, mask, query_embeddings, support_embeddings):
        # Apply mask to filter valid entries
        metric = metric[mask]                  # (B_valid, S)
        label = label[mask]                    # (B_valid,)
        query_embeddings = query_embeddings[mask]        # (B_valid, D)
        support_embeddings = support_embeddings[mask]
        variants = variants[mask] 
           # (B_valid, S, D)

        # Scale the similarity scores by the temperature
        scaled_metric = metric / self.temperature


        # Cross-entropy loss on similarity scores
        loss = []
        uniques, counts = variants.unique(return_counts=True)
        for i, (v, c) in enumerate(zip(uniques, counts)):
            v_mask = (variants == v)
            loss.append( torch.nn.functional.cross_entropy(metric[v_mask][1:], label[v_mask][1:].float(), reduction="mean"))
        

        return torch.mean(torch.stack(loss))


class MinimizingPredictor(nn.Module):
    def __init__(self):
        super(MinimizingPredictor, self).__init__()

    def forward(self, metric, variants):
        preds2 = []
        uniques, counts = variants.unique(return_counts=True)
        for i, (v, c) in enumerate(zip(uniques, counts)):
            if v == -100:
                continue
            v_mask = (variants == v)
            # Example prediction
            pred = torch.argmin(metric[v_mask][1:]) #Ignore the first element, which is the query image
            preds2.append(nn.functional.one_hot(pred+1, c)) #Add 1 to the prediction to account for the query image being at index 0
        return torch.cat(preds2)

class MaximizingPredictor(nn.Module):
    def __init__(self):
        super(MaximizingPredictor, self).__init__()

    def forward(self, metric, variants):
        preds2 = []
        uniques, counts = variants.unique(return_counts=True)
        for i, (v, c) in enumerate(zip(uniques, counts)):
            if v == -100:
                continue
            v_mask = (variants == v)
            # Example prediction
            pred = torch.argmax(metric[v_mask][1:]) #Ignore the first element, which is the query image
            preds2.append(nn.functional.one_hot(pred+1, c)) #Add 1 to the prediction to account for the query image being at index 0
        return torch.cat(preds2)

def accuracy_func(preds, labels):
    """
    Computes accuracy for a batch of predictions and labels.

    Args:
        preds (torch.Tensor): One hot encoding of predicted class for all images (B,).
        labels (torch.Tensor): truth labels (B,).

    Returns:
        torch.Tensor: Accuracy as a scalar tensor.
    """
    # Assuming predictions and labels are 1D tensors of same shape
    pred_is_one = preds == 1
    label_is_one = labels == 1

    hits = (pred_is_one & label_is_one).sum()
    total_predictions_as_one = pred_is_one.sum()

    # To avoid division by zero
    return hits.float() / total_predictions_as_one.float() if total_predictions_as_one > 0 else torch.tensor(0.0)

class PackedVariant(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        metric: nn.Module,
        loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        predictor: nn.Module,
        max_group_size=10,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "metric"])

        self.encoder = encoder
        self.metric = metric
        self.loss = loss
        self.predictor = predictor

        self.train_acc = 0
        self.val_acc = 0 
        self.test_acc = 0

        # self.train_acc = Accuracy(task="multiclass", num_classes=max_group_size)
        # self.val_acc = Accuracy(task="multiclass", num_classes=max_group_size)
        # self.test_acc = Accuracy(task="multiclass", num_classes=max_group_size)

        # test = tuple([torch.zeros(8, self.encoder.num_channels, 100, 300),  torch.tensor([   0,    0,    0,    1,    1,    1,    -100,    -100]), torch.tensor([   0,    0,    1,    0,    0,    1, -100, -100] )])
        # self.example_input_array = (test,)

    def on_train_start(self) -> None:
        # Reset validation metrics at the start of training to avoid effects of sanity batches
        #self.val_acc.reset()
        self.val_acc = 0

    def forward(self, batch):
        images, variants, labels = batch         
        images_embeddings = self.encoder(images)

        #Separate query and support embeddings
        
        #Removing padding
        #query_embeddings = images_embeddings[torch.masked_select(variants, variants != -100)]
        #support_embeddings = images_embeddings[ :query_embeddings.size()[0]]

        #Not removing paddings
        query_embeddings = images_embeddings[torch.masked_select(variants, variants != -100)]
        padding = (0,) * (2 * len(images_embeddings.shape) - 1) + (images_embeddings.size()[0]- query_embeddings.size()[0],)
        query_embeddings  =  torch.nn.functional.pad(query_embeddings, padding, mode='constant', value=0)
        support_embeddings = images_embeddings

        # We would want to generalize this to any metric, here we use dot product as an example
        # batched_dot = torch.vmap(torch.dot)
       
        metric = self.metric(query_embeddings, support_embeddings)
        return (metric, query_embeddings, support_embeddings)
        


    def _model_step(self, batch, batch_idx):
        """
        Performs a single forward pass and computes loss and predictions.

        Args:
            batch (Tuple): A batch of data containing:
                - Images: (B, C, H, W) query images + support images flattened
                - variants: (B,) variant index for each image in the batch
                - labels: (B,) true = index of the correct support image for each variant group

        Returns:
            Tuple:
                - loss (Tensor): Scalar loss value.
                - preds (Tensor): Predicted class index for each query (B,).
                - label (Tensor): Ground truth labels (B,).
        """
        # Unpack the batch    
        images, variants, labels = batch

        # Compute metric
        # Forward pass: compute similarity metric between query and support embeddings by passing the batch to forward
        metric, query_embeddings, support_embeddings = self(batch)
        
        
        # Mask padding
        padding_positions = (variants == -100).nonzero(as_tuple=True)[0]
        if len(padding_positions) > 0:
            padding_start_idx = padding_positions[0]
        else:
            padding_start_idx = len(variants)
        mask = torch.arange(len(variants), device=variants.device) < padding_start_idx

        # --- Compute loss and predictions ---
        # Custom loss function may use embeddings and mask

        print("\nVariants", variants)
        
        loss = self.loss(metric, labels, mask, query_embeddings, support_embeddings, variants)
        print("\nLOSS", loss)
        # Predictor outputs predicted class index per query (e.g. argmax over masked metric)
        
        #preds = self.predictor(metric, mask)
        preds = self.predictor(metric, variants)
        print("\nPREDS", preds)
        print("\nMETRIC", metric)
        print("\nLABELS", labels[mask])
        

        return (loss, preds, labels[mask])

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, label = self._model_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        self.train_acc = accuracy_func(preds, label)
        print("\nTRAIN ACC", self.train_acc)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, label = self._model_step(batch, batch_idx)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        #self.val_acc(preds, label)
        self.val_acc = accuracy_func(preds, label)
        print("\nVAL ACC", self.val_acc)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, label = self._model_step(batch, batch_idx)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_acc = accuracy_func(preds, label)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        metric, *_ = self(batch)
        return metric, batch
    
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return { "optimizer": optimizer }
    

class GroupedVariant(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        metric: nn.Module,
        loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        predictor: nn.Module,
        max_group_size=6,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "metric"])

        self.encoder = encoder
        self.metric = metric
        self.loss = loss
        self.predictor = predictor

        self.train_acc = Accuracy(task="multiclass", num_classes=max_group_size)
        self.val_acc = Accuracy(task="multiclass", num_classes=max_group_size)
        self.test_acc = Accuracy(task="multiclass", num_classes=max_group_size)

        self.example_input_array = (torch.zeros(1, self.encoder.num_channels, 100, 300), torch.zeros(1, max_group_size, self.encoder.num_channels, 100, 300))

    def on_train_start(self) -> None:
        # Reset validation metrics at the start of training to avoid effects of sanity batches
        self.val_acc.reset()

    def forward(self, query, support):
        query_embeddings = self.encoder(query)

        # https://github.com/pytorch/pytorch/issues/1927#issuecomment-1245392571
        support = support.transpose(0, 1)
        support_embeddings = torch.stack([self.encoder(s) for s in support], dim=0)
        support_embeddings = support_embeddings.transpose(0, 1)
        metric = self.metric(query_embeddings, support_embeddings)

        return (metric, query_embeddings, support_embeddings)

    def _model_step(self, batch, batch_idx):
        query, support, num_support, label, *_ = batch
        metric, query_embeddings, support_embeddings = self(query, support)
        # Create a mask for the valid support images in each group by filling ones out to the
        # last valid support image (via "exclusive cumsum")
        mask = torch.zeros(metric.shape, dtype=torch.long, device=metric.device)
        mask[(torch.arange(metric.shape[0]), num_support - 1)] = 1
        mask = (1 - (mask.cumsum(dim=-1) - mask)).to(torch.bool)

        loss = self.loss(metric, label, mask, query_embeddings, support_embeddings)
        preds = self.predictor(metric, mask)

        return (loss, preds, label)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, label = self._model_step(batch, batch_idx)

        self.log("train_loss", loss, prog_bar=True)
        self.train_acc(preds, label)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, label = self._model_step(batch, batch_idx)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_acc(preds, label)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, label = self._model_step(batch, batch_idx)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_acc(preds, label)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        query, support, num_support, label, total_support, key = batch
        metric, *_ = self(query, support)
        return metric, num_support, label, total_support, key

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return { "optimizer": optimizer }
    



# def train(cfg, output_dir=None, **kw_args):
#     dm = hydra.utils.instantiate(cfg.data)
#     model = hydra.utils.instantiate(cfg.model)

#     # Overwrite existing checkpoints, instead of creating new versions
#     checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(dirpath=output_dir, enable_version_counter=False)

#     if cfg.data.validation_urls:
#         limit_val_batches = OmegaConf.select(cfg, "data.limit_val_batches", default=1.0)
#         num_sanity_val_steps = OmegaConf.select(cfg, "data.num_sanity_val_steps", default=2)
#     else:
#         # Skip validation if no validation data provided
#         limit_val_batches = num_sanity_val_steps = 0

#     if cfg.data.test_urls:
#         limit_test_batches = OmegaConf.select(cfg, "data.limit_test_batches", default=1.0)
#     else:
#         # Skip testing if no testing data provided
#         limit_test_batches = 0

#     trainer =  hydra.utils.instantiate(cfg.trainer, callbacks=[checkpoint_callback], limit_val_batches=limit_val_batches, num_sanity_val_steps=num_sanity_val_steps, limit_test_batches=limit_test_batches, **kw_args)

#     # TODO: Check if we have reached the final, if not, continue training by setting ckpt_path
#     # https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html#resume-training-state
#     trainer.fit(model=model, datamodule=dm)

#     return checkpoint_callback.best_model_path

def predict(cfg, **kw_args):
    dm = hydra.utils.instantiate(cfg.data)

    model_cls = hydra.utils.get_class(cfg.model._target_)
    # We need to instantiate any of ignored components in the model
    model = model_cls.load_from_checkpoint(cfg.model.checkpoint, encoder=hydra.utils.instantiate(cfg.model.encoder), metric=hydra.utils.instantiate(cfg.model.metric))
    print(model)
    trainer = L.Trainer(limit_predict_batches=2)
    predictions = trainer.predict(model, dm)
    print(predictions)


def test(cfg, **kw_args):
    dm = hydra.utils.instantiate(cfg.data)

    model_cls = hydra.utils.get_class(cfg.model._target_)
    # We need to instantiate any of ignored components in the model
    model = model_cls.load_from_checkpoint(cfg.model.checkpoint, encoder=hydra.utils.instantiate(cfg.model.encoder), metric=hydra.utils.instantiate(cfg.model.metric))

    trainer = L.Trainer(**kw_args)
    trainer.test(model, dm)
