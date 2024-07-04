import numpy as np
import webdataset as wds

def _split_and_pad_support(data, max_genotypes=6, padding_value=0):
    # G R H W C
    for sample in data:
        query, support, label = sample
        genotypes, replicates, *image_size = support.shape
        print(genotypes, replicates, image_size)
        assert len(image_size) == 3
        assert label < genotypes

        i = 0
        while i < genotypes:
            # Make sure there is a positive support image in each yielded example
            indices = list(range(i, min(i + max_genotypes, genotypes)))
            if i <= label < i + max_genotypes:
                # positive support image is already present in this group
                group_label = label - i
                i += len(indices)
            else:
                # Swap positive support image into group
                group_label = len(indices) - 1
                i, indices[group_label] = indices[group_label], label

            # yield an example for each replicate
            num_support = len(indices)
            padding = [(0, max_genotypes - num_support)] + [(0, 0)] * len(image_size)
            for j in range(replicates):
                yield query, np.pad(
                    support[indices, j],
                    pad_width=padding,
                    mode='constant',
                    constant_values=padding_value,
                ), num_support, group_label


split_and_pad_support = wds.pipelinefilter(_split_and_pad_support)

# class PileupImageData(pl.LightningDataModule):
#     def __init__(self, cfg, shards, **kw):
#         super().__init__(self)

#         self.cfg = cfg
#         self.training_urls = shards

#     def make_loader(self, cfg, urls):
#         # Adapted from: https://github.com/webdataset/webdataset/blob/main/examples/out/train-resnet50-multiray-wds.ipynb

#         # Do we need a cache_dir?
#         # trainset = wds.WebDataset(trainset_url, resampled=True, cache_dir=cache_dir, nodesplitter=wds.split_by_node)

#         dataset = (
#             wds.WebDataset(urls)
#             .shuffle(shuffle)
#             .decode()
#             .to_tuple("image.npy.gz", "sim.images.npy.gz", "label.cls")
#             .then(split_and_pad_support)
#             .batched(self.batch_size, partial=False)
#         )

#         # We unbatch, shuffle, and rebatch to mix samples from different workers as shown in webdataset examples
#         loader = (
#             wds.WebLoader(
#                 dataset,
#                 batch_size=None,
#                 num_workers=self.num_workers,
#             )
#             .unbatched()
#             .shuffle(1000)
#             .batched(batch_size)
#         )

#         return loader

#     def train_dataloader(self):
#         return self.make_loader(self.training_urls, mode="train")
