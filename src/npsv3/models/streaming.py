from typing import Any, Optional, Self

import lightning as L
import numpy as np
import streaming
import torch
from numpy.typing import NDArray
from torch.utils import data
from torchvision.transforms import v2 as transforms


class PackableMDSWriter(streaming.MDSWriter):
    format = "pmds"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_lengths = []

    def write(self, sample: dict[str, Any], sample_length: int) -> None:
        self.sample_lengths.append(sample_length)
        super().write(sample)

    def flush_shard(self) -> None:
        if len(self.sample_lengths) > len(self.new_samples):
            # A shard is being flushed as a sample is written, don't include last sample length
            super().flush_shard()
            self.shards[-1]["sample_lengths"] = self.sample_lengths[:-1]
            self.sample_lengths = self.sample_lengths[-1:]
        else:
            super().flush_shard()
            self.shards[-1]["sample_lengths"] = self.sample_lengths
            self.sample_lengths = []


class PackableMDSReader(streaming.base.format.mds.MDSReader):
    def __init__(self, *args, sample_lengths: list[int], **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_lengths = sample_lengths

    @classmethod
    def from_json(cls, dirname: str, split: Optional[str], obj: dict[str, Any]) -> Self:
        # Set the format to the expected "mds" for the underlying MDSReader
        obj["format"] = streaming.MDSWriter.format
        return super().from_json(dirname, split, obj)


# "Register" our custom pmds packable MDS format
streaming.base.format._readers.update({"pmds": PackableMDSReader})  # noqa: SLF001


class PackableStreamingDataset(streaming.StreamingDataset):
    def __init__(self, *args, packing_method: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if packing_method is not None:
            self.batching_method = packing_method

    def sample_to_length(self, sample_ids):
        # Global sample IDs are generated from the concatenation of all shards. Construct the corresponding sample lengths.
        all_sample_lengths = np.concat([shard.sample_lengths for shard in self.shards])
        return np.where(sample_ids != -1, all_sample_lengths[sample_ids], 0)

    def get_item(self, sample_id: int) -> Any:
        print(f"Fetching sample {sample_id}")
        sample = super().get_item(sample_id)
        print(sample)
        return sample


def _greedy_knapsack(samples: NDArray[np.int64], lengths: NDArray[np.int64], max_length: int) -> NDArray[np.int64]:
    cuml_length = lengths.cumsum()
    curr_idx = 0
    target_length = max_length
    packed_batches = []
    while curr_idx < len(samples):
        next_idx = np.searchsorted(cuml_length[curr_idx : curr_idx + max_length], target_length, side="right")
        packed_batches.append(
            np.pad(samples[curr_idx : curr_idx + next_idx], (0, max_length - next_idx), constant_values=-1)
        )
        curr_idx += next_idx
        target_length += cuml_length[curr_idx - 1]
    return np.concatenate(packed_batches)


def generate_work_random_packed_sample_batching(
    dataset: PackableStreamingDataset, world: streaming.base.world.World, epoch: int, sample_in_epoch: int
) -> NDArray[np.int64]:
    # (physical nodes, ranks per node, workers per rank, batches per worker, batch size)
    initial_work = streaming.base.batching.random.generate_work_random_batching(dataset, world, epoch, 0)
    (n_nodes, n_ranks, n_workers, n_batches, batch_size) = initial_work.shape

    # Determine the actual length of each worker's data in batches using sequencing packing algorithm.
    # This will increase the number of batches as the samples are expanded.
    work_length = dataset.sample_to_length(initial_work)

    # TODO: Undo the worker-level interleaving to pack sequences in shard-aware order
    packed_work = []
    for rank_work, rank_work_length in zip(
        initial_work.reshape(n_nodes * n_ranks * n_workers, -1),
        work_length.reshape(n_nodes * n_ranks * n_workers, -1),
        strict=True,
    ):
        packed_rank_work = _greedy_knapsack(rank_work, rank_work_length, dataset.batch_size)
        packed_work.append(packed_rank_work)

    # Equalize work across ranks
    max_samples = max(len(rw) for rw in packed_work)
    packed_work = [np.pad(rw, (0, max_samples - len(rw))[sample_in_epoch:], constant_values=-1) for rw in packed_work]
    packed_work = np.reshape(np.concatenate(packed_work), (n_nodes, n_ranks, n_workers, -1, batch_size))

    # We would need to equalize with valid samples here as Streaming errors out with any padding other than -1,
    # but drops all -1 samples, so it is not possible to have exclusively padding batches.

    # Not clear that this will work for us. A possible alternative is Megatron-Energon, but how shards and samples
    # are split is hard to find/control (here? https://github.com/NVIDIA/Megatron-Energon/blob/412a6cf38c2e64ef32fdf447cf53071f7311ec92/src/megatron/energon/flavors/webdataset/base_webdataset.py#L147)
    # Although it has a notion of sequence packing, it does not appear to have a clear mechanism for equalizing work across ranks.
    # It appears we effectively would need to "pre-run" the packing to determine the lengths, possibly equalize and then actually perform the loading.
    # Not clear we can do that within streaming or Megatron-Energon.

    # 1. Sort shards and partition by rank
    # 2. Interleave? samples across workers within a shard.
    # 3. Shuffle samples within a worker's allocation.
    # 4. Compute packed length and deterministically adjust number of batches per worker to equalize across ranks (eg. from largest to smallest).

    # Can do so once in LightningDataModule setup prepare_data hook, but that is run once, not every epoch. No hook is available for the latter. Would need to do so
    # during iterator creation, but that occurs in every worker. We could possibly force it to reload every epoch with reload_dataloaders_every_epoch. That could reset
    # the sampling. That would presumably occur in every process (unlike setup). We could possibly access the synchronization provided by the Lightning strategy to compute in a single
    # process and broadcast the result. We would use the stateful dataloader and save the RNG state to use to recreate the distribution.

    return packed_work


streaming.base.batching.batching_methods.update({"packed_samples": generate_work_random_packed_sample_batching})


class EmptyDataset(data.Dataset):
    def __init__(self):
        super(EmptyDataset).__init__()

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


class PackableDataModule(L.LightningDataModule):
    def __init__(self, *, train_urls: Optional[str] = None, mean=(0.5,), std=(0.5,)):
        super().__init__()
        self.save_hyperparameters(ignore=["training_urls", "validation_urls", "prediction_urls", "test_urls"])
        self.train_urls = train_urls

        self.transforms = transforms.Compose(
            [
                transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input
                transforms.Normalize(mean=self.hparams.mean, std=self.hparams.std),
            ]
        )

    def prepare_data(self):
        # Download index files to a local cache if needed
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # Rank 0 processes load the index
        print(self.trainer)
        print(self.trainer.is_global_zero)

    def train_dataloader(self):
        if self.trainer.is_global_zero:
            print("Generate loading schedule with equalized number of batches")
            schedule = np.array([1, 2, 3])
        else:
            schedule = None
        # It appears we can use broadcast to share the schedule within the datamodule.
        schedule = self.trainer.strategy.broadcast(schedule, src=0)
        print(f"Using schedule: {schedule}")
