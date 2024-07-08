import pytest
import webdataset as wds
import torch
from torchvision.transforms import v2 as transforms

from npsv3.models.paired import transform_images, split_and_pad_support, GroupedImageDataModule

from .. import data_path, result_path

@pytest.mark.cfg_overrides(
    f"pileup.image_channels=[0,1,2,3,4,5,6,7]",
)
class TestPairedDataLoader:
    def test_paired_loader(self, cfg):
        dm = GroupedImageDataModule(cfg, data_path("images-0000.tar"), batch_size=1)
        dm.prepare_data()

        dm.setup(stage="fit")

        for _i, batch in enumerate(dm.train_dataloader()):
            query, support, num_support, label = batch

            assert query.shape == (1, len(cfg.pileup.image_channels), cfg.pileup.image_height, cfg.pileup.image_width)
            assert support.shape == (1, 6, len(cfg.pileup.image_channels), cfg.pileup.image_height, cfg.pileup.image_width)
            assert num_support == torch.tensor([4]), "Only 4 genotypes in support data"
            assert label == torch.tensor([3])

        assert _i == 1, "The two replicates become 2 examples in the dataset"

        dm.teardown(stage="fit")