import pytest
import torch

from npsv3.models.paired import GroupedImageDataModule
from npsv3.models.runners import train

from .. import data_path


@pytest.mark.cfg_overrides(
    "pileup.image_channels=[0,1,2,3,4,5,6,7]",
)
class TestPairedDataLoader:
    def test_paired_loader(self, cfg):
        dm = GroupedImageDataModule(data_path("images-0000.tar"), batch_size=1, num_workers=cfg.threads)
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

class TestPairedCNNModel:
    @pytest.mark.cfg_overrides(
        "model=paired_inception_contrastive",
        f"data.train_urls={data_path('images-0000.tar')}",
        f"data.validate_urls={data_path('images-0000.tar')}",
        "data.batch_size=2",
        "trainer=paired",
    )
    def test_paired_cnn_contrastive_model(self, cfg):
        train(cfg, fast_dev_run=True)

    @pytest.mark.cfg_overrides(
        "model=paired_inception_npairs",
        f"data.train_urls={data_path('images-0000.tar')}",
        f"data.validate_urls={data_path('images-0000.tar')}",
        "data.batch_size=2",
        "trainer=paired",
    )
    def test_paired_cnn_npairs_model(self, cfg):
        train(cfg, fast_dev_run=True)
