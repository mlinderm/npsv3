import pytest

from npsv3.models.dvae import RealImageDataModule
from npsv3.models.runners import train

from .. import data_path

@pytest.mark.cfg_overrides(
    f"pileup.image_channels=[0,1,2,3,4,5,6,7]",
)
class TestRealImageDataLoader:
    def test_real_image_loader(self, cfg):
        dm = RealImageDataModule(data_path("images-0000.tar"), num_channels=len(cfg.pileup.image_channels), batch_size=1, num_workers=cfg.threads)
        dm.prepare_data()
        
        dm.setup(stage="fit")
        for _i, batch in enumerate(dm.train_dataloader()):
            image, = batch
            assert image.shape == (1, len(cfg.pileup.image_channels), 224, 224)

        dm.teardown(stage="fit")


class TestDVAEModel:
    @pytest.mark.cfg_overrides(
        "pileup.image_channels=[0,1,2,3,4,5,6,7]",
        "model=vqvae",
        "data=real_image",
        f"data.training_urls={'::'.join([data_path('images-0000.tar')]*2)}",
        "data.batch_size=2",
        "trainer=dvae",
    )
    def test_vqvae_model(self, cfg):
        # Need to stack up the data and reduce the batch size so we get at least one batch
        train(
            cfg,
            fast_dev_run=True
        )