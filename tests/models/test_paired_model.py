import pytest
import lightning as L

from npsv3.models.paired import train

from .. import data_path, result_path

class TestPairedCNNModel:
    @pytest.mark.cfg_overrides(
        f"model=paired_inception_contrastive",
        f"data.training_urls={data_path('images-0000.tar')}",
        f"data.validation_urls={data_path('images-0000.tar')}",
        "data.batch_size=2",
    )
    def test_paired_cnn_contrastive_model(self, cfg):
        train(cfg, fast_dev_run=True)

    @pytest.mark.cfg_overrides(
        f"model=paired_inception_npairs",
        f"data.training_urls={data_path('images-0000.tar')}",
        f"data.validation_urls={data_path('images-0000.tar')}",
        "data.batch_size=2",
    )
    def test_paired_cnn_npairs_model(self, cfg):
        train(cfg, fast_dev_run=True)