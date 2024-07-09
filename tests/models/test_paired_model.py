import pytest
import lightning as L

from npsv3.models.paired import train

from .. import data_path, result_path

class TestPairedCNNModel:
    @pytest.mark.cfg_overrides(
        f"data.training_urls={data_path('images-0000.tar')}",
    )
    def test_paired_cnn_model(self, cfg):
        train(cfg, fast_dev_run=True)