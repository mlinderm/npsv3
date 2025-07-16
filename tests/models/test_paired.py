import os

import pytest

from npsv3.models.runners import train

from .. import EXPERIMENTS_DIR

TEST_IMAGES = os.path.join(EXPERIMENTS_DIR, "training/freeze4.sv.alt.passing.training.hg38.images/HG00731/generator=coverage,pileup=unphased,simulation.replicates=1/images-0000.tar")

class TestPackedVariant:
    @pytest.mark.skipif(not os.path.exists(TEST_IMAGES), reason="Skip if experiments directory does not exist")
    @pytest.mark.cfg_overrides(
        "model=paired_packed_inception_contrastive",
        "data=packed_images",
        "data.batch_size=64",
        f'data.train_urls="{TEST_IMAGES}"',
        "trainer=paired",
    )
    def test_paired_cnn_contrastive(self, cfg):
        train(cfg, fast_dev_run=True)

    @pytest.mark.skipif(not os.path.exists(TEST_IMAGES), reason="Skip if experiments directory does not exist")
    @pytest.mark.cfg_overrides(
        "model=paired_packed_inception_infonce",
        "data=packed_images",
        "data.batch_size=64",
        f'data.train_urls="{TEST_IMAGES}"',
        "trainer=paired",
    )
    def test_paired_cnn_infonce(self, cfg):
        train(cfg, fast_dev_run=True)
