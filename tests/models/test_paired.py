import os

import pytest
import hydra

from npsv3.models.runners import test, train

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
        "model=paired_packed_vit_infonce",
        f'model.encoder.checkpoint_path="{os.path.join(EXPERIMENTS_DIR, "training/hgsvc3-hprc-2024-02-23-mc-chm13.GRCh38.vcfbub.a100k.wave.passing.training.hg38.models/data.batch_size=512,data.epoch_batches=16000,data.resampled=True,data=masked_images,model=masked_vit,trainer.max_epochs=10,trainer=vit_pretraining/model.ckpt")}"',
        "data=packed_images",
        "data.batch_size=64",
        f'data.train_urls="{TEST_IMAGES}"',
        "trainer=paired",
    )
    def test_paired_vit_infonce(self, cfg):
        train(cfg, fast_dev_run=True)
