import os

import pytest

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
        "model=paired_packed_inception_infonce",
        "data=packed_images",
        "data.batch_size=64",
        f'data.train_urls="{TEST_IMAGES}"',
        "trainer=paired",
    )
    def test_paired_cnn_infonce(self, cfg):
        train(cfg, fast_dev_run=True)

    @pytest.mark.skipif(not os.path.exists(os.path.join(EXPERIMENTS_DIR, "training/freeze4.sv.alt.passing.training.hg38.models/data.batch_size=1024,data=packed_images,model=paired_packed_inception_infonce,trainer.max_epochs=10/epoch=9-step=112683.ckpt")), reason="Model does not exist")
    @pytest.mark.cfg_overrides(
        "command=test",
        "model=paired_packed_inception_infonce",
		f'model.checkpoint="{os.path.join(EXPERIMENTS_DIR, "training/freeze4.sv.alt.passing.training.hg38.models/data.batch_size=1024,data=packed_images,model=paired_packed_inception_infonce,trainer.max_epochs=10/epoch=9-step=112683.ckpt")}"',
        "data=packed_images",
        "data.batch_size=64",
         f'data.test_urls="{TEST_IMAGES}"',
        "trainer=paired",
    )
    def test_paired_cnn_infonce_testing(self, cfg):
        test(cfg, limit_test_batches=1)

    @pytest.mark.cfg_overrides(
        "model=paired_packed_inception_infonce",
        "data=packed_images",
        "data.batch_size=64",
        f'data.train_urls="{os.path.join(EXPERIMENTS_DIR, "training/freeze3.sv.alt.passing.training.hg38.DEL.images/HG00096/+pileup.snv_input=True,generator=single_depth_phaseread,pileup.discrete_mapq=True,pileup.render_snv=True,simulation.augment=True,simulation.chrom_norm_covg=True,simulation.replicates=5/images.tar")}"',
        "trainer=paired",
        "model.encoder.num_channels=9",
    )
    def test_paired_cnn_infonce_npsv2(self, cfg):
        train(cfg, fast_dev_run=50)
