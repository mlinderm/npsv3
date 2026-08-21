import pytest
# from npsv3.models.runners import predict
from npsv3.models.runners import predict, train
from npsv3.models.runners import test as model_test


@pytest.mark.cfg_overrides(
        "pileup=unphased_variable",
        "model=paired_packed_inout_contrastive.yaml",
        # "data.patch_size=16",
        "data=packed_images",
        # "data._target_=npsv3.models.transformer.RealImageDataModule",
        "data.test_urls='/storage/mlinderman/projects/sv/npsv3-experiments/training/hgsvc3-hprc-2024-02-23.dipcall.passing.hg38.eval-images/HG00733/generator=coverage,pileup=unphased_variant,simulation.replicates=1/images-0000.tar'",
        "data.train_urls='/storage/mlinderman/projects/sv/npsv3-experiments/training/hgsvc3-hprc-2024-02-23.dipcall.passing.hg38.eval-images/HG00733/generator=coverage,pileup=unphased_variable,simulation.replicates=1/images-0000.tar'",
        "data.batch_size=64",
        "trainer.max_epochs=1",
        # '+model.checkpoint="/storage/mlinderman/projects/sv/npsv3-experiments/training/hgsvc3-hprc-2024-02-23-mc-chm13.GRCh38.vcfbub.a100k.wave.passing.training.hg38.models2/+train=0009,data.epoch_batches=2720,data.resampled=True/epoch=19-step=54400.ckpt"',
    )


class TestAccuracy:
    @pytest.mark.skip()
    def test_accuracy(self, tmp_path, cfg):
        model_test(cfg)

    # @pytest.mark.skip()
    def test_train(self, cfg):
        train(cfg, limit_train_batches=50)