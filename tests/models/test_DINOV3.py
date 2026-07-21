import pytest
# from npsv3.models.runners import predict
from npsv3.models.runners import predict
from npsv3.models.runners import test as model_test

class TestAccuracy:
    @pytest.mark.cfg_overrides(
        "pileup=unphased_variant",
        "model=DINOv3",
        # "data.patch_size=16",
        "data=packed_images",
        # "data._target_=npsv3.models.transformer.RealImageDataModule",
        "data.test_urls='/storage/mlinderman/projects/sv/npsv3-experiments/training/hgsvc3-hprc-2024-02-23.dipcall.passing.hg38.eval-images/HG00733/generator=coverage,pileup=unphased_variable,simulation.replicates=1/images-0000.tar'",
        # "data.batch_size=4",
        # '+model.checkpoint="DINOv3.ckpt"',
    )
    def test_accuracy(self, tmp_path, cfg):
        model_test(cfg, limit_predict_batches=2)