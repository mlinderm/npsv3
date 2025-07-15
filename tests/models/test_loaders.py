import torch

from npsv3.models.loaders import PackedImageDataModule

from .. import data_path


class TestPackedDataLoader:
    def test_packed_loader(self, cfg):
        dm = PackedImageDataModule(train_urls=data_path("images-0000.tar"), batch_size=64, num_workers=cfg.threads)
        dm.prepare_data()

        dm.setup(stage="fit")

        for _i, batch in enumerate(dm.train_dataloader()):
            images, labels, offsets, *addl_fields = batch

            b, c, h, w = images.shape
            assert b == 2*4+1, "Batch size must be 2 replicates of 4 genotypes + 1 query image"
            assert labels.shape == (2*4,), "Labels must be 1D tensor with 2 replicates of 4 genotypes"
            assert labels.equal(torch.tensor([0,0, 0,0, 0,0, 1,1], dtype=torch.long)), "All replicates of correct genotype (3) should have 1 label"
            assert offsets.equal(torch.tensor([0, 8], dtype=torch.long)), "Offsets should match the support images"
            assert addl_fields[0] == ["12_22129565_22130387"], "Additional fields should include key or region strings"

        assert _i == 0, "The 8 support images are packed into a single batch with the query image"

        dm.teardown(stage="fit")

    def test_packed_and_paired_loader(self, cfg):
        dm = PackedImageDataModule(train_urls=data_path("images-0000.tar"), batch_size=64, pad=True, num_workers=cfg.threads)
        dm.prepare_data()

        dm.setup(stage="fit")

        for _i, batch in enumerate(dm.train_dataloader()):
            images, labels, offsets, *addl_fields = batch

            b, c, h, w = images.shape
            assert b == 64, "Batch size must be padded to max batch size"
            assert labels.shape == (b,), "Labels must be 1D tensor matching image batch size"
            assert labels.equal(torch.tensor([0,0, 0,0, 0,0, 1,1] + [-100]*(64-8), dtype=torch.long)), "All replicates of correct genotype (3) should have 1 label"
            assert offsets.equal(torch.tensor([0, 8], dtype=torch.long)), "Offsets should match the support images"
            assert addl_fields[0] == ["12_22129565_22130387"], "Additional fields should include key or region strings"

        assert _i == 0, "The 8 support images are packed into a single batch with the query image"

        dm.teardown(stage="fit")
