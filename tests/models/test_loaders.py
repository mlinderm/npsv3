import os

import pytest
import torch
import webdataset as wds

from npsv3.images.example import (
    example_to_image,
)
from npsv3.models.loaders import AllImageDataModule, MaskedImageDataModule, PackedImageDataModule

from .. import EXPERIMENTS_DIR, data_path, result_path


class TestPackedImageDataLoader:
    def test_packed_loader(self, cfg):
        dm = PackedImageDataModule(train_urls=data_path("images-0000.tar"), batch_size=64, num_workers=cfg.threads)
        dm.prepare_data()

        dm.setup(stage="fit")

        for _i, batch in enumerate(dm.train_dataloader()):
            images, labels, offsets, *addl_fields = batch

            b, c, h, w = images.shape
            assert b == 2*4+1, "Batch size must be 2 replicates of 4 genotypes + 1 query image"
            assert labels.shape == (2*4,), "Labels must be 1D tensor with 2 replicates of 4 genotypes"
            assert labels.equal(torch.tensor([0,0, 3,3, 3,3, 1,1], dtype=torch.long)), "All replicates of correct genotype (3) should have 1 label"
            assert offsets.equal(torch.tensor([0, 8], dtype=torch.long)), "Offsets should match the support images"
            assert addl_fields[0] == ["12_22129565_22130387"], "Additional fields should include key or region strings"

        assert _i == 0, "The 8 support images are packed into a single batch with the query image"

        dm.teardown(stage="fit")

    def test_packed_and_padded_loader(self, cfg):
        dm = PackedImageDataModule(train_urls=data_path("images-0000.tar"), batch_size=64, pad=True, num_workers=cfg.threads)
        dm.prepare_data()

        dm.setup(stage="fit")

        for _i, batch in enumerate(dm.train_dataloader()):
            images, labels, offsets, *addl_fields = batch

            b, c, h, w = images.shape
            assert b == 64, "Batch size must be padded to max batch size"
            assert labels.shape == (b,), "Labels must be 1D tensor matching image batch size"
            assert labels.equal(torch.tensor([0,0, 3,3, 3,3, 1,1] + [-100]*(64-8), dtype=torch.long)), "All replicates of correct genotype (3) should have 1 label"
            assert offsets.equal(torch.tensor([0, 8], dtype=torch.long)), "Offsets should match the support images"
            assert addl_fields[0] == ["12_22129565_22130387"], "Additional fields should include key or region strings"

        assert _i == 0, "The 8 support images are packed into a single batch with the query image"

        dm.teardown(stage="fit")

class TestMaskedImageDataLoader:
    def test_masked_loader(self, cfg):
        dm = MaskedImageDataModule(train_urls=data_path("images-0000.tar"), batch_size=2, num_workers=cfg.threads, patch_size=(16,16), mask_fraction=0.5)
        dm.prepare_data()

        dm.setup(stage="fit")

        for _i, batch in enumerate(dm.train_dataloader()):
            images, masks = batch

            b, _c, h, w = images.shape
            assert b == 2

            num_patches = (h // 16) * (w // 16)
            assert masks.shape == (b, num_patches), f"Masks must be 2D tensor with {num_patches} patches per image"
            assert torch.equal(masks.sum(dim=1), torch.tensor([int(0.5 * num_patches)]*b)), "Each image should have 50% of patches masked"

        assert _i == 3, "The 9 images packed into 4 complete batches of 2 images each, with the remaining partial batch dropped"

        dm.teardown(stage="fit")

    def test_partial_masked_loader(self, cfg):
        dm = MaskedImageDataModule(test_urls=data_path("images-0000.tar"), batch_size=64, num_workers=cfg.threads, patch_size=(8,8), mask_fraction=0.8)
        dm.prepare_data()

        # Ensure partial batches are included
        dm.setup(stage="test")

        for _i, batch in enumerate(dm.test_dataloader()):
            images, masks = batch

            b, _c, h, w = images.shape
            assert b == 2*4+1, "Batch size must be 2 replicates of 4 genotypes + 1 query image"

            num_patches = (h // 8) * (w // 8)
            assert masks.shape == (b, num_patches), f"Masks must be 2D tensor with {num_patches} patches per image"
            assert torch.equal(masks.sum(dim=1), torch.tensor([int(0.8 * num_patches)]*b)), "Each image should have 80% of patches masked"


        assert _i == 0, "All 9 images are packed into a single partial batch"

        dm.teardown(stage="test")

class TestNPSV2Examples:
    @pytest.mark.skipif(
        not os.path.exists(os.path.join(EXPERIMENTS_DIR, "training/freeze3.sv.alt.passing.training.hg38.DEL.images/HG00096/+pileup.snv_input=True,generator=single_depth_phaseread,pileup.discrete_mapq=True,pileup.render_snv=True,simulation.augment=True,simulation.chrom_norm_covg=True,simulation.replicates=5/images.tar")),
        reason="NPSV2 images dataset required"
    )
    def test_visualize_converted_images(self, tmp_path, cfg):
        urls = os.path.join(EXPERIMENTS_DIR, "training/freeze3.sv.alt.passing.training.hg38.DEL.images/HG00096/+pileup.snv_input=True,generator=single_depth_phaseread,pileup.discrete_mapq=True,pileup.render_snv=True,simulation.augment=True,simulation.chrom_norm_covg=True,simulation.replicates=5/images.tar")
        dataset = wds.WebDataset(urls, shardshuffle=False).decode()
        for _i, sample in enumerate(dataset):
            #png_path = str(tmp_path / "test.png")
            png_path = result_path("test.png")

            example_to_image(
                cfg,
                {"image": sample["image.npy.gz"], "sim.images": sample["sim.images.npy.gz"] },
                png_path, with_simulations=True, render_channels=True, select_channels=[0, 1, 5], # ALIGNED, PAIRED, ALLELE
            )
            assert os.path.exists(png_path)
            break

    def test_packed_loader(self, tmp_path, cfg):
        urls = os.path.join(EXPERIMENTS_DIR, "training/freeze3.sv.alt.passing.training.hg38.DEL.images/HG00096/+pileup.snv_input=True,generator=single_depth_phaseread,pileup.discrete_mapq=True,pileup.render_snv=True,simulation.augment=True,simulation.chrom_norm_covg=True,simulation.replicates=5/images.tar")
        dm = PackedImageDataModule(train_urls=urls, batch_size=64, num_workers=cfg.threads)
        dm.prepare_data()

        dm.setup(stage="fit")

        for _i, batch in enumerate(dm.train_dataloader()):
            images, labels, offsets, *addl_fields = batch
            print(images.shape, labels.shape, offsets, addl_fields)
           

            break

        dm.teardown(stage="fit")