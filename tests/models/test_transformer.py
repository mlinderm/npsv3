import os
import io

import torch
import pytest

from npsv3.models.transformer import RealImageDataModule, reconstruct
from .. import B37_REF_FASTA, data_path, result_path
from PIL import Image
import random
import requests
import webdataset as wds
import numpy as np
from npsv3.models.runners import train, assess_accuracy
from npsv3.images.example import example_to_image

from omegaconf import OmegaConf

def torch_decode(key, data):
    # Use custom decoder to eliminate a warning with torch.load about weights_only
    if key.endswith((".pth", ".pt")):
        stream = io.BytesIO(data)
        return torch.load(stream, weights_only=True, map_location='cpu')

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
test_image = Image.open(requests.get(url, stream=True).raw)

@pytest.mark.skip()
@pytest.mark.cfg_overrides(
    "pileup=unphased_variant",
)
class TestRealImageDataLoader:
    def test_real_image_loader(self, cfg):
        dm = RealImageDataModule(
            data_path("images-0000.tar"),
            num_channels=len(cfg.pileup.image_channels),
            batch_size=1,
            num_workers=cfg.threads,
        )
        dm.prepare_data()

        dm.setup(stage="fit")
        for _i, batch in enumerate(dm.train_dataloader()):
            keys, images, *_ = batch
            # assert images.shape == (1, len(cfg.pileup.image_channels), cfg.pileup.image_height, cfg.pileup.image_width)

        dm.teardown(stage="fit")

@pytest.mark.skip()
class TestMiM:
    @pytest.mark.cfg_overrides(
        "pileup=unphased_variant",
        "model=MiM",
        "data=real_image",
        "data._target_=npsv3.models.transformer.RealImageDataModule",
        f"data.train_urls={'::'.join([data_path('unphased_variant_images-0000.tar')]*2)}",
        "data.batch_size=2",
        "data.patch_size=16",
        "trainer=transformer",
    )
    def test_MiM(self, cfg):
        train(cfg, fast_dev_run=True)

@pytest.mark.skip()
class TestClassifier:
    @pytest.mark.cfg_overrides(
        "pileup=unphased_variant",
        "model=classifier",
        "data.patch_size=16",
        "data=real_image",
        "data._target_=npsv3.models.transformer.RealImageDataModule",
        f"data.train_urls={'::'.join([data_path('unphased_variant_images-0000.tar')]*2)}",
        "data.batch_size=2",
        "trainer=transformer",
        # "pretrained=classifier"
    )
    def test_classifier(self, cfg):
        train(cfg, fast_dev_run=True)

@pytest.mark.skip()
class TestAccuracy:
    @pytest.mark.cfg_overrides(
        "pileup=unphased_variant",
        "model=classifier",
        "data.patch_size=16",
        "data=real_image",
        "data._target_=npsv3.models.transformer.RealImageDataModule",
        "data.predict_urls='/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze4.sv.alt.passing.training.hg38.images/NA19983/generator=coverage,pileup=unphased_variant,simulation.replicates=1/images-0000.tar'",
        "data.batch_size=1",
        '+model.checkpoint="/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze4.sv.alt.passing.training.hg38.models/data._target_=npsv3.models.transformer.RealImageDataModule,data.batch_size=256,data.mask_scheme=[random,50],data.patch_size=16,data=real_image,model=MiM,pileup=unphased_variant,trainer.max_epochs=50/full_train-step=407400.ckpt"',
    )
    def test_accuracy(self, tmp_path, cfg):
        # output_dir = str(tmp_path / "shards")
        assess_accuracy(cfg, cfg.model.checkpoint, limit_predict_batches=100)

# @pytest.mark.skip()
@pytest.mark.cfg_overrides(
    f"reference={B37_REF_FASTA}",
    "simulation.replicates=0",
    "pileup=unphased_variant",
    "model=MiM",
    "data.patch_size=16",
    "data.mask_scheme=[\"random\", 50]",
    '+model.checkpoint="/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze4.sv.alt.passing.training.hg38.models/10Epoch_50R_16_AdamW_BEST/pretrained_MiM-step=101840.ckpt"',
    "data=real_image",
    "data._target_=npsv3.models.transformer.RealImageDataModule",
    "data.batch_size=1",
)
@pytest.mark.usefixtures("ray_setup")
class TestTransformerReconstruction:
    def test_transformer_reconstruction(self, tmp_path, cfg, hg002_sample):
        # Generate image for variant
        output_dir = str(tmp_path) # / "shards")

        print("Temp path: ", tmp_path)

        # Write reconstructed images to a WebDataset file
        local_conf = OmegaConf.from_dotlist([
            # f"data.predict_urls=/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze4.sv.alt.passing.training.hg38.images/HG00096/generator=coverage,pileup=unphased_variant,simulation.replicates=1/images-0000.tar",
            f"data.predict_urls=/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze3.sv.alt.passing.training.hg38.DEL.images/HG00096/+pileup.snv_input=True,generator=single_depth_phaseread,pileup.discrete_mapq=True,pileup.render_snv=True,simulation.augment=True,simulation.chrom_norm_covg=True,simulation.replicates=5/images.tar",
        ])
        local_cfg = OmegaConf.merge(cfg, local_conf)
        mask_path = result_path("mask.png")

        reconstruct(local_cfg, output_dir, limit_predict_batches=1, mask_path=mask_path)
        reconstructions_path = os.path.join(output_dir, "reconstructions-0000.tar.gz")
        assert os.path.exists(reconstructions_path)

        dataset = wds.WebDataset(reconstructions_path, shardshuffle=False).decode(torch_decode)
        for _i, sample in enumerate(dataset):
            orig = sample["image.npy"]
            recon = sample["recon_image.npy"]
            assert orig.shape == recon.shape

            # Convert tensors to "false color" images and save to a PNG file
            [orig_image, recon_image] = [
                example_to_image(
                    cfg,
                    {"image": x},
                    with_simulations=False,
                    render_channels=False,
                    select_channels=[0, 1, 5],  # ALIGNED, PAIRED, ALLELE
                )
                for x in (orig, recon)
            ]
            mask = Image.open(result_path("mask.png"))
            pixel_array = np.array(mask)

            for i in range (len(pixel_array)):
                for j in range (len(pixel_array[0])):
                    bool_index = (i // cfg.data.patch_size) * (len(pixel_array[0]) // cfg.data.patch_size) + (j // cfg.data.patch_size)
                    # case where pixel should be masked
                    if (mask.getpixel((j, i)) == (255, 255, 255)):
                        pass
                    else:
                        mask.putpixel((j, i), orig_image.getpixel((j, i)))


            png_path = result_path("test_recon3.png")
            combined = Image.new(orig_image.mode, (orig_image.width + recon_image.width + mask.width + 20, orig_image.height))
            combined.paste(orig_image, (0, 0))
            combined.paste(mask, (orig_image.width + 10, 0))
            combined.paste(recon_image, (orig_image.width + 10 + mask.width + 10, 0))
            combined.save(png_path)
            # break
        assert _i == 0, "Only one sample in dataset"


@pytest.mark.skip()
@pytest.mark.cfg_overrides(
    f"reference={B37_REF_FASTA}",
    "simulation.replicates=0",
    "pileup=unphased_variant",
    "model=MiM",
    "data.patch_size=8",
    '+model.checkpoint="/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze4.sv.alt.passing.training.hg38.models/data._target_=npsv3.models.transformer.RealImageDataModule,data.batch_size=256,data=real_image,model=MiM,pileup=unphased_variant,trainer.max_epochs=5/epoch=4-step=58770.ckpt"',
    "data=real_image",
    "data._target_=npsv3.models.transformer.RealImageDataModule",
    "data.batch_size=1",
)
class TestMasking:
    def test_masking(self, cfg):
        data_path = "/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze4.sv.alt.passing.training.hg38.images/HG00096/generator=coverage,pileup=unphased_variant,simulation.replicates=1/images-0000.tar"
        dataset = wds.WebDataset(data_path, shardshuffle=False).decode(torch_decode)

        random_num = random.randint(0, 999)

        for _i, sample in enumerate(dataset):
            
            if (_i == random_num):

                orig = sample["image.npy.gz"]

                num_patches = (orig.shape[0] // cfg.data.patch_size) * (orig.shape[1] // cfg.data.patch_size)
                # bool_masked_pos = torch.randint(low=0, high=2, size=(num_patches, )).bool()
                vals = [True, False]
                top_weights = [3, 1]
                bottom_weights = [1, 2]
                top_masked_pos = random.choices(vals, weights=top_weights, k=num_patches//2)
                bottom_masked_pos = random.choices(vals, weights=bottom_weights, k=num_patches//2)
                bool_masked_pos = torch.tensor(top_masked_pos+bottom_masked_pos).bool()
                print(bool_masked_pos)

                # Convert tensors to "false color" images and save to a PNG file
                orig_image = (
                    example_to_image(
                        cfg,
                        {"image": orig},
                        with_simulations=False,
                        render_channels=False,
                        select_channels=[0, 1, 5],  # ALIGNED, PAIRED, ALLELE
                    )
                )

                pixel_array = np.array(orig_image)
                masked_image = Image.new(orig_image.mode, (orig_image.width, orig_image.height))

                # Iterate through each pixel and make a copy image with the masking
                for i in range (len(pixel_array)):
                    for j in range (len(pixel_array[0])):

                        bool_index = (i // cfg.data.patch_size) * (len(pixel_array[0]) // cfg.data.patch_size) + (j // cfg.data.patch_size)

                        # case where pixel should be masked
                        if (bool_masked_pos[bool_index] == True):
                            masked_image.putpixel((j, i), (255, 255, 255))
                        else:
                            masked_image.putpixel((j, i), orig_image.getpixel((j, i)))
                            

                png_path = result_path("masking_test.png")
                # combined = Image.new(orig_image.mode, (orig_image.width, orig_image.height))
                combined = Image.new(orig_image.mode, (orig_image.width +  masked_image.width + 10, orig_image.height))
                combined.paste(orig_image, (0, 0))
                combined.paste(masked_image, (orig_image.width + 10, 0))
                combined.save(png_path)
                break
            
            else:
                continue
            # assert _i == 0, "Only one sample in dataset"
