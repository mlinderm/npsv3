import os
import io

import torch
import pytest
import lightning as L

from npsv3.models.transformer import RealImageDataModule, Transformer
from .. import B37_REF_FASTA, data_path, result_path
from PIL import Image
import requests
import webdataset as wds
from npsv3.models.runners import train
from npsv3.images.example import (
    vcf_to_variant_examples,
    example_to_image
)

from omegaconf import OmegaConf

from npsv3.models.dvae import EncodingToWebDatasetCallback
from npsv3.models.transformer import RealImageDataModule, reconstruct

import hydra

def torch_decode(key, data):
    # Use custom decoder to eliminate a warning with torch.load about weights_only
    if key.endswith((".pth", ".pt")):
        stream = io.BytesIO(data)
        return torch.load(stream, weights_only=True, map_location='cpu')

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
test_image = Image.open(requests.get(url, stream=True).raw)

@pytest.mark.cfg_overrides(
    "pileup=unphased_variant",
)
class TestRealImageDataLoader:
    def test_real_image_loader(self, cfg):
        dm = RealImageDataModule(
            data_path("unphased_variant_images-0000.tar"),
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

class TestTransformer:
    @pytest.mark.cfg_overrides(
        "pileup=unphased_variant",
        "model=MiM",
        "data=real_image",
        "data._target_=npsv3.models.transformer.RealImageDataModule",
        f"data.train_urls={'::'.join([data_path('unphased_variant_images-0000.tar')]*2)}",
        "data.batch_size=2",
        "trainer=transformer",
    )
    def test_transformer(self, cfg):
        train(cfg, fast_dev_run=True)


@pytest.mark.cfg_overrides(
    f"reference={B37_REF_FASTA}",
    "simulation.replicates=0",
    "pileup=unphased_variant",
    "model=MiM",
    '+model.checkpoint="/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze4.sv.alt.passing.training.hg38.models/data._target_=npsv3.models.transformer.RealImageDataModule,data.batch_size=256,data=real_image,model=MiM,pileup=unphased_variant,trainer.max_epochs=1/epoch=0-step=11754.ckpt"',
    "data=real_image",
    "data._target_=npsv3.models.transformer.RealImageDataModule",
    "data.batch_size=1",
)
@pytest.mark.usefixtures("ray_setup")
class TestTransformerReconstruction:
    def test_transformer_reconstruction(self, tmp_path, cfg, hg002_sample):
        # Generate image for variant
        output_dir = str(tmp_path) # / "shards")
        # vcf_to_variant_examples(
        #     cfg,
        #     data_path("12_22127565_22132387.bam"),
        #     hg002_sample,
        #     data_path("12_22129565_22130387.vcf.gz"),
        #     output_dir,
        #     background_vcf=data_path("12_22129565_22130387.background.vcf.gz"),
        # )
        # images_path = os.path.join(output_dir, "images-0000.tar")
        # assert os.path.exists(images_path)

        # Write reconstructed images to a WebDataset file
        local_conf = OmegaConf.from_dotlist([
            f"data.predict_urls=/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze4.sv.alt.passing.training.hg38.images/HG00096/generator=coverage,pileup=unphased_variant,simulation.replicates=1/images-0000.tar",
        ])
        local_cfg = OmegaConf.merge(cfg, local_conf)
        reconstruct(local_cfg, output_dir, limit_predict_batches=1)
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

            png_path = result_path("test.png")
            combined = Image.new(orig_image.mode, (orig_image.width + recon_image.width + 10, orig_image.height))
            combined.paste(orig_image, (0, 0))
            combined.paste(recon_image, (orig_image.width + 10, 0))
            combined.save(png_path)
            assert os.path.exists(png_path)
        assert _i == 0, "Only one sample in dataset"



