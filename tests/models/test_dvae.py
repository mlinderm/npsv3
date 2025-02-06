import pytest
import hydra
import io
import os
import torch
from omegaconf import OmegaConf
import lightning as L
import webdataset as wds
from PIL import Image

from npsv3.models.dvae import RealImageDataModule, EncodingToWebDatasetCallback, reconstruct
from npsv3.models.runners import train
from npsv3.simulation import bwa_index_loaded
from npsv3.images.example import (
    example_to_image,
    vcf_to_variant_examples,
)

from .. import B37_REF_FASTA, data_path, result_path, RESULT_DIR


def torch_decode(key, data):
    # Use custom decoder to eliminate a warning with torch.load about weights_only
    if key.endswith((".pth", ".pt")):
        stream = io.BytesIO(data)
        return torch.load(stream, weights_only=True, map_location='cpu')


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
            assert images.shape == (1, len(cfg.pileup.image_channels), cfg.pileup.image_height, cfg.pileup.image_width)

        dm.teardown(stage="fit")


class TestDVAEModel:
    @pytest.mark.cfg_overrides(
        "pileup=unphased_variant",
        "model=vqvae",
        "data=real_image",
        f"data.train_urls={'::'.join([data_path('unphased_variant_images-0000.tar')]*2)}",
        "data.batch_size=2",
        "trainer=dvae",
    )
    def test_vqvae_model(self, cfg):
        # Need to stack up the data and reduce the batch size so we get at least one batch
        train(cfg, fast_dev_run=True)

    @pytest.mark.cfg_overrides(
        "pileup=unphased_variant",
        "model=vqvae",
        "data=real_image",
        f"data.predict_urls={data_path('unphased_variant_images-0000.tar')}",
        "data.batch_size=1",
    )
    def test_vqvae_predict_to_file(self, tmp_path, cfg):
        # Create model without decoder, mimic-ing encoding-only mode
        model = hydra.utils.instantiate(cfg.model, decoder=None)
        dm = hydra.utils.instantiate(cfg.data)

        writer = EncodingToWebDatasetCallback(output_dir=tmp_path)
        trainer = L.Trainer(callbacks=[writer])

        trainer.predict(model=model, datamodule=dm, return_predictions=False)

        expected_path = str(tmp_path / "encodings-0000.tar.gz")
        assert os.path.exists(expected_path)

        dataset = wds.WebDataset(expected_path, shardshuffle=False).decode(torch_decode)
        for _i, sample in enumerate(dataset):
            assert sample["image.encoded.pth"].shape == (6, 18)  # 96x288 with 16x16 patches
        assert _i == 0, "Only one sample in dataset"


@pytest.mark.skipif(
    (
        not os.path.exists(B37_REF_FASTA)
        or not bwa_index_loaded(B37_REF_FASTA)
        or not os.path.exists(
            "/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze4.sv.alt.passing.training.hg38.models/data.batch_size=256,data=real_image,model.quantizer.num_embeddings=4096,model=vqvae,pileup=unphased_variant,trainer.max_epochs=5/epoch=4-step=58770.ckpt"
        )
    ),
    reason="B37 reference in SHM required",
)
@pytest.mark.cfg_overrides(
    f"reference={B37_REF_FASTA}",
    "simulation.replicates=0",
    "pileup=unphased_variant",
    "model=vqvae",
    "model.quantizer.num_embeddings=4096",
    '+model.checkpoint="/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze4.sv.alt.passing.training.hg38.models/data.batch_size=256,data=real_image,model.quantizer.num_embeddings=4096,model=vqvae,pileup=unphased_variant,trainer.max_epochs=5/epoch=4-step=58770.ckpt"',
    "data=real_image",
    "data.batch_size=1",
)
@pytest.mark.usefixtures("ray_setup")
class TestDVAEReconstruction:
    def test_vqae_reconstruction(self, tmp_path, cfg, hg002_sample):
        # Generate image for variant
        output_dir = str(tmp_path / "shards")
        vcf_to_variant_examples(
            cfg,
            data_path("12_22127565_22132387.bam"),
            hg002_sample,
            data_path("12_22129565_22130387.vcf.gz"),
            output_dir,
            background_vcf=data_path("12_22129565_22130387.background.vcf.gz"),
        )
        images_path = os.path.join(output_dir, "images-0000.tar")
        assert os.path.exists(images_path)

        # Write reconstructed images to a WebDataset file
        local_conf = OmegaConf.from_dotlist([
            f"data.predict_urls={os.path.join(output_dir, 'images-0000.tar')}",
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
