import io
import os
import random

import hydra
import lightning as L
import pytest
import torch
import webdataset as wds
from omegaconf import OmegaConf
from PIL import Image

from npsv3.images.example import example_to_image
from npsv3.models.loaders import ntuple
from npsv3.models.runners import train, predict
from npsv3.models.transformer import MaskedImageReconstructionToWebDatasetCallback

from .. import B37_REF_FASTA, EXPERIMENTS_DIR, RESULT_DIR, data_path, result_path

TEST_IMAGES = os.path.join(EXPERIMENTS_DIR, "training/freeze4.sv.alt.passing.training.hg38.images/HG00731/generator=coverage,pileup=unphased,simulation.replicates=1/images-0000.tar")

class TestMaskableVisionTransformer:
    @pytest.mark.skipif(not os.path.exists(TEST_IMAGES), reason="Skip if test images do not exist")
    @pytest.mark.cfg_overrides(
        "model=masked_vit",
        "data=masked_images",
        "data.batch_size=2",
        f'data.train_urls="{TEST_IMAGES}"',
    )
    def test_maskable_encoder(self, cfg):
        model = hydra.utils.instantiate(cfg.model.encoder)
        patch_size = ntuple(cfg.data.patch_size, 2)
        dm = hydra.utils.instantiate(cfg.data)
        dm.prepare_data()

        dm.setup(stage="fit")

        for _i, batch in enumerate(dm.train_dataloader()):
            images, masks = batch
            features = model.forward_features(images, bool_masked_pos=masks)

            b, _c, h, w = images.shape
            num_embeddings = (h // patch_size[0]) * (w // patch_size[1]) + model.num_prefix_tokens
            assert features.shape == (b, num_embeddings, cfg.model.encoder.embed_dim), "Features should have shape (batch_size, num_patches, embed_dim)"
            break

        dm.teardown(stage="fit")

    @pytest.mark.skipif(not os.path.exists(TEST_IMAGES), reason="Skip if test images do not exist")
    @pytest.mark.cfg_overrides(
        "model=masked_vit",
        "model.encoder.use_mask_token=false",
        "data=masked_images",
        "data.batch_size=2",
        f'data.train_urls="{TEST_IMAGES}"',
    )
    def test_unmasked_encoder(self, cfg):
        model = hydra.utils.instantiate(cfg.model.encoder)
        patch_size = ntuple(cfg.data.patch_size, 2)
        dm = hydra.utils.instantiate(cfg.data)
        dm.prepare_data()

        dm.setup(stage="fit")

        for _i, batch in enumerate(dm.train_dataloader()):
            images, *_ = batch
            features = model.forward_features(images)

            b, _c, h, w = images.shape
            num_embeddings = (h // patch_size[0]) * (w // patch_size[1]) + model.num_prefix_tokens
            assert features.shape == (b, num_embeddings, cfg.model.encoder.embed_dim), "Features should have shape (batch_size, num_patches, embed_dim)"
            break

        dm.teardown(stage="fit")

    @pytest.mark.skipif(not os.path.exists(TEST_IMAGES), reason="Skip if test images do not exist")
    @pytest.mark.cfg_overrides(
        "model=masked_vit",
        "data=masked_images",
        "data.batch_size=2",
        f'data.train_urls="{TEST_IMAGES}"',
    )
    def test_masked_modeling(self, cfg):
        model = hydra.utils.instantiate(cfg.model)
        dm = hydra.utils.instantiate(cfg.data)
        dm.prepare_data()

        dm.setup(stage="fit")

        for _i, batch in enumerate(dm.train_dataloader()):
            images, masks = batch
            loss, reconstructed_images = model._model_step(batch, _i)
            assert reconstructed_images.shape == images.shape, "Reconstructed images should have the same shape as input images"
            break

        dm.teardown(stage="fit")

    @pytest.mark.skipif(not os.path.exists(TEST_IMAGES), reason="Skip if test images do not exist")
    @pytest.mark.cfg_overrides(
        "model=masked_vit",
        "data=masked_images",
        "data.batch_size=2",
        f'data.train_urls="{TEST_IMAGES}"',
        "trainer=vit_pretraining",
    )
    def test_masked_modeling_training(self, cfg):
        train(cfg, fast_dev_run=True)

class TestLearningRateSchedulers:
    @pytest.mark.cfg_overrides(
        "model=masked_vit",
        "model.scheduler.lr_min=0.00001",
        "model.scheduler.t_initial=5",
        "model.scheduler.cycle_limit=2",
        "model.scheduler.warmup_t=1",
        "model.scheduler.warmup_lr_init=0.0001",
        "model.scheduler.warmup_prefix=true",
    )
    def test_cosine_annealing(self, cfg):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        scheduler = hydra.utils.instantiate(cfg.model.scheduler)(optimizer=optimizer)

        lrs = []
        for e in range(20):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step(epoch=e + 1) # Mimic the `lr_scheduler_step` method in the lightning module

        assert lrs[0] == 0.0001, "Initial learning rate should be the warmup learning rate"
        assert lrs[1] == 0.001, "Learning rate should be the initial learning rate after warmup"
        assert lrs[1+5] == 0.001, "Second cycle should have the same initial learning rate"
        assert lrs[1+5+5] == 0.00001, "Learning rate should hit minimum at the the cycle limit"

    @pytest.mark.cfg_overrides(
        "model=masked_vit",
    )
    def test_default_schedule(self, cfg):
        model = torch.nn.Linear(1, 1)
        optimizer = hydra.utils.instantiate(cfg.model.optimizer)(model.parameters())
        scheduler = hydra.utils.instantiate(cfg.model.scheduler)(optimizer=optimizer)

        lrs = []
        for e in range(10):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step(epoch=e + 1)

def _reconstruct_image(cfg: OmegaConf, output_dir: str, reconstructions_path: str, spacing: int = 10):
    """Generate composite real, mask and reconstructed PNG images from a WebDataset of reconstructions.

    Args:
        cfg (OmegaConf): Configuration object.
        output_dir (str): Directory to save output images.
        reconstructions_path (str): Path to the WebDataset of reconstructions.
        spacing (int, optional): Spacing between images in the composite. Defaults to 10.
    """
    images = []
    dataset = wds.WebDataset(reconstructions_path, shardshuffle=False).decode()
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
        mask_image = Image.fromarray(sample["mask.npy"].astype("uint8") * 255)

        png_path = os.path.join(output_dir, f"reconstruction{sample['__key__']}.png")
        combined = Image.new(orig_image.mode, (orig_image.width + mask_image.width + recon_image.width + 2*spacing, orig_image.height))
        combined.paste(orig_image, (0, 0))
        combined.paste(mask_image, (orig_image.width + spacing, 0))
        combined.paste(recon_image, (orig_image.width + mask_image.width + 2*spacing, 0))
        combined.save(png_path)
        images.append(png_path)
    return images

class TestMaskedReconstruction:
    @pytest.mark.skipif(not os.path.exists(TEST_IMAGES), reason="Skip if test images do not exist")
    @pytest.mark.cfg_overrides(
        "model=masked_vit",
        #f'+model.checkpoint="{os.path.join(EXPERIMENTS_DIR,"training/hgsvc3-hprc-2024-02-23-mc-chm13.GRCh38.vcfbub.a100k.wave.passing.training.hg38.models/data.batch_size=512,data=masked_images,model=masked_vit,trainer.max_epochs=11,trainer=vit_pretraining/epoch=0-step=32419.ckpt")}"',
        f'+model.checkpoint="{os.path.join(EXPERIMENTS_DIR,"training/hgsvc3-hprc-2024-02-23-mc-chm13.GRCh38.vcfbub.a100k.wave.passing.training.hg38.models/data.batch_size=512,data.epoch_batches=16000,data.resampled=True,data=masked_images,model=masked_vit,trainer.max_epochs=10,trainer=vit_pretraining/model.ckpt")}"',
        "data=masked_images",
        "data.batch_size=1",
        f'data.predict_urls="{TEST_IMAGES}"',
        "trainer=vit_pretraining",
    )
    def test_reconstruct_image(self, tmp_path, cfg):
        if not os.path.exists(cfg.model.checkpoint):
            pytest.skip(f"Checkpoint {cfg.model.checkpoint} does not exist, skipping test")

        # Note that the masks are generated by the data loader and so will be different each time
        recon_output_dir = str(tmp_path)
        trainer_args ={
            "callbacks": [MaskedImageReconstructionToWebDatasetCallback(recon_output_dir, patch_size=cfg.data.patch_size)],
            "limit_predict_batches": 1,
        }
        predict(cfg, **trainer_args)
        images = _reconstruct_image(cfg, RESULT_DIR, os.path.join(recon_output_dir, "reconstructions-0000.tar.gz"))
        assert len(images) == cfg.data.batch_size, "Should have one image in the output"
        for image in images:
            assert os.path.exists(image), f"Image {image} should exist"


# def torch_decode(key, data):
#     # Use custom decoder to eliminate a warning with torch.load about weights_only
#     if key.endswith((".pth", ".pt")):
#         stream = io.BytesIO(data)
#         return torch.load(stream, weights_only=True, map_location='cpu')

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# test_image = Image.open(requests.get(url, stream=True).raw)


# @pytest.mark.skip()
# class TestMiM:
#     @pytest.mark.cfg_overrides(
#         "pileup=unphased_variant",
#         "model=MiM",
#         "data=real_image",
#         "data._target_=npsv3.models.transformer.RealImageDataModule",
#         f"data.train_urls={'::'.join([data_path('unphased_variant_images-0000.tar')]*2)}",
#         "data.batch_size=2",
#         "data.patch_size=16",
#         "trainer=transformer",
#     )
#     def test_MiM(self, cfg):
#         train(cfg, fast_dev_run=True)

# @pytest.mark.skip()
# class TestClassifier:
#     @pytest.mark.cfg_overrides(
#         "pileup=unphased_variant",
#         "model=classifier",
#         "data.patch_size=16",
#         "data=real_image",
#         "data._target_=npsv3.models.transformer.RealImageDataModule",
#         f"data.train_urls={'::'.join([data_path('unphased_variant_images-0000.tar')]*2)}",
#         "data.batch_size=2",
#         "trainer=transformer",
#         # "pretrained=classifier"
#     )
#     def test_classifier(self, cfg):
#         train(cfg, fast_dev_run=True)

# @pytest.mark.skip()
# class TestAccuracy:
#     @pytest.mark.cfg_overrides(
#         "pileup=unphased_variant",
#         "model=classifier",
#         "data.patch_size=16",
#         "data=real_image",
#         "data._target_=npsv3.models.transformer.RealImageDataModule",
#         "data.predict_urls='/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze4.sv.alt.passing.training.hg38.images/NA19983/generator=coverage,pileup=unphased_variant,simulation.replicates=1/images-0000.tar'",
#         "data.batch_size=1",
#         '+model.checkpoint="/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze4.sv.alt.passing.training.hg38.models/10Epoch_50R_16_AdamW_BEST/Node20/full_train-step=117510-train_loss=0.2303631603717804.ckpt"',
#     )
#     def test_accuracy(self, tmp_path, cfg):
#         # output_dir = str(tmp_path / "shards")
#         assess_accuracy(cfg, cfg.model.checkpoint, limit_predict_batches=100)

# def reconstruct_image(cfg, reconstructions_path, idx, iter):
#     dataset = wds.WebDataset(reconstructions_path, shardshuffle=False).decode(torch_decode)
#     for _i, sample in enumerate(dataset):
#         orig = sample["image.npy"]
#         recon = sample["recon_image.npy"]
#         assert orig.shape == recon.shape

#         # Convert tensors to "false color" images and save to a PNG file
#         [orig_image, recon_image] = [
#             example_to_image(
#                 cfg,
#                 {"image": x},
#                 with_simulations=False,
#                 render_channels=False,
#                 select_channels=[0, 1, 5],  # ALIGNED, PAIRED, ALLELE
#             )
#             for x in (orig, recon)
#         ]
#         mask = Image.open(result_path("mask.png"))
#         pixel_array = np.array(mask)

#         for i in range (len(pixel_array)):
#             for j in range (len(pixel_array[0])):
#                 bool_index = (i // cfg.data.patch_size) * (len(pixel_array[0]) // cfg.data.patch_size) + (j // cfg.data.patch_size)
#                 # case where pixel should be masked
#                 if (mask.getpixel((j, i)) == (255, 255, 255)):
#                     pass
#                 else:
#                     mask.putpixel((j, i), orig_image.getpixel((j, i)))


#         png_path = result_path("test_recon_model"+ str(idx)+"_img"+str(iter) +".png")
#         combined = Image.new(orig_image.mode, (orig_image.width + recon_image.width + mask.width + 20, orig_image.height))
#         combined.paste(orig_image, (0, 0))
#         combined.paste(mask, (orig_image.width + 10, 0))
#         combined.paste(recon_image, (orig_image.width + 10 + mask.width + 10, 0))
#         combined.save(png_path)
#         # break
#     assert _i == 0, "Only one sample in dataset"

# @pytest.mark.skip()
# @pytest.mark.cfg_overrides(
#     f"reference={B37_REF_FASTA}",
#     "simulation.replicates=0",
#     "pileup=unphased_variant",
#     "model=MiM",
#     "data.patch_size=16",
#     "data.mask_scheme=[\"random\", 50]",
#     '+model.checkpoint=null',
#     "data=real_image",
#     "data._target_=npsv3.models.transformer.RealImageDataModule",
#     "data.batch_size=1",
# )
# @pytest.mark.usefixtures("ray_setup")
# class TestTransformerReconstruction:
#     def test_transformer_reconstruction(self, tmp_path, cfg, hg002_sample):
#         # Generate image for variant
#         output_dir = str(tmp_path) # / "shards")

#         # Write reconstructed images to a WebDataset file
#         local_conf = OmegaConf.from_dotlist([
#             f"data.predict_urls=/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze4.sv.alt.passing.training.hg38.images/HG00096/generator=coverage,pileup=unphased_variant,simulation.replicates=1/images-0000.tar",
#             # f"data.predict_urls=/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze3.sv.alt.passing.training.hg38.DEL.images/HG00096/+pileup.snv_input=True,generator=single_depth_phaseread,pileup.discrete_mapq=True,pileup.render_snv=True,simulation.augment=True,simulation.chrom_norm_covg=True,simulation.replicates=5/images.tar",
#         ])
#         local_cfg = OmegaConf.merge(cfg, local_conf)
#         mask_path = result_path("mask.png")

#         model_checkpoints = [
#             "/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze4.sv.alt.passing.training.hg38.models/10Epoch_50R_16_AdamW_BEST/Node20/pretrained_MiM-step=117510-train_loss=0.048836905509233475.ckpt",
#             "/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze4.sv.alt.passing.training.hg38.models/data._target_=npsv3.models.transformer.RealImageDataModule,data.batch_size=256,data.mask_scheme=[random,50],data.patch_size=16,data=real_image,model=MiM,pileup=unphased_variant,trainer.max_epochs=20/pretrained_MiM-step=188016-train_loss=0.058955565094947815.ckpt",
#             "/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze4.sv.alt.passing.training.hg38.models/data._target_=npsv3.models.transformer.RealImageDataModule,data.batch_size=256,data.mask_scheme=[random,50],data.patch_size=16,data=real_image,model=MiM,pileup=unphased_variant,trainer.max_epochs=20/pretrained_MiM-step=235020-train_loss=0.07813941687345505.ckpt",
#             ]
#         # Generates a seed that will be used across iterations
#         torch_seed = random.randint(1, 10000000)
#         iterations = 2
#         for i in range(len(model_checkpoints)):
#             # Sets the seed such that each model checkpoint will be given the same images with the same masks
#             torch.manual_seed(torch_seed)
#             OmegaConf.update(local_cfg, "model.checkpoint", model_checkpoints[i], merge=False)
#             for j in range(iterations):
#                 reconstruct(local_cfg, output_dir, limit_predict_batches=1, mask_path=mask_path)
#                 reconstructions_path = os.path.join(output_dir, "reconstructions-0000.tar.gz")
#                 assert os.path.exists(reconstructions_path)

#                 reconstruct_image(cfg, reconstructions_path, i, j)


# # @pytest.mark.skip()
# @pytest.mark.cfg_overrides(
#     f"reference={B37_REF_FASTA}",
#     "simulation.replicates=0",
#     "pileup=unphased_variant",
#     "model=MiM",
#     "data.patch_size=8",
#     '+model.checkpoint="/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze4.sv.alt.passing.training.hg38.models/data._target_=npsv3.models.transformer.RealImageDataModule,data.batch_size=256,data=real_image,model=MiM,pileup=unphased_variant,trainer.max_epochs=5/epoch=4-step=58770.ckpt"',
#     "data=real_image",
#     "data._target_=npsv3.models.transformer.RealImageDataModule",
#     "data.batch_size=1",
# )
# class TestMasking:
#     def test_masking(self, cfg):
#         data_path = "/storage/mlinderman/projects/sv/npsv3-experiments/training/freeze4.sv.alt.passing.training.hg38.images/HG00096/generator=coverage,pileup=unphased_variant,simulation.replicates=1/images-0000.tar"
#         dataset = wds.WebDataset(data_path, shardshuffle=False).decode(torch_decode)

#         random_num = random.randint(0, 999)

#         for _i, sample in enumerate(dataset):

#             if (_i == random_num):

#                 orig = sample["image.npy.gz"]

#                 num_patches = (orig.shape[0] // cfg.data.patch_size) * (orig.shape[1] // cfg.data.patch_size)
#                 # bool_masked_pos = torch.randint(low=0, high=2, size=(num_patches, )).bool()
#                 vals = [True, False]
#                 top_weights = [3, 1]
#                 bottom_weights = [1, 2]
#                 top_masked_pos = random.choices(vals, weights=top_weights, k=num_patches//2)
#                 bottom_masked_pos = random.choices(vals, weights=bottom_weights, k=num_patches//2)
#                 bool_masked_pos = torch.tensor(top_masked_pos+bottom_masked_pos).bool()
#                 print(bool_masked_pos)

#                 # Convert tensors to "false color" images and save to a PNG file
#                 orig_image = (
#                     example_to_image(
#                         cfg,
#                         {"image": orig},
#                         with_simulations=False,
#                         render_channels=False,
#                         select_channels=[0, 1, 5],  # ALIGNED, PAIRED, ALLELE
#                     )
#                 )

#                 pixel_array = np.array(orig_image)
#                 masked_image = Image.new(orig_image.mode, (orig_image.width, orig_image.height))

#                 # Iterate through each pixel and make a copy image with the masking
#                 for i in range (len(pixel_array)):
#                     for j in range (len(pixel_array[0])):

#                         bool_index = (i // cfg.data.patch_size) * (len(pixel_array[0]) // cfg.data.patch_size) + (j // cfg.data.patch_size)

#                         # case where pixel should be masked
#                         if (bool_masked_pos[bool_index] == True):
#                             masked_image.putpixel((j, i), (255, 255, 255))
#                         else:
#                             masked_image.putpixel((j, i), orig_image.getpixel((j, i)))


#                 png_path = result_path("masking_test.png")
#                 # combined = Image.new(orig_image.mode, (orig_image.width, orig_image.height))
#                 combined = Image.new(orig_image.mode, (orig_image.width +  masked_image.width + 10, orig_image.height))
#                 combined.paste(orig_image, (0, 0))
#                 combined.paste(masked_image, (orig_image.width + 10, 0))
#                 combined.save(png_path)
#                 break

#             else:
#                 continue
#             # assert _i == 0, "Only one sample in dataset"
