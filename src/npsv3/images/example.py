import os
import pathlib
import shutil
import sys
import tempfile
import typing

import hydra
import numpy as np
import pysam
import ray
from PIL import Image
from streaming import MDSWriter
from streaming.base.util import merge_index
from tqdm import tqdm

from npsv3.images.generator import ImageGenerator
from npsv3.util.range import Range
from npsv3.util.reads import downsample_reads, haplotag_reads
from npsv3.util.sample import Sample
from npsv3.variant import Variant

MDS_COMPRESSION = "zstd"
MDS_HASHES = "sha1", "xxh64"


def _reference_sequence(reference_fasta: str, region: Range) -> str:
    with pysam.FastaFile(reference_fasta) as ref_fasta:
        # Make sure reference sequence is all upper case
        return ref_fasta.fetch(reference=region.contig, start=region.start, end=region.end).upper()


def example_to_image(
    cfg, example, out_path: str, with_simulations=False, margin=10, max_replicates=1, render_channels=False
):
    generator = hydra.utils.instantiate(cfg.generator, cfg, _recursive_=False)

    image_tensor = example["image"]
    real_image = generator.render(image_tensor, render_channels=render_channels)

    _, replicates, *_ = example["sim/images"].shape if with_simulations and "sim/images" in example else (3, 0)
    if replicates > 0:
        width, height = real_image.size
        replicates = min(replicates, max_replicates)

        image = Image.new(real_image.mode, (width + 2 * (width + margin), height + replicates * (height + margin)))
        image.paste(real_image, (width + margin, 0))

        synth_tensor = example["sim/images"]
        for ac in range(3):
            for repl in range(replicates):
                synth_image_tensor = synth_tensor[ac, repl]
                synth_image = generator.render(synth_image_tensor, render_channels=render_channels)

                coord = (ac * (width + margin), (repl + 1) * (height + margin))
                image.paste(synth_image, coord)
    else:
        image = real_image

    image.save(out_path)


def make_example_from_region(
    cfg,
    region: Range,
    read_path: str,
    sample: Sample,
    background_vcf: typing.Optional[str] = None,
    generator: ImageGenerator = None,
    addl_features: typing.Optional[dict] = None,
):
    # Create generator if not provided
    generator = generator or hydra.utils.instantiate(cfg.generator, cfg=cfg, _recursive_=False)

    # Construct image for "real" data
    with tempfile.TemporaryDirectory() as tempdir:
        local_read_path = read_path
        if cfg.pileup.downsample < 1.0:
            # Downsample reads if specified
            local_read_path = downsample_reads(
                local_read_path,
                region.expand(cfg.pileup.fetch_flank),
                tempdir,
                downsample=cfg.pileup.downsample,
            )

        if cfg.pileup.haplotag_reads and background_vcf is not None:
            # Haplotag reads on the fly using whatshap
            local_read_path = haplotag_reads(
                cfg.reference, sample, local_read_path, background_vcf, region.expand(cfg.pileup.fetch_flank), tempdir
            )

        image_tensor = generator.generate(
            local_read_path,
            sample,
            region,
            ref_seq=_reference_sequence(cfg.reference, region),
        )

    example = {"region": str(region), "image": image_tensor}

    return example


columns = {
    "region": "str",
    "image": "ndarray",
}


@ray.remote
class ExhaustiveExampleActor:
    def __init__(self, index: int, output_dir: str, cfg, *args):
        self.output_path = os.path.join(output_dir, f"{index:03}")
        self.cfg = cfg
        self.args = args

        # MDSWriter requires the output directory to not exist
        shutil.rmtree(self.output_path, ignore_errors=True)
        self._writer = MDSWriter(out=self.output_path, columns=columns, compression=MDS_COMPRESSION, hashes=MDS_HASHES)

    def from_region(self, region: Range):
        try:
            example = make_example_from_region(self.cfg, region, *self.args)
            self._writer.write(example)
        except:
            # Mimic the behavior of the with statement
            if not self._writer.__exit__(*sys.exc_info()):
                raise

    def cleanup(self):
        self._writer.__exit__(None, None, None)


def vcf_to_region_examples(
    cfg,
    read_path: str,
    sample: Sample,
    inference_vcf: str,
    output_dir: str,
    background_vcf: typing.Optional[str] = None,
    progress_bar: bool = False,
):

    regions = []
    with pysam.VariantFile(inference_vcf) as vcf_file:
        for record in vcf_file:
            variant = Variant.from_pysam(record)
            regions.append(variant.reference_region.expand(cfg.pileup.variant_padding))

    os.makedirs(output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as ray_dir:
        # We currently just use ray for the CPU-side work, specifically simulating the SVs. We use a private temporary directory
        # to avoid conflicts between clusters running on the same node.
        ray.init(num_cpus=cfg.threads, num_gpus=0, _temp_dir=ray_dir, ignore_reinit_error=True, include_dashboard=False)

        actors = [
            ExhaustiveExampleActor.remote(i, output_dir, cfg, read_path, sample, background_vcf)
            for i in range(cfg.threads)
        ]
        pool = ray.util.ActorPool(actors)

        gen = pool.map_unordered(lambda actor, region: actor.from_region.remote(region), regions)
        for _ in tqdm(gen, total=len(regions), disable=not progress_bar):
            pass

        # Make sure all the writers are cleaned up
        ray.wait([actor.cleanup.remote() for actor in actors], num_returns=len(actors))

    # Combine the streaming MDS files into a single dataset
    pathlib.Path(output_dir, "index.json").unlink(missing_ok=True)
    merge_index(output_dir, keep_local=True)
