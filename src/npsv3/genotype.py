import logging
import os
import tempfile
import weakref
from collections.abc import Sequence
from typing import Optional

import lightning as L
import pysam
import ray
import torch
from pysam import bcftools
from torchvision.transforms import v2 as transforms

from npsv3.graphs.graph_constructor import variant_path_to_allele, variant_path_to_id
from npsv3.images.example import make_graph_example_from_region
from npsv3.models.loaders import _pack_and_pad_images, to_tensor
from npsv3.models.runners import load_model_from_checkpoint
from npsv3.util.config import setup_resolvers
from npsv3.util.range import Range
from npsv3.util.sample import Sample
from npsv3.util.vcf import bcftools_format, bcftools_index, index_variant_file
from npsv3.variant import Variant, overlapping_records, vg_variant_id

PLOIDY = 2
VCF_HEADER_TYPES_TO_COPY = frozenset(["GENERIC", "STRUCTURED", "INFO", "FILTER", "CONTIG"])


@ray.remote
class VariantExamples:
    """Ray actor for generating packed image batches for each variant individually in a VCF file"""

    def __init__(
        self,
        cfg,
        *,
        read_path: str,
        sample: Sample,
        inference_vcf: str,
        background_vcf: Optional[str] = None,
        mean=(0.5,),
        std=(0.5,),
        batch_size=64,
    ):
        self.cfg = cfg
        self.read_path = read_path
        self.sample = sample
        self.src_vcf_file = pysam.VariantFile(inference_vcf)
        self._src_vcf_file_finalizer = weakref.finalize(self, self.src_vcf_file.close)
        self._src_vcf_file_header = self.src_vcf_file.header
        self.background_vcf = background_vcf

        self.transforms = transforms.Compose(
            [
                transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.batch_size = batch_size

    def _example_generator(self, region: Range):
        for record in self.src_vcf_file.fetch(**region.pysam_fetch):
            variant = Variant.from_pysam(record)
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as dst_dir:
                dst_vcf = os.path.join(dst_dir, "variant.vcf.gz")
                with pysam.VariantFile(dst_vcf, mode="wz", header=self._src_vcf_file_header) as dst_vcf_file:
                    dst_vcf_file.write(record)
                index_variant_file(dst_vcf)

                example = make_graph_example_from_region(
                    self.cfg,
                    variant.reference_region,
                    self.read_path,
                    self.sample,
                    background_vcf=self.background_vcf,
                    inference_vcf=dst_vcf,
                )
                query_image = to_tensor(example["image"])
                support_images = to_tensor(example["sim.images"])
                label = torch.from_numpy(example["label.rank"])
                yield (query_image, support_images, label, example["region"], example["label.alleles"])

    def from_region(self, region: Range):
        # Perform the packing and padding in the actor to maximize the work done in parallel
        yield from _pack_and_pad_images(
            self._example_generator(region), batch_size=self.batch_size, image_transform=self.transforms, pad=False
        )


class OnlinePackedImageDataModule(L.LightningDataModule):
    """Lightning DataModule for performing prediction on packed images generated on-the-fly using Ray actors"""

    def __init__(self, cfg, *, inference_vcf: str, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.inference_vcf = inference_vcf
        self.kwargs = kwargs

    def setup(self, **_kwargs):
        # We use Ray actors to parallelize example generation (instead of multiple DataLoader workers)
        self._ray_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        ray.init(
            num_cpus=self.cfg.threads,
            num_gpus=0,
            _temp_dir=self._ray_dir.name,
            ignore_reinit_error=True,
            include_dashboard=False,
            runtime_env=ray.runtime_env.RuntimeEnv(worker_process_setup_hook=setup_resolvers),
        )

        group_padding = self.cfg.pileup.variant_padding // 2
        self.regions = [region for region, *_ in overlapping_records(self.inference_vcf, flank=group_padding)]
        logging.info("Genotyping variants in %d regions", len(self.regions))

    def predict_dataloader(self):
        actors = [
            VariantExamples.remote(self.cfg, inference_vcf=self.inference_vcf, **self.kwargs)
            for i in range(self.cfg.threads)
        ]
        pool = ray.util.ActorPool(actors)
        for region in self.regions:
            pool.submit(lambda actor, region: actor.from_region.remote(region), region)

        # Yield batches from any actor as they become available (each actor produces a generator with possibly multiple batches)
        # https://docs.ray.io/en/latest/ray-core/ray-generator.html#how-to-wait-for-generator-without-blocking-a-thread-compatibility-to-ray-wait-and-ray-get
        ready, unready = [], [pool.get_next_unordered()]
        while unready:
            # Yield any ready examples
            ready, unready = ray.wait(unready)
            for ready_gen in ready:
                try:
                    yield ray.get(next(ready_gen))
                except StopIteration:
                    pass
                else:
                    unready.append(ready_gen)
            # Check for additional generators available from the pool. Don't wait (timeout=0) if we already have unready generators to avoid blocking.
            # Otherwise block until a new generator is available (timeout=None).
            try:
                while pool.has_next():
                    unready.append(pool.get_next_unordered(timeout=0 if unready else None))
            except TimeoutError:
                pass

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # We currently only need to transfer the images to the device, e.g., GPU
        query, support, *addl_fields = batch
        return (query.to(device), support.to(device), *addl_fields)

    def teardown(self, **_kwargs):
        ray.shutdown()
        self._ray_dir.cleanup()


def _dict_of_alleles(genotype: Sequence[Sequence[str]]) -> dict[str, tuple[Optional[str], ...]]:
    """Create a dictionary mapping variant IDs to allele indices in the predicted genotype

    For example, `(("alt_abc_0", "alt_def_1"), ("alt_abc_1", "alt_def_1"))` returns
    `{ "abc": (0, 1), "def": (1,1) }`
    """
    dict_of_alleles = {}
    for g, alleles in enumerate(genotype):
        for a in alleles:
            var_id = variant_path_to_id(a)
            curr = dict_of_alleles.get(var_id, ())
            dict_of_alleles[var_id] = (*curr, *((None,) * (g - len(curr))), variant_path_to_allele(a))
    return dict_of_alleles


class VCFWriterCallback(L.Callback):
    """Lightning callback for writing genotyping results to a VCF file during prediction"""

    def __init__(self, sample: Sample, inference_vcf: str, output_path: str):
        self.sample = sample
        self.inference_vcf = inference_vcf
        self.output_path = output_path

    def setup(self, trainer, pl_module, stage):  # noqa: ARG002
        # Initialize output VCF file from existing VCF
        self.src_vcf_file = pysam.VariantFile(self.inference_vcf, drop_samples=True)

        # Create header for destination file, copying existing header fields, samples, and
        # adding NPSV-specific header lines TODO: Add metadata about the model, etc.
        self._dst_header = pysam.VariantHeader()
        src_header = self.src_vcf_file.header
        for record in src_header.records:
            if record.type in VCF_HEADER_TYPES_TO_COPY:
                self._dst_header.add_record(record)
        self._dst_header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        self._dst_header.add_line(
            '##FORMAT=<ID=MT,Number=.,Type=Float,Description="Metric between real and simulated data">'
        )
        self._dst_header.add_sample(self.sample.name)

        self.dst_vcf_file = pysam.VariantFile(self.output_path, mode="wz", header=self._dst_header)

    def teardown(self, trainer, pl_module, stage):  # noqa: ARG002
        self.dst_vcf_file.close()
        self.src_vcf_file.close()

    def on_predict_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):  # noqa: ARG002
        for metric, pred, _label, region, alleles in zip(*outputs, strict=True):
            # Fetch the corresponding variants in the region to create destination records
            records = list(self.src_vcf_file.fetch(region=region))
            assert len(records) > 0, f"Didn't find any records in region {region}"

            # Link the predictions to the relevant records/alleles based on the VG allele paths in the predicted haplotypes
            var_to_gts = _dict_of_alleles(alleles[pred.item()])

            sample_metric = metric.round(decimals=4).tolist()  # Since records were grouped, all share metric
            for record in records:
                # In "variant" mode we can get other records in the region that were not included in this genotyping
                # run, so we skip records without genotypes.
                var_id = vg_variant_id(record)
                if var_id not in var_to_gts:
                    continue
                dst_samples = [{"GT": var_to_gts[var_id], "MT": sample_metric}]
                # Create and write new record with genotypes
                dst_record = self._dst_header.new_record(
                    contig=record.contig,
                    start=record.start,
                    stop=record.stop,
                    alleles=record.alleles,
                    id=record.id,
                    qual=record.qual,
                    filter=record.filter,
                    info=record.info,
                    samples=dst_samples,
                )
                self.dst_vcf_file.write(dst_record)


def genotype(
    cfg, read_path: str, sample: Sample, inference_vcf: str, output_path: str, *, background_vcf: Optional[str] = None
):
    assert cfg.simulation.replicates >= 1, "At least one replicate is required for genotyping"
    torch.set_num_threads(min(4, cfg.threads))

    # Create the datamodule and genotyper model
    datamodule = OnlinePackedImageDataModule(
        cfg,
        read_path=read_path,
        sample=sample,
        inference_vcf=inference_vcf,
        background_vcf=background_vcf,
        batch_size=cfg.data.batch_size,
    )
    model = load_model_from_checkpoint(cfg, strict=True)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as output_dir:
        unsorted_output_path = os.path.join(output_dir, "genotypes.vcf.gz")
        trainer_args = {
            "callbacks": [
                L.pytorch.callbacks.TQDMProgressBar(refresh_rate=50),  # Reduce refresh rate to mitigate performance issues
                VCFWriterCallback(sample=sample, inference_vcf=inference_vcf, output_path=unsorted_output_path),
            ],
            "logger": False, # Disable logging (we don't need it for prediction)
        }

        trainer = L.Trainer(**trainer_args)
        trainer.predict(model=model, datamodule=datamodule)

        # Sort output file and index (if relevant)
        bcftools.sort(
            *bcftools_index(output_path),
            "-O",
            bcftools_format(output_path),
            "-o",
            output_path,
            "-T",
            output_dir,
            unsorted_output_path,
            catch_stdout=False,
        )
