import itertools
import logging
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
import webdataset as wds
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

from npsv3.images.generator import ImageGenerator
from npsv3.util.range import Range
from npsv3.util.reads import downsample_reads, haplotag_reads
from npsv3.util.sample import Sample
from npsv3.util.timeout import Timeout
from npsv3.variant import Variant, overlapping_records
from npsv3.graph import Graph
from npsv3.graph import Graph
from npsv3.pileup import AlleleAssignment, BaseAlignment, FragmentTracker, ReadPileup, Strand
from npsv3.realigner import AlleleRealignment, FragmentRealigner, realign_fragment
from npsv3.simulation import augment_sample, simulate_variant_sequencing


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

    genotypes, replicates, *_ = example["sim/images"].shape if with_simulations and "sim/images" in example else (3, 0)
    if replicates > 0:
        width, height = real_image.size
        replicates = min(replicates, max_replicates)

        image = Image.new(real_image.mode, (width + (genotypes - 1) * (width + margin), height + replicates * (height + margin)))
        image.paste(real_image, (width + margin, 0))

        synth_tensor = example["sim/images"]
        for gt in range(genotypes):
            for repl in range(replicates):
                synth_image_tensor = synth_tensor[gt, repl]
                synth_image = generator.render(synth_image_tensor, render_channels=render_channels)

                coord = (gt * (width + margin), (repl + 1) * (height + margin))
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
    **kwargs,
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

def make_graph_example_from_region(
    cfg,
    region: Range,
    read_path: str,
    sample: Sample,
    background_vcf: str,
    inference_vcf: str,
    ploidy: int = 2,
    generator: ImageGenerator = None,
    addl_features: typing.Optional[dict] = None,
    **kwargs,
):
    # Create generator if not provided
    generator = generator or hydra.utils.instantiate(cfg.generator, cfg=cfg, _recursive_=False)

    # Generate graph from the VCF(s)
    graph = Graph.from_vcf(
        cfg.reference, background_vcf, region.expand(cfg.pileup.graph_flank), inference_vcf=inference_vcf
    )

    # Set up image flanks to minimize compression
    example_region = generator.image_region(region)

    with tempfile.TemporaryDirectory() as tempdir:
        # Generate haplotypes for re-alignment, i.e., with reference as the background (as opposed to a specific haplotype)
        assert graph.is_bubble_path(region.contig), "Graph over region must form bubble for reference background"
        realign_haplotypes = graph.generate_possible_haplotypes(inference_vcf, region.contig, example_region)
        assert len(realign_haplotypes) >= 1
        assert realign_haplotypes[0].nodes == graph.nodes_on_path(
            region.contig
        ), "First haplotype must be the reference"

        realign_fasta_path = os.path.join(tempdir, "realign.fasta")
        with open(realign_fasta_path, "w") as fasta:
            for i, haplotype in enumerate(realign_haplotypes):
                fasta.write(f">seq{i}\n")
                fasta.write(haplotype.sequence() + "\n")

        # Construct realigner once for all images for this variant,
        # TODO: Incorporate IUPAC FASTA for scoring?
        addl_args = {"num_alts": len(realign_haplotypes) - 1}  # This is needed to prevent C++ errors
        realigner = FragmentRealigner(realign_fasta_path, sample.mean_insert_size, sample.std_insert_size, **addl_args)

        # Extract the reference sequence from the first haplotype
        ref_seq = realign_haplotypes[0].sequence()[
            example_region.start - graph.region.start : example_region.end - graph.region.end
        ]
        assert len(ref_seq) == example_region.length, "Reference sequence length does not match the region size"

    # Construct image for "real" data
    with tempfile.TemporaryDirectory() as tempdir:
        local_read_path = read_path
        if cfg.pileup.downsample < 1.0:
            # Downsample reads if specified
            local_read_path = _downsample_reads(
                local_read_path,
                example_region.expand(cfg.pileup.fetch_flank),
                tempdir,
                downsample=cfg.pileup.downsample,
            )

        if cfg.pileup.haplotag_reads:
            # Haplotag reads on the fly using whatshap
            local_read_path = haplotag_reads(
                cfg.reference, sample, local_read_path, background_vcf, graph.region, tempdir
            )

        image_tensor = generator.generate(
            local_read_path,
            sample,
            example_region,
            realigner=realigner,
            ref_seq=ref_seq,
            compress=True,
        )

    example = {"region": str(region), "image": image_tensor}

    # If we are augmenting the simulated data, use the provided statistics for the first example, so it
    # will hopefully be most similar to the real data and then augment the remaining replicates
    if cfg.simulation.augment:
        repl_samples = augment_sample(sample, cfg.simulation.replicates, keep_original=True)
    else:
        repl_samples = [sample] * cfg.simulation.replicates

    # Generate the possible haplotypes for this region on the possible backgrounds
    # TODO: Check if the backgrounds are identical, if so, we can generate the haplotypes once
    backgrounds = [
        graph.generate_possible_haplotypes(inference_vcf, f"{sample.name}#{i}#{region.contig}", example_region)
        for i in range(ploidy)
    ]

    # Generate the relevant sequences once, which are then combined to create the fasta for simulation.
    # TODO: Do we need pad out the shorter sequences?
    sequences = [[haplotype.sequence() for haplotype in background] for background in backgrounds]

    # For fully labeled data, one of the haplotypes should be the true haplotype
    labels = []
    for allele, haplotypes in enumerate(backgrounds):
        base_path_nodes = graph.shortest_path(f"{sample.name}#{allele}#{region.contig}")
        for allele_index, haplotype in enumerate(haplotypes):
            if haplotype.nodes == base_path_nodes:
                labels.append(allele_index)
                break
        else:
            raise ValueError("True haplotype not found in possible haplotypes")
    if len(labels) == ploidy:
        example["label"] = np.ravel_multi_index(tuple([i] for i in labels), tuple(len(b) for b in backgrounds)).item()

    # TODO: Do we want to only have one of 0/1, 1/0?
    alleles_encoded_images = []
    for allele_indices in itertools.product(*(range(len(haplotypes)) for haplotypes in backgrounds)):
        # Get the sequences for this haplotype combination
        gt_sequences = [sequences[i][allele_index] for i, allele_index in enumerate(allele_indices)]
        with tempfile.TemporaryDirectory() as tempdir:
            # Write the fasta file for this haplotype combination
            fasta_path = os.path.join(tempdir, "haplotypes.fasta")
            with open(fasta_path, "w") as fasta:
                for i, sequence in enumerate(gt_sequences):
                    fasta.write(f">{allele_indices[i]}#{i}#{graph.region.contig}#0\n")
                    fasta.write(sequence + "\n")

            repl_encoded_images = []
            for repl, repl_sample in enumerate(repl_samples):
                # Simulate reads from this haplotype combination
                try:
                    sample_coverage = (
                        repl_sample.chrom_mean_coverage(graph.region.contig)
                        if cfg.simulation.chrom_norm_covg
                        else repl_sample.mean_coverage
                    )
                    replicate_bam_path = simulate_variant_sequencing(
                        fasta_path,
                        (sample_coverage * cfg.pileup.downsample) / ploidy,
                        repl_sample,
                        reference=cfg.reference,
                        shared_reference=cfg.shared_reference,
                        dir=tempdir,
                        stats_path=cfg.stats_path if cfg.simulation.gc_norm_covg else None,
                        region=example_region.expand(cfg.pileup.realigner_flank),
                        phase_vcf_path=background_vcf if cfg.pileup.haplotag_sim else None,
                        aligner=cfg.pileup.aligner,
                    )
                except ValueError:
                    logging.error(
                        "Failed to synthesize data for alleles (%d,%d) in graph region %s",
                        *allele_indices,
                        str(graph.region),
                    )
                    raise

                synth_image_tensor = generator.generate(
                    replicate_bam_path,
                    repl_sample,
                    example_region,
                    realigner=realigner,
                    ref_seq=ref_seq,
                    compress=True,
                )
                repl_encoded_images.append(synth_image_tensor)

                # TODO: Get "graph name" into output file
                if not OmegaConf.is_missing(cfg.simulation, "save_sim_bam_dir"):
                    sim_bam_path = os.path.join(cfg.simulation.save_sim_bam_dir, f"{'_'.join(allele_indices)}_{repl}.bam")
                    shutil.copy(replicate_bam_path, sim_bam_path)
                    shutil.copy(f"{replicate_bam_path}.bai", f"{sim_bam_path}.bai")

            # Stack all of the image replicates into a tensor
            alleles_encoded_images.append(np.stack(repl_encoded_images))

        # Stack the replicated images for each genotype into a tensor
        sim_image_tensor = np.stack(alleles_encoded_images)
        example["sim.images"] = sim_image_tensor
        
        # Add any additional (extension) features
        if addl_features:
            example.update(addl_features)

    return example

class ExampleActor:
    def __init__(self, index: int, output_dir: str, cfg, *args, **kwargs):
        self.output_path = os.path.join(output_dir, f"images-{index:04d}.tar")
        self.cfg = cfg
        self.args = args
        self.kwargs = kwargs

        self._writer = wds.TarWriter(self.output_path)
    
    def cleanup(self):
        self._writer.close()

@ray.remote
class RegionWriter(ExampleActor):
    def from_region(self, region: Range):
        example = make_example_from_region(self.cfg, region, *self.args, **self.kwargs)
        self._writer.write({
            "__key__": region.slug,
            "image.npy.gz": example["image"],
        })
 
@ray.remote
class GraphWriter(ExampleActor): 
    def from_region(self, region: Range):
        try:
            # Attempt to timeout long running regions.
            with Timeout(self.cfg.timeout):
                example = make_graph_example_from_region(self.cfg, region, *self.args, **self.kwargs)
            sample = {
                "__key__": region.slug,
                "image.npy.gz": example["image"],
            }
            if "label" in example:
                sample["label.cls"] = example["label"]
            if "sim.images" in example:
                sample["sim.images.npy.gz"] = example["sim.images"]
            self._writer.write(sample)
        except TimeoutError:
            logging.error("Timeout error for region %s", region)

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
            RegionWriter.remote(i, output_dir, cfg, read_path, sample, background_vcf=background_vcf, inference_vcf=inference_vcf)
            for i in range(cfg.threads)
        ]
        pool = ray.util.ActorPool(actors)

        gen = pool.map_unordered(lambda actor, region: actor.from_region.remote(region), regions)
        for _ in tqdm(gen, total=len(regions), disable=not progress_bar):
            pass

        # Make sure all the writers are cleaned up
        ray.wait([actor.cleanup.remote() for actor in actors], num_returns=len(actors))


def vcf_to_graph_examples(
    cfg,
    read_path: str,
    sample: Sample,
    inference_vcf: str,
    output_dir: str,
    background_vcf: typing.Optional[str] = None,
    progress_bar: bool = False,
    ploidy: int = 2,
):
    # Identify regions in the inference with overlapping SVs (i.e., identify bubbles)
    regions, search_regions = [], []
    running_total = running_max = 0
    group_padding = cfg.pileup.variant_padding // 2
    for region, records in overlapping_records(inference_vcf, flank=group_padding):
        count = len(records)

        # TODO: Filter out Ns
        # for record in records:
        #     if vg_variant_id(record) == "cd8536300a04f95e268065d36bb58150081a8f65":
        #         print(region, record)
        #print(region.expand(-group_padding))        

        running_total += count
        running_max = max(running_max, count)

        (regions if count <= cfg.pileup.max_exhaustive_records else search_regions).append(region.expand(-group_padding))

    logging.info(
        "Identified %d regions with mean %f and max %d records",
        running_total,
        running_total / (len(regions) + len(search_regions)),
        running_max,
    )
     # We can't exhaustively generate examples for regions with too many records, so we skip any more complex regions
    logging.info(
        "Generating exhaustive images for %d regions (across %d threads)", len(regions), cfg.threads
    )

    os.makedirs(output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as ray_dir:
        # We currently just use ray for the CPU-side work, specifically simulating the SVs. We use a private temporary directory
        # to avoid conflicts between clusters running on the same node.
        ray.init(num_cpus=cfg.threads, num_gpus=0, _temp_dir=ray_dir, ignore_reinit_error=True, include_dashboard=False)

        actors = [
            GraphWriter.remote(i, output_dir, cfg, read_path, sample, background_vcf=background_vcf, inference_vcf=inference_vcf)
            for i in range(cfg.threads)
        ]
        pool = ray.util.ActorPool(actors)

        gen = pool.map_unordered(lambda actor, region: actor.from_region.remote(region), regions)
        for _ in tqdm(gen, total=len(regions), disable=not progress_bar):
            pass

        # Make sure the tfrecords files are closed
        ray.wait([actor.cleanup.remote() for actor in actors], num_returns=len(actors))