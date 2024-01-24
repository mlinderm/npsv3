import itertools
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import typing
from collections import defaultdict
from shlex import quote

import hydra
import numpy as np
import pysam
import ray
import tensorflow as tf
from omegaconf import OmegaConf
from PIL import Image
from scipy.stats import fisher_exact
from tqdm import tqdm

from npsv3.graph import Graph
from npsv3.pileup import AlleleAssignment, BaseAlignment, FragmentTracker, ReadPileup, Strand
from npsv3.realigner import AlleleRealignment, FragmentRealigner, realign_fragment
from npsv3.simulation import augment_sample, simulate_variant_sequencing
from npsv3.util.range import Range
from npsv3.util.sample import Sample
from npsv3.variant import generate_allele_indices, genotype_field_index, overlapping_records, vg_variant_id

MAX_PIXEL_VALUE = 254.0  # Adapted from DeepVariant

ALIGNED_CHANNEL = 0
ALLELE_CHANNEL = 1
READ_ALLELE_CHANNEL = 2
PAIRED_CHANNEL = 3
MAPQ_CHANNEL = 4
STRAND_CHANNEL = 5
BASEQ_CHANNEL = 6
PHASE_CHANNEL = 7

MAX_NUM_CHANNELS = 8


def _fragment_zscore(sample: Sample, fragment_length: int, fragment_delta=0):
    return (fragment_length + fragment_delta - sample.mean_insert_size) / sample.std_insert_size


# def _realigner(variant, sample: Sample, reference, flank=1000, snv_vcf_path: str=None, alleles: typing.AbstractSet[int]={1}, realignment_bam_dir: str=None):
#     with tempfile.TemporaryDirectory() as dir:
#         # Generate index fasta with contigs filtered by alleles. The fasta should include the reference sequence and the
#         # sequence of the alternate alleles specified in `alleles`
#         fasta_alleles = sorted({0}.union(alleles))
#         assert len(fasta_alleles) >= 2
#         fasta_path, ref_contig, alt_contig = variant.synth_fasta(reference_fasta=reference, alleles=fasta_alleles, dir=dir, flank=flank, index_mode=True)

#         addl_args = {}
#         if snv_vcf_path:
#             iupac_fasta_path, *_ = variant.synth_fasta(reference_fasta=reference, alleles=fasta_alleles, dir=dir, flank=flank, ref_contig=ref_contig, alt_contig=alt_contig, snv_vcf_path=snv_vcf_path, index_mode=True)
#             addl_args["iupac_fasta_path"] = iupac_fasta_path

#         if realignment_bam_dir:
#             addl_args["alt_alignment_paths"] = [os.path.join(realignment_bam_dir, f"{variant.name}_{i}.bam") for i in range(len(alleles))]
#             shutil.copy(fasta_path, os.path.join(realignment_bam_dir, f"{variant.name}.fasta"))

#         return FragmentRealigner(fasta_path, sample.mean_insert_size, sample.std_insert_size, **addl_args)


def _fisher_strand(table):
    _, pvalue = fisher_exact(table)
    return -10.0 * math.log10(pvalue)


def _strand_orientation_bias(table, pseudo=1):
    table = np.array(table) + pseudo

    symmetric_ratio = (table[0, 0] * table[1, 1]) / (table[0, 1] * table[1, 0])
    symmetric_ratio += 1 / symmetric_ratio

    allele_ratio = np.log(np.min(table, axis=1) / np.max(table, axis=1))

    return math.log(symmetric_ratio) + allele_ratio[0] - allele_ratio[1]


def _fetch_reads(read_path: str, fetch_region: Range, reference: typing.Optional[str] = None) -> FragmentTracker:
    fragments = FragmentTracker()
    with pysam.AlignmentFile(read_path, reference_filename=reference) as alignment_file:
        for read in alignment_file.fetch(**fetch_region.pysam_fetch):
            if read.is_duplicate or read.is_qcfail or read.is_unmapped or read.is_secondary or read.is_supplementary:
                # TODO: Potentially recover secondary/supplementary alignments if primary is outside pileup region
                continue

            fragments.add_read(read)
    return fragments


# https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
class AnnotatedArray(np.ndarray):
    def __new__(cls, input_array, fisher_strand=None, strand_orientation_bias=None):
        obj = np.asarray(input_array).view(cls)
        # Genomic attributes
        obj.fisher_strand = fisher_strand
        obj.strand_orientation_bias = strand_orientation_bias

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.fisher_strand = getattr(obj, "fisher_strand", None)
        self.strand_orientation_bias = getattr(obj, "strand_orientation_bias", None)


class ImageGenerator:
    def __init__(self, cfg):
        self._cfg = cfg
        self._image_shape = (
            self._cfg.pileup.image_height,
            self._cfg.pileup.image_width,
            len(self._cfg.pileup.image_channels),
        )
        assert all(
            channel in range(MAX_NUM_CHANNELS) for channel in self._cfg.pileup.image_channels
        ), "Invalid channel indices specified"

        # Helper dictionaries to map to pixel values
        self._aligned_to_pixel = {
            BaseAlignment.ALIGNED: self._cfg.pileup.aligned_base_pixel,
            BaseAlignment.MATCH: self._cfg.pileup.match_base_pixel
            if self._cfg.pileup.render_snv
            else self._cfg.pileup.aligned_base_pixel,
            BaseAlignment.MISMATCH: self._cfg.pileup.mismatch_base_pixel
            if self._cfg.pileup.render_snv
            else self._cfg.pileup.aligned_base_pixel,
            BaseAlignment.SOFT_CLIP: self._cfg.pileup.soft_clip_base_pixel,
            BaseAlignment.INSERT: self._cfg.pileup.insert_base_pixel,
        }

        self._allele_to_pixel = {
            AlleleAssignment.AMB: self._cfg.pileup.amb_allele_pixel,
            AlleleAssignment.REF: self._cfg.pileup.ref_allele_pixel,
            AlleleAssignment.ALT: self._cfg.pileup.alt_allele_pixel,
            None: 0,
        }

        self._strand_to_pixel = {
            Strand.POSITIVE: self._cfg.pileup.positive_strand_pixel,
            Strand.NEGATIVE: self._cfg.pileup.negative_strand_pixel,
            None: 0.0,
        }

    def _align_pixel(self, align):
        if isinstance(align, BaseAlignment):
            return self._aligned_to_pixel[align]
        else:
            return [self._aligned_to_pixel[a] for a in align]

    def _zscore_pixel(self, zscore):
        if zscore is None:
            return 0
        else:
            return np.clip(
                self._cfg.pileup.insert_size_mean_pixel + zscore * self._cfg.pileup.insert_size_sd_pixel,
                1,
                MAX_PIXEL_VALUE,
            )

    def _allele_pixel(self, realignment: AlleleRealignment):
        if self._cfg.pileup.binary_allele:
            return self._allele_to_pixel[realignment.allele]
        elif (
            realignment is None
            or realignment.ref_quality is None
            or math.isnan(realignment.ref_quality)
            or realignment.alt_quality is None
            or math.isnan(realignment.alt_quality)
        ):
            return 0
        else:
            return np.clip(
                (realignment.alt_quality - realignment.ref_quality)
                / self._cfg.pileup.max_alleleq
                * self._cfg.pileup.allele_pixel_range
                + self._cfg.pileup.amb_allele_pixel,
                1,
                MAX_PIXEL_VALUE,
            )

    def _strand_pixel(self, read: pysam.AlignedSegment):
        return self._strand_to_pixel[Strand.NEGATIVE if read.is_reverse else Strand.POSITIVE]

    def _qual_pixel(self, qual, max_qual: int):
        if qual is None:
            return 0
        else:
            return np.minimum(np.array(qual) / max_qual, 1.0) * MAX_PIXEL_VALUE

    def _mapq_pixel(self, qual):
        if qual is None:
            return 0
        elif self._cfg.pileup.discrete_mapq:
            if qual == 0:
                return self._cfg.pileup.mapq0_pixel
            else:
                return np.minimum((np.array(qual) / self._cfg.pileup.max_mapq) * 127 + 128, MAX_PIXEL_VALUE)
        else:
            return self._qual_pixel(qual, self._cfg.pileup.max_mapq)

    def _phase_pixel(self, hp):
        if hp is None:
            return 0
        else:
            return (hp / 2.0) * MAX_PIXEL_VALUE

    def _flatten_image(self, image_tensor, render_channels=False, margin=5):
        if tf.is_tensor(image_tensor):
            image_tensor = image_tensor.numpy()

        # TODO: Better combine all the channels into a single image, perhaps ALIGNED, REF_PAIRED_CHANNEL, ALLELE (with mapq as alpha)...
        channels = [ALIGNED_CHANNEL, PAIRED_CHANNEL, ALLELE_CHANNEL]
        combined_image = Image.fromarray(image_tensor[:, :, channels], mode="RGB")

        if render_channels:
            height, width, num_channels = image_tensor.shape
            image = Image.new(combined_image.mode, (width + (num_channels - 1) * (width + margin), 2 * height + margin))
            image.paste(combined_image, ((image.width - width) // 2, 0))

            for i in range(num_channels):
                channel_image = Image.fromarray(image_tensor[:, :, [i] * len(channels)], mode=combined_image.mode)
                coord = (i * (width + margin), height + margin)
                image.paste(channel_image, coord)

            return image
        else:
            return combined_image

    @property
    def image_shape(self):
        return self._image_shape

    def image_region(self, region) -> Range:
        # Try to minimize compression by setting right padding to exact width...
        to_pad = self._cfg.pileup.image_width - region.length
        left_padding = max((to_pad + 1) // 2, self._cfg.pileup.variant_padding)
        right_padding = max(to_pad // 2, self._cfg.pileup.variant_padding)
        return region.expand(left_padding, right_padding)

    def render(self, image_tensor, **kwargs) -> Image:
        shape = image_tensor.shape
        assert len(shape) == 3
        return self._flatten_image(image_tensor, **kwargs)

    def generate(self, read_path, sample: Sample, region: Range, realigner: FragmentRealigner, **kwargs):
        image_tensor = self._generate(read_path, sample, region, realigner, **kwargs)

        # Create consistent image size
        if image_tensor.shape != self.image_shape:
            # resize converts to float directly (however convert_image_dtype assumes floats are in [0-1]) so
            # we use cast instead
            image_tensor = AnnotatedArray(
                tf.cast(
                    tf.image.resize(image_tensor[:, :, self._cfg.pileup.image_channels], self.image_shape[:2]),
                    dtype=tf.uint8,
                ).numpy(),
                fisher_strand=image_tensor.fisher_strand,
                strand_orientation_bias=image_tensor.strand_orientation_bias,
            )

        return image_tensor


class CoverageImageGenerator(ImageGenerator):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _generate(
        self,
        read_path,
        sample: Sample,
        region: Range,
        realigner: FragmentRealigner,
        ref_seq: typing.Optional[str] = None,
        **kwargs,
    ):
        image_height, _, _ = self.image_shape
        image_tensor = np.zeros((image_height, region.length, MAX_NUM_CHANNELS), dtype=np.uint8)

        fragments = _fetch_reads(read_path, region.expand(self._cfg.pileup.fetch_flank), reference=self._cfg.reference)

        # Construct the pileup from the fragments, realigning fragments to assign reads to the reference and alternate alleles
        pileup = ReadPileup(region)

        for fragment in fragments:
            # At present we render reads based on the original alignment so we only realign (and track) fragments that could overlap
            # the image window. If we render "insert" bases, then we look if any part of the fragment overlaps the region
            if fragment.fragment_overlaps(region, read_overlap_only=not self._cfg.pileup.insert_bases):
                realignment, read1_realignment, read2_realignment = realign_fragment(
                    realigner, fragment, assign_delta=self._cfg.pileup.assign_delta
                )

                insert_zscore = _fragment_zscore(sample, fragment.fragment_length)

                # Render "insert" bases for overlapping fragments without reads in the region (and thus would not
                # otherwise be represented)
                add_insert = self._cfg.pileup.insert_bases and not fragment.reads_overlap(region)

                pileup.add_fragment(
                    fragment,
                    add_insert=add_insert,
                    allele=realignment,
                    insert_zscore=insert_zscore,
                    phase_tag=self._cfg.pileup.phase_tag,
                    read1_realignment=read1_realignment,
                    read2_realignment=read2_realignment,
                )

        # Read level statistics
        allele_strand = defaultdict(int)

        # Add pileup bases to the image (downsample reads based on simple coverage-based heuristic)
        max_reads = (region.length * image_height) // sample.read_length
        row_idxs = np.full((region.length,), 0)
        for read in pileup.overlapping_reads(region, max_reads=max_reads):
            for col_slice, aligned, read_slice in pileup.read_columns(region, read, ref_seq):
                col_idxs = range(col_slice.start, col_slice.stop)
                image_tensor[row_idxs[col_slice], col_idxs, ALIGNED_CHANNEL] = self._align_pixel(aligned)
                image_tensor[row_idxs[col_slice], col_idxs, ALLELE_CHANNEL] = self._allele_pixel(read.allele)
                image_tensor[row_idxs[col_slice], col_idxs, READ_ALLELE_CHANNEL] = self._allele_pixel(read.read_allele)
                image_tensor[row_idxs[col_slice], col_idxs, PAIRED_CHANNEL] = self._zscore_pixel(read.insert_zscore)
                image_tensor[row_idxs[col_slice], col_idxs, MAPQ_CHANNEL] = self._mapq_pixel(read.mapq)
                image_tensor[row_idxs[col_slice], col_idxs, STRAND_CHANNEL] = self._strand_to_pixel[read.strand]
                image_tensor[row_idxs[col_slice], col_idxs, BASEQ_CHANNEL] = self._qual_pixel(
                    read.baseq(read_slice), self._cfg.pileup.max_baseq
                )
                image_tensor[row_idxs[col_slice], col_idxs, PHASE_CHANNEL] = self._phase_pixel(read.phase)

                # Increment the 'current' row for the bases we just added to the pileup, overwrite the last row if we exceed
                # the maximum coverage
                row_idxs[col_slice] = np.clip(row_idxs[col_slice] + 1, 0, image_height - 1)

            # Compute other read metrics
            if read.read_allele is not None:
                allele_strand[(read.read_allele.allele, read.strand)] += 1

        strand_contingency = [
            [
                allele_strand[(AlleleAssignment.REF, Strand.POSITIVE)],
                allele_strand[(AlleleAssignment.REF, Strand.NEGATIVE)],
            ],
            [
                allele_strand[(AlleleAssignment.ALT, Strand.POSITIVE)],
                allele_strand[(AlleleAssignment.ALT, Strand.NEGATIVE)],
            ],
        ]
        return AnnotatedArray(
            image_tensor,
            fisher_strand=_fisher_strand(strand_contingency),
            strand_orientation_bias=_strand_orientation_bias(strand_contingency),
        )


def _haplotag_reads(reference: str, sample: Sample, read_path: str, vcf_path: str, region: Range, dir) -> str:
    tagged_bam = tempfile.NamedTemporaryFile(delete=False, suffix=".bam", dir=dir)
    tagged_bam.close()

    whatshap_commandline = f"whatshap haplotag \
        --tag-supplementary \
        --reference {quote(reference)} \
        --regions {region} \
        --sample {sample.name} \
        --output {quote(tagged_bam.name)} \
        {quote(vcf_path)} \
        {quote(read_path)}"

    haplotag_result = subprocess.run(whatshap_commandline, shell=True, stderr=subprocess.PIPE)
    if haplotag_result.returncode != 0 or not os.path.exists(tagged_bam.name):
        print(haplotag_result.stderr)
        print(region)
        msg = "Failed to haplotag read file"
        raise RuntimeError(msg)
    pysam.index(tagged_bam.name)
    return tagged_bam.name


def _downsample_reads(read_path: str, region: Range, dir, downsample: float = 1.0) -> str:
    downsampled_bam = tempfile.NamedTemporaryFile(delete=False, suffix=".bam", dir=dir)
    downsampled_bam.close()

    samtools_commandline = (
        f"samtools view -b -o {quote(downsampled_bam.name)} -s {downsample} {quote(read_path)} {region}"
    )
    samtools_result = subprocess.run(samtools_commandline, shell=True, stderr=subprocess.PIPE)
    if samtools_result.returncode != 0 or not os.path.exists(downsampled_bam.name):
        print(samtools_result.stderr)
        msg = "Failed to downsample read file"
        raise RuntimeError(msg)
    pysam.index(downsampled_bam.name)
    return downsampled_bam.name


# Adapted from DeepVariant
def _bytes_feature(list_of_strings):
    """Returns a bytes_list from a list of string / byte."""
    if isinstance(list_of_strings, type(tf.constant(0))):
        list_of_strings = [list_of_strings.numpy()]  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_strings))


def _int_feature(list_of_ints):
    """Returns a int64_list from a list of int / bool."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def _float_feature(list_of_floats):
    """Returns a float_list from a list of int / bool."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def make_example_from_region(
    cfg,
    region: Range,
    background_vcf: str,
    inference_vcf: str,
    read_path: str,
    sample: Sample,
    ploidy: int = 2,
    generator: ImageGenerator = None,
    addl_features: typing.Optional[dict] = None,
):
    print(region, file=sys.stderr)
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
            local_read_path = _haplotag_reads(
                cfg.reference, sample, local_read_path, background_vcf, graph.region, tempdir
            )

        image_tensor = generator.generate(
            local_read_path,
            sample,
            example_region,
            realigner,
            ref_seq=ref_seq,
        )

    feature = {
        # "variant/encoded": _bytes_feature([variant.as_proto().SerializeToString()]),
        "image/shape": _int_feature(image_tensor.shape),
        "image/encoded": _bytes_feature(tf.io.serialize_tensor(image_tensor)),
        "addl/fisher_strand": _float_feature([getattr(image_tensor, "fisher_strand", 0.0)]),
        "addl/strand_orientation_bias": _float_feature([getattr(image_tensor, "strand_orientation_bias", 0.0)]),
    }

    # If we are augmenting the simulated data, use the provided statistics for the first example, so it
    # will hopefully be most similar to the real data and then augment the remaining replicates
    if cfg.simulation.augment:
        repl_samples = augment_sample(sample, cfg.simulation.replicates, keep_original=True)
    else:
        repl_samples = [sample] * cfg.simulation.replicates

    # Generate the possible haplotypes for this region on the possible backgrounds
    # TODO: Check if the backgrounds are identical, if so, we can generate the haplotypes once
    assert all(graph.is_bubble_path(f"{sample.name}#{i}#{region.contig}#0") for i in range(ploidy)), "Graph must form bubble for haplotype paths"
    backgrounds = [
        graph.generate_possible_haplotypes(inference_vcf, f"{sample.name}#{i}#{region.contig}#0", example_region)
        for i in range(ploidy)
    ]

    # Generate the relevant sequences once, which are then combined to create the fasta for simulation.
    # TODO?: Do we need pad out the shorter sequences?
    sequences = [[haplotype.sequence() for haplotype in background] for background in backgrounds]

    # For fully labeled data, one of the haplotypes should be the true haplotype
    labels = []
    for allele, haplotypes in enumerate(backgrounds):
        base_path_nodes = graph.nodes_on_path(f"{sample.name}#{allele}#{region.contig}#0")
        for allele_index, haplotype in enumerate(haplotypes):
            if haplotype.nodes == base_path_nodes:
                labels.append(allele_index)
                break

    if len(labels) == ploidy:
        feature["label"] = _int_feature(np.ravel_multi_index(tuple([i] for i in labels), tuple(len(b) for b in backgrounds)))

    # TODO?: Do we want to only have one of 0/1, 1/0?
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

            # Simulate reads from this haplotype combination
            repl_sample = repl_samples[0]
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
                realigner,
                ref_seq=ref_seq,
            )
            repl_encoded_images.append(synth_image_tensor)

            # TODO: Get "graph name" into output file
            if not OmegaConf.is_missing(cfg.simulation, "save_sim_bam_dir"):
                sim_bam_path = os.path.join(cfg.simulation.save_sim_bam_dir, f"{'_'.join(allele_indices)}.bam")
                shutil.copy(replicate_bam_path, sim_bam_path)
                shutil.copy(f"{replicate_bam_path}.bai", f"{sim_bam_path}.bai")

            # Stack all of the image replicates into a tensor
            alleles_encoded_images.append(np.stack(repl_encoded_images))

        # Stack the replicated images for each genotype into a tensor
        sim_image_tensor = np.stack(alleles_encoded_images)
        feature["sim/images/encoded"] = _bytes_feature(tf.io.serialize_tensor(sim_image_tensor))
        feature["sim/images/shape"] = _int_feature(
            (len(alleles_encoded_images), cfg.simulation.replicates, *image_tensor.shape)
        )

        # Add any additional (extension) features
        if addl_features:
            feature.update(addl_features)

    return tf.train.Example(features=tf.train.Features(feature=feature))


def _filename_to_compression(filename: str) -> typing.Optional[str]:
    if filename.endswith(".gz"):
        return "GZIP"
    else:
        return None


@ray.remote
class ExhaustiveExampleActor:
    def __init__(
        self, index: int, cfg, background_vcf: str, inference_vcf: str, read_path: str, sample: Sample, output_dir: str
    ):
        self.cfg = cfg
        self.background_vcf = background_vcf
        self.inference_vcf = inference_vcf
        self.read_path = read_path
        self.sample = sample

        self.output_path = os.path.join(output_dir, f"{index:03}.tfrecords.gz")

        try:
            # Try to reduce the number of threads TF creates since we are running multiple instances of TF via Ray
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
        except RuntimeError:
            pass

        self._tfrecordwritier = tf.io.TFRecordWriter(self.output_path, _filename_to_compression(self.output_path))

    def from_region(self, region: Range):
        example = make_example_from_region(
            self.cfg, region, self.background_vcf, self.inference_vcf, self.read_path, self.sample
        )
        self._tfrecordwritier.write(example.SerializeToString())

    def cleanup(self):
        self._tfrecordwritier.close()


def vcf_to_tfrecords(
    cfg,
    background_vcf: str,
    inference_vcf: str,
    read_path: str,
    sample: Sample,
    output_dir: str,
    ploidy: int = 2,
    progress_bar: bool = False,
):
    # Idenitify regions in the inference with overlapping SVs (i.e., identify bubbles)
    exhaustive_regions, search_regions = [], []
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

        (exhaustive_regions if count <= cfg.pileup.max_exhaustive_records else search_regions).append(region.expand(-group_padding))

    #assert False

    logging.info(
        "Identified %d regions with mean %f and max %d records",
        running_total,
        running_total / (len(exhaustive_regions) + len(search_regions)),
        running_max,
    )

    # We can't exhaustively generate examples for regions with too many records, so we skip any more complex regions
    logging.info(
        "Generating exhaustive images for %d regions (across %d threads)", len(exhaustive_regions), cfg.threads
    )
    os.makedirs(output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as ray_dir:
        # We currently just use ray for the CPU-side work, specifically simulating the SVs. We use a private temporary directory
        # to avoid conflicts between clusters running on the same node.
        ray.init(num_cpus=cfg.threads, num_gpus=0, _temp_dir=ray_dir, ignore_reinit_error=True, include_dashboard=False)

        actors = [
            ExhaustiveExampleActor.remote(i, cfg, background_vcf, inference_vcf, read_path, sample, output_dir)
            for i in range(cfg.threads)
        ]
        pool = ray.util.ActorPool(actors)

        gen = pool.map_unordered(lambda actor, region: actor.from_region.remote(region), exhaustive_regions)
        for _ in tqdm(gen, total=len(exhaustive_regions), disable=not progress_bar):
            pass

        # Make sure the tfrecords files are closed
        ray.wait([actor.cleanup.remote() for actor in actors], num_returns=len(actors))


def _example_image_shape(example: tf.train.Example):
    return tuple(example.features.feature["image/shape"].int64_list.value)


def _example_image(example: tf.train.Example):
    image_data = tf.io.parse_tensor(example.features.feature["image/encoded"].bytes_list.value[0], tf.uint8).numpy()
    return image_data


def _example_sim_images_shape(example: tf.train.Example):
    if "sim/images/shape" in example.features.feature:
        return tuple(example.features.feature["sim/images/shape"].int64_list.value)
    else:
        return (3, 0, None, None, None)


def _example_sim_images(example):
    image_data = tf.io.parse_tensor(
        example.features.feature["sim/images/encoded"].bytes_list.value[0], tf.uint8
    ).numpy()
    return image_data


def _example_addl_attribute(example: tf.train.Example, attr: str, prefix: str = "addl/"):
    return float(example.features.feature[prefix + attr].float_list.value[0])


def _example_label(example):
    return int(example.features.feature["label"].int64_list.value[0])


def _extract_metadata_from_first_example(filename, pileup_image_channels=None):
    raw_example = next(
        iter(tf.data.TFRecordDataset(filenames=filename, compression_type=_filename_to_compression(filename)))
    )
    example = tf.train.Example.FromString(raw_example.numpy())

    image_shape = _example_image_shape(example)
    genotypes, replicates, *sim_image_shape = _example_sim_images_shape(example)
    if replicates > 0:
        assert genotypes >= 3, "Incorrect number of genotypes in simulated data"
        assert image_shape == tuple(sim_image_shape), "Simulated and actual image shapes don't match"
    if pileup_image_channels:
        assert len(pileup_image_channels) <= image_shape[-1], "More channels requested than available"
        image_shape = image_shape[:-1] + (len(pileup_image_channels),)

    return image_shape, replicates


def features_to_image(
    cfg, features, out_path: str, with_simulations=False, margin=10, max_replicates=1, render_channels=False
):
    generator = hydra.utils.instantiate(cfg.generator, cfg, _recursive_=False)

    image_tensor = features["image"]
    real_image = generator.render(image_tensor, render_channels=render_channels)

    _, replicates, *_ = features["sim/images"].shape if with_simulations and "sim/images" in features else (3, 0)
    if replicates > 0:
        width, height = real_image.size
        replicates = min(replicates, max_replicates)

        image = Image.new(real_image.mode, (width + 2 * (width + margin), height + replicates * (height + margin)))
        image.paste(real_image, (width + margin, 0))

        synth_tensor = features["sim/images"]
        for ac in range(3):
            for repl in range(replicates):
                synth_image_tensor = synth_tensor[ac, repl]
                synth_image = generator.render(synth_image_tensor, render_channels=render_channels)

                coord = (ac * (width + margin), (repl + 1) * (height + margin))
                image.paste(synth_image, coord)
    else:
        image = real_image

    image.save(out_path)


def example_to_image(cfg, example: tf.train.Example, out_path: str, with_simulations=False, **kwargs):
    features = {
        "image": _example_image(example),
    }
    _, replicates, *_ = _example_sim_images_shape(example)
    if with_simulations and replicates > 0:
        features["sim/images"] = _example_sim_images(example)

    features_to_image(cfg, features, out_path, with_simulations=with_simulations and replicates > 0, **kwargs)


def load_example_dataset(
    filenames, with_label=False, with_simulations=False, num_parallel_reads=None, pileup_image_channels=None, ploidy=2,
) -> tf.data.Dataset:
    if isinstance(filenames, str):
        filenames = [filenames]
    assert len(filenames) > 0

    # Extract image shape from the first example
    shape, replicates = _extract_metadata_from_first_example(filenames[0], pileup_image_channels=pileup_image_channels)

    proto_features = {
        # "variant/encoded": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        "image/encoded": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
        "image/shape": tf.io.FixedLenFeature(shape=(len(shape),), dtype=tf.int64),
    }
    if with_label:
        proto_features["label"] = tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
    if with_simulations and replicates > 0:
        proto_features.update(
            {
                "sim/images/shape": tf.io.FixedLenFeature(shape=(len(shape) + 2,), dtype=tf.int64),
                "sim/images/encoded": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            }
        )

    # Adapted from Nucleus example
    def _process_input(proto_string):
        """Helper function for input function that parses a serialized example."""

        parsed_features = tf.io.parse_single_example(serialized=proto_string, features=proto_features)

        features = {
            # "variant/encoded": parsed_features["variant/encoded"],
            "image": tf.io.parse_tensor(parsed_features["image/encoded"], tf.uint8),
        }
        if with_simulations:
            features["sim/images"] = tf.io.parse_tensor(parsed_features["sim/images/encoded"], tf.uint8)

        if pileup_image_channels:
            features["image"] = tf.gather(features["image"], indices=list(pileup_image_channels), axis=-1)
            if with_simulations:
                features["sim/images"] = tf.gather(features["sim/images"], indices=list(pileup_image_channels), axis=-1)

        if with_label:
            return features, parsed_features["label"]
        else:
            return features, None

    compression = _filename_to_compression(filenames[0])
    num_parallel_calls = (
        num_parallel_reads
        if num_parallel_reads is None or num_parallel_reads == tf.data.experimental.AUTOTUNE
        else min(len(filenames), num_parallel_reads)
    )
    return tf.data.Dataset.from_tensor_slices(filenames).interleave(
        lambda filename: tf.data.TFRecordDataset(filename, compression_type=compression).map(
            _process_input, num_parallel_calls=1
        ),
        cycle_length=len(filenames),
        num_parallel_calls=num_parallel_calls,
    )
