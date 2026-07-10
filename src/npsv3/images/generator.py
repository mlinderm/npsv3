import math

import numpy as np
import pysam
import torch
from PIL import Image
from torchvision.transforms import v2 as transforms

from npsv3.images.annotated_array import AnnotatedArray
from npsv3.pileup import AlleleAssignment, BaseAlignment, ReadPileup, Strand, fetch_reads
from npsv3.realigner import AlleleRealignment, realign_fragment
from npsv3.util.range import Range
from npsv3.util.sample import Sample

MAX_PIXEL_VALUE = 254.0  # Adapted from DeepVariant

ALIGNED_CHANNEL = 0
PAIRED_CHANNEL = 1
MAPQ_CHANNEL = 2
STRAND_CHANNEL = 3
BASEQ_CHANNEL = 4

PHASE_CHANNEL = 5

ALLELE_CHANNEL = 6
READ_ALLELE_CHANNEL = 7

MAX_NUM_CHANNELS = 8

def _fragment_zscore(sample: Sample, fragment_length: int, fragment_delta=0):
    return (fragment_length + fragment_delta - sample.mean_insert_size) / sample.std_insert_size


class ImageGenerator:
    def __init__(self, cfg):
        self._cfg = cfg
        self._image_shape = (
            self._cfg.pileup.image_height,
            self._cfg.pileup.image_width,
            len(self._cfg.pileup.image_channels),
        )

        # Helper dictionaries to map to pixel values
        # fmt: off
        self._aligned_to_pixel = {
            BaseAlignment.ALIGNED: self._cfg.pileup.aligned_base_pixel,
            BaseAlignment.MATCH: self._cfg.pileup.match_base_pixel if self._cfg.pileup.render_snv else self._cfg.pileup.aligned_base_pixel,
            BaseAlignment.MISMATCH: self._cfg.pileup.mismatch_base_pixel if self._cfg.pileup.render_snv else self._cfg.pileup.aligned_base_pixel,
            BaseAlignment.SOFT_CLIP: self._cfg.pileup.soft_clip_base_pixel,
            BaseAlignment.INSERT: self._cfg.pileup.insert_base_pixel,
        }
        # fmt: on

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
        return [self._aligned_to_pixel[a] for a in align]

    def _zscore_pixel(self, zscore):
        if zscore is None:
            return 0
        return np.clip(
            self._cfg.pileup.insert_size_mean_pixel + zscore * self._cfg.pileup.insert_size_sd_pixel,
            1,
            MAX_PIXEL_VALUE,
        )

    def _allele_pixel(self, realignment: AlleleRealignment):
        if self._cfg.pileup.binary_allele:
            return self._allele_to_pixel[realignment.allele]
        if (
            realignment is None
            or realignment.ref_quality is None
            or math.isnan(realignment.ref_quality)
            or realignment.alt_quality is None
            or math.isnan(realignment.alt_quality)
        ):
            return 0
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
        return np.minimum(np.array(qual) / max_qual, 1.0) * MAX_PIXEL_VALUE

    def _mapq_pixel(self, qual):
        if qual is None:
            return 0
        if self._cfg.pileup.discrete_mapq:
            if qual == 0:
                return self._cfg.pileup.mapq0_pixel
            return np.minimum((np.array(qual) / self._cfg.pileup.max_mapq) * 127 + 128, MAX_PIXEL_VALUE)
        return self._qual_pixel(qual, self._cfg.pileup.max_mapq)

    def _phase_pixel(self, hp):
        if hp is None:
            return 0
        return (hp / 2.0) * MAX_PIXEL_VALUE

    def _flatten_image(
        self,
        image_tensor: np.ndarray,
        render_channels=False,
        select_channels=[ALIGNED_CHANNEL, PAIRED_CHANNEL, ALLELE_CHANNEL],
        margin=5,
    ):
        # TODO: Better combine all the channels into a single image, perhaps ALIGNED, PAIRED_CHANNEL, ALLELE (with mapq as alpha)...
        combined_image = Image.fromarray(image_tensor[:, :, select_channels])

        if render_channels:
            height, width, num_channels = image_tensor.shape
            image = Image.new(combined_image.mode, (width + (num_channels - 1) * (width + margin), 2 * height + margin))
            image.paste(combined_image, ((image.width - width) // 2, 0))

            for i in range(num_channels):
                channel_image = Image.fromarray(image_tensor[:, :, [i] * 3])
                coord = (i * (width + margin), height + margin)
                image.paste(channel_image, coord)

            return image
        return combined_image

    def image_region(self, region) -> Range:
        # Try to minimize compression by setting right padding to exact width...
        to_pad = self._cfg.pileup.image_width - region.length
        left_padding = max((to_pad + 1) // 2, self._cfg.pileup.variant_padding)
        right_padding = max(to_pad // 2, self._cfg.pileup.variant_padding)
        return region.expand(left_padding, right_padding)

    def image_region_variable(self, region) -> Range:
        # May need something to prevent padding when larger than the maximum image width
        # Pads the image by at least 96 pixels and rounds up to the nearest value divisible by 16
        to_pad = 2 * self._cfg.pileup.variant_padding + (16 - (((region.length-1) % 16) + 1))
        # print("rounded padding:", (16 - (((region.length-1) % 16) + 1)))
        # print("region length:", region.length)
        # print("to pad:",to_pad)
        return region.expand((to_pad + 1) // 2, to_pad // 2)

    def render(self, image_tensor, **kwargs) -> Image:
        assert len(image_tensor.shape) == 3
        return self._flatten_image(image_tensor, **kwargs)

    def generate(self, read_path, sample: Sample, region: Range, compress=False, **kwargs):
        image = self._generate(read_path, sample, region, **kwargs)
        image_array = image[:, :, self._cfg.pileup.image_channels]

        # Create consistent image size
        if compress and image_array.shape != (self._cfg.pileup.image_height, self._cfg.pileup.max_image_width, len(self._cfg.pileup.image_channels)):
            # Need to convert from HWC to CHW (and back again)
            in_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
            out_tensor = transforms.functional.resize(in_tensor, [self._cfg.pileup.image_height, self._cfg.pileup.max_image_width])
            image_array = out_tensor.permute(1, 2, 0).numpy()

        return AnnotatedArray(
            image_array,
            fisher_strand=image.fisher_strand,
            strand_orientation_bias=image.strand_orientation_bias,
        )


class CoverageImageGenerator(ImageGenerator):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _generate(
        self,
        read_path,
        sample: Sample,
        region: Range,
        realigner: AlleleRealignment | None = None,
        ref_seq: str | None = None,
        **kwargs,
    ):
        image_height = self._cfg.pileup.image_height
        image_tensor = np.zeros((image_height, region.length, MAX_NUM_CHANNELS), dtype=np.uint8)

        fragments = fetch_reads(read_path, region.expand(self._cfg.pileup.fetch_flank), reference=self._cfg.reference)

        # Construct the pileup from the fragments
        pileup = ReadPileup(region)

        for fragment in fragments:
            # At present we render reads based on the original alignment so we only realign (and track) fragments that could overlap
            # the image window. If we render "insert" bases, then we look if any part of the fragment overlaps the region
            if fragment.fragment_overlaps(region, read_overlap_only=not self._cfg.pileup.insert_bases):
                insert_zscore = _fragment_zscore(sample, fragment.fragment_length)

                # Render "insert" bases for overlapping fragments without reads in the region (and thus would not
                # otherwise be represented)
                add_insert = self._cfg.pileup.insert_bases and not fragment.reads_overlap(region)

                if realigner is not None:
                    realignment, read1_realignment, read2_realignment = realign_fragment(
                        realigner, fragment, assign_delta=self._cfg.pileup.assign_delta
                    )
                    realign_args = { "allele": realignment, "read1_realignment": read1_realignment, "read2_realignment": read2_realignment }
                else:
                    realign_args = {}

                pileup.add_fragment(
                    fragment,
                    add_insert=add_insert,
                    insert_zscore=insert_zscore,
                    phase_tag=self._cfg.pileup.phase_tag,
                    **realign_args,
                )

        # Add pileup bases to the image (downsample reads based on simple coverage-based heuristic)
        max_reads = (region.length * image_height) // sample.read_length
        row_idxs = np.full((region.length,), 0)
        for read in pileup.overlapping_reads(region, max_reads=max_reads):
            for col_slice, aligned, read_slice in pileup.read_columns(region, read, ref_seq):
                col_idxs = range(col_slice.start, col_slice.stop)
                image_tensor[row_idxs[col_slice], col_idxs, ALIGNED_CHANNEL] = self._align_pixel(aligned)
                image_tensor[row_idxs[col_slice], col_idxs, PAIRED_CHANNEL] = self._zscore_pixel(read.insert_zscore)
                image_tensor[row_idxs[col_slice], col_idxs, MAPQ_CHANNEL] = self._mapq_pixel(read.mapq)
                image_tensor[row_idxs[col_slice], col_idxs, STRAND_CHANNEL] = self._strand_to_pixel[read.strand]
                image_tensor[row_idxs[col_slice], col_idxs, BASEQ_CHANNEL] = self._qual_pixel(
                    read.baseq(read_slice), self._cfg.pileup.max_baseq
                )
                image_tensor[row_idxs[col_slice], col_idxs, PHASE_CHANNEL] = self._phase_pixel(read.phase)
                image_tensor[row_idxs[col_slice], col_idxs, ALLELE_CHANNEL] = self._allele_pixel(read.allele)
                image_tensor[row_idxs[col_slice], col_idxs, READ_ALLELE_CHANNEL] = self._allele_pixel(read.read_allele)

                # Increment the 'current' row for the bases we just added to the pileup, overwrite the last row if we exceed
                # the maximum coverage
                row_idxs[col_slice] = np.clip(row_idxs[col_slice] + 1, 0, image_height - 1)

        return AnnotatedArray(image_tensor)
