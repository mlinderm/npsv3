import sys
import tempfile
import typing

import hydra
import numpy as np
import pysam
from PIL import Image

from npsv3.images.annotated_array import AnnotatedArray
from npsv3.pileup import AlleleAssignment, BaseAlignment, FragmentTracker, ReadPileup, Strand#, fetch_reads
from npsv3.realigner import AlleleRealignment
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

    def _flatten_image(
        self,
        image_tensor: np.ndarray,
        render_channels=False,
        select_channels=[ALIGNED_CHANNEL, PAIRED_CHANNEL, PHASE_CHANNEL],
        margin=5,
    ):
        # TODO: Better combine all the channels into a single image, perhaps ALIGNED, PAIRED_CHANNEL, ALLELE (with mapq as alpha)...
        combined_image = Image.fromarray(image_tensor[:, :, select_channels], mode="RGB")

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

    def render(self, image_tensor, **kwargs) -> Image:
        assert len(image_tensor.shape) == 3
        return self._flatten_image(image_tensor, **kwargs)

    def generate(self, read_path, sample: Sample, region: Range, **kwargs):
        return self._generate(read_path, sample, region, **kwargs)


class CoverageImageGenerator(ImageGenerator):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _generate(
        self,
        read_path,
        sample: Sample,
        region: Range,
        ref_seq: typing.Optional[str] = None,
        **kwargs,
    ):
        image_height = self._cfg.pileup.image_height
        image_tensor = np.zeros((image_height, region.length, 6), dtype=np.uint8)

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

                pileup.add_fragment(
                    fragment,
                    add_insert=add_insert,
                    insert_zscore=insert_zscore,
                    phase_tag=self._cfg.pileup.phase_tag,
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

                # Increment the 'current' row for the bases we just added to the pileup, overwrite the last row if we exceed
                # the maximum coverage
                row_idxs[col_slice] = np.clip(row_idxs[col_slice] + 1, 0, image_height - 1)

        return AnnotatedArray(image_tensor)