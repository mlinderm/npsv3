import math
import os
import random
from collections.abc import Sequence

import numpy as np
import pysam
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2 as transforms

from npsv3.images.annotated_array import AnnotatedArray
from npsv3.pileup import AlleleAssignment, BaseAlignment, ReadPileup, Strand, fetch_reads
from npsv3.realigner import AlleleRealignment, realign_fragment
from npsv3.types import PathType
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

# Stage 1 ("Cheaper allele evidence for image generation"): per-base k-mer allele-evidence vote channel,
# a cheaper alternative/complement to the per-fragment BWA realignment channels above. See
# KmerEvidenceIndex (native/kmer_evidence.hpp) and ImageGenerator._kmer_evidence_pixel.
KMER_EVIDENCE_CHANNEL = 8

MAX_NUM_CHANNELS = 9

def _fragment_zscore(sample: Sample, fragment_length: int, fragment_delta=0):
    return (fragment_length + fragment_delta - sample.mean_insert_size) / sample.std_insert_size


class ImageGenerator:
    def __init__(self, cfg):
        # Resolve the (OmegaConf) pileup settings used inside the per-read/per-column rendering hot loop into
        # plain Python attributes once, rather than repeatedly resolving them via DictConfig on every pixel.
        self._cfg = cfg

        self._reference = cfg.reference

        self._image_height = cfg.pileup.image_height
        self._image_width = cfg.pileup.image_width
        self._image_channels = list(cfg.pileup.image_channels)
        self._image_shape = (self._image_height, self._image_width, len(self._image_channels))

        self._fetch_flank = cfg.pileup.fetch_flank
        self._insert_bases = cfg.pileup.insert_bases
        self._assign_delta = cfg.pileup.assign_delta
        self._phase_tag = cfg.pileup.phase_tag
        self._variant_padding = cfg.pileup.variant_padding

        self._max_baseq = cfg.pileup.max_baseq
        self._max_mapq = cfg.pileup.max_mapq
        self._discrete_mapq = cfg.pileup.discrete_mapq
        self._mapq0_pixel = cfg.pileup.mapq0_pixel

        self._binary_allele = cfg.pileup.binary_allele
        self._max_alleleq = cfg.pileup.max_alleleq
        self._allele_pixel_range = cfg.pileup.allele_pixel_range
        self._amb_allele_pixel = cfg.pileup.amb_allele_pixel

        self._evidence_min_baseq = cfg.pileup.evidence_min_baseq
        self._kmer_evidence_center_pixel = cfg.pileup.kmer_evidence_center_pixel
        # Scale so a saturated tent apex (k-1 votes, see plan 1a) lands near the pixel rails.
        self._kmer_evidence_scale = cfg.pileup.kmer_evidence_pixel_range / max(1, cfg.pileup.evidence_kmer_size - 1)

        self._insert_size_mean_pixel = cfg.pileup.insert_size_mean_pixel
        self._insert_size_sd_pixel = cfg.pileup.insert_size_sd_pixel

        # Helper dictionaries to map to pixel values
        # fmt: off
        self._aligned_to_pixel = {
            BaseAlignment.ALIGNED: cfg.pileup.aligned_base_pixel,
            BaseAlignment.MATCH: cfg.pileup.match_base_pixel if cfg.pileup.render_snv else cfg.pileup.aligned_base_pixel,
            BaseAlignment.MISMATCH: cfg.pileup.mismatch_base_pixel if cfg.pileup.render_snv else cfg.pileup.aligned_base_pixel,
            BaseAlignment.SOFT_CLIP: cfg.pileup.soft_clip_base_pixel,
            BaseAlignment.INSERT: cfg.pileup.insert_base_pixel,
        }
        # fmt: on
        self._aligned_value_to_pixel = np.zeros(max(a.value for a in BaseAlignment) + 1)
        for align, pixel in self._aligned_to_pixel.items():
            self._aligned_value_to_pixel[align.value] = pixel

        self._allele_to_pixel = {
            AlleleAssignment.AMB: cfg.pileup.amb_allele_pixel,
            AlleleAssignment.REF: cfg.pileup.ref_allele_pixel,
            AlleleAssignment.ALT: cfg.pileup.alt_allele_pixel,
            None: 0,
        }

        self._strand_to_pixel = {
            Strand.POSITIVE: cfg.pileup.positive_strand_pixel,
            Strand.NEGATIVE: cfg.pileup.negative_strand_pixel,
            None: 0.0,
        }

    def _align_pixel(self, align):
        if isinstance(align, BaseAlignment):
            return self._aligned_to_pixel[align]
        # An array of BaseAlignment values (see pileup._refine_base_alignment)
        return self._aligned_value_to_pixel[align]

    def _zscore_pixel(self, zscore):
        if zscore is None:
            return 0
        return min(max(self._insert_size_mean_pixel + zscore * self._insert_size_sd_pixel, 1), MAX_PIXEL_VALUE)

    def _allele_pixel(self, realignment: AlleleRealignment):
        if self._binary_allele:
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
            / self._max_alleleq
            * self._allele_pixel_range
            + self._amb_allele_pixel,
            1,
            MAX_PIXEL_VALUE,
        )

    def _kmer_evidence_pixel(self, ref_votes: np.ndarray | None, alt_votes: np.ndarray | None):
        # Plan 1b encoding: pixel 0 means "no informative k-mer window covers this base" (the same
        # reserved-0 convention _allele_pixel/clip already establish for channels 6/7); otherwise the
        # magnitude is the raw vote-count contrast so the tent's apex height (and thus shape) survives.
        if ref_votes is None or alt_votes is None:
            return 0
        depth = ref_votes + alt_votes
        pixel = np.zeros_like(depth)
        informative = depth > 0
        pixel[informative] = np.clip(
            self._kmer_evidence_center_pixel + self._kmer_evidence_scale * (alt_votes[informative] - ref_votes[informative]),
            1,
            MAX_PIXEL_VALUE,
        )
        return pixel

    def _scan_kmer_evidence(self, kmer_index, read: pysam.AlignedSegment):
        seq = read.query_sequence
        if not seq:
            return None
        base_qualities = list(read.query_qualities) if self._evidence_min_baseq > 0 and read.query_qualities else None
        ref_votes, alt_votes, multi_occurrence_hits = kmer_index.scan(seq, base_qualities, self._evidence_min_baseq)
        return np.asarray(ref_votes, dtype=np.float32), np.asarray(alt_votes, dtype=np.float32), multi_occurrence_hits

    def _strand_pixel(self, read: pysam.AlignedSegment):
        return self._strand_to_pixel[Strand.NEGATIVE if read.is_reverse else Strand.POSITIVE]

    def _qual_pixel(self, qual, max_qual: int):
        if qual is None:
            return 0
        return np.minimum(np.array(qual) / max_qual, 1.0) * MAX_PIXEL_VALUE

    def _mapq_pixel(self, qual):
        if qual is None:
            return 0
        if self._discrete_mapq:
            if qual == 0:
                return self._mapq0_pixel
            return np.minimum((np.array(qual) / self._max_mapq) * 127 + 128, MAX_PIXEL_VALUE)
        return self._qual_pixel(qual, self._max_mapq)

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
        to_pad = self._image_width - region.length
        left_padding = max((to_pad + 1) // 2, self._variant_padding)
        right_padding = max(to_pad // 2, self._variant_padding)
        return region.expand(left_padding, right_padding)

    def render(self, image_tensor, **kwargs) -> Image:
        assert len(image_tensor.shape) == 3
        return self._flatten_image(image_tensor, **kwargs)

    def generate(self, read_path: PathType | Sequence[PathType], sample: Sample, region: Range, compress=False, return_raw=False, **kwargs):
        image = self._generate(read_path, sample, region, **kwargs)
        image_array = image[:, :, self._image_channels]

        # Create consistent image size
        if compress and image_array.shape != self._image_shape:
            image_array = self._resize(image_array)

        result = AnnotatedArray(
            image_array,
            fisher_strand=image.fisher_strand,
            strand_orientation_bias=image.strand_orientation_bias,
        )
        if return_raw:
            # The full, pre-channel-select, pre-resize tensor (all MAX_NUM_CHANNELS channels, at
            # region.length columns) -- used by the dual-run agreement harness's pixel-correlation report,
            # which needs to compare channels 6/7 against channel 8 before either selection or resize
            # would destroy the comparison.
            return result, image
        return result

    def _resize(self, image_array: np.ndarray) -> np.ndarray:
        # Plan constraint 3: bilinear area-averaging is fine for every existing channel, but it would
        # flatten the k-mer evidence channel's interpretable quantity -- the tent's apex height -- toward
        # its (uninformative) midpoint. So that channel gets its own pooling, keyed on distance from the
        # neutral center pixel ("strongest evidence, either direction, wins") rather than raw pixel
        # magnitude (which would silently bias every downsample toward the alt allele).
        pool_channels = {KMER_EVIDENCE_CHANNEL}
        selected = self._image_channels
        bilinear_idx = [i for i, ch in enumerate(selected) if ch not in pool_channels]
        pool_idx = [i for i, ch in enumerate(selected) if ch in pool_channels]

        out = np.empty((self._image_height, self._image_width, len(selected)), dtype=image_array.dtype)
        if bilinear_idx:
            in_tensor = torch.from_numpy(image_array[:, :, bilinear_idx]).permute(2, 0, 1)
            out_tensor = transforms.functional.resize(in_tensor, [self._image_height, self._image_width])
            out[:, :, bilinear_idx] = out_tensor.permute(1, 2, 0).numpy()
        if pool_idx:
            out[:, :, pool_idx] = self._pool_by_evidence_strength(image_array[:, :, pool_idx])
        return out

    def _pool_by_evidence_strength(self, channels: np.ndarray) -> np.ndarray:
        # channels: (H, W, C) uint8, pixel 0 reserved for "no informative k-mer window" (plan 1b). Within
        # each output cell's window, keep the input pixel with the largest |value - center| (the strongest
        # ref/alt contrast in either direction); a window that is entirely 0 stays 0.
        raw = torch.from_numpy(channels).permute(2, 0, 1).unsqueeze(0).float()  # 1, C, H, W
        distance = torch.where(raw == 0, torch.full_like(raw, -1.0), (raw - self._kmer_evidence_center_pixel).abs())

        pooled_distance, indices = F.adaptive_max_pool2d(
            distance, (self._image_height, self._image_width), return_indices=True
        )
        flat_raw = raw.view(1, raw.shape[1], -1)
        flat_indices = indices.view(1, raw.shape[1], -1)
        gathered = torch.gather(flat_raw, 2, flat_indices).view_as(pooled_distance)

        pooled = torch.where(pooled_distance < 0, torch.zeros_like(gathered), gathered)
        return pooled.squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy()


class CoverageImageGenerator(ImageGenerator):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _generate(
        self,
        read_path: PathType | Sequence[PathType],
        sample: Sample,
        region: Range,
        realigner: AlleleRealignment | None = None,
        kmer_index=None,
        ref_seq: str | None = None,
        realign_cache: dict | None = None,
        fragment_cache: dict | None = None,
        agreement_sink=None,
        **kwargs,
    ):
        image_height = self._image_height
        image_tensor = np.zeros((image_height, region.length, MAX_NUM_CHANNELS), dtype=np.uint8)

        read_paths = [read_path] if isinstance(read_path, (str, os.PathLike)) else list(read_path)

        # Construct the pileup from the fragments, merging fragments fetched from all of the BAM/CRAM files into a
        # single pileup so that, e.g., per-haplotype read sets can be rendered as one combined image
        pileup = ReadPileup(region)

        # Collect the overlapping fragments first and subsample *before* realigning. Realignment is the
        # expensive step and the pileup only ever renders up to `max_reads` reads, so realigning fragments
        # that would just be discarded by the pileup's own downstream subsampling wastes that cost.
        max_reads = (region.length * image_height) // sample.read_length

        overlapping_fragments = []
        for path in read_paths:
            # The same per-haplotype split BAM is rendered in every genotype that includes that haplotype, so reuse
            # its (overlapping) fragments, and their k-mer evidence below, rather than re-reading and re-scanning it
            cache_key = (str(path), str(region))
            if fragment_cache is not None and cache_key in fragment_cache:
                overlapping_fragments.extend(fragment_cache[cache_key])
                continue

            path_fragments = []
            fragments = fetch_reads(path, region.expand(self._fetch_flank), reference=self._reference)
            for fragment in fragments:
                # At present we render reads based on the original alignment so we only realign (and track) fragments that could overlap
                # the image window. If we render "insert" bases, then we look if any part of the fragment overlaps the region
                if fragment.fragment_overlaps(region, read_overlap_only=not self._insert_bases):
                    path_fragments.append(fragment)
            if fragment_cache is not None:
                fragment_cache[cache_key] = path_fragments
            overlapping_fragments.extend(path_fragments)

        if len(overlapping_fragments) > max_reads:
            overlapping_fragments = random.sample(overlapping_fragments, k=max_reads)

        for fragment in overlapping_fragments:
            insert_zscore = _fragment_zscore(sample, fragment.fragment_length)

            # Render "insert" bases for overlapping fragments without reads in the region (and thus would not
            # otherwise be represented)
            add_insert = self._insert_bases and not fragment.reads_overlap(region)

            if realigner is not None:
                realignment, read1_realignment, read2_realignment = realign_fragment(
                    realigner, fragment, assign_delta=self._assign_delta, cache=realign_cache
                )
                realign_args = { "allele": realignment, "read1_realignment": read1_realignment, "read2_realignment": read2_realignment }
            else:
                realign_args = {}

            if kmer_index is not None:
                kmer_evidence = getattr(fragment, "_kmer_evidence", None) if fragment_cache is not None else None
                if kmer_evidence is None:
                    kmer_evidence = (
                        self._scan_kmer_evidence(kmer_index, fragment.read1) if fragment.read1 else None,
                        self._scan_kmer_evidence(kmer_index, fragment.read2) if fragment.read2 else None,
                    )
                    if fragment_cache is not None:
                        fragment._kmer_evidence = kmer_evidence
                read1_kmer_evidence, read2_kmer_evidence = kmer_evidence
            else:
                read1_kmer_evidence = None
                read2_kmer_evidence = None

            if agreement_sink is not None:
                agreement_sink.record(
                    fragment,
                    realign_args.get("allele"),
                    realign_args.get("read1_realignment"),
                    realign_args.get("read2_realignment"),
                    read1_kmer_evidence,
                    read2_kmer_evidence,
                )

            pileup.add_fragment(
                fragment,
                add_insert=add_insert,
                insert_zscore=insert_zscore,
                phase_tag=self._phase_tag,
                read1_kmer_evidence=read1_kmer_evidence,
                read2_kmer_evidence=read2_kmer_evidence,
                **realign_args,
            )

        # Add pileup bases to the image (downsample reads based on simple coverage-based heuristic; this is now a
        # no-op in the common case since the fragment-level subsample above already bounds the read count, but it
        # remains the final authority since a fragment can contribute up to two reads)
        # Only render the selected channels (unless the agreement harness needs the full, raw tensor), writing all of
        # them for a read's slice with a single assignment
        render_channels = list(range(MAX_NUM_CHANNELS)) if agreement_sink is not None else self._image_channels
        render_channel_idxs = np.array(render_channels)[None, :]
        row_idxs = np.full((region.length,), 0)
        for read in pileup.overlapping_reads(region, max_reads=max_reads):
            # Per-read (constant) pixels
            read_pixels = {
                PAIRED_CHANNEL: self._zscore_pixel(read.insert_zscore),
                MAPQ_CHANNEL: self._mapq_pixel(read.mapq),
                STRAND_CHANNEL: self._strand_to_pixel[read.strand],
            }
            if PHASE_CHANNEL in render_channels:
                read_pixels[PHASE_CHANNEL] = self._phase_pixel(read.phase)
            if ALLELE_CHANNEL in render_channels:
                read_pixels[ALLELE_CHANNEL] = self._allele_pixel(read.allele)
            if READ_ALLELE_CHANNEL in render_channels:
                read_pixels[READ_ALLELE_CHANNEL] = self._allele_pixel(read.read_allele)

            for col_slice, aligned, read_slice in pileup.read_columns(region, read, ref_seq):
                pixels = np.empty((col_slice.stop - col_slice.start, len(render_channels)), dtype=np.float64)
                for i, channel in enumerate(render_channels):
                    if channel == ALIGNED_CHANNEL:
                        pixels[:, i] = self._align_pixel(aligned)
                    elif channel == BASEQ_CHANNEL:
                        pixels[:, i] = self._qual_pixel(read.baseq(read_slice), self._max_baseq)
                    elif channel == KMER_EVIDENCE_CHANNEL:
                        pixels[:, i] = self._kmer_evidence_pixel(*read.kmer_votes(read_slice))
                    else:
                        pixels[:, i] = read_pixels[channel]
                col_idxs = np.arange(col_slice.start, col_slice.stop)[:, None]
                image_tensor[row_idxs[col_slice][:, None], col_idxs, render_channel_idxs] = pixels

                # Increment the 'current' row for the bases we just added to the pileup, overwrite the last row if we exceed
                # the maximum coverage
                row_idxs[col_slice] = np.minimum(row_idxs[col_slice] + 1, image_height - 1)

        return AnnotatedArray(image_tensor)
