"""Score the predictions and write the genotyped VCF (stage 5 of the genotyper).

Each scored region's (R, G) similarities become a probability for each of its diplotypes (`score_diplotypes`), and the
most probable is the region's call. Every variant in the region is then reported with the alleles on the called diplotype's
two haplotypes, and with qualities from the marginal probability of each of its unphased genotypes (`genotype_variant`).

Records in the output:

- Every record of the genotyping VCF in the graph of a genotyped region. Each region is one phase block.
- Regions where only a single haplotype was found are calls: every variant is `0|0`, without qualities, with the INFO flag
  `SINGLE_HAPLOTYPE`.
- Every record of the genotyping VCF with an inference allele (`INFERENCE_ALLELE`, see `npsv3.genotyping.inputs`), even if
  it wasn't genotyped. Those have a missing genotype and a FILTER explaining why: `NoRegion` (in no genotyped region),
  `NotInGraph` (excluded from the graph), or `InsufficientHaplotypes`, `Timeout` or `Rejected` (its region wasn't scored).

Every inference allele is in exactly one record of the genotyping VCF, so every record in the inference VCF is reported
once, possibly as part of a record with other alleles (e.g., the population record it merged with), or split across
records. Records are reported as they are in the genotyping VCF (alleles, ID and INFO, including `INFERENCE_ALLELE` and
`INFERENCE_QUAL`), and the genotypes of a subset of a record's alleles aren't calculated.

Other records in regions that weren't genotyped aren't reported.
"""

import bisect
import collections
import dataclasses
import hashlib
import itertools
import json
import logging
import math
import os
import tempfile
from collections.abc import Sequence

import hydra
import numpy as np
import pandas as pd
import pysam
from megatron.energon.cache.file_store import WebdatasetFileStore
from megatron.energon.epathlib import EPath
from omegaconf import DictConfig, OmegaConf
from pysam import bcftools

from npsv3.constants import HAPLOTYPES_SUFFIX
from npsv3.genotyping.inputs import has_inference_allele
from npsv3.images.example import FAILURE_INSUFFICIENT_HAPLOTYPES, FAILURE_TIMEOUT, energon_key
from npsv3.images.haplotypes import decode_haplotype_record
from npsv3.types import PathType
from npsv3.util.range import Range
from npsv3.util.variant import VariantFileReader, generate_allele_indices, vg_variant_id
from npsv3.util.vcf import bcftools_format, bcftools_index

FILTER_NO_REGION = "NoRegion"
FILTER_NOT_IN_GRAPH = "NotInGraph"
FILTER_INSUFFICIENT_HAPLOTYPES = "InsufficientHaplotypes"
FILTER_TIMEOUT = "Timeout"
FILTER_REJECTED = "Rejected"

INFO_SINGLE_HAPLOTYPE = "SINGLE_HAPLOTYPE"

MAX_PHRED = 999
"""Phred-scaled qualities are capped, since a genotype no candidate diplotype produces has a probability of zero"""

AGGREGATIONS = {"mean": lambda metric: metric.mean(axis=0), "max": lambda metric: metric.max(axis=0)}
"""How to summarize the (R, G) similarities as one per genotype. "max" is the model's own `MaximizingPredictor`."""

_HEADER_TYPES_TO_COPY = frozenset(["GENERIC", "STRUCTURED", "INFO", "FILTER", "CONTIG"])
_INFO_LINES = {
    INFO_SINGLE_HAPLOTYPE: "Number=0,Type=Flag,Description=\"Only one candidate haplotype was found in the region, so it wasn't scored\"",
}
_FILTER_LINES = {
    FILTER_NO_REGION: "Not in any genotyped region",
    FILTER_NOT_IN_GRAPH: "Excluded from the region's graph",
    FILTER_INSUFFICIENT_HAPLOTYPES: "Too few candidate haplotypes were found to genotype the region",
    FILTER_TIMEOUT: "Timed out generating the region's images",
    FILTER_REJECTED: "The region's images couldn't be packed for inference",
}
_FORMAT_LINES = (
    'ID=GT,Number=1,Type=String,Description="Genotype"',
    'ID=PS,Number=1,Type=Integer,Description="Phase set, the POS of the first record in the region"',
    'ID=GQ,Number=1,Type=Integer,Description="Genotype quality, from the marginal probability of the unphased genotype"',
    'ID=DQ,Number=1,Type=Float,Description="Phred-scaled probability that the called diplotype of the region is wrong"',
    'ID=PL,Number=G,Type=Integer,Description="Normalized, Phred-scaled marginal probabilities of the unphased genotypes"',
    'ID=MT,Number=1,Type=Float,Description="Similarity of the real data to the simulated called diplotype"',
    'ID=MG,Number=1,Type=Float,Description="Margin between the similarities of the best and second-best diplotypes"',
)


@dataclasses.dataclass(frozen=True)
class ScoringOptions:
    """How to turn the similarities into calls and qualities"""

    temperature: float
    """Softmax temperature"""
    aggregate: str = "mean"
    """How to combine a genotype's replicates, one of `AGGREGATIONS`"""
    max_gq: int = 99
    """Maximum GQ"""

    def __post_init__(self):
        if self.aggregate not in AGGREGATIONS:
            msg = f"scoring.aggregate must be one of {sorted(AGGREGATIONS)}, not {self.aggregate!r}"
            raise ValueError(msg)
        if self.temperature <= 0:
            msg = f"The softmax temperature must be positive, not {self.temperature}"
            raise ValueError(msg)


def scoring_options(cfg: DictConfig) -> ScoringOptions:
    """The scoring options in `cfg.scoring`. The temperature defaults to the one the model's loss was trained with.

    Raises:
        ValueError: If the temperature isn't specified and the model's loss doesn't have a single temperature
    """
    temperature = OmegaConf.select(cfg, "scoring.temperature", default=None)
    if temperature is None:
        # Usually not in the configuration (it is the loss's default), so ask the loss
        loss = hydra.utils.instantiate(cfg.model.loss)
        temperature = getattr(loss, "temperature", None)
        if isinstance(temperature, bool) or not isinstance(temperature, int | float):
            msg = f"Loss {type(loss).__name__} has no scalar temperature, so scoring.temperature is required"
            raise ValueError(msg)
    return ScoringOptions(
        temperature=float(temperature),
        aggregate=OmegaConf.select(cfg, "scoring.aggregate", default="mean"),
        max_gq=OmegaConf.select(cfg, "scoring.max_gq", default=99),
    )


def _log_sum(log_values: np.ndarray) -> float:
    """log(sum(exp(`log_values`))), -inf if empty"""
    return float(np.logaddexp.reduce(log_values)) if log_values.size else -math.inf


def _phred(log_probability: float) -> float:
    """The Phred-scaled probability, capped at `MAX_PHRED`"""
    return min(MAX_PHRED, max(0.0, -10.0 * log_probability / math.log(10)))


@dataclasses.dataclass
class DiplotypeCall:
    """A region's call"""

    genotypes: list[tuple[int, ...]]
    """The candidate haplotypes of each diplotype"""
    similarities: np.ndarray
    """Each diplotype's similarity (`s_g`)"""
    log_probabilities: np.ndarray
    """Each diplotype's log probability"""
    called: int
    """The index of the called diplotype"""

    @property
    def diplotype(self) -> tuple[int, ...]:
        """The called diplotype's candidate haplotypes"""
        return self.genotypes[self.called]

    @property
    def diplotype_quality(self) -> float:
        """DQ: Phred-scaled probability that the called diplotype is wrong"""
        return _phred(_log_sum(np.delete(self.log_probabilities, self.called)))

    @property
    def similarity(self) -> float:
        """MT: The called diplotype's similarity"""
        return float(self.similarities[self.called])

    @property
    def margin(self) -> float | None:
        """MG: The margin between the best and second-best similarities (None if there is only one diplotype)"""
        if self.similarities.size < 2:  # noqa: PLR2004
            return None
        second, best = np.partition(self.similarities, -2)[-2:]
        return float(best - second)


def score_diplotypes(metric: np.ndarray, genotypes: Sequence[tuple[int, ...]], options: ScoringOptions) -> DiplotypeCall:
    """Call the most probable of a region's diplotypes.

    Args:
        metric: (R, G) similarities of the real image to each replicate's simulated image of each diplotype
        genotypes: The candidate haplotypes of each of the G diplotypes
        options: The scoring options

    Returns:
        The call. With ties, the earliest diplotype.
    """
    metric = np.asarray(metric, dtype=np.float64)
    if metric.ndim != 2 or metric.shape[1] != len(genotypes):  # noqa: PLR2004
        msg = f"Expected (R, {len(genotypes)}) similarities, not {metric.shape}"
        raise ValueError(msg)
    similarities = AGGREGATIONS[options.aggregate](metric)
    logits = similarities / options.temperature
    log_probabilities = logits - _log_sum(logits)
    return DiplotypeCall(
        genotypes=list(genotypes),
        similarities=similarities,
        log_probabilities=log_probabilities,
        called=int(np.argmax(log_probabilities)),
    )


@dataclasses.dataclass
class VariantGenotype:
    """A variant's genotype in a region's call"""

    alleles: tuple[int, ...]
    """The alleles on the called diplotype's haplotypes, in order (i.e., phased)"""
    quality: float
    """QUAL: Phred-scaled probability of the homozygous reference genotype"""
    genotype_quality: int
    """GQ: Phred-scaled probability that the unphased genotype is wrong, capped"""
    likelihoods: list[int]
    """PL: Normalized Phred-scaled marginal probabilities, in VCF order"""


def genotype_variant(
    call: DiplotypeCall, haplotype_alleles: Sequence[int], num_alleles: int, *, max_gq: int = 99
) -> VariantGenotype:
    """The genotype of one variant in a region's call. The marginal probability of each of the variant's unphased genotypes
    is the sum over the diplotypes that produce it. Because the call is the most probable *diplotype*, another unphased
    genotype can have a higher marginal probability than the called one (and so have the zero PL).

    Args:
        call: The region's call
        haplotype_alleles: The variant's allele on each of the region's candidate haplotypes
        num_alleles: The number of alleles of the variant (including the reference)
        max_gq: Maximum GQ

    Returns:
        The variant's genotype
    """
    unphased = [tuple(sorted(haplotype_alleles[haplotype] for haplotype in diplotype)) for diplotype in call.genotypes]
    alleles = tuple(haplotype_alleles[haplotype] for haplotype in call.diplotype)
    called = tuple(sorted(alleles))

    log_marginals = {
        genotype: _log_sum(call.log_probabilities[[g == genotype for g in unphased]]) for genotype in set(unphased)
    }
    phreds = [_phred(log_marginals.get(g, -math.inf)) for g in generate_allele_indices(num_alleles, len(alleles))]
    best = min(phreds)
    return VariantGenotype(
        alleles=alleles,
        quality=_phred(log_marginals.get((0,) * len(alleles), -math.inf)),
        genotype_quality=min(max_gq, round(_phred(_log_sum(call.log_probabilities[[g != called for g in unphased]])))),
        likelihoods=[round(phred - best) for phred in phreds],
    )


class _RegionIndex:
    """Determine the FILTER of an inference record that wasn't written, from the disjoint genotyping regions"""

    def __init__(self, regions: Sequence[Range], filters: dict[str, str | None]):
        """`filters` is each region's (string) FILTER if it wasn't genotyped, and otherwise None"""
        by_contig = collections.defaultdict(list)
        for region in regions:
            by_contig[region.contig].append((region.start, region.end, filters[str(region)]))
        self._regions = {contig: sorted(spans, key=lambda span: span[0]) for contig, spans in by_contig.items()}
        self._starts = {contig: [span[0] for span in spans] for contig, spans in self._regions.items()}

    def filter(self, region: Range) -> str:
        """The FILTER for a record with reference region `region`"""
        spans = self._regions.get(region.contig, [])
        # Insertions have an empty reference region
        end = max(region.end, region.start + 1)
        overlapping = False
        for index in range(bisect.bisect_left(self._starts.get(region.contig, []), end) - 1, -1, -1):
            start, stop, region_filter = spans[index]
            if stop <= region.start:
                break
            if start < region.start and region.end < stop:
                # Inside the graph's region. If its region was genotyped, it would have been written if it was in the graph.
                return region_filter or FILTER_NOT_IN_GRAPH
            overlapping = True
        return FILTER_NOT_IN_GRAPH if overlapping else FILTER_NO_REGION


def _output_header(vcf_file: pysam.VariantFile, sample_name: str, checkpoint: PathType | None) -> pysam.VariantHeader:
    """The header for the output: The source header, with our own FORMAT fields (replacing any in the source, which is
    likely to define the same fields differently) and INFO and FILTER fields, and identifying the model"""
    header = pysam.VariantHeader()
    for record in vcf_file.header.records:
        if record.type not in _HEADER_TYPES_TO_COPY:
            continue
        if (record.type == "INFO" and record.get("ID") in _INFO_LINES) or (
            record.type == "FILTER" and record.get("ID") in _FILTER_LINES
        ):
            continue
        header.add_record(record)
    for info_id, line in _INFO_LINES.items():
        header.add_line(f"##INFO=<ID={info_id},{line}>")
    for filter_id, description in _FILTER_LINES.items():
        header.add_line(f'##FILTER=<ID={filter_id},Description="{description}">')
    for line in _FORMAT_LINES:
        header.add_line(f"##FORMAT=<{line}>")
    if checkpoint is not None:
        with open(checkpoint, "rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        header.add_line(f"##npsv3_checkpoint={os.path.abspath(checkpoint)}")
        header.add_line(f"##npsv3_checkpoint_sha256={digest}")
    header.add_sample(sample_name)
    return header


def _write_not_genotyped(
    dst_file: pysam.VariantFile,
    header: pysam.VariantHeader,
    inference_sites: PathType,
    written: set[str],
    index: "_RegionIndex",
    counts: collections.Counter,
    ploidy: int,
) -> None:
    """Write the records with inference alleles that weren't written, with a missing genotype and the FILTER of why"""
    with pysam.VariantFile(os.fspath(inference_sites)) as sites_file, VariantFileReader.open(
        os.fspath(inference_sites)
    ) as reader:
        for record, variant in zip(sites_file.fetch(), reader.fetch(), strict=True):
            if variant.variant_id in written or not has_inference_allele(record):
                continue
            # The (padding-trimmed) reference region, as is used to decide if it is in a graph
            record_filter = index.filter(variant.reference_region())
            dst_file.write(
                header.new_record(
                    contig=record.contig,
                    start=record.start,
                    stop=record.stop,
                    alleles=record.alleles,
                    id=record.id,
                    filter=record_filter,
                    info=dict(record.info),
                    samples=[{"GT": (None,) * ploidy}],
                )
            )
            counts[record_filter] += 1


def _region_filter(failure: dict) -> str:
    reasons = {FAILURE_INSUFFICIENT_HAPLOTYPES: FILTER_INSUFFICIENT_HAPLOTYPES, FAILURE_TIMEOUT: FILTER_TIMEOUT}
    if (region_filter := reasons.get(failure["reason"])) is None:
        msg = f"Unexpected failure in genotyping region {failure['region']}: {failure['reason']}"
        raise ValueError(msg)
    return region_filter


def _read_example_metadata(images_dir: str) -> dict[str, dict]:
    with open(os.path.join(images_dir, "meta_ds", "train.jsonl")) as file:
        return {row["key"]: row for row in map(json.loads, filter(str.strip, file))}


def write_genotypes_vcf(  # noqa: PLR0913
    vcf_path: PathType,
    output_path: PathType,
    *,
    sample_name: str,
    regions: Sequence[Range],
    images_dir: PathType,
    predictions: pd.DataFrame | None,
    rejected: Sequence[dict],
    failures: Sequence[dict],
    options: ScoringOptions,
    inference_sites: PathType | None = None,
    checkpoint: PathType | None = None,
    ploidy: int = 2,
) -> collections.Counter:
    """Write the sorted (and if compressed, indexed) genotyped VCF. See the module documentation for the records written.

    Args:
        vcf_path: The genotyping VCF
        output_path: The output VCF
        sample_name: The sample being genotyped
        regions: Every region to genotype
        images_dir: The image dataset's directory, with its haplotype records
        predictions: The predictions (see `npsv3.genotyping.predict.read_predictions`), None if there are none
        rejected: The regions the loader rejected
        failures: The regions without images (see `npsv3.images.example.read_failures`)
        options: The scoring options
        inference_sites: The genotyping VCF's records with inference alleles (see `GenotypingInputs.inference_sites`), if any
        checkpoint: The model's checkpoint, recorded in the header
        ploidy: The sample's ploidy

    Returns:
        The number of records written by outcome: "called", INFO_SINGLE_HAPLOTYPE and the FILTER of the others

    Raises:
        ValueError: If any of the `regions` isn't predicted, rejected or a failure, i.e., can't be accounted for
    """
    output_path, images_dir = os.fspath(output_path), os.fspath(images_dir)

    predicted = {} if predictions is None else {row.key: row for row in predictions.itertuples()}
    region_failures = {row["key"]: row for row in failures}
    rejected_keys = {row["key"] for row in rejected}
    keys = [energon_key(sample_name, region) for region in regions]
    accounted = predicted.keys() | region_failures.keys() | rejected_keys
    if missing := [str(region) for region, key in zip(regions, keys, strict=True) if key not in accounted]:
        msg = f"{len(missing)} regions have no prediction, failure or rejection, e.g., {missing[:5]}"
        raise ValueError(msg)

    def single_haplotype(key: str) -> dict | None:
        """The haplotype record of a region with a single candidate haplotype, the only failure that is still a call"""
        failure = region_failures.get(key)
        if failure is None or failure["reason"] != FAILURE_INSUFFICIENT_HAPLOTYPES:
            return None
        record = failure.get("haplotypes")
        return record if record and record["paths"] else None

    # The region's FILTER if it wasn't genotyped (the single-haplotype regions are calls)
    region_filters: dict[str, str | None] = {}
    for region, key in zip(regions, keys, strict=True):
        if key in predicted or single_haplotype(key) is not None:
            region_filters[str(region)] = None
        else:
            region_filters[str(region)] = FILTER_REJECTED if key in rejected_keys else _region_filter(region_failures[key])

    examples = _read_example_metadata(images_dir) if predicted else {}
    store = WebdatasetFileStore(EPath(os.path.join(images_dir, "imgs_ds"))) if predicted else None

    counts: collections.Counter = collections.Counter()
    written: set[str] = set()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir, pysam.VariantFile(
        os.fspath(vcf_path), drop_samples=True
    ) as vcf_file:
        header = _output_header(vcf_file, sample_name, checkpoint)
        unsorted_path = os.path.join(tmp_dir, "genotypes.vcf.gz")
        with pysam.VariantFile(unsorted_path, mode="wz", header=header) as dst_file:
            for region, key in zip(regions, keys, strict=True):
                if region_filters[str(region)] is not None:
                    continue

                if key in predicted:
                    row = predicted[key]
                    assert store is not None
                    haplotypes = decode_haplotype_record(store[f"{key}.{HAPLOTYPES_SUFFIX}"][0])
                    possible = list(itertools.combinations_with_replacement(range(len(haplotypes["paths"])), ploidy))
                    genotype_idxs = examples[key].get("genotype_idxs")
                    genotypes = possible if genotype_idxs is None else [possible[idx] for idx in genotype_idxs]
                    call = score_diplotypes(row.metric, genotypes, options)
                else:
                    haplotypes, call = single_haplotype(key), None

                variant_index = {variant_id: i for i, variant_id in enumerate(haplotypes["variant_ids"])}
                phase_set = None
                for record in vcf_file.fetch(**region.pysam_fetch):
                    if record.alts is None or (index := variant_index.get(variant_id := vg_variant_id(record))) is None:
                        continue
                    haplotype_alleles = [alleles[index] for alleles in haplotypes["variant_alleles"]]
                    if any(allele is None for allele in haplotype_alleles):
                        continue  # Excluded from the graph

                    # Every record in the region is in the same phase block, identified by its first record
                    phase_set = record.pos if phase_set is None else phase_set
                    info = dict(record.info)

                    fields: dict = {"PS": phase_set}
                    if call is None:
                        fields = {"GT": (0,) * ploidy, **fields}
                        info[INFO_SINGLE_HAPLOTYPE] = True
                        quality = None
                        counts[INFO_SINGLE_HAPLOTYPE] += 1
                    else:
                        genotype = genotype_variant(call, haplotype_alleles, len(record.alleles), max_gq=options.max_gq)
                        fields = {
                            "GT": genotype.alleles,
                            **fields,
                            "GQ": genotype.genotype_quality,
                            "DQ": round(call.diplotype_quality, 2),
                            "PL": genotype.likelihoods,
                            "MT": round(call.similarity, 4),
                            **({} if call.margin is None else {"MG": round(call.margin, 4)}),
                        }
                        quality = round(genotype.quality, 2)
                        counts["called"] += 1

                    dst_record = header.new_record(
                        contig=record.contig,
                        start=record.start,
                        stop=record.stop,
                        alleles=record.alleles,
                        id=record.id,
                        qual=quality,
                        filter="PASS",
                        info=info,
                        samples=[fields],
                    )
                    dst_record.samples[0].phased = True
                    dst_file.write(dst_record)
                    written.add(variant_id)

            # The rest of the records with inference alleles, which weren't genotyped
            if inference_sites is not None:
                _write_not_genotyped(
                    dst_file, header, inference_sites, written, _RegionIndex(regions, region_filters), counts, ploidy
                )

        bcftools.sort(
            *bcftools_index(output_path),
            "-O",
            bcftools_format(output_path),
            "-o",
            output_path,
            "-T",
            tmp_dir,
            unsorted_path,
            catch_stdout=False,
        )

    logging.info("Wrote %s: %s", output_path, dict(counts))
    return counts
