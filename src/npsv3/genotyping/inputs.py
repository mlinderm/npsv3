"""Prepare the inputs for genotyping a sample: the VCF the graphs are built from and the regions to genotype.

Genotyping builds graphs from a phased population VCF, as for training, with two differences:

1. The sample being genotyped is excluded from the population, i.e., its columns are dropped before graph construction,
   so its own haplotypes can't be among the candidates or inform the population prior.
2. An optional inference VCF can add variants to genotype that are not in the population VCF. Its records are merged
   into the population VCF as sites only (any genotypes it carries are dropped), so it is effectively an excluded sample:
   the population haplotypes carry none of its novel alleles (the prior penalizes them with the sampler's fallback
   penalty). Records in both VCFs (identical position and alleles, and so the same variant ID) are merged into one.

The regions are defined exactly as for training (`overlapping_variants` of the analysis variants), but over the merged
VCF. With an inference VCF, only the regions containing at least one of its records are genotyped.
"""

import bisect
import collections
import dataclasses
import logging
import os
import tempfile
from collections.abc import Iterable, Sequence

import pysam
import yaml
from pysam import bcftools

from npsv3.types import PathType
from npsv3.util.range import Range
from npsv3.util.variant import Variant, VariantFileReader, overlapping_variants
from npsv3.util.vcf import index_variant_file

GENOTYPING_VCF = "genotyping.vcf.gz"
"""The merged (and sample excluded) VCF graphs are built from, in the working directory's inputs directory"""

PROVENANCE_FILE = "provenance.yaml"


@dataclasses.dataclass
class GenotypingInputs:
    """The prepared inputs for genotyping"""

    vcf_path: str
    """VCF for graph construction and haplotype sampling, i.e., the population VCF without the sample, merged with the
    inference VCF's sites (if any). The population VCF itself, if neither change was needed."""
    inference_ids: frozenset[str] | None
    """Variant IDs of the inference VCF's records, None to genotype all of the analysis variants"""
    excluded_sample: bool
    """Whether the sample was in (and so was removed from) the population VCF"""


def _vcf_samples(vcf_path: PathType) -> list[str]:
    with pysam.VariantFile(str(vcf_path)) as vcf_file:
        return list(vcf_file.header.samples)


def _variant_ids(vcf_path: PathType) -> frozenset[str]:
    with VariantFileReader.open(str(vcf_path)) as vcf_file:
        return frozenset(variant.variant_id for variant in vcf_file.fetch())


PLACEHOLDER_SAMPLE = "__npsv3_inference_sites__"
"""Sample added to the inference sites while merging, so every merged record has a GT field (see `_write_placeholder_sites`)"""


def _write_placeholder_sites(inference_vcf: str, output_path: str) -> str:
    """Write `inference_vcf`'s records, without its samples, but with a `PLACEHOLDER_SAMPLE` with a missing genotype.

    Merging records without a GT field produces records without one, which graph construction rejects. With the
    placeholder, the merged inference-only records have missing GTs for every population sample (and the placeholder,
    which is then removed).
    """
    with pysam.VariantFile(inference_vcf, drop_samples=True) as src_file:
        # A copy of the source header would keep its samples, so copy just the meta-information lines
        header = pysam.VariantHeader()
        for header_record in src_file.header.records:
            header.add_record(header_record)
        if "GT" not in header.formats:
            header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
        header.add_sample(PLACEHOLDER_SAMPLE)
        with pysam.VariantFile(output_path, mode="wz", header=header) as dst_file:
            for record in src_file:
                dst_file.write(
                    header.new_record(
                        contig=record.contig,
                        start=record.start,
                        stop=record.stop,
                        alleles=record.alleles,
                        id=record.id,
                        qual=record.qual,
                        filter=list(record.filter),
                        info=dict(record.info),
                        samples=[{"GT": (None, None)}],
                    )
                )
    index_variant_file(output_path)
    return output_path


def _bcftools_to_vcf(*args: str, output_path: str, threads: int = 1) -> str:
    """Run bcftools subcommand `args` writing an indexed, bgzipped VCF to `output_path`"""
    command, *rest = args
    getattr(bcftools, command)("--threads", str(threads), "-O", "z", "-o", output_path, *rest, catch_stdout=False)
    index_variant_file(output_path)
    return output_path


def prepare_genotyping_vcf(
    population_vcf: PathType,
    output_dir: PathType,
    *,
    sample_name: str,
    inference_vcf: PathType | None = None,
    threads: int = 1,
) -> GenotypingInputs:
    """Create the VCF for genotyping `sample_name` in `output_dir` (see the module documentation).

    Args:
        population_vcf: Phased population VCF (bgzipped and indexed)
        output_dir: Directory for the prepared VCF (and any intermediate files)
        sample_name: The sample being genotyped, removed from the population VCF if present
        inference_vcf: Optional VCF (bgzipped and indexed) with the variants to genotype, which may include variants
            not in the population VCF. If None, all of the population's analysis variants are genotyped.
        threads: Threads for bcftools

    Returns:
        GenotypingInputs: The prepared inputs
    """
    population_vcf = os.fspath(population_vcf)
    excluded_sample = sample_name in _vcf_samples(population_vcf)
    if not excluded_sample and inference_vcf is None:
        # Nothing to change, so avoid re-writing a potentially very large VCF
        return GenotypingInputs(vcf_path=population_vcf, inference_ids=None, excluded_sample=False)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, GENOTYPING_VCF)
    with tempfile.TemporaryDirectory(dir=output_dir, ignore_cleanup_errors=True) as tmp_dir:
        source_vcf = population_vcf
        if excluded_sample:
            logging.info("Excluding sample %s from the population VCF %s", sample_name, population_vcf)
            # Don't trim ALT alleles only the sample carried (`-a`), since that would change the variant IDs. Those
            # alleles are kept, just without any population support.
            source_vcf = _bcftools_to_vcf(
                "view",
                "-s",
                f"^{sample_name}",
                population_vcf,
                output_path=output_path if inference_vcf is None else os.path.join(tmp_dir, "population.vcf.gz"),
                threads=threads,
            )

        if inference_vcf is not None:
            inference_vcf = os.fspath(inference_vcf)
            logging.info("Merging the sites in inference VCF %s into the population VCF", inference_vcf)
            sites_vcf = _write_placeholder_sites(inference_vcf, os.path.join(tmp_dir, "inference.sites.vcf.gz"))
            # Never combine records into multi-allelic records (`-m none`), only merge identical records, so every
            # record keeps its alleles (and variant ID). Then drop the placeholder sample.
            merged_vcf = _bcftools_to_vcf(
                "merge", "-m", "none", source_vcf, sites_vcf, output_path=os.path.join(tmp_dir, "merged.vcf.gz"), threads=threads,
            )
            _bcftools_to_vcf("view", "-s", f"^{PLACEHOLDER_SAMPLE}", merged_vcf, output_path=output_path, threads=threads)

    inference_ids = None if inference_vcf is None else _variant_ids(inference_vcf)
    if inference_ids is not None:
        logging.info("Genotyping the regions with the %d variants in %s", len(inference_ids), inference_vcf)
    return GenotypingInputs(vcf_path=output_path, inference_ids=inference_ids, excluded_sample=excluded_sample)


class _InferenceIndex:
    """Find the regions containing (variant reference regions of) a set of variants"""

    def __init__(self, variants: Iterable[Variant]):
        by_contig: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        for variant in variants:
            ref_region = variant.reference_region()
            by_contig[ref_region.contig].append((ref_region.start, ref_region.end))
        self._starts: dict[str, list[int]] = {}
        self._ends: dict[str, list[int]] = {}
        for contig, spans in by_contig.items():
            spans.sort()
            self._starts[contig] = [start for start, _ in spans]
            self._ends[contig] = [end for _, end in spans]

    def contains_any(self, region: Range) -> bool:
        """Whether `region` strictly contains any of the variants, the condition for graph construction to include it"""
        starts = self._starts.get(region.contig)
        if not starts:
            return False
        ends = self._ends[region.contig]
        for i in range(bisect.bisect_right(starts, region.start), bisect.bisect_left(starts, region.end)):
            if ends[i] < region.end:
                return True
        return False


def genotyping_regions(
    vcf_path: PathType,
    *,
    flank: int,
    min_variant_size: int,
    inference_vcf: PathType | None = None,
    region: Range | None = None,
) -> list[Range]:
    """The regions to genotype in `vcf_path` (the prepared genotyping VCF), defined as for training data generation.

    Args:
        vcf_path: The prepared genotyping VCF
        flank: Padding around the analysis variants, i.e., `pileup.variant_padding`
        min_variant_size: Minimum size of the analysis variants
        inference_vcf: If provided, only the regions containing at least one of its records (of any size)
        region: Only the regions in this region

    Returns:
        list[Range]: The regions, in VCF order
    """
    index = None
    if inference_vcf is not None:
        with VariantFileReader.open(os.fspath(inference_vcf)) as inference_file:
            index = _InferenceIndex(inference_file.fetch(region=region))

    with VariantFileReader.open(os.fspath(vcf_path)) as vcf_file:
        return [
            variants_region
            for variants_region, _ in overlapping_variants(vcf_file, region=region, flank=flank, min_variant_size=min_variant_size)
            if index is None or index.contains_any(variants_region)
        ]


def file_provenance(path: PathType | None) -> dict | None:
    """Identify an input file by its path, size and modification time (hashing whole-genome inputs is too slow)"""
    if path is None:
        return None
    stat = os.stat(path)
    return {"path": os.path.abspath(path), "size": stat.st_size, "mtime": int(stat.st_mtime)}


def write_provenance(directory: PathType, provenance: dict) -> None:
    with open(os.path.join(directory, PROVENANCE_FILE), "w") as file:
        yaml.safe_dump(provenance, file, sort_keys=True)


def check_provenance(directory: PathType, provenance: dict, *, keys: Sequence[str] | None = None) -> None:
    """Raise ValueError if `directory`'s recorded provenance differs from `provenance` (in just `keys`, if provided)"""
    path = os.path.join(directory, PROVENANCE_FILE)
    if not os.path.exists(path):
        msg = f"{directory} has no {PROVENANCE_FILE}, so can't check it was generated from the same inputs"
        raise ValueError(msg)
    with open(path) as file:
        recorded = yaml.safe_load(file)
    # Round-trip so values are compared exactly as they would be after being recorded
    current = yaml.safe_load(yaml.safe_dump(provenance))
    keys = sorted(recorded.keys() | current.keys()) if keys is None else list(keys)
    if differ := [key for key in keys if recorded.get(key) != current.get(key)]:
        msg = (
            f"{directory} was generated with different inputs or settings ({', '.join(differ)}): "
            f"{ {key: recorded.get(key) for key in differ} } vs. the current { {key: current.get(key) for key in differ} }"
        )
        raise ValueError(msg)
