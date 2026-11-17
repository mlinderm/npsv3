"""Prepare the inputs for genotyping a sample: the VCF the graphs are built from and the regions to genotype.

Genotyping builds graphs from a phased population VCF, as for training. How the population VCF is prepared depends on
whether there is an inference VCF:

1. Without an inference VCF, the population VCF is likely also the truth set, so it is used unchanged. The sample's
   private variants and alleles remain as candidates, while its genotypes are excluded from the population prior
   (`graph.population_excluded_samples`) and its own haplotypes from its candidate haplotypes
   (`haplotype.add_true_haplotypes`), see `npsv3.genotyping.images.genotyping_config`.
2. With an inference VCF, the sample's private variants and alleles are also dropped, so the graphs are built as if the
   sample was never in the population. The inference VCF's records are then merged in as sites only (its samples are never
   read), so it is effectively an excluded sample: the population haplotypes carry none of its novel alleles (the prior
   penalizes them with the sampler's fallback penalty).

Each inference ALT allele is tagged with `INFERENCE_ALLELE` (Number=A), `<vg variant ID of its record>_<allele index>`,
and its record's QUAL in `INFERENCE_QUAL` (Number=A), so the merged records map back to every inference allele.
`bcftools merge -m none` never combines records in the same file, but does merge a population record with an inference
record at the same position that shares at least one ALT allele (after normalizing any differences in the REF padding),
and the merged record has the union of their ALT alleles. So an inference record can be contained in a merged record with
other alleles, and the reverse. Since each record is merged at most once, the same allele can still end up in two records
at the same position, e.g., an inference record split into biallelic records merged with a multi-allelic population record.
Those are resolved so every allele is in one record (see `_resolve_shared_alleles`).

The regions are defined exactly as for training (`overlapping_variants` of the analysis variants), but over the merged
VCF. With an inference VCF, the regions can be restricted to those with an analysis variant that has an inference allele.
"""

import collections
import dataclasses
import itertools
import logging
import os
import tempfile
from collections.abc import Sequence

import pysam
import yaml
from pysam import bcftools

from npsv3.types import PathType
from npsv3.util.range import Range
from npsv3.util.variant import (
    VariantFileReader,
    generate_allele_indices,
    genotype_field_index,
    overlapping_variants,
    vg_variant_id,
)
from npsv3.util.vcf import bcftools_index, index_variant_file

GENOTYPING_VCF = "genotyping.vcf.gz"
"""The merged (and sample excluded) VCF graphs are built from, in the working directory's inputs directory"""

INFERENCE_SITES_VCF = "inference.sites.vcf.gz"
"""The records of the genotyping VCF with inference alleles, as sites only, in the inputs directory"""

PROVENANCE_FILE = "provenance.yaml"

INFO_INFERENCE_ALLELE = "INFERENCE_ALLELE"
INFO_INFERENCE_QUAL = "INFERENCE_QUAL"
INFERENCE_ALLELE_SEPARATOR = "|"
"""Joins the inference alleles of one ALT allele, if the inference VCF had the same allele in more than one record"""

_INFERENCE_INFO_LINES = {
    INFO_INFERENCE_ALLELE: (
        "A",
        "String",
        "The inference VCF allele(s) of each ALT allele, as <vg variant ID of the inference record>_<allele index>, "
        f"separated by '{INFERENCE_ALLELE_SEPARATOR}' if more than one",
    ),
    INFO_INFERENCE_QUAL: ("A", "Float", "QUAL of the inference VCF record of each ALT allele (the maximum if more than one)"),
}


@dataclasses.dataclass
class GenotypingInputs:
    """The prepared inputs for genotyping"""

    vcf_path: str
    """VCF for graph construction and haplotype sampling, i.e., the population VCF without the sample, merged with the
    inference VCF's sites. The population VCF itself (possibly including the sample) without an inference VCF."""
    inference_sites: str | None
    """The records of `vcf_path` with an inference allele, as sites only. None without an inference VCF."""


def inference_allele_id(record: pysam.VariantRecord, allele_index: int) -> str:
    """The `INFERENCE_ALLELE` of the `allele_index` allele of an inference VCF record"""
    return f"{vg_variant_id(record)}_{allele_index}"


def _missing(value) -> bool:
    # pysam reports missing strings in a list as "."
    return value is None or value == "."


def _info_values(record: pysam.VariantRecord, key: str) -> list:
    """The Number=A INFO field `key` of `record` with None for missing values"""
    values = record.info.get(key)
    if values is None:
        return [None] * len(record.alts or ())
    return [None if _missing(value) else value for value in values]


def has_inference_allele(record: pysam.VariantRecord) -> bool:
    """Whether a record of the genotyping VCF has any inference alleles"""
    return any(not _missing(value) for value in record.info.get(INFO_INFERENCE_ALLELE, ()))


def _vcf_samples(vcf_path: PathType) -> list[str]:
    with pysam.VariantFile(str(vcf_path)) as vcf_file:
        return list(vcf_file.header.samples)


def _bcftools(*args: str, output_path: str, output_type: str = "z", threads: int = 1, index: bool = True) -> str:
    """Run bcftools subcommand `args` writing `output_type` (e.g., "z" for bgzipped VCF) to `output_path`, and if `index`,
    its index as it is written (instead of a second pass over the output)"""
    command, *rest = args
    index_args = bcftools_index(output_path) if index else ()
    getattr(bcftools, command)(
        "--threads", str(threads), "-O", output_type, "-o", output_path, *index_args, *rest, catch_stdout=False
    )
    return output_path


def _add_inference_info(header: pysam.VariantHeader) -> None:
    """Add the inference INFO fields to `header`, if not already defined identically (e.g., genotyping our own output)"""
    for key, (number, value_type, description) in _INFERENCE_INFO_LINES.items():
        if key in header.info:
            existing = header.info[key]
            if (str(existing.number), existing.type) != (number, value_type):
                msg = f"The inference VCF already defines INFO/{key} with Number={existing.number},Type={existing.type}"
                raise ValueError(msg)
        else:
            header.info.add(key, number, value_type, description)


def _sites_header(header: pysam.VariantHeader) -> pysam.VariantHeader:
    """A copy of `header` without any samples"""
    sites_header = pysam.VariantHeader()
    for header_record in header.records:
        sites_header.add_record(header_record)
    return sites_header


def _sites_record(header: pysam.VariantHeader, record: pysam.VariantRecord, info: dict) -> pysam.VariantRecord:
    """A copy of `record` without any samples (for `header`), with INFO `info`"""
    return header.new_record(
        contig=record.contig,
        start=record.start,
        stop=record.stop,
        alleles=record.alleles,
        id=record.id,
        qual=record.qual,
        filter=list(record.filter),
        info=info,
    )


def _write_inference_sites(inference_vcf: str, output_path: str) -> int:
    """Write the inference VCF's records as sites only (bgzipped and indexed), with `INFERENCE_ALLELE` and `INFERENCE_QUAL`
    for each ALT allele. The samples are never read, so the inference VCF's genotypes can't contribute to the population.

    Returns:
        The number of records written. Records without an ALT allele aren't variants, and so are skipped.
    """
    written = 0
    with pysam.VariantFile(inference_vcf, drop_samples=True) as src_file:
        header = _sites_header(src_file.header)
        _add_inference_info(header)
        with pysam.VariantFile(output_path, mode="wz", header=header) as dst_file:
            for record in src_file:
                if not record.alts:
                    continue
                num_alts = len(record.alts)
                info = dict(record.info)
                info[INFO_INFERENCE_ALLELE] = tuple(inference_allele_id(record, i) for i in range(1, num_alts + 1))
                if record.qual is not None:
                    info[INFO_INFERENCE_QUAL] = (record.qual,) * num_alts
                dst_file.write(_sites_record(header, record, info))
                written += 1
    index_variant_file(output_path)
    return written


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
        sample_name: The sample being genotyped, removed from the population VCF if present and there is an inference VCF
        inference_vcf: Optional VCF (bgzipped and indexed) with the variants to genotype, which may include variants
            not in the population VCF. If None, all of the population's analysis variants are genotyped.
        threads: Threads for bcftools and compression

    Returns:
        GenotypingInputs: The prepared inputs
    """
    population_vcf = os.fspath(population_vcf)
    if inference_vcf is None:
        # Probably genotyping with the population VCF as the truth set, so use it unchanged (avoiding re-writing a
        # potentially very large VCF). The sample's private variants and alleles remain as candidates, so they can be
        # genotyped, while its genotypes are excluded from the population prior and its candidate haplotypes by the
        # genotyping configuration (see `npsv3.genotyping.images.genotyping_config`).
        return GenotypingInputs(vcf_path=population_vcf, inference_sites=None)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, GENOTYPING_VCF)
    inference_vcf = os.fspath(inference_vcf)
    sites_path = os.path.join(output_dir, INFERENCE_SITES_VCF)
    with tempfile.TemporaryDirectory(dir=output_dir, ignore_cleanup_errors=True) as tmp_dir:
        source_vcf = population_vcf
        if sample_name in _vcf_samples(population_vcf):
            logging.info("Excluding sample %s and its private alleles from the population VCF %s", sample_name, population_vcf)
            # Remove the alleles only the sample carried (`-a`) and the records left without any ALT alleles (`--min-ac`),
            # so the graphs are built as if the sample was never in the population. Otherwise its private alleles would
            # still be candidates. The inference VCF can still add those alleles back, as sites without population support.
            # BCF, since it is only read by bcftools merge.
            source_vcf = _bcftools(
                "view", "-s", f"^{sample_name}", "-a", "--min-ac", "1", population_vcf,
                output_path=os.path.join(tmp_dir, "population.bcf"), output_type="b", threads=threads,
            )

        logging.info("Merging the sites in inference VCF %s into the population VCF", inference_vcf)
        inference_sites = _write_inference_sites(inference_vcf, os.path.join(tmp_dir, "inference.vcf.gz"))
        # Never combine records in the same file (`-m none`). Records from the two files at the same position are merged
        # if they share any ALT allele, into a record with the union of their ALT alleles (with the INFERENCE_ALLELE tags
        # of the inference alleles). The records only in the inference VCF have no genotype fields, which graph
        # construction treats as missing genotypes. Uncompressed, since it is only read once, below.
        merged_vcf = _bcftools(
            "merge", "-m", "none", source_vcf, os.path.join(tmp_dir, "inference.vcf.gz"),
            output_path=os.path.join(tmp_dir, "merged.bcf"), output_type="u", threads=threads, index=False,
        )
        counts = _resolve_shared_alleles(merged_vcf, output_path, sites_path, threads=threads)

    logging.info(
        "Merged %d inference records into %d records with inference alleles (%d resolved shared alleles)",
        inference_sites, counts["inference"], counts["moved"],
    )
    return GenotypingInputs(vcf_path=output_path, inference_sites=sites_path)


def _allele_key(ref: str, alt: str) -> tuple[str, str] | None:
    """The allele normalized for differences in the REF padding (the common suffix), as `bcftools merge` does, so alleles
    at the same position can be compared. None for alleles that can't be compared by sequence (symbolic or '*')."""
    if alt == "*" or alt.startswith("<") or "[" in alt or "]" in alt:
        return None
    ref, alt = ref.upper(), alt.upper()
    trim = 0
    while trim < min(len(ref), len(alt)) - 1 and ref[-1 - trim] == alt[-1 - trim]:
        trim += 1
    return (ref[: len(ref) - trim], alt[: len(alt) - trim])


def _subset_field(value, number: str, keep: Sequence[int], ploidy: int):
    """Subset an INFO or FORMAT `value` with VCF `number` to the `keep` alleles (old indices, including the REF)"""
    if value is None or not isinstance(value, tuple):
        return value
    if number == "A":
        return tuple(value[allele - 1] for allele in keep[1:])
    if number == "R":
        return tuple(value[allele] for allele in keep)
    if number == "G":
        return tuple(
            value[genotype_field_index([keep[allele] for allele in genotype])]
            for genotype in generate_allele_indices(len(keep), ploidy)
        )
    return value


def _remove_alleles(header: pysam.VariantHeader, record: pysam.VariantRecord, removed: set[int], info: dict):
    """A copy of `record` without the `removed` ALT alleles (that no genotype refers to), with INFO `info`, or None if it
    would have no ALT alleles"""
    keep = [allele for allele in range(len(record.alleles)) if allele not in removed]
    if len(keep) == 1:
        return None
    new_index = {old: new for new, old in enumerate(keep)}
    samples, phased = [], []
    for call in record.samples.values():
        genotype = call.get("GT")
        ploidy = len(genotype) if genotype else 2
        fields = {}
        for key, value in call.items():
            if key == "GT":
                fields[key] = tuple(None if allele is None else new_index[allele] for allele in value)
            else:
                fields[key] = _subset_field(value, str(header.formats[key].number), keep, ploidy)
        samples.append(fields)
        phased.append(call.phased)
    new_record = header.new_record(
        contig=record.contig,
        start=record.start,
        stop=record.stop,
        alleles=[record.alleles[allele] for allele in keep],
        id=record.id,
        qual=record.qual,
        filter=list(record.filter),
        info={key: _subset_field(value, str(header.info[key].number), keep, 2) for key, value in info.items()},
        samples=samples,
    )
    for call, is_phased in zip(new_record.samples.values(), phased, strict=True):
        call.phased = is_phased
    return new_record


def _resolve_group(header: pysam.VariantHeader, group: list[pysam.VariantRecord], counts: collections.Counter):
    """Resolve the alleles shared by records at the same position so each allele with an inference allele is in only one
    record (see `_resolve_shared_alleles`), returning the records to write"""
    if len(group) < 2:  # noqa: PLR2004
        return group  # Almost every record, so check first
    tags = [_info_values(record, INFO_INFERENCE_ALLELE) for record in group]
    if not any(any(value is not None for value in record_tags) for record_tags in tags):
        return group

    occurrences = collections.defaultdict(list)
    for r, record in enumerate(group):
        for allele, alt in enumerate(record.alts, start=1):
            if (key := _allele_key(record.ref, alt)) is not None:
                occurrences[key].append((r, allele))
    shared = {key: alleles for key, alleles in occurrences.items() if len(alleles) > 1}
    if not any(tags[r][allele - 1] is not None for alleles in shared.values() for r, allele in alleles):
        return group  # Alleles only the population shares between records are left as they are

    quals = [_info_values(record, INFO_INFERENCE_QUAL) for record in group]
    referenced = [
        {allele for call in record.samples.values() for allele in (call.get("GT") or ()) if allele is not None}
        for record in group
    ]
    removed: list[set[int]] = [set() for _ in group]
    for alleles in shared.values():
        if all(tags[r][allele - 1] is None for r, allele in alleles):
            continue
        # The allele stays in the records whose genotypes refer to it (and so can't be removed), or otherwise the first
        supported = [(r, allele) for r, allele in alleles if allele in referenced[r]]
        home_r, home_allele = supported[0] if supported else alleles[0]
        for r, allele in alleles:
            if (r, allele) == (home_r, home_allele) or (r, allele) in supported:
                continue
            tag, qual = tags[r][allele - 1], quals[r][allele - 1]
            if tag is not None:
                home_tag = tags[home_r][home_allele - 1]
                tags[home_r][home_allele - 1] = tag if home_tag is None else f"{home_tag}{INFERENCE_ALLELE_SEPARATOR}{tag}"
            if qual is not None:
                home_qual = quals[home_r][home_allele - 1]
                quals[home_r][home_allele - 1] = qual if home_qual is None else max(home_qual, qual)
            removed[r].add(allele)
            counts["moved"] += 1

    resolved = []
    for record, record_tags, record_quals, record_removed in zip(group, tags, quals, removed, strict=True):
        info = dict(record.info)
        for key, values in ((INFO_INFERENCE_ALLELE, record_tags), (INFO_INFERENCE_QUAL, record_quals)):
            info.pop(key, None)
            if any(value is not None for value in values):
                # pysam writes a missing string in a list as an empty string, rather than "."
                info[key] = tuple("." if value is None and key == INFO_INFERENCE_ALLELE else value for value in values)
        if record_removed:
            if (new_record := _remove_alleles(header, record, record_removed, info)) is not None:
                resolved.append(new_record)
        else:
            for key in (INFO_INFERENCE_ALLELE, INFO_INFERENCE_QUAL):
                if key in info:
                    record.info[key] = info[key]
                elif key in record.info:
                    del record.info[key]
            resolved.append(record)
    return resolved


def _resolve_shared_alleles(
    merged_vcf: str, output_path: str, sites_path: str, *, threads: int = 1
) -> collections.Counter:
    """Write the merged VCF so that every allele with an inference allele is in just one record (bgzipped and indexed),
    and its records with an inference allele as sites only to `sites_path`.

    `bcftools merge` only merges each record once, so an allele can be in more than one record at the same position. If
    any of those has an inference allele, the allele is kept in the records whose genotypes refer to it (otherwise the
    first) and removed from the others, which can't have any genotype with it. Its INFERENCE_ALLELE and INFERENCE_QUAL
    move to the kept record, and records left without any ALT allele are dropped. For example:

    - Population ACGT>A,AC and inference ACGT>A and ACGT>AC (split): the merged ACGT>A,AC and the inference-only ACGT>AC.
      The latter's allele is moved to the former, which is then the only record.
    - Population ACGT>A and ACGT>AC (split) and inference ACGT>A,AC: the merged ACGT>A,AC and the population ACGT>AC. The
      AC allele is removed from the former, since only the latter's genotypes refer to it, so the inference record is
      then in two records.

    Alleles only the population shares between records are left as they are.

    Returns:
        The number of records with inference alleles ("inference") and of alleles removed from a record ("moved")
    """
    counts: collections.Counter = collections.Counter()
    # The merged VCF isn't indexed, since it's only read sequentially, so don't warn about the missing index
    verbosity = pysam.set_verbosity(0)
    try:
        src_file = pysam.VariantFile(merged_vcf, threads=threads)
    finally:
        pysam.set_verbosity(verbosity)
    with src_file:
        header = src_file.header
        sites_header = _sites_header(header)

        with pysam.VariantFile(output_path, mode="wz", header=header, threads=threads) as dst_file, pysam.VariantFile(
            sites_path, mode="wz", header=sites_header
        ) as sites_file:
            for _, group in itertools.groupby(src_file, key=lambda record: (record.rid, record.pos)):
                for record in _resolve_group(header, list(group), counts):
                    dst_file.write(record)
                    if has_inference_allele(record):
                        counts["inference"] += 1
                        sites_file.write(_sites_record(sites_header, record, dict(record.info)))
    index_variant_file(output_path)
    index_variant_file(sites_path)
    return counts


def _inference_variant_ids(inference_sites: PathType, region: Range | None) -> frozenset[str]:
    with pysam.VariantFile(os.fspath(inference_sites)) as vcf_file:
        records = vcf_file.fetch(**region.pysam_fetch) if region is not None else vcf_file.fetch()
        return frozenset(vg_variant_id(record) for record in records if has_inference_allele(record))


def genotyping_regions(
    vcf_path: PathType,
    *,
    flank: int,
    min_variant_size: int,
    inference_sites: PathType | None = None,
    region: Range | None = None,
) -> list[Range]:
    """The regions to genotype in `vcf_path` (the prepared genotyping VCF), defined as for training data generation.

    Args:
        vcf_path: The prepared genotyping VCF
        flank: Padding around the analysis variants, i.e., `pileup.variant_padding`
        min_variant_size: Minimum size of the analysis variants
        inference_sites: If provided (see `GenotypingInputs.inference_sites`), only the regions with an analysis variant
            that has an inference allele
        region: Only the regions in this region

    Returns:
        list[Range]: The regions, in VCF order
    """
    inference_ids = None if inference_sites is None else _inference_variant_ids(inference_sites, region)
    with VariantFileReader.open(os.fspath(vcf_path)) as vcf_file:
        return [
            variants_region
            for variants_region, variants in overlapping_variants(
                vcf_file, region=region, flank=flank, min_variant_size=min_variant_size
            )
            if inference_ids is None or any(variant.variant_id in inference_ids for variant in variants)
        ]


def file_provenance(path: PathType | None) -> dict | None:
    """Identify an input file by its path, size and modification time (hashing whole-genome inputs can be slow)"""
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
