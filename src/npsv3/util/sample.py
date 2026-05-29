import io
import json
import logging
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from shlex import quote

import numpy as np
import omegaconf
import pandas as pd
import pybedtools.bedtool as bed
import pysam

_REQUIRED_STATS_FIELDS = ("sequencer", "read_length", "mean_coverage", "mean_insert_size", "std_insert_size")


def sample_name_from_bam(bam_path) -> str:
    """Extract sample name from BAM"""
    with pysam.AlignmentFile(bam_path) as bam:
        read_groups = bam.header["RG"]
        samples = {rg["SM"] for rg in read_groups}
        assert len(samples) == 1, f"BAM file {bam.filename} must contain a single sample"
        (sample,) = samples  # Extract single value from set
        return sample


@dataclass
class Sample:
    name: str
    sequencer: str
    read_length: int
    mean_coverage: float
    mean_insert_size: float
    std_insert_size: float
    bam: str|None = None
    sex: int = 0
    chrom_normalized_coverage: dict = field(default_factory=dict)
    gc_normalized_coverage: dict = field(default_factory=dict)
    kmc_prefix: str|None = None
    kmer_coverage: float|None = None

    def chrom_mean_coverage(self, chrom: str) -> float:
        """Return mean coverage for specific chromosome, defaulting to overall mean coverage.

        Args:
            chrom (str): Chromosome

        Returns:
            float: Mean coverage
        """
        return self.chrom_normalized_coverage.get(chrom, 1.0) * self.mean_coverage

    @classmethod
    def from_json(cls, json_path: str, min_gc_bin=100, max_gc_error=0.01) -> "Sample":
        with open(json_path) as file:
            sample_info = json.load(file)

            fields = {k: sample_info[k] for k in _REQUIRED_STATS_FIELDS}

            # Optional fields
            fields["bam"] = sample_info.get("bam", None)
            fields["sex"] = sample_info.get("sex", 0)
            fields["chrom_normalized_coverage"] = sample_info.get("chrom_normalized_coverage", {})
            fields["kmc_prefix"] = sample_info.get("kmc_prefix", None)
            fields["kmer_coverage"] = sample_info.get("kmer_coverage", None)

            # Filter GC entries with limited data
            gc_normalized_coverage = {}
            for gc, norm_covg in sample_info.get("gc_normalized_coverage", {}).items():
                if (
                    sample_info.get("gc_bin_count", {}).get(gc, 0) >= min_gc_bin
                    and sample_info.get("gc_normalized_coverage_error", {}).get(gc, 0) <= max_gc_error
                ):
                    gc_normalized_coverage[round(float(gc) * 100)] = norm_covg
            fields["gc_normalized_coverage"] = gc_normalized_coverage

            return cls(sample_info["sample"], **fields)


def _compute_coverage_with_samtools(
    read_path: str, fasta_path: str, covg_regions=5000, depth_samples=200, min_samples_per_chrom=5, threads=1
) -> tuple[float, dict, dict, dict]:
    """Compute coverage across windows with samtools (run via goleft)

    Args:
        read_path (str): Read file (BAM or CRAM)
        fasta_path (str): Reference fasta
        covg_regions (int, optional): Split genome in buckets of equal amounts of data . Defaults to 5000.
        depth_samples (int, optional): Sample regions to determine coverage. Defaults to 200.
        min_samples_per_chrom (int, optional): Minimum sampled regions for each chromosome. Defaults to 5.
        threads (int, optional): Number of threads. Defaults to 1.

    Returns:
        typing.Tuple[float,dict,dict,dict]: Mean coverage, dictionaries of chromosome and GC-normalized coverage, count of each GC bin
    """
    with tempfile.TemporaryDirectory() as output_dir:
        indexsplit_commandline = f"goleft \
            indexsplit \
            --n {covg_regions} \
            --fai {quote(fasta_path + '.fai')} \
           {read_path + '.crai' if read_path.endswith('cram') else read_path}"
        indexsplit = subprocess.check_output(indexsplit_commandline, shell=True, universal_newlines=True)
        indexsplit_table = pd.read_csv(
            io.StringIO(indexsplit), sep="\t", names=["chrom", "start", "end", "sum", "split"]
        )

        # Drop entries outside main chromosomes and randomly sample remaining regions, ensuring similar number of regions from each chromosome
        indexsplit_table = indexsplit_table[
            indexsplit_table.chrom.str.contains(r"^(?:chr)?(?:\d{1,2}|[XY])$", regex=True)
        ]

        if depth_samples < covg_regions:
            depth_regions = indexsplit_table.sample(depth_samples)

            # Make sure we have a minimum number of regions for each chrom
            indexsplit_group = indexsplit_table.groupby("chrom")

            def _min_sample(table):
                return (
                    indexsplit_group.get_group(table.name).sample(min_samples_per_chrom)
                    if table.shape[0] < min_samples_per_chrom
                    else table
                )

            depth_regions = depth_regions.groupby("chrom").apply(_min_sample).reset_index(0, drop=True)
        else:
            depth_regions = indexsplit_table

        regions_file = os.path.join(output_dir, "regions.bed")
        depth_regions.to_csv(regions_file, columns=["chrom", "start", "end"], header=False, index=False, sep="\t")

        prefix = os.path.join(output_dir, "depth")
        depth_commandline = f"goleft \
            depth \
            --processes {threads} \
            --stats \
            --reference {quote(fasta_path)} \
            --prefix {prefix} \
            --bed {regions_file} \
           {read_path}"

        subprocess.check_call(
            depth_commandline,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        depth_table = pd.read_csv(
            prefix + ".depth.bed",
            sep="\t",
            names=["chrom", "start", "end", "mean", "GC", "CpG", "Masked"],
            dtype={"chrom": str},
        )
        depth_table["len"] = depth_table.end - depth_table.start

        def _mean_coverage_group(table):
            return np.sum(table["len"] * table["mean"]) / np.sum(table["len"])

        # Global mean coverage
        mean_coverage = _mean_coverage_group(depth_table)

        # Compute normalized per-chromosome coverage
        chrom_norm_covg = (depth_table.groupby("chrom").apply(_mean_coverage_group) / mean_coverage).to_dict()

        # Compute GC normalized coverage (using only the autosome)
        autosome_depth_table = depth_table[depth_table.chrom.str.contains(r"^(?:chr)?(?:\d{1,2})$", regex=True)]
        gc_bin = np.int64(autosome_depth_table.GC.round(2) * 100)

        gc_norm_covg = (autosome_depth_table.groupby(gc_bin).apply(_mean_coverage_group) / mean_coverage).to_dict()
        gc_norm_covg_count = autosome_depth_table.groupby(gc_bin).size().to_dict()

        return mean_coverage, chrom_norm_covg, gc_norm_covg, gc_norm_covg_count


def _compute_coverage_with_indexcov(read_path: str, fasta_path: str) -> dict:
    # Generate GC and chromosome normalized coverage for the entire BAM file
    with tempfile.TemporaryDirectory() as output_dir:
        prefix = os.path.basename(output_dir)

        indexcov_commandline = f"goleft \
            indexcov \
            --extranormalize \
            --directory {output_dir} \
            --fai {quote(fasta_path + '.fai')} \
            {quote(read_path + '.crai' if read_path.endswith('cram') else read_path)}"

        subprocess.check_call(
            indexcov_commandline,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Compute chromosome and GC normalized coverage
        windows_table = (
            # pylint: disable=unexpected-keyword-arg
            bed.BedTool(fn=os.path.join(output_dir, prefix + "-indexcov.bed.gz"))
            .nucleotide_content(fi=fasta_path)
            .to_dataframe(
                index_col=False,
                header=0,
                usecols=[0, 1, 2, 3, 7, 8, 10, 11, 12],
                names=[
                    "chrom",
                    "start",
                    "end",
                    "norm_covg",
                    "num_C",
                    "num_G",
                    "num_N",
                    "num_oth",
                    "seq_len",
                ],
                dtype={"chrom": str},
            )
        )
        if windows_table.shape[0] == 0:
            return {}

        # Remove windows with no alignable data
        windows_table["align_len"] = windows_table.seq_len - windows_table.num_N - windows_table.num_oth
        windows_table = windows_table[windows_table.align_len != 0]

        def norm_coverage_group(table):
            weights = table.align_len / np.sum(table.align_len)
            return np.sum(weights * table.norm_covg)

        norm_coverage_by_chrom = windows_table.groupby("chrom").apply(norm_coverage_group).to_dict()

        return norm_coverage_by_chrom


def _kmc_db_kmer_size(prefix: str) -> int | None:
    """Return the k-mer size stored in an existing KMC database, or None if unreadable."""
    pre_path = f"{prefix}.kmc_pre"
    suf_path = f"{prefix}.kmc_suf"
    if not (os.path.exists(pre_path) and os.path.exists(suf_path)):
        return None
    try:
        with open(pre_path, "rb") as f:
            if f.read(4) != b"KMCP":
                return None
            f.seek(-4, 2)
            if f.read(4) != b"KMCP":
                return None
            # KMC stores a version uint32 at -12 and a 1-byte header_offset at -8;
            # kmer_length (uint32) is the first field in the footer at -(header_offset+8).
            f.seek(-12, 2)
            (kmc_version,) = struct.unpack("<I", f.read(4))
            if kmc_version not in (0, 0x200):
                return None
            f.seek(-8, 2)
            header_offset = f.read(1)[0]
            f.seek(-(header_offset + 8), 2)
            (kmer_length,) = struct.unpack("<I", f.read(4))
            return kmer_length
    except (OSError, struct.error, IndexError):
        return None


def _find_or_create_kmc_db(
    read_path: str,
    output_prefix: str,
    reference_path: str,
    kmer_size: int = 31,
    canonicalize: bool = False,
    min_count: int = 1,
    tmp_dir: str | None = None,
    threads: int = 1,
    max_memory: int = 12,
) -> str:
    """Create a KMC k-mer database from a BAM or CRAM file.

    Args:
        read_path: Path to BAM or CRAM file.
        output_prefix: Output path prefix for the KMC database (without extension).
        reference_path: Reference FASTA used to decode CRAM. Required for CRAM input.
        kmer_size: K-mer length. Defaults to 31.
        min_count: Minimum k-mer count to retain. Defaults to 1.
        threads: Number of threads for both samtools and KMC. Defaults to 1.
        max_memory: Maximum memory in GB for KMC. Defaults to 12.

    Returns:
        str: The output_prefix, pointing to the created KMC database.
    """
    if _kmc_db_kmer_size(output_prefix) == kmer_size:
        logging.debug("Reusing existing KMC database at %s (k=%d)", output_prefix, kmer_size)
        return output_prefix

    logging.info("Building KMC k-mer database at %s (k=%d)", output_prefix, kmer_size)
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp_dir:
        if read_path.endswith(".cram"):
            # KMC does not support CRAM so convert to BAM. We don't convert to FASTQ because that requires sorting.
            bam_path = os.path.join(tmp_dir, "reads.bam")
            convert_commandline = f"samtools view -@ {threads} --bam --reference {quote(reference_path)} -o {bam_path} {quote(read_path)}"
            subprocess.check_call(
                convert_commandline,
                shell=True,
                stderr=subprocess.DEVNULL,
            )
            read_path = bam_path

        kmc_commandline = f"kmc \
            -t{threads} \
            -m{max_memory} -sm \
            -k{kmer_size} \
            {'-b' if not canonicalize else ""} \
            -ci{min_count} \
            -fbam \
            {quote(read_path)} \
            {quote(output_prefix)} \
            {quote(tmp_dir)}"
        subprocess.check_call(
            kmc_commandline,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return output_prefix

def _estimate_coverage_from_histogram(histogram: "pd.Series[int]") -> int:
    """Estimate haploid k-mer coverage from a count histogram (Sirén et al. §2.1).

    Args:
        histogram: Mapping of count -> frequency, with singletons (count=1) already excluded.

    Returns:
        float: Estimated haploid coverage.

    Raises:
        ValueError: If the coverage cannot be determined from the histogram shape.
    """
    if histogram.empty:
        raise ValueError("K-mer count histogram is empty; cannot auto-estimate coverage")

    mode_count = int(histogram.idxmax())
    mode_freq = histogram[mode_count]

    # Weighted median: smallest count where cumulative frequency >= 50% of total
    median_idx = histogram.cumsum().searchsorted(histogram.sum() // 2, side="left")
    median_count = int(histogram.index[int(median_idx)])  # Need int conversion to silence type errors

    # From Sirén et al. §2.1:
    # Let N be the most common count and m the median count. If N ≥ m, we assume that most k-mers that
    # are present in the sample are homozygous and use N as the estimate of k-mer coverage.
    if mode_count >= median_count:
        return mode_count

    # Otherwise we consider the case that most present k-mers are heterozygous. Let
    # N′ = argmax_{1.7N≤n≤2.3N}f(n) be the secondary peak near 2N. If N′ ≥ m and f(N′) ≥ 0.5f(N ),
    # we assume that N′ is the most common count of homozygous k-mers and use it as the estimate.
    search_idxs = (histogram.index >= 1.7 * mode_count) & (histogram.index <= 2.3 * mode_count)
    window = histogram[search_idxs]
    if not window.empty:
        prime_count = int(window.idxmax())
        if prime_count >= median_count and histogram[prime_count] >= 0.5 * mode_freq:
            return prime_count

    raise ValueError(
        "Cannot auto-estimate coverage from k-mer count distribution; please supply kmer_coverage explicitly"
    )


def _estimate_kmer_coverage(kmc_prefix: str, threads: int = 1) -> float:
    """Estimate haploid k-mer coverage from an existing KMC database using the Sirén et al. §2.1
    peak-finding algorithm

    Args:
        kmc_prefix (str): Path prefix for an existing KMC database (without extension).
        threads (int): Number of threads to use for kmc_tools.

    Returns:
        float: Estimated k-mer coverage.
    """
    with tempfile.NamedTemporaryFile(suffix=".hist", delete=False) as hist_file:
        hist_path = hist_file.name
    try:
        subprocess.check_call(
            f"kmc_tools -t{threads} -hp transform {quote(kmc_prefix)} histogram {quote(hist_path)}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        histogram = pd.read_csv(hist_path, sep="\t", index_col=0, names=["count", "freq"], dtype=int)["freq"]
    finally:
        os.unlink(hist_path)

    return _estimate_coverage_from_histogram(histogram)


def compute_read_stats(cfg: omegaconf.DictConfig, read_path: str, kmc_prefix: str|None = None) -> dict:
    """Compute read stats for a aligned read file

    Args:
        cfg (omegaconf.DictConfig): Configuration object.
        read_path (str): Path to the aligned read file.
        kmc_prefix (str | None, optional): Path prefix for an existing KMC database (without extension). Defaults to None.

    Returns:
        dict: Dictionary containing the computed stats.
    """    

    # Generate stats for the entire read file using goleft covstats
    logging.info("Computing coverage and insert size statistics with goleft")
    covstats_commandline = f"goleft \
        covstats \
        --fasta {quote(cfg.reference)} \
        {read_path}"

    covstats = subprocess.check_output(covstats_commandline, shell=True, universal_newlines=True)
    covstats_table = pd.read_csv(io.StringIO(covstats), sep="\t")
    (covstats_record,) = covstats_table[covstats_table.bam == read_path].to_dict("records")

    # Compute the chromosomal and GC normalized coverage. Use all the regions to get
    # more consistent depth estimates.
    logging.info("Computing normalized coverage with parallelized samtools")
    mean_coverage, chrom_norm_covg, gc_norm_covg, gc_norm_covg_count = _compute_coverage_with_samtools(
        read_path, cfg.reference, covg_regions=5000, depth_samples=5000, min_samples_per_chrom=0, threads=cfg.threads
    )

    # Construct stats dictionary that can be written to JSON
    stats = {
        "sample": covstats_record["sample"],
        "sequencer": cfg.sequencer,
        "bam": read_path,
        "read_length": covstats_record["read_length"],
        "mean_insert_size": covstats_record["template_mean"],
        "std_insert_size": covstats_record["template_sd"],
        "mean_coverage": mean_coverage,
        "chrom_normalized_coverage": chrom_norm_covg,
        "gc_normalized_coverage": gc_norm_covg,
        "gc_bin_count": gc_norm_covg_count,
    }

    if kmc_prefix is not None:
        _find_or_create_kmc_db(
            read_path,
            kmc_prefix,
            reference_path=cfg.reference,
            kmer_size=cfg.kmer.kmer_size,
            threads=cfg.threads,
        )
        stats["kmc_prefix"] = kmc_prefix
        logging.info("Estimating k-mer coverage from KMC database")
        stats["kmer_coverage"] = _estimate_kmer_coverage(kmc_prefix, threads=cfg.threads)

    return stats

def filter_kmc_database(
    genome_db_prefix: str|os.PathLike[str],
    unique_kmers,
    k: int,
    output_db_prefix: str|os.PathLike[str],
    canonicalize: bool = False,
    tmp_dir: str | None = None,
    threads: int = 1,
    max_memory: int = 12,
) -> str|os.PathLike[str]:
    """Create a KMC database containing only the subset of genome_db_prefix with unique k-mers from the graph

    Args:
        genome_db_prefix: Path prefix for the genome KMC database (without suffixes).
        unique_kmers: UniqueKmersOverlay object containing the unique k-mers.
        k: K-mer length used when building the graph unique k-mers.
        output_db_prefix: Path prefix for the filtered KMC database (without suffixes).
        tmp_dir: Directory for temporary files, Python default directory if None.
        threads: Number of threads for KMC tools. Defaults to 1.
        max_memory: Maximum memory to use for KMC tools. Defaults to 12.
    Returns:
        str|os.PathLike[str]: output_db_prefix, pointing to the filtered KMC database.
    """
    with tempfile.TemporaryDirectory(dir=tmp_dir) as work_dir:

        # Write graph unique k-mer sequences as FASTA (one entry per k-mer).
        fasta_path = os.path.join(work_dir, "graph_kmers.fa")
        unique_kmers.save_fasta(fasta_path)

        # Build a small KMC database from the FASTA.
        graph_db = os.path.join(work_dir, "graph_kmers")
        kmc_tmp = os.path.join(work_dir, "graph_kmers_tmp")
        os.makedirs(kmc_tmp)

        subprocess.check_call(
            f"kmc -t{threads} -m{max_memory} -sm -k{k} {'-b' if not canonicalize else ''} -ci0 -cs65535 -fa {quote(fasta_path)} {quote(graph_db)} {quote(kmc_tmp)}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Intersect with the genome database, keeping database counts
        os.makedirs(os.path.dirname(output_db_prefix), exist_ok=True)
        subprocess.check_call(
            f"kmc_tools -t{threads} -hp simple {quote(str(genome_db_prefix))} {quote(str(graph_db))} intersect {quote(str(output_db_prefix))} -ocleft",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return output_db_prefix
