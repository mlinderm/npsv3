import glob
import os
from typing import TYPE_CHECKING

import pytest

from npsv3.util.config import setup_resolvers

if TYPE_CHECKING:
    from npsv3.util.range import Range
    from npsv3.util.sample import Sample


RESOURCES_DIR = "/storage/mlinderman/projects/sv/npsv3-experiments/resources"
EXPERIMENTS_DIR = "/storage/mlinderman/projects/sv/npsv3-experiments"

def _first_existing(*paths: str) -> str|None:
    """Return the first path that exists, or None if none exist."""
    for path in paths:
        if os.path.exists(path):
            return path
    return None


B37_REF_FASTA = _first_existing(
    "/data/human_g1k_v37.fasta",
    os.path.join(RESOURCES_DIR, "human_g1k_v37.fasta"),
    "/storage/mlinderman/projects/sv/npsv2-experiments/resources/human_g1k_v37.fasta",
)

HG38_REF_FASTA = _first_existing(
    "/data/Homo_sapiens_assembly38.fasta",
    os.path.join(RESOURCES_DIR, "Homo_sapiens_assembly38.fasta"),
    "/storage/mlinderman/projects/sv/npsv2-experiments/resources/Homo_sapiens_assembly38.fasta",
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")

HG00731_VCF = _first_existing(
    "/data/HG00731.freeze4.alt.passing.training.hg38.vcf.gz",
    "/storage/mlinderman/projects/sv/npsv3-experiments/training/training_vcfs/HG00731.freeze4.alt.passing.training.hg38.vcf.gz",
)

HG00731_SV_VCF = _first_existing(
    "/data/HG00731.freeze4.sv.alt.passing.training.hg38.vcf.gz",
    "/storage/mlinderman/projects/sv/npsv3-experiments/training/training_vcfs/HG00731.freeze4.sv.alt.passing.training.hg38.vcf.gz",
)

SYNDIP_VCF = _first_existing(os.path.join(RESOURCES_DIR, "syndip.genotyped.passing.b37.vcf.gz"))
SYNDIP_SV_VCF = _first_existing(os.path.join(RESOURCES_DIR, "syndip.sv.genotyped.passing.b37.vcf.gz"))
SYNDIP_BAM = _first_existing(os.path.join(RESOURCES_DIR, "sequence/CHM1_CHM13_2.b37.bam"))

HG002_GIAB_VCF = _first_existing(os.path.join(RESOURCES_DIR, "HG002.ashkenazi-gatk-haplotype-annotated.phased.vcf.gz"))
HG002_GIAB_SV_VCF = _first_existing(os.path.join(RESOURCES_DIR, "HG002_SVs_Tier1_v0.6.genotyped.passing.tier1and2.b37.vcf.gz"))
HG002_B37_BAM = _first_existing(os.path.join(RESOURCES_DIR, "sequence/HG002-ready.b37.bam"))

HG002_DIPCALL_VCF = _first_existing(os.path.join(RESOURCES_DIR, "hg002v1.1.dipcall.passing.hg38.vcf.gz"))
HG002_DIPCALL_SV_VCF = _first_existing(os.path.join(RESOURCES_DIR, "hg002v1.1.dipcall.passing.sv.hg38.vcf.gz"))
HG002_HG38_BAM = _first_existing(os.path.join(RESOURCES_DIR, "sequence/HG002-ready.hg38.bam"))

NA12878_VCF = _first_existing(os.path.join(RESOURCES_DIR, "NA12878.freeze4.alt.passing.hg38.vcf.gz"))
NA12878_SV_VCF = _first_existing(os.path.join(RESOURCES_DIR, "NA12878.freeze4.sv.alt.passing.hg38.vcf.gz"))
NA12878_BAM = _first_existing(os.path.join(RESOURCES_DIR, "sequence/NA12878.final.cram"))

HG00733_DIPCALL_VCF = _first_existing(os.path.join(RESOURCES_DIR, "HG00733.hgsvc3-hprc-2024-02-23.dipcall.passing.hg38.vcf.gz"))
HG00733_DIPCALL_SV_VCF = _first_existing(os.path.join(RESOURCES_DIR, "HG00733.hgsvc3-hprc-2024-02-23.dipcall.passing.sv.hg38.vcf.gz"))
HG00733_HG38_BAM = _first_existing(os.path.join(RESOURCES_DIR, "sequence/HG00733.final.cram"))

HG00731_TRAINING_VCF = _first_existing(os.path.join(EXPERIMENTS_DIR, "training", "training_vcfs", "HG00731.hgsvc3-hprc-2024-02-23-mc-chm13.GRCh38.vcfbub.a100k.wave.passing.training.hg38.vcf.gz"))
HG00731_TRAINING_SV_VCF = _first_existing(os.path.join(EXPERIMENTS_DIR, "training", "training_vcfs", "HG00731.hgsvc3-hprc-2024-02-23-mc-chm13.GRCh38.vcfbub.a100k.wave.sv.passing.training.hg38.vcf.gz"))
HG00731_HG38_BAM = _first_existing(os.path.join(RESOURCES_DIR, "sequence/HG00731.final.cram"))

def data_path(path: str) -> str:
    """Return path to file in the test data directory"""
    return os.path.join(DATA_DIR, path)


def result_path(path: str) -> str:
    """Return path to file in the test results directory, creating the directory if it doesn't exist"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    return os.path.join(RESULT_DIR, path)


def create_vcf(tmp_path: str, vcf: bytes, name: str = "test.vcf.gz") -> str:
    """Create and return path to bgzip-compressed VCF file at tmp_path/name containing vcf"""
    from pysam import BGZFile

    from npsv3.util.vcf import index_variant_file

    vcf_path = os.path.join(tmp_path, name)
    assert vcf_path.endswith(".vcf.gz"), "VCF path must end with .vcf.gz"
    with BGZFile(vcf_path, "wb", index=None) as vcf_file:
        vcf_file.write(vcf)
    index_variant_file(vcf_path)
    return vcf_path

# Register resolvers for OmegaConf
setup_resolvers()

def hash_vcf_file(vcf_path: str) -> str:
    """Return the sha256 hash of the VCF file contents."""
    import hashlib
    with open(vcf_path, "rb") as f:
        digest = hashlib.file_digest(f, "sha256")
        return digest.hexdigest()


def cache_filter_kmc_database(cfg, sample: "Sample", vcf_path: str, region: "Range", unique_kmers, tmp_path) -> str:
    """Compute and save filtered kmers in the results directory using a hash of the VCF file to avoid collisions.

    Args:
        cfg (_type_): Hydra configuration
        sample (Sample): Sample object containing with KMC database prefix
        vcf_path (str): VCF file path to use for hashing
        region (Range): Region object to use for hashing
        unique_kmers (_type_): Unique k-mers to filter the KMC database
        tmp_path (_type_): Temporary path to use for intermediate files

    Returns:
        str: Filtered KMC database prefix
    """
    # Incorporate hash into custom VCF file into cache directory to avoid collisions
    results_directory = result_path(f"{region.slug}.{hash_vcf_file(vcf_path)}.{sample.name}.k{cfg.graph.kmer_size}")
    os.makedirs(results_directory, exist_ok=True)
    filtered_kmer_path = os.path.join(results_directory, "filtered_kmers")
    if not os.path.exists(filtered_kmer_path + ".kmc_pre"):
        # Generate filtered k-mers if not already available (a slow step, so we cache the results)
        from npsv3.util.sample import filter_kmers_by_unique_kmers
        if not sample.kmc_prefix:
            pytest.skip(f"KMC database for {sample.name} not found")
            return
        filter_kmers_by_unique_kmers(
            sample.kmc_prefix,
            unique_kmers,
            cfg.graph.kmer_size,
            filtered_kmer_path,
            tmp_dir=tmp_path,
            threads=cfg.threads,
        )

    return filtered_kmer_path

def cache_graph_and_filter_kmc_database(cfg, sample: "Sample", vcf_path: str, region: "Range"):
    # Incorporate hash into custom VCF file into cache directory to avoid collisions
    results_directory = result_path(f"{region.slug}.{hash_vcf_file(vcf_path)}.{sample.name}.k{cfg.graph.kmer_size}")
    os.makedirs(results_directory, exist_ok=True)
    filtered_kmer_path = os.path.join(results_directory, "filtered_kmers")
    graph_shards = glob.glob(os.path.join(results_directory, "graphs-*-*.tar.gz"))
    if not graph_shards or not os.path.exists(f"{filtered_kmer_path}.kmc_pre"):
        if not sample.kmc_prefix:
            pytest.skip(f"KMC database for {sample.name} not found")
            return
        from npsv3.graphs.haplotype import serialize_graph_and_unique_kmers
        from npsv3.util.sample import kmc_filter
        graph_shards, unique_kmer_path, _region_count = serialize_graph_and_unique_kmers(
            cfg,
            cfg.input,
            ref_kmer_counts_path=cfg.graph.ref_kmer_counts_kmc_prefix,
            output_dir=results_directory,
            pool_kmers=True,
            region=region,
        )
        kmc_filter(sample.kmc_prefix, unique_kmer_path, filtered_kmer_path, threads=cfg.threads)  # type: ignore

    return graph_shards, filtered_kmer_path
