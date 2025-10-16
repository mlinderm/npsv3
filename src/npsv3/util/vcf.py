import typing

from pysam import bcftools


def index_variant_file(filename: str):
    """Index variant file"""
    # There is a file handle leak in pysam.tabix_index, so we use bcftools index directly
    if filename.endswith("vcf.gz"):
        bcftools.index("-t", filename, catch_stdout=False)
    elif filename.endswith(".bcf"):
        bcftools.index("-c", filename, catch_stdout=False)

def bcftools_format(filename: str) -> str:
    """Return bcftools type string based on filename extensions, e.g. 'vcf.gz' -> 'z'"""
    if filename.endswith("vcf.gz"):
        return "z"
    if filename.endswith(".bcf"):
        return "b"
    return "v"

def bcftools_index(filename: str) -> tuple[str]:
    """"Return bcftools index arguments based on filename extensions, e.g. 'vcf.gz' -> ('-Wtbi',)"""
    if filename.endswith("vcf.gz"):
        return ("-Wtbi",)
    if filename.endswith(".bcf"):
        return ("-Wcsi",)
    return ()

def pysam_write_mode(filename: str|typing.TextIO) -> str:
    """Return pysam write mode based on filename extensions, e.g. 'vcf.gz' -> 'wz'"""
    if isinstance(filename, str):
        if filename.endswith("vcf.gz"):
            return "wz"
        if filename.endswith(".bcf"):
            return "wb"
    return "w"  # uncompressed VCF or writing to a file-like object
