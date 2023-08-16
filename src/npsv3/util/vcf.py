from pysam import bcftools


def index_variant_file(filename: str):
    """Index variant file"""

    # There is a file handle leak in pysam.tabix_index, so we use bcftools index directly
    if filename.endswith("vcf.gz"):
        bcftools.index("-t", filename, catch_stdout=False)
    elif filename.endswith(".bcf"):
        bcftools.index("-c", filename, catch_stdout=False)
