import os
import subprocess
import tempfile
import typing
from shlex import quote

import pysam

from npsv3.util.range import Range
from npsv3.util.sample import Sample


def haplotag_reads(reference: str, sample: Sample, read_path: str, vcf_path: str, region: Range, dir) -> str:
    tagged_bam = tempfile.NamedTemporaryFile(delete=False, suffix=".bam", dir=dir)
    tagged_bam.close()

    whatshap_commandline = f"whatshap haplotag \
        --tag-supplementary \
        --reference {quote(reference)} \
        --regions {region} \
        --sample {sample.name} \
        --output {quote(tagged_bam.name)} \
        {quote(vcf_path)} \
        {quote(read_path)}"

    haplotag_result = subprocess.run(whatshap_commandline, shell=True, stderr=subprocess.PIPE, check=False)
    if haplotag_result.returncode != 0 or not os.path.exists(tagged_bam.name):
        raise RuntimeError("Failed to haplotag read file")
    pysam.index(tagged_bam.name)
    return tagged_bam.name


def downsample_reads(read_path: str, region: Range, dir, downsample: float = 1.0) -> str:
    downsampled_bam = tempfile.NamedTemporaryFile(delete=False, suffix=".bam", dir=dir)
    downsampled_bam.close()

    samtools_commandline = (
        f"samtools view -b -o {quote(downsampled_bam.name)} -s {downsample} {quote(read_path)} {region}"
    )
    samtools_result = subprocess.run(samtools_commandline, shell=True, stderr=subprocess.PIPE, check=False)
    if samtools_result.returncode != 0 or not os.path.exists(downsampled_bam.name):
        raise RuntimeError("Failed to downsample read file")
    pysam.index(downsampled_bam.name)
    return downsampled_bam.name
