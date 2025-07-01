# SPDX-FileCopyrightText: 2023-present Michael Linderman <mlinderman@middlebury.edu>
#
# SPDX-License-Identifier: MIT
import os

from omegaconf import OmegaConf

RESOURCES_DIR = "/storage/mlinderman/projects/sv/npsv3-experiments/resources"


def _first_existing(*paths):
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

HG002_DIPCALL_VCF = _first_existing(os.path.join(RESOURCES_DIR, "hg002v1.1.dipcall.passing.hg38.vcf.gz"))
HG002_DIPCALL_SV_VCF = _first_existing(os.path.join(RESOURCES_DIR, "hg002v1.1.dipcall.passing.sv.hg38.vcf.gz"))

def data_path(path: str) -> str:
    return os.path.join(DATA_DIR, path)


def result_path(path: str) -> str:
    os.makedirs(RESULT_DIR, exist_ok=True)
    return os.path.join(RESULT_DIR, path)


# Match resolvers available in main.py
OmegaConf.register_new_resolver("len", lambda arg: len(arg))
