# SPDX-FileCopyrightText: 2023-present Michael Linderman <mlinderman@middlebury.edu>
#
# SPDX-License-Identifier: MIT
import os

def _first_existing(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None

B37_REF_FASTA = "/data/human_g1k_v37.fasta"
for alt_ref in (
    "/storage/mlinderman/projects/sv/npsv3-experiments/resources/human_g1k_v37.fasta",
    "/storage/mlinderman/projects/sv/npsv2-experiments/resources/human_g1k_v37.fasta",
):
    if not os.path.exists(B37_REF_FASTA) and os.path.exists(alt_ref):
        B37_REF_FASTA = alt_ref
        break

HG38_REF_FASTA = "/data/Homo_sapiens_assembly38.fasta"
for alt_ref in (
    "/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.fasta",
    "/storage/mlinderman/projects/sv/npsv2-experiments/resources/Homo_sapiens_assembly38.fasta",
):
    if not os.path.exists(HG38_REF_FASTA) and os.path.exists(alt_ref):
        HG38_REF_FASTA = alt_ref
        break

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")

HG00731_VCF = _first_existing(
    "/data/HG00731.freeze4.alt.passing.training.hg38.vcf.gz",
    "/storage/mlinderman/projects/sv/npsv3-experiments/training/HGSVC2_training_vcfs/HG00731.freeze4.alt.passing.training.hg38.vcf.gz",
)

HG00731_SV_VCF = _first_existing(
    "/data/HG00731.freeze4.sv.alt.passing.training.hg38.vcf.gz",
    "/storage/mlinderman/projects/sv/npsv3-experiments/training/HGSVC2_training_vcfs/HG00731.freeze4.sv.alt.passing.training.hg38.vcf.gz",
)


def data_path(path: str) -> str:
    return os.path.join(DATA_DIR, path)


def result_path(path: str) -> str:
    os.makedirs(RESULT_DIR, exist_ok=True)
    return os.path.join(RESULT_DIR, path)
