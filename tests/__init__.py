# SPDX-FileCopyrightText: 2023-present Michael Linderman <mlinderman@middlebury.edu>
#
# SPDX-License-Identifier: MIT
import os

B37_REF_FASTA = "/data/human_g1k_v37.fasta"
HG38_REF_FASTA = "/data/Homo_sapiens_assembly38.fasta"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")


def data_path(path: str) -> str:
    return os.path.join(DATA_DIR, path)


def result_path(path: str) -> str:
    os.makedirs(RESULT_DIR, exist_ok=True)
    return os.path.join(RESULT_DIR, path)
