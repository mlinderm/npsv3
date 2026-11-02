import json
import os
from unittest import mock

import pysam
import pytest

from npsv3 import simulation
from npsv3.simulation import bwa_index_loaded, simulate_variant_sequencing, split_bam_by_tag
from npsv3.util.sample import Sample

from . import HG38_BWA_MEM3_INDEX, HG38_REF_FASTA


class TestSharedMemoryIndex:
    @pytest.fixture
    def lock_file(self, tmp_path):
        return str(tmp_path / "bwa.lock")

    def test_index_names_match_exactly(self):
        # bwa and bwa-mem3 share a registry, so a bwa-mem3 index whose name extends the bwa index's name must not match
        listing = "Homo_sapiens_assembly38.fasta.bwa-mem3\t11261248768\n"
        with mock.patch("subprocess.check_output", return_value=listing) as mock_check_output:
            assert not simulation._is_bwa_index_loaded("Homo_sapiens_assembly38.fasta", "bwa-mem3")
            assert simulation._is_bwa_index_loaded("Homo_sapiens_assembly38.fasta.bwa-mem3", "bwa-mem3")
        assert mock_check_output.call_args.args[0] == "bwa-mem3 shm -l"

    def test_unsupported_aligner(self):
        with pytest.raises(ValueError, match="novo"):
            bwa_index_loaded("/data/test.fasta", aligner="novo")

    @pytest.mark.parametrize("aligner", ["bwa-mem3", "bwa"])
    def test_load_and_unload(self, lock_file, aligner):
        with (
            mock.patch("npsv3.simulation._is_bwa_index_loaded", return_value=False),
            mock.patch("subprocess.run") as mock_run,
            mock.patch("atexit.register") as mock_register,
        ):
            assert simulation._bwa_index_load("/data/test.fasta", lock_file=lock_file, aligner=aligner) == "test.fasta"
            mock_run.assert_called_with(f"{aligner} shm /data/test.fasta", shell=True, check=True)
            mock_register.assert_called_once_with(simulation._bwa_index_unload, "test.fasta", lock_file, aligner)
            with open(lock_file) as file:
                assert json.load(file) == {"test.fasta": 1}

            simulation._bwa_index_unload("test.fasta", lock_file, aligner)
            mock_run.assert_called_with(f"{aligner} shm -d", shell=True, check=True)
            assert not os.path.exists(lock_file)

    def test_unload_preserves_index_in_use_by_npsv2(self, lock_file):
        # `shm -d` drops every index (bwa and bwa-mem3), so it must not be run while npsv2 has its bwa index in use
        with open(lock_file, "w") as file:
            json.dump({"Homo_sapiens_assembly38.fasta": 1, "Homo_sapiens_assembly38_bwa3.fasta": 1}, file)

        with mock.patch("subprocess.run") as mock_run:
            simulation._bwa_index_unload("Homo_sapiens_assembly38_bwa3.fasta", lock_file, "bwa-mem3")
        mock_run.assert_not_called()
        with open(lock_file) as file:
            assert json.load(file) == {"Homo_sapiens_assembly38.fasta": 1, "Homo_sapiens_assembly38_bwa3.fasta": 0}


@pytest.mark.skipif(
    not HG38_REF_FASTA or not HG38_BWA_MEM3_INDEX or not bwa_index_loaded(HG38_BWA_MEM3_INDEX),
    reason="HG38 bwa-mem3 index in SHM required",
)
class TestSimulateVariantSequencing:
    @pytest.fixture
    def sample(self):
        return Sample(
            "synth", sequencer="HSXn", read_length=150, mean_coverage=20, mean_insert_size=454.57, std_insert_size=106.63
        )

    @pytest.fixture
    def fasta_path(self, tmp_path):
        """Reference haplotypes for an unremarkable region, named as for the haplotypes of allele 0"""
        with pysam.FastaFile(HG38_REF_FASTA) as reference:
            sequence = reference.fetch("chr5", 109362000, 109368000)
        path = tmp_path / "simulation.fasta"
        path.write_text("".join(f">seq-0-{i}\n{sequence}\n" for i in range(2)))
        return str(path)

    @pytest.mark.parametrize("aligner_preset", ["bcbio", "nygc"])
    def test_simulate_and_split(self, tmp_path, sample, fasta_path, aligner_preset):
        bam_path = simulate_variant_sequencing(
            fasta_path,
            10.0,
            sample,
            reference=HG38_REF_FASTA,
            dir=str(tmp_path),
            aligner="bwa-mem3",
            aligner_index=HG38_BWA_MEM3_INDEX,
            aligner_preset=aligner_preset,
        )
        assert os.path.exists(f"{bam_path}.bai")

        with pysam.AlignmentFile(bam_path) as bam:
            assert bam.header["HD"]["SO"] == "coordinate"
            reads = list(bam)
        assert reads, "Expected simulated reads"
        for read in reads:
            haplotype = read.query_name.split("-")[2]
            assert read.get_tag("SQ") == f"0_{haplotype}"
            assert read.get_tag("HP") == (int(haplotype) % 2) + 1
        # The NYGC preset (-Y) marks split alignments as supplementary, never secondary
        if aligner_preset == "nygc":
            assert not any(read.is_secondary for read in reads)

        splits = split_bam_by_tag(bam_path, tmp_path, tag="SQ")
        assert set(splits) == {"0_0", "0_1"}
