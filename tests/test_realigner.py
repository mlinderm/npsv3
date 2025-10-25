import os
import tempfile
from collections import Counter, defaultdict

import pysam
import pytest

from npsv3.pileup import FragmentTracker
from npsv3.realigner import (
    AlleleAssignment,
    FragmentRealigner,
    realign_fragment,
)
from npsv3._native_realign import test_realign_read_pair as realign_read_pair, test_score_alignment as score_alignment

from . import data_path


class TestRealigner:
    def test_alignment_scoring(self):
        try:
            # Create SAM file with a single read
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".sam") as sam_file:
                # fmt: off
                print("@HD", "VN:1.3", "SO:coordinate", sep="\t", file=sam_file)
                print("@SQ", "SN:1", "LN:249250621", sep="\t", file=sam_file)
                print("@RG", "ID:synth1", "LB:synth1", "PL:illumina", "PU:ART", "SM:HG002", sep="\t", file=sam_file)
                print(
                    "ref-354", "147", "1", "1", "60", "148M", "=", "2073433", "-435",
                    "AGCAGCCGAAGCGCCTCCTTTCACTCTAGGGTCCAGGCATCCAGCAGCCGAAGCGCCTCCTTTCAATCCAGGGTCCACACATCCAGCAGCCGAAGCGCCCTCCTTTCAATCCAGGGTCCAGGCATCTAGCAGCCGAAGCGCCTCCTTT",
                    "GG8CCGGGCGGGCGGGGGCGGCGGGGGGGGGGGCGGCGGGG=GGGJCCJGGGGGGCGGGGGGCG1GGCGG8GGCGC1GGCGJGCCGGJGJGJGGCGCJGJGJJCGGJJCJJGJJGJJJGJGCJJJGGJJJJGJJJGGGCGGGCGGCCC",
                    "RG:Z:synth1",
                    sep="\t",
                    file=sam_file,
                )
                # fmt: on

            # Read was aligned to very beginning of reference, so using read as reference should be all matches
            ref_sequence = "AGCAGCCGAAGCGCCTCCTTTCACTCTAGGGTCCAGGCATCCAGCAGCCGAAGCGCCTCCTTTCAATCCAGGGTCCACACATCCAGCAGCCGAAGCGCCCTCCTTTCAATCCAGGGTCCAGGCATCTAGCAGCCGAAGCGCCTCCTTT"
            scores = score_alignment(ref_sequence, sam_file.name)
            assert len(scores) == 1
            assert -9 < scores[0] < -8
        finally:
            os.remove(sam_file.name)

    def test_realign_read_pair(self, tmp_path, hg002_sample):
        # FASTA has a 3000bp flank
        fasta_path = data_path("1_899922_899992.fasta")

        header = pysam.AlignmentHeader.from_dict(
            {
                "HD": {"VN": "1.5", "SO": "coordinate"},
                "SQ": [{"SN": "1", "LN": 249250621}],
            }
        )

        read1 = pysam.AlignedSegment.fromstring(
            "HISEQ1:18:H8VC6ADXX:1:2103:1867:53768	163	1	899944	60	67M1I47M33S	=	900366	570	GTCCGCAGTGGGGCTGTGGGAGGGGTCCGCGCGTCCGCAGTGGGGCTGTGCTGCGGGAAGGGGGGGGCCGGGCCCGCAGTGGGGATGTGCTGCCGGGAGGGGGGCGCGGGTCCGCGGGGGGGCGGGGCCGCCGGCGGGGGGGCGCGGG	CCCFFFFFHFHHGHIIIJIJGEHIIHIGIGIDGBEEBC/>;>C?:;@B<CA@C37@B6&8?BDBDD@505;@&50&0)9(+9?>(08(4:(4++055&005)05.&0)&5058&&)&&&)93&&&&&)&&)0&)&&)&&&&&&&&)&)",
            header=header,
        )
        read2 = pysam.AlignedSegment.fromstring(
            "HISEQ1:18:H8VC6ADXX:1:2103:1867:53768	83	1	900366	60	148M	=	899944	-570	TGGACGGATGGTTGTACGCCGTGGGGGGTAACGACGGTAGCTCCAGCCTCAACTCCATCGAGAAGTACAACCCGAGGACCAACAAGTGGGTGGCCGCATCCTGCATGTTCACCCGGCGCAGCAGTGTGGGTGTGGCGGTGCTGGAGCT	8>825AA:@DC@?950555B99@DDDDDDDDDB@BCDDBCBBDDDBCDDDCDDCDEDCDDDDEEEDDDDDDDBDCCC?DDDDDDDDDDDDBDDBBDDEDEFFFHHHHFHHBJJJJJIJJJJJJJJJJJJJJJJJJHHHHHFFFFFC@C",
            header=header,
        )

        read1_seq = read1.query_sequence
        read1_qual = "".join([chr(c) for c in read1.query_qualities])
        assert len(read1_seq) == len(read1_qual)

        read2_seq = read2.query_sequence
        read2_qual = "".join([chr(c) for c in read2.query_qualities])
        assert len(read2_seq) == len(read2_qual)

        results = realign_read_pair(
            fasta_path,
            read1.query_name,
            read1_seq,
            read1_qual,
            read2_seq=read2_seq,
            read2_qual=read2_qual,
            fragment_mean=hg002_sample.mean_insert_size,
            fragment_sd=hg002_sample.std_insert_size,
            offset=0,  # Conversion already performed by pySAM
            alt_alignment_paths=[str(tmp_path / "realignment.bam")],
        )

        assert os.path.exists(tmp_path / "realignment.bam")

        read_scores = results[4]
        assert read_scores[0] > read_scores[2], "Read 1 should be assigned to the reference allele"
        assert read_scores[1] == pytest.approx(read_scores[3]), "Read 2 is equally aligned to both alleles"

    def test_realign_reads(self, hg002_sample):
        fasta_path = data_path("1_899922_899992.fasta")
        bam_path = data_path("1_896922_902998.bam")

        fragments = FragmentTracker()
        with pysam.AlignmentFile(bam_path, "rb") as bam_file:
            for read in bam_file:
                if (
                    read.is_duplicate
                    or read.is_qcfail
                    or read.is_unmapped
                    or read.is_secondary
                    or read.is_supplementary
                ):
                    # TODO: Potentially recover secondary/supplementary alignments if primary is outside pileup region
                    continue
                fragments.add_read(read)

        realigner = FragmentRealigner(fasta_path, hg002_sample.mean_insert_size, hg002_sample.std_insert_size)

        allele_counts = Counter()
        allele_strand = defaultdict(int)
        for fragment in fragments:
            realignment, read1_realignment, read2_realignment = realign_fragment(realigner, fragment, assign_delta=1.0)
            allele_counts[realignment.allele] += 1

            allele_strand[(read1_realignment.allele, fragment.read1.is_forward)] += 1
            if fragment.read2:
                allele_strand[(read2_realignment.allele, fragment.read2.is_forward)] += 1

        assert allele_counts[AlleleAssignment.REF] == 12
        assert allele_counts[AlleleAssignment.ALT] == 8

        # Example strand bias test...
        from scipy.stats import fisher_exact

        strand_bias = [
            [allele_strand[(AlleleAssignment.REF, True)], allele_strand[(AlleleAssignment.REF, False)]],
            [allele_strand[(AlleleAssignment.ALT, True)], allele_strand[(AlleleAssignment.ALT, False)]],
        ]
        odds, p = fisher_exact(strand_bias)
        assert p > 0.05
