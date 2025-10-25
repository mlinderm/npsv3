import numpy as np
import pysam
from scipy.special import logsumexp

from npsv3._native_realign import FragmentRealigner
from npsv3.pileup import AlleleAssignment, AlleleRealignment, Fragment


def _quality_string(read: pysam.AlignedSegment) -> str:
    return "".join([chr(c) for c in read.query_qualities])


def _realignment_assignment(ref_quality, alt_quality, assign_delta) -> AlleleAssignment:
    delta = alt_quality - ref_quality
    if delta > assign_delta:
        return AlleleAssignment.ALT
    if delta < -assign_delta:
        return AlleleAssignment.REF
    return AlleleAssignment.AMB


def _read_realignment(scores, assign_delta) -> AlleleRealignment:
    # Convert the read scores to relative phred-scaled qualities
    with np.errstate(divide="ignore"):
        qualities = np.clip(np.log10(1 - np.power(10.0, np.array(scores) - logsumexp(scores))) * -10.0, 0.0, 40.0)
    return AlleleRealignment(*qualities, _realignment_assignment(*qualities, assign_delta))


def realign_fragment(realigner: FragmentRealigner, fragment: Fragment, assign_delta=1):
    name = fragment.query_name
    read1_seq = fragment.read1.query_sequence
    read1_qual = _quality_string(fragment.read1)

    kw = {"offset": 0}  # Conversion already performed by pySAM
    if fragment.read2:
        kw["read2_seq"] = fragment.read2.query_sequence
        kw["read2_qual"] = _quality_string(fragment.read2)

    ref_quality, _, alt_quality, _, read_scores = realigner.realign_read_pair(name, read1_seq, read1_qual, **kw)
    #import pytest; pytest.set_trace()
    assign = _realignment_assignment(ref_quality, alt_quality, assign_delta=assign_delta)

    # Compute read allele assignment to facilitate strand bias analysis
    return (
        AlleleRealignment(ref_quality, alt_quality, assign),
        _read_realignment(read_scores[0::2], assign_delta),
        _read_realignment(read_scores[1::2], assign_delta),
    )
