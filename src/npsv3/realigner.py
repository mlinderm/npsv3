import numpy as np
import pysam

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
    # Convert the read scores (base-10 log probabilities) to relative phred-scaled qualities
    with np.errstate(divide="ignore"):
        scores = np.array(scores)
        normalizer = np.log10(np.sum(np.power(10.0, scores)))
        qualities = np.clip(np.log10(1 - np.power(10.0, scores - normalizer)) * -10.0, 0.0, 40.0)
    return AlleleRealignment(*qualities, _realignment_assignment(*qualities, assign_delta))


def realign_fragment(realigner: FragmentRealigner, fragment: Fragment, assign_delta=1, cache: dict | None = None):
    read1_seq = fragment.read1.query_sequence
    read1_qual = _quality_string(fragment.read1)
    read2_seq = fragment.read2.query_sequence if fragment.read2 else None
    read2_qual = _quality_string(fragment.read2) if fragment.read2 else None

    # Identical (fragment sequence, quality) pairs realign identically regardless of read name -- e.g. the same
    # simulated haplotype read set is reused, unmodified, across the several genotypes that share that haplotype.
    cache_key = (read1_seq, read1_qual, read2_seq, read2_qual) if cache is not None else None
    if cache_key is not None and cache_key in cache:
        return cache[cache_key]

    name = fragment.query_name
    kw = {"offset": 0}  # Conversion already performed by pySAM
    if fragment.read2:
        kw["read2_seq"] = read2_seq
        kw["read2_qual"] = read2_qual

    ref_quality, _, alt_quality, _, read_scores = realigner.realign_read_pair(name, read1_seq, read1_qual, **kw)
    assign = _realignment_assignment(ref_quality, alt_quality, assign_delta=assign_delta)

    # Compute read allele assignment to facilitate strand bias analysis
    result = (
        AlleleRealignment(ref_quality, alt_quality, assign),
        _read_realignment(read_scores[0::2], assign_delta),
        _read_realignment(read_scores[1::2], assign_delta),
    )
    if cache_key is not None:
        cache[cache_key] = result
    return result
