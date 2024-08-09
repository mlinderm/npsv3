"""Construct a graph from a VCF, treating it as a sorted DAG

Replaces VG construct command. The resulting graph has explicit zero-length nodes for deletion alternate alleles
and insertion reference alleles. These nodes don't change the genomic sequences, but facilitate haplotype
generation.

# TODO: Replace VG for translating VCF to sample path
"""

import itertools
import os
import subprocess
import sys
import tempfile
from bisect import bisect_left
from collections import defaultdict
from collections.abc import MutableSequence
from contextlib import nullcontext
from dataclasses import dataclass
from shlex import quote
from typing import Optional, TextIO, Tuple, Union

import pysam

from npsv3.util.range import Range
from npsv3.variant import Variant

# We would like to use the interval tree library, but it doesn't support "null" intervals,
# i.e., with zero size, which we need for insertions. So instead we use a sorted list as the core
# data structure and binary search.


class GraphConstructor:
    def __init__(self, region: Range, graph_vcf: str):
        """Construct a graph from graph_vcf variants in region. Assumes VCF is indexed and in sorted order."""
        self.region = region

        # Start with a single span for the entire reference region (i.e. no variation)
        self.spans: MutableSequence[ReferenceSpan] = [ReferenceSpan(region)]
        self.spans[0].names.add(self.region.contig)

        self._construct_from_vcf(graph_vcf)

    def _construct_from_vcf(self, vcf_path: str):
        with pysam.VariantFile(vcf_path, drop_samples=True) as vcf_file:
            for record in vcf_file.fetch(**self.region.pysam_fetch):
                variant = Variant.from_pysam(record)
                for allele_idx, allele_len in enumerate(variant.length_change(allele=None), start=1):
                    if allele_len is not None: # Ignore * alleles
                        # Find and split the corresponding span
                        alt_region = variant.alt_reference_region(allele_idx)
                        alt = AltPath(
                            variant_path_name(variant.vg_variant_id, allele_idx),
                            alt_region.end,
                            variant.alt_seq(allele_idx),
                        )
                        self._split_spans(alt_region, variant.vg_variant_id, alt)

    @property
    def num_spans(self) -> int:
        return len(self.spans)

    def get_span_region(self, idx: int) -> Range:
        return self.spans[idx].region

    def find_overlapping_spans(self, region: Range) -> tuple[int, int]:
        """Return inclusive indices the spans overlapping region"""
        assert region.contig == self.region.contig
        if len(region) == 0:  # A "region" must match or be between spans
            start_idx = bisect_left(self.spans, region.start, key=span_between_point_key)
            return (start_idx, start_idx)
        else:
            # TODO: Linear search likely faster for end coordinate
            start_idx = bisect_left(self.spans, region.start, key=span_start_point_key)
            end_idx = bisect_left(self.spans, region.end, key=span_end_point_key)
            return (start_idx, end_idx)

    def find_target_span(self, target_start: int) -> int:
        """Find the leftmost span that starts at the target position, including any null regions"""
        return bisect_left(self.spans, target_start, key=span_between_point_key)

    def to_gfa(self, ref_fasta: str, out_file: Union[str, TextIO] = sys.stdout):
        ref_seq = _reference_sequence(ref_fasta, self.region)

        def get_ref_seq(region: Range):
            return (
                "*" if len(region) == 0 else ref_seq[region.start - self.region.start : region.end - self.region.start]
            )

        paths = defaultdict(list)

        with open(out_file, "w") if isinstance(out_file, str) else nullcontext(out_file) as gfa_file:
            print("H", "VN:Z:1.0", sep="\t", file=gfa_file)
            print(f"# Region: {self.region}", file=gfa_file)
            # Assign node ids to spans and each span's alternate alleles, associating those nodes with paths
            node_id_gen = itertools.count(1)
            for span in self.spans:
                span.node_id = next(node_id_gen)
                for name in span.names:
                    paths[name].append(span.node_id)

                for alt in span.alts:
                    alt.node_id = next(node_id_gen)
                    paths[alt.name].append(alt.node_id)

            # Emit nodes from spans and their alternate alleles
            for span in self.spans:
                print("S", span.node_id, get_ref_seq(span.region), sep="\t", file=gfa_file)
                for alt in span.alts:
                    print("S", alt.node_id, alt.sequence or "*", sep="\t", file=gfa_file)

            assert len(self.spans[0].alts) == 0 and len(self.spans[-1].alts) == 0, "Graph does not form a bubble"
            # Link up nodes (the last node shouldn't have any outgoing edges)
            for i, span in enumerate(self.spans[:-1]):
                next_span = self.spans[i + 1]
                # Link to the next span and its alternate nodes
                print("L", span.node_id, "+", next_span.node_id, "+", "0M", sep="\t", file=gfa_file)
                for next_alt in next_span.alts:
                    print("L", span.node_id, "+", next_alt.node_id, "+", "0M", sep="\t", file=gfa_file)

                # Link out of the alternate node, avoid self-loops for null intervals
                # TODO: Combine shared prefixes into a single node?
                for alt in span.alts:
                    if len(span.region) == 0 and alt.target == span.region.start:
                        target_span = next_span
                    else:
                        target_span = self.spans[self.find_target_span(alt.target)]
                        assert (
                            target_span.region.start == alt.target
                        ), f"Target {alt.target} not at start of span {target_span.region}"
                    assert target_span.node_id != span.node_id, "Self-loop detected"
                    print("L", alt.node_id, "+", target_span.node_id, "+", "0M", sep="\t", file=gfa_file)
                    for next_alt in target_span.alts:
                        print("L", alt.node_id, "+", next_alt.node_id, "+", "0M", sep="\t", file=gfa_file)

            # Emit paths
            for path, nodes in paths.items():
                print("P", path, ",".join(f"{n}+" for n in nodes), "*", sep="\t", file=gfa_file)

    def _split_spans(self, variant_region: Range, variant_id: str, alt: "AltPath") -> None:
        ref_name = variant_path_name(variant_id, 0)
        start_idx, end_idx = self.find_overlapping_spans(variant_region)

        if start_idx == end_idx:
            source_span = self.spans[start_idx]
            if source_span.region == variant_region:
                source_span.names.add(ref_name)
                source_span.alts.append(alt)
            else:
                # Splitting a single node into shortened original, the variant region and the remainder
                assert variant_region < source_span.region, f"{variant_region} not a subset of {source_span.region}"

                variant_span = ReferenceSpan(variant_region, source_span=source_span)
                variant_span.names.add(ref_name)
                variant_span.alts.append(alt)

                if variant_region.start == source_span.start:
                    # Only split into two nodes at the start of the source span
                    remainder_span = ReferenceSpan(
                        Range(variant_region.contig, variant_region.end, source_span.end), source_span=source_span
                    )

                    source_span.names.add(ref_name)  # Add the variant to the source span
                    source_span.alts.append(alt)
                    source_span.region = Range(source_span.contig, source_span.start, variant_region.end)

                    self.spans.insert(start_idx + 1, remainder_span)
                elif variant_region.end == source_span.end:
                    # Only split into two nodes at the end of the source span
                    self.spans.insert(start_idx + 1, variant_span)
                    source_span.region = Range(source_span.contig, source_span.start, variant_region.start)
                else:
                    # Split into three nodes
                    remainder_span = ReferenceSpan(
                        Range(variant_region.contig, variant_region.end, source_span.end), source_span=source_span
                    )
                    source_span.region = Range(source_span.contig, source_span.start, variant_region.start)

                    self.spans.insert(start_idx + 1, remainder_span)
                    self.spans.insert(start_idx + 1, variant_span)

        else:
            # Variant extends across multiple spans
            assert len(variant_region) > 0
            start_source_span = self.spans[start_idx]
            end_source_span = self.spans[end_idx]

            # Insert nodes working from the end of the spans forwards
            if variant_region.end == end_source_span.end:
                # Don't need to split the end_source span, there are identical ending points
                end_source_span.names.add(ref_name)
            else:
                remainder_span = ReferenceSpan(
                    Range(variant_region.contig, variant_region.end, end_source_span.end), source_span=end_source_span
                )

                end_source_span.names.add(ref_name)
                end_source_span.region = Range(end_source_span.contig, end_source_span.start, variant_region.end)

                self.spans.insert(end_idx + 1, remainder_span)

            if variant_region.start == start_source_span.start:
                # Don't need to split the start_source span, there are identical starting points
                start_source_span.names.add(ref_name)
                start_source_span.alts.append(alt)
            else:
                start_variant_span = ReferenceSpan(
                    Range(variant_region.contig, variant_region.start, start_source_span.end),
                    source_span=start_source_span,
                )
                start_variant_span.names.add(ref_name)
                start_variant_span.alts.append(alt)
                self.spans.insert(start_idx + 1, start_variant_span)
                start_source_span.region = Range(
                    start_source_span.contig, start_source_span.start, variant_region.start
                )

            # Add name to intermediate nodes
            for i in range(start_idx + 1, end_idx):
                self.spans[i].names.add(ref_name)


class ReferenceSpan:
    def __init__(self, region: Range, source_span: Optional["ReferenceSpan"] = None):
        self.region: Range = region
        self.names: set[str] = set(source_span.names) if source_span else set()
        self.alts: list[AltPath] = []
        self.node_id: Optional[int] = None

    @property
    def contig(self) -> str:
        return self.region.contig

    @property
    def start(self) -> int:
        return self.region.start

    @property
    def end(self) -> int:
        return self.region.end


@dataclass
class AltPath:
    name: str
    target: int
    sequence: str
    node_id: Optional[int] = None


class _StartPointRegionCmp:
    def __init__(self, region: Range):
        self.region = region

    def __lt__(self, point: int):
        return self.region.end <= point

    def __gt__(self, point: int):
        return self.region.start > point

    def __eq__(self, point: int):
        return self.region.start <= point < self.region.end


def span_start_point_key(span: ReferenceSpan):
    """Key function to find spans with inclusive start from a non-empty span"""
    return _StartPointRegionCmp(span.region)


class _EndPointRegionCmp:
    def __init__(self, region: Range):
        self.region = region

    def __lt__(self, point: int):
        return self.region.end < point

    def __gt__(self, point: int):
        return self.region.start >= point

    def __eq__(self, point: int):
        return self.region.start < point <= self.region.end


def span_end_point_key(span: ReferenceSpan):
    """Key function to find spans with exclusive end from a non-empty span"""
    return _EndPointRegionCmp(span.region)


class _BetweenPointRegionCmp:
    def __init__(self, region: Range):
        self.region = region

    def __lt__(self, point: int):
        return self.region.end < point if (self.region.start == self.region.end) else self.region.end <= point

    def __gt__(self, point: int):
        return self.region.start > point

    def __eq__(self, point: int):
        if self.region.start == self.region.end:
            return self.region.start == point
        else:
            return self.region.start <= point < self.region.end


def span_between_point_key(span: ReferenceSpan):
    """Key function to find spans with point before (in-between) span"""
    return _BetweenPointRegionCmp(span.region)


def _reference_sequence(reference_fasta: str, region: Range) -> str:
    with pysam.FastaFile(reference_fasta) as ref_fasta:
        # Make sure reference sequence is all upper case
        return ref_fasta.fetch(reference=region.contig, start=region.start, end=region.end).upper()


def variant_path_name(variant_id: str, allele: int) -> str:
    return f"_alt_{variant_id}_{allele}"


def gfa_to_xg(gfa_path: str, xg_path: str):
    with open(xg_path, "w") as xg_file:
        convert_command = f"vg convert --gfa-in {gfa_path} --xg-out"
        convert_result = subprocess.run(convert_command, shell=True, stdout=xg_file, stderr=subprocess.PIPE)
        if convert_result.returncode != 0 or not os.path.exists(xg_path):
            print(convert_result.stderr, file=sys.stderr)
            raise RuntimeError("Failed to convert GFA to XG")


def vcf_to_gbwt(xg_path: str, vcf_path: str, region: Range, gbwt_path: str):
    # VG drops the '*' alleles as produced by HaplotypeCaller or other tools, producing
    # incorrect haplotypes
    gbwt_command = f"vg gbwt \
        --xg-name {quote(xg_path)} \
        --vcf-input {quote(vcf_path)} \
        --vcf-region {quote(str(region))} \
        --ignore-missing \
        --output {quote(gbwt_path)}"
    subprocess.run(gbwt_command, shell=True, check=True)


def vcf_to_paths(gfa_path: str, vcf_path: str, region: Range):
    with tempfile.TemporaryDirectory() as tmp_dir:
        xg_path = os.path.join(tmp_dir, "test.xg")
        gfa_to_xg(gfa_path, xg_path)

        gbwt_path = os.path.join(tmp_dir, "test.gbwt")
        vcf_to_gbwt(xg_path, vcf_path, region, gbwt_path)

        threads_command = f"vg paths --extract-gaf --xg {quote(xg_path)} --gbwt {quote(gbwt_path)}"
        with subprocess.Popen(threads_command, shell=True, stdout=subprocess.PIPE, text=True) as threads:
            while True:
                line = threads.stdout.readline()
                if not line and threads.poll() is not None:
                    break
                elif not line:
                    continue
                path_name, length, _, _, strand, nodes, *_ = line.split("\t", 6)
                if int(length) > 0:
                    yield (path_name, strand, nodes[1:].split(">"))  # Drop leading ">"

def add_haplotypes_to_gfa(gfa_path: str, vcf_path: str, region: Range):
    with open(gfa_path, "a") as gfa_file:
        for name, strand, nodes in vcf_to_paths(gfa_path, vcf_path, region):
            print("P", name, ",".join(f"{n}{strand}" for n in nodes), "*", sep="\t", file=gfa_file)    
