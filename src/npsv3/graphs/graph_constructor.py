"""Construct a graph from a VCF, treating it as a sorted DAG

Replaces VG construct command. The resulting graph has explicit zero-length nodes for deletion alternate alleles
and insertion reference alleles. These nodes don't change the genomic sequences, but facilitate haplotype
generation.
"""

import copy
import functools
import itertools
import logging
import os
import subprocess
import sys
import tempfile
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Callable, Iterator, MutableSequence, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum, Flag, auto
from shlex import quote
from types import MappingProxyType
from typing import Optional, TextIO

import pysam

from npsv3.util.range import Range
from npsv3.variant import VARIANT_ID_LENGTH, Variant

# We would like to use the interval tree library, but it doesn't support "null" intervals,
# i.e., with zero size, which we need for insertions. So instead we use a sorted list as the core
# data structure and binary search.


class VariantOverlap(Flag):
    OVERLAP = auto()
    STAR_ALLELE = auto()

class GraphConstructor:
    """Construct a pangenome graph from graph_vcf variants in a given region.

    VCF features supported:
    * Global phasing (|) and local phasing (|) with a PS field.
    * Star alleles and alleles overlapped by a deletion more generally. If genotypes are not explicitly phased, we try to find
      a consistent phasing of the alleles in the overlapping variants.

    Known limitations:
    * We don't attempt to change smaller homozygous variants contained within larger heterozygous variants, e.g.,
      a het. SV DEL. These are reported 'Found non-reference allele overlapped by another non-reference allele'. These
      are a frequent source of the above warnings.
    """
    def __init__(self, region: Range, graph_vcf: str):
        """Construct a graph from graph_vcf variants in region. Assumes VCF is indexed and in sorted order."""
        self.region = region

        # Start with a single span for the entire reference region (i.e. no variation)
        self.spans: MutableSequence[ReferenceSpan] = [ReferenceSpan(region)]
        self.spans[0].names.add(self.region.contig)

        self.paths = defaultdict(list)

        self._vcf_to_spans(graph_vcf)
        self._assign_nodes()
        self._extract_paths()
        self._extract_haplotypes(graph_vcf)

    def _vcf_to_spans(self, vcf_path: str):
        """Split the reference region into spans based on variants in vcf_path."""
        with pysam.VariantFile(vcf_path, drop_samples=True) as vcf_file:
            for variant, _ in sort_variant_reference_region(vcf_file.fetch(**self.region.pysam_fetch)):
                has_star_allele = variant.has_star_allele
                for allele_idx, allele_len in enumerate(variant.length_change(allele=None), start=1):
                    try:
                        # When the * alleles also create paths that include the padding bases (noted with "rec" prefix).
                        # This crudely create extra copies of some nodes, but those could be pruned later by removing
                        # the "rec" paths and then any nodes that are no longer referenced in any path.
                        if has_star_allele and allele_len is not None:
                            # Find and split the corresponding reference span (which includes the padding bases)
                            alt_region = variant.record_reference_region #TODO: Should this be based on the alternative alle?
                            alt = AltPath(
                                variant_path_name(variant.vg_variant_id, allele_idx, prefix="rec"),
                                alt_region.end,
                                variant.allele(allele_idx), # Include all the padding bases in the sequence
                            )
                            self._split_spans(alt_region, variant.vg_variant_id, alt, path_prefix="rec")

                        # "Normal" allele addition without padding bases
                        if allele_len is not None: # Ignore * alleles
                            # Find and split the corresponding span
                            alt_region = variant.alt_reference_region(allele_idx)
                            alt = AltPath(
                                variant_path_name(variant.vg_variant_id, allele_idx),
                                alt_region.end,
                                variant.alt_seq(allele_idx),
                            )
                            self._split_spans(alt_region, variant.vg_variant_id, alt)
                    except Exception as e:
                        e.add_note(f"in variant spanning {variant.reference_region} and allele {allele_idx}") # Python 3.11+ (alternately use e.args)
                        raise


    def _assign_nodes(self):
        """Assign unique node IDs to spans and their alternate alleles."""
        node_id_gen = itertools.count(1)
        for span in self.spans:
            span.node_id = next(node_id_gen)
            for alt in span.alts:
                alt.node_id = next(node_id_gen)

    def _extract_paths(self):
        """Extract paths from the spans to create a mapping of path names to node IDs."""
        for span in self.spans:
            for name in span.names:
                self.paths[name].append(span.node_id)
            for alt in span.alts:
                self.paths[alt.name].append(alt.node_id)

        # Extend trimmed (padding removed) alternate paths to match the beginning and end points of the reference path. We need to
        # do this because we maintain a single reference path for each variant, as opposed to an allele-specific reference path.
        # The result should be that the reference and alternate paths begin and end at the same points in the graph.
        for name, nodes in self.paths.items():
            if not name.startswith("_"):
                continue # Skip paths that aren't variants, e.g., _alt_...
            allele = int(name[5+VARIANT_ID_LENGTH+1:])
            if allele > 0:
                # Extend the alternate path to match the start and end of the reference paths, i.e.
                # make sure the paths span equivalent portions of the graph.
                ref_path = name[:5+VARIANT_ID_LENGTH] + "_0"
                ref_nodes = self.paths[ref_path]

                span_idx, alt_path = self.find_node_span(nodes[0])
                assert span_idx is not None and span_idx > 0 and alt_path is not None  # noqa: PT018
                target_span = self.spans[span_idx - 1]
                # If the preceding span id is in the references nodes, extend the alternate path to include those nodes to the beginning
                # of the reference path
                try:
                    ref_nodes_index = ref_nodes.index(target_span.node_id)
                    self.paths[name][:0] = ref_nodes[:ref_nodes_index + 1] # Insert elements at the beginning of the list
                except ValueError:
                    pass

                span_idx, alt_path = self.find_node_span(nodes[-1])
                assert span_idx is not None and span_idx < len(self.spans) - 1 and alt_path is not None  # noqa: PT018
                span_region = self.spans[span_idx].region
                if len(span_region) == 0 and alt_path.target == span_region.start:
                    target_span = self.spans[span_idx + 1]
                else:
                    target_span = self.spans[self.find_target_span(alt_path.target)]

                # If the target span id is in the references nodes, extend the alternate path to include those nodes to the end
                try:
                    ref_nodes_index = ref_nodes.index(target_span.node_id)
                    self.paths[name].extend(ref_nodes[ref_nodes_index:])
                except ValueError:
                    pass


    def _extract_haplotypes(self, vcf_path: str, samples: Sequence[str] | None=None, ploidy=2):
        """Extract haplotypes from vcf_path for samples as paths in the graph"""
        with pysam.VariantFile(vcf_path) as vcf_file:
            if samples:
                vcf_file.subset_samples(samples)

            current_samples = list(vcf_file.header.samples)

            ref_nodes = self.paths[self.region.contig]
            polytypes  = [PolytypePaths(MappingProxyType(self.paths), ref_nodes, sample, ploidy) for sample in current_samples]

            current_range = None
            for variant, record in sort_variant_reference_region(vcf_file.fetch(**self.region.pysam_fetch)):
                # We can also have overlap alleles that are not specifically encoded in the VCF with "*", but instead
                # are still marked with a genotype, e.g., 0/0, even it latter should formally be 0/*. These include insertions
                # that have identical "zero width" regions
                has_star = False
                try:
                    alt_alleles = record.alts
                    if alt_alleles == ("*",):
                        continue  # Skip variants that only have * alternate alleles
                    star_idx = alt_alleles.index("*") + 1
                    allele_indices = set(itertools.chain.from_iterable(genotype.allele_indices for genotype in record.samples.itervalues()))
                    if allele_indices == { star_idx }:
                        continue  # Skip variants with genotypes that are all * alleles
                    if star_idx in allele_indices:
                        # Only consider * alleles actually present in one of the genotypes, i.e., not just in the ALTS
                        has_star = True
                except ValueError:
                    pass

                # TODO: This currently doesn't handle the case where we are checking an insertion overlaps a multi-allelic variant
                # containing the same insertion.
                variant_range = variant.reference_region
                if current_range is not None and (current_range.overlaps(variant_range) or current_range == variant_range):
                    current_range = current_range.union(variant_range)
                    overlap =  (VariantOverlap.OVERLAP | VariantOverlap.STAR_ALLELE) if has_star else VariantOverlap.OVERLAP
                elif current_range is None and has_star:
                    current_range = variant_range
                    overlap = VariantOverlap.OVERLAP | VariantOverlap.STAR_ALLELE
                else:
                    if has_star:
                        variant_range = variant.record_reference_region
                        if current_range.overlaps(variant_range):
                            # The '*' allele overlaps if we include the padding base. Add genotypes with "rec" prefix paths that include
                            # those padding bases
                            for g, genotype in enumerate(record.samples.itervalues()):
                                polytypes[g].add_genotype(variant, genotype, VariantOverlap.OVERLAP | VariantOverlap.STAR_ALLELE, path_prefix="rec")
                            current_range = current_range.union(variant_range)
                        else:
                            logging.warning("Found non-overlapping variant with * allele in %s, skipping", self.region)
                        continue
                    current_range = variant_range
                    overlap = VariantOverlap(0)

                for g, genotype in enumerate(record.samples.itervalues()):
                    polytypes[g].add_genotype(variant, genotype, overlap)

        for polytype in polytypes:
            polytype.finish_paths()
            self.paths.update(polytype.gfa_paths(self.region))


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
        # TODO: Linear search likely faster for end coordinate
        start_idx = bisect_left(self.spans, region.start, key=span_start_point_key)
        end_idx = bisect_left(self.spans, region.end, key=span_end_point_key)
        return (start_idx, end_idx)

    def find_target_span(self, target_start: int) -> int:
        """Find the leftmost span that starts at the target position, including any null regions"""
        return bisect_left(self.spans, target_start, key=span_between_point_key)

    def find_node_span(self, node: int) -> "tuple[int|None,AltPath|None]":
        """Find the span or alternate path with the given node ID."""
        for i, span in enumerate(self.spans):
            if span.node_id == node:
                return (i,None)
            for alt in span.alts:
                if alt.node_id == node:
                    return (i,alt)
        return (None,None)

    def to_gfa(self, ref_fasta: str, out_file: str | TextIO = sys.stdout) -> None:
        """Write the graph in GFA format to out_file using reference sequence from ref_fasta."""
        ref_seq = _reference_sequence(ref_fasta, self.region)

        def get_ref_seq(region: Range):
            return (
                "*" if len(region) == 0 else ref_seq[region.start - self.region.start : region.end - self.region.start]
            )

        with open(out_file, "w") if isinstance(out_file, str) else nullcontext(out_file) as gfa_file:
            print("H", "VN:Z:1.0", sep="\t", file=gfa_file)
            print(f"# Region: {self.region}", file=gfa_file)

            # Emit nodes from spans and their alternate alleles
            for span in self.spans:
                print("S", span.node_id, get_ref_seq(span.region), sep="\t", file=gfa_file)
                for alt in span.alts:
                    print("S", alt.node_id, alt.sequence or "*", sep="\t", file=gfa_file)

            assert len(self.spans[0].alts) == 0 and len(self.spans[-1].alts) == 0, "Graph does not form a bubble"  # noqa: PT018
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
                        # Don't create links between alternate alleles of the same multi-allelic variant since that is not consistent with
                        # with the VCF
                        if alt.name[5:5+VARIANT_ID_LENGTH] != next_alt.name[5:5+VARIANT_ID_LENGTH]:
                            print("L", alt.node_id, "+", next_alt.node_id, "+", "0M", sep="\t", file=gfa_file)

            # Emit paths
            for path, nodes in self.paths.items():
                print("P", path, ",".join(f"{n}+" for n in nodes), "*", sep="\t", file=gfa_file)

    def _split_spans(self, variant_region: Range, variant_id: str, alt: "AltPath", path_prefix: str = "alt") -> None:
        ref_name = variant_path_name(variant_id, 0, prefix=path_prefix)
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

            # Add name to any intermediate nodes
            for i in range(start_idx + 1, end_idx):
                self.spans[i].names.add(ref_name)

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



class ReferenceSpan:
    """Reference span in the graph with its associated names and alternate paths"""
    def __init__(self, region: Range, source_span: Optional["ReferenceSpan"] = None):
        self.region: Range = region
        self.names: set[str] = set(source_span.names) if source_span else set()
        self.alts: list[AltPath] = []
        self.node_id: int | None = None

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
    node_id: int | None = None


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
        return self.region.start <= point < self.region.end


def span_between_point_key(span: ReferenceSpan):
    """Key function to find spans with point before (in-between) span"""
    return _BetweenPointRegionCmp(span.region)


def _reference_sequence(reference_fasta: str, region: Range) -> str:
    with pysam.FastaFile(reference_fasta) as ref_fasta:
        # Make sure reference sequence is all upper case
        return ref_fasta.fetch(reference=region.contig, start=region.start, end=region.end).upper()

# For reference on converting VCF to graph paths, the approach in the GBWT library
# https://github.com/jltsiren/gbwt/blob/bde6858046580d1b9dbfa54f48ab187c85998ffe/src/variants.cpp#L826


class Phasing(Enum):
    """Possible phasing for a genotype"""
    UNPHASED = 0
    GLOBAL = 1
    LOCAL = 2
    IMPLICIT = 3  # Homozygous genotypes are implicitly phased


@dataclass
class PhaseState:
    """Phase state for a genotype, including local phase set (PS) if applicable"""
    phase: Phasing
    phase_set: int | None = None  # Phase set ID, if applicable (this is really any type used in the VCF PS field)

    def __eq__(self, other):
        if isinstance(other, PhaseState):
            return self.phase == other.phase and (self.phasing != Phasing.LOCAL or self.phase_set == other.phase_set)
        return False

    def next_state(self, variant_phase: "PhaseState") -> tuple["PhaseState", bool]:
        """ Return the next phase based on the current phase and the variant's phase and whether to break the haplotype"""
        state_and_trans = _NEXT_PHASE_STATE[self.phase, variant_phase.phase]
        if isinstance(state_and_trans, tuple):
            next_phase, break_before, = state_and_trans
            return PhaseState(next_phase), break_before
        if callable(state_and_trans):
            return state_and_trans(self, variant_phase)
        raise ValueError


def _compare_local_phase_sets(phase: PhaseState, variant_phase: PhaseState) -> tuple["PhaseState", bool]:
    assert phase.phase == variant_phase.phase == Phasing.LOCAL, "Can only compare local phases"
    return (variant_phase, variant_phase.phase_set != phase.phase_set)

# Transition table for the next phase state based on the (current phase, variant's phase)
# The values should be tuples of (next_phase, break_before) or a function that takes the current phase and variant phase
# and returns the next phase and whether to break the haplotype.
_NEXT_PHASE_STATE : dict[tuple[Phasing, Phasing], tuple["PhaseState", bool]|Callable]= {
    (Phasing.UNPHASED, Phasing.UNPHASED): (Phasing.UNPHASED, True),
    (Phasing.UNPHASED, Phasing.GLOBAL): (Phasing.GLOBAL, True),
    (Phasing.UNPHASED, Phasing.LOCAL): lambda _, n: (n, True),
    (Phasing.UNPHASED, Phasing.IMPLICIT): (Phasing.UNPHASED, False),
    (Phasing.GLOBAL, Phasing.UNPHASED): (Phasing.UNPHASED, True),
    (Phasing.GLOBAL, Phasing.GLOBAL): (Phasing.GLOBAL, False),
    (Phasing.GLOBAL, Phasing.LOCAL): lambda _, n: (n, True),
    (Phasing.GLOBAL, Phasing.IMPLICIT): (Phasing.GLOBAL, False),
    (Phasing.LOCAL, Phasing.UNPHASED): (Phasing.UNPHASED, True),
    (Phasing.LOCAL, Phasing.GLOBAL): (Phasing.GLOBAL, True),
    (Phasing.LOCAL, Phasing.LOCAL): _compare_local_phase_sets,
    (Phasing.LOCAL, Phasing.IMPLICIT): lambda c, _: (c, False),
    (Phasing.IMPLICIT, Phasing.UNPHASED): (Phasing.UNPHASED, False),
    (Phasing.IMPLICIT, Phasing.GLOBAL): (Phasing.GLOBAL, False),
    (Phasing.IMPLICIT, Phasing.LOCAL): lambda _, n: (n, False),
    (Phasing.IMPLICIT, Phasing.IMPLICIT): (Phasing.IMPLICIT, False),
}

class NonRefAlleleOverlappingNonRefError(Exception):
    """Non-reference allele overlapping another non-reference allele"""
    # In a well formed VCF this should not occur, but in practice if the variants are not phased
    # we can obtain a non-reference allele that overlaps another non-reference allele. Report this
    # error explicitly so we can possibly fix it.


class HaplotypePaths:
    """Construct a sequence of (dis)connected paths making up a single haplotype by adding differently phased alleles"""
    def __init__(self, ref_path: list[int]):
        self._ref_path = ref_path
        self.current_path = []
        self.haplotypes = [self.current_path]
        self.current_ref_idx = 0
        self.phase = PhaseState(Phasing.IMPLICIT)  # Start with implicit phasing, i.e., no breaks in the haplotype

    def apply_allele(self, ref_nodes: Sequence[int], phase: PhaseState, overlap: VariantOverlap, alt_nodes: Sequence[int]|None = None, variant: Variant = None, *, break_inconsistent = False):
        # Find replacement index, allowing for imprecisely defined overlapping alleles
        try:
            if ref_nodes[0] < self._ref_path[self.current_ref_idx]:
                # We appear to have overlapping alleles. Trim any trailing reference nodes from the current path
                # to try to eliminate the overlap
                while self.current_ref_idx > 0 and len(self.current_path) > 0 and self._ref_path[self.current_ref_idx - 1] == self.current_path[-1]:
                    self.current_ref_idx -= 1
                    self.current_path.pop()
            ref_idx = self._ref_path.index(ref_nodes[0], self.current_ref_idx)
            assert self._ref_path[ref_idx:ref_idx + len(ref_nodes)] == ref_nodes, "Reference nodes don't match reference path"
        except (AssertionError, IndexError) as e:
            e.add_note(f"in variant spanning {variant.reference_region}") # Python 3.11+ (alternately use e.args)
            raise
        except ValueError as e:
            if VariantOverlap.OVERLAP in overlap and alt_nodes is None:
                # We have likely found an overlapping reference allele (without an explicit * allele). Skip it.
                return

             # Found an inconsistent allele. If specified, break the current haplotype and reset to add the allele. Otherwise raise Overlapping error
             # to try another permutation of the genotype that might be consistent.
            if break_inconsistent:
                self.current_path = []
                self.haplotypes.append(self.current_path)
                self.phase = PhaseState(Phasing.IMPLICIT)
                ref_idx = self._ref_path.index(ref_nodes[0])
                assert self._ref_path[ref_idx:ref_idx + len(ref_nodes)] == ref_nodes, "Reference nodes don't match reference path"
                self.current_ref_idx = ref_idx
            else:
                msg = f"Non-reference allele overlapped by another non-reference allele in {variant.reference_region}"
                raise NonRefAlleleOverlappingNonRefError(msg) from e

        # Fill in intervening reference nodes
        self.current_path.extend(self._ref_path[self.current_ref_idx:ref_idx])

        # Introduce a break in the haplotype
        next_phase, break_before = self.phase.next_state(phase)
        if break_before and len(self.current_path) > 0:
            self.current_path = []
            self.haplotypes.append(self.current_path)
        self.phase = next_phase

        # Append new nodes and update current reference index to after the replaced nodes
        self.current_path.extend(alt_nodes if alt_nodes is not None else ref_nodes)
        self.current_ref_idx = ref_idx + len(ref_nodes)

    def finish_paths(self):
        # Fill in any remaining reference nodes
        self.current_path.extend(self._ref_path[self.current_ref_idx:])

    def gfa_paths(self, sample: str, chrom: int, contig: str) -> list[str]:
        """Return dict of GFA path name -> nodes for the haplotype"""
        return {f"{sample}#{chrom}#{contig}#{i}": nodes for i, nodes in enumerate(self.haplotypes)}

class PolytypePaths:
    """Construct haplotypes by adding polyploid genotypes for a sequence of variants"""
    def __init__(self, paths: dict[Sequence[int]], ref_nodes: Sequence[int], sample: str, max_ploidy: int = 2):
        self.paths = paths
        self.sample = sample
        self.max_ploidy = max_ploidy
        self.haplotypes = [HaplotypePaths(ref_nodes) for _ in range(max_ploidy)]

    def add_genotype(self, variant: Variant, genotype: pysam.VariantRecordSample, overlap: VariantOverlap, path_prefix: str = "alt", max_gt_permutations: int = 2):
        ref_nodes = self.paths[variant_path_name(variant.vg_variant_id, 0, prefix=path_prefix)]
        indices = genotype.allele_indices

        # A variant can be explicitly globally or locally phased, or implicitly phased if it has a overlapping '*' 'allele
        # or if all alleles are the same (e.g., 0/0 genotype)
        if genotype.phased:
            phase_set = genotype.get("PS")
            phase = PhaseState(Phasing.GLOBAL if phase_set is None else Phasing.LOCAL, phase_set)
        elif VariantOverlap.STAR_ALLELE in overlap or len(set(indices)) == 1:
            # Heterozygous variants should only be implicitly phased if there are actual overlapping alleles
            # where we can make inferences about the phase. At present we only assume that for explicit * alleles.
            phase = PhaseState(Phasing.IMPLICIT)
        else:
            phase = PhaseState(Phasing.UNPHASED)

        # A NonRefAlleleOverlappingNonRefError indicates alleles in overlapping variants are not consistently phased.
        # If the genotype was originally unphased, try permutations of the alleles to try to find a consistent phasing.
        for local_indices in itertools.islice(itertools.permutations(indices), 0, max_gt_permutations if not genotype.phased else 1):
            try:
                local_haplotypes = copy.deepcopy(self.haplotypes) if VariantOverlap.OVERLAP in overlap else self.haplotypes
                for i, index in itertools.zip_longest(range(self.max_ploidy), local_indices, fillvalue=-1):
                    if index is None:
                        continue  # Skip undefined (".") alleles
                    if VariantOverlap.STAR_ALLELE in overlap and variant.allele(index) == "*":
                        continue  # Don't apply explicitly overlapped "*"" allele
                    if index != -1:
                        alt_path = variant_path_name(variant.vg_variant_id, index, prefix=path_prefix)
                        local_haplotypes[i].apply_allele(ref_nodes, phase, overlap, alt_nodes=self.paths[alt_path] if index > 0 else None, variant=variant)
                    else:
                        msg = f"Haploid genotypes are currently not supported ({variant.reference_region})"
                        raise NotImplementedError(msg)
                # Successfully applied all the alleles, update the haplotypes and terminate the retry loop
                self.haplotypes = local_haplotypes
                break
            except NonRefAlleleOverlappingNonRefError:
                # Try another permutation of the genotype to see if we can find a local phasing consistent with the overlap
                continue
        else:
            # We tried different permutations of the genotype, but couldn't find a consistent phasing. Re-add the alleles, but breaking
            # the haplotype on failure.
            for i, index in itertools.zip_longest(range(self.max_ploidy), indices, fillvalue=-1):
                if index is None:
                    continue  # Skip undefined (".") alleles
                if VariantOverlap.STAR_ALLELE in overlap and variant.allele(index) == "*":
                    continue  # Don't apply explicitly overlapped "*"" allele
                if index != -1:
                    alt_path = variant_path_name(variant.vg_variant_id, index, prefix=path_prefix)
                    self.haplotypes[i].apply_allele(ref_nodes, phase, overlap, alt_nodes=self.paths[alt_path] if index > 0 else None, variant=variant, break_inconsistent=True)
                else:
                    msg = f"Haploid genotypes are currently not supported ({variant.reference_region})"
                    raise NotImplementedError(msg)

    def finish_paths(self):
        for haplotype in self.haplotypes:
            haplotype.finish_paths()

    def gfa_paths(self, region) -> dict[str, list[int]]:
        """Return dict of VG-style GFA path name -> nodes for the haplotypes in the region"""
        paths = {}
        for h, haplotype in enumerate(self.haplotypes):
            paths.update(haplotype.gfa_paths(self.sample, h, region.contig))
        return paths


def variant_path_name(variant_id: str, allele: int, prefix: str="alt") -> str:
    """Construct a path name for a variant and allele, e.g., alt_123_1"""
    return f"_{prefix}_{variant_id}_{allele}"

def variant_path_to_id(path_name: str, prefix: str="alt") -> str:
    """Extract the variant ID from a path name, e.g., alt_123_1 -> 123"""
    prefix = f"_{prefix}_"
    assert path_name.startswith(prefix)
    return path_name[len(prefix):len(prefix) + VARIANT_ID_LENGTH]

def variant_path_to_allele(path_name: str, prefix: str="alt") -> int:
    """Extract the allele index from a path name, e.g., alt_123_1 -> 1"""
    prefix = f"_{prefix}_"
    assert path_name.startswith(prefix)
    return int(path_name[len(prefix) + VARIANT_ID_LENGTH + 1:])

def _nesting_reference_region_cmp(a: tuple[Variant, pysam.VariantRecord], b: tuple[Variant, pysam.VariantRecord]) -> int:
    region_a = a[0].reference_region
    region_b = b[0].reference_region
    if region_a.start == region_b.start:
        # Order zero-length regions first, the in reverse order of region length (so enclosing regions come first)
        if region_a.length == 0:
            return -1 if region_b.length > 0 else 0
        if region_b.length == 0:
            return 1
        return region_b.length - region_a.length
    return region_a.start - region_b.start


def sort_variant_reference_region(records: Iterator[pysam.VariantRecord]) -> Iterator[tuple[Variant, pysam.VariantRecord]]:
    """Adaptor to sort pysam.VariantRecords by their reference region start position (not VCF start position)"""
    # TODO: Potentially replace with online sorting to reduce memory usage
    var_iter = ((Variant.from_pysam(record), record) for record in records)
    return sorted(var_iter, key=functools.cmp_to_key(_nesting_reference_region_cmp))


# VG helper functions to build GFA from a VCF file

def gfa_to_xg(gfa_path: str, xg_path: str):
    with open(xg_path, "w") as xg_file:
        convert_command = f"vg convert --gfa-in {gfa_path} --xg-out"
        convert_result = subprocess.run(convert_command, shell=True, stdout=xg_file, stderr=subprocess.PIPE, check=False)
        if convert_result.returncode != 0 or not os.path.exists(xg_path):
            raise RuntimeError("Failed to convert GFA to XG")


def vcf_to_gbwt(xg_path: str, vcf_path: str, region: Range, gbwt_path: str):
    # vg drops the '*' alleles (e.g., as produced by HaplotypeCaller) producing
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
    """Append haplotype paths from vcf_path in region to gfa_path"""
    with open(gfa_path, "a") as gfa_file:
        for name, strand, nodes in vcf_to_paths(gfa_path, vcf_path, region):
            print("P", name, ",".join(f"{n}{strand}" for n in nodes), "*", sep="\t", file=gfa_file)

