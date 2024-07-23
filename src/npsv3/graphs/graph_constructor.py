from typing import Optional, TextIO, Union
from bisect import bisect_left
from collections import defaultdict
from collections.abc import MutableSequence
from contextlib import nullcontext
from dataclasses import dataclass
import sys

import pysam

from npsv3.variant import Variant
from npsv3.util.range import Range
from npsv3.graph import variant_path_name

# We would like to use the interval tree library, but it doesn't support "null" intervals,
# i.e., with zero size, which we need for insertions.

class GraphConstructor:
    def __init__(self, region: Range, graph_vcf: str):
        self.region = region

        # Start with a single span for the entire reference region (i.e. no variation)
        self.spans: MutableSequence["ReferenceSpan"] = [ReferenceSpan(region)]
        self.spans[0].names.add(self.region.contig)

        self._construct_from_vcf(graph_vcf)

    def _construct_from_vcf(self, vcf_path: str):
        with pysam.VariantFile(vcf_path, drop_samples=True) as vcf_file:
            for record in vcf_file.fetch(**self.region.pysam_fetch):
                variant = Variant.from_pysam(record)
                for allele_idx, allele_len in enumerate(variant.length_change(allele=None), start=1):
                    # Find and split the corresponding span
                    alt_region = variant.alt_reference_region(allele_idx)
                    alt = AltPath(f"_alt_{variant.vg_variant_id}_{allele_idx}", alt_region.end, variant.alt_seq(allele_idx))
                    
                    self._split_spans(alt_region, variant.vg_variant_id, alt)

    @property
    def num_spans(self) -> int:
        return len(self.spans)
    
    def get_span_region(self, idx: int) -> Range:
        return self.spans[idx].region

    def find_leftmost_span(self, start: int) -> int:
        start_idx = bisect_left(self.spans, start, key=span_start_point_key)
        # Shift left if there is a preceding null region
        if start_idx > 0 and self.spans[start_idx-1].region.start == start:
            assert len(self.spans[start_idx-1].region) == 0
            return start_idx-1
        return start_idx

    def find_spans(self, region: Range) -> tuple[int, int]:
        assert region.contig == self.region.contig
        start_idx = bisect_left(self.spans, region.start, key=span_start_point_key)
        # If we are searching for a null region, check if we actually match the region to the left
        if len(region) == 0 and start_idx > 0 and self.spans[start_idx-1].region == region:
            return (start_idx-1, start_idx-1)
        end_idx = bisect_left(self.spans, region.end, key=span_end_point_key)
        return (start_idx, end_idx)

    def to_gfa(self, ref_fasta: str, out_file: Union[str,TextIO] = sys.stdout):
        ref_seq = _reference_sequence(ref_fasta, self.region)
        def get_ref_seq(region: Range):
            return "*" if len(region) == 0 else ref_seq[region.start - self.region.start:region.end - self.region.start]
        
        paths = defaultdict(list)

        with open(out_file, "w") if isinstance(out_file, str) else nullcontext(out_file) as gfa_file:
            print("H","VN:Z:1.0", sep="\t", file=gfa_file)
            
            for span_id, span in enumerate(self.spans[:-1], start=1):
                print("S", span_id, get_ref_seq(span.region), sep="\t", file=gfa_file)
                print("L", span_id, "+", span_id+1, "+", "0M", sep="\t", file=gfa_file)
                for name in span.names:
                    paths[name].append(span_id)
            print("S", len(self.spans), get_ref_seq(span.region), sep="\t", file=gfa_file)
            
            # Link up alternate alleles
            curr_id = len(self.spans) + 1
            for span_id, span in enumerate(self.spans, start=1):
                # TODO: Combine shared prefixes into a single node
                for alt in span.alts:
                    print("S", curr_id, alt.sequence or "*", sep="\t", file=gfa_file)
                    print("L", span_id, "+", curr_id, "+", "0M", sep="\t", file=gfa_file)
                    target_span = self.find_leftmost_span(alt.target)
                    print("L", curr_id, "+", target_span+1, "+", "0M", sep="\t", file=gfa_file)
                    paths[alt.name].append(curr_id)
                    curr_id += 1 

            # Emit paths
            for path, nodes in paths.items():
                print("P", path, ",".join(f"{n}+" for n in nodes), "*", sep="\t", file=gfa_file)

    def _split_spans(self, variant_region: Range, variant_id: str, alt: "AltPath"):
        ref_name = variant_path_name(variant_id, 0)
        start_idx, end_idx = self.find_spans(variant_region)
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
                    # Only split into two nodes
                    self.spans.insert(start_idx, variant_span)
                    source_span.region = Range(source_span.contig, variant_region.end, source_span.end)
                else:
                    # Split into three nodes
                    remainder_span = ReferenceSpan(
                        Range(variant_region.contig, variant_region.end, source_span.end), source_span=source_span
                    )
                    source_span.region = Range(source_span.contig, source_span.start, variant_region.start)

                    self.spans.insert(start_idx+1, remainder_span)
                    self.spans.insert(start_idx+1, variant_span)
            
        else:
            # Variant extends across multiple spans
            assert len(variant_region) > 0
            start_source_span = self.spans[start_idx]
            end_source_span = self.spans[end_idx]
            
            # Insert nodes working from the end of the spans forwards
            assert variant_region.end > end_source_span.start
            end_variant_span = ReferenceSpan(Range(variant_region.contig, end_source_span.start, variant_region.end), source_span=end_source_span)
            end_variant_span.names.add(ref_name)
            self.spans.insert(end_idx, end_variant_span)
            
            end_source_span.region = Range(end_source_span.contig, variant_region.end, end_source_span.end)
            
            if variant_region.start == start_source_span.start:
                # Don't need to split the source span, there are identical starting points
                start_source_span.names.add(ref_name)
                start_source_span.alts.append(alt)
            else:
                start_variant_span = ReferenceSpan(Range(variant_region.contig, variant_region.start, start_source_span.end), source_span=start_source_span)
                start_variant_span.names.add(ref_name)
                start_variant_span.alts.append(alt)
                self.spans.insert(start_idx+1, start_variant_span)

                start_source_span.region = Range(start_source_span.contig, start_source_span.start, variant_region.start)
            
            # Add name to intermediate nodes
            for i in range(start_idx+1, end_idx):
                self.spans[i].names.add(ref_name)

class ReferenceSpan:
    def __init__(self, region: Range, source_span: Optional["ReferenceSpan"] = None):
        self.region = region
        self.names = set(source_span.names) if source_span else set()
        self.alts = []

    @property
    def contig(self):
        return self.region.contig

    @property
    def start(self):
        return self.region.start

    @property
    def end(self):
        return self.region.end

@dataclass
class AltPath:
    name: str
    target: int
    sequence: str

class StartPointRegionCmp:
    def __init__(self, region: Range):
        self.region = region

    def __lt__(self, point: int):
        return self.region.end <= point

    def __gt__(self, point: int):
        return self.region.start > point

    def __eq__(self, point: int):
        return self.region.contains(point)

def span_start_point_key(span: ReferenceSpan):
    return StartPointRegionCmp(span.region)

class EndPointRegionCmp:
    def __init__(self, region: Range):
        self.region = region

    def __lt__(self, point: int):
        return self.region.end < point

    def __gt__(self, point: int):
        return self.region.start >= point

    def __eq__(self, point: int):
        return self.region.start < point <= self.region.end

def span_end_point_key(span: ReferenceSpan):
    return EndPointRegionCmp(span.region)

def _reference_sequence(reference_fasta: str, region: Range) -> str:
    with pysam.FastaFile(reference_fasta) as ref_fasta:
        # Make sure reference sequence is all upper case
        return ref_fasta.fetch(reference=region.contig, start=region.start, end=region.end).upper()
