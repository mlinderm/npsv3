import itertools
import operator
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from shlex import quote
from typing import Union

import odgi
import pysam
from pysam import bcftools

from npsv3.graphs.graph_constructor import GraphConstructor, variant_path_name
from npsv3.util.range import Range
from npsv3.util.vcf import index_variant_file
from npsv3.variant import Variant

HandleOrIntType = Union[odgi.handle, int]

# Length of SHA1 hex digest used for variants IDs by vg
VARIANT_ID_LENGTH = 40

VCF_HEADER_TYPES_TO_COPY = frozenset(["GENERIC", "STRUCTURED", "INFO", "FILTER", "CONTIG"])
VCF_INFO_FIELDS_TO_COPY = frozenset(["SVTYPE", "SVLEN", "END", "CIPOS", "CIEND"])

SAMPLE_PATH_REGEX = re.compile(r"^(?P<sample>[^#]+)#(?P<allele>\d)#(?P<contig>[^#]+)#(?P<count>\d+)$")

class Graph:
    def __init__(self, gfa_path: str, region: Range):
        with tempfile.TemporaryDirectory() as temp_dir:
            og_path = os.path.join(temp_dir, "graph.og")
            # Build graph with odgi executable
            build_command = f"odgi build --sort --optimize -g {quote(gfa_path)} -o {quote(og_path)}"
            subprocess.run(build_command, shell=True, check=True)

            # Load graph into Python object
            self._graph = odgi.graph()
            self._graph.load(og_path)

        self.region = region

    def _sort_and_compact(self):
        """Topologically order nodes and compact ids into [1-max node id] space.

        After this operation, nodes ids should occupy a contiguous range and
        iterating through the nodes with `for_each_handle` will be in topological order.
        """
        self._graph.apply_ordering(self._graph.topological_order(), compact_ids=True)
        # Since we change the node ids, we need to reset any cached node sets
        # (adapted from https://stackoverflow.com/a/73131568)
        cls = self.__class__
        attrs = [a for a in dir(self) if isinstance(getattr(cls, a, cls), cached_property) and a in self.__dict__]
        for a in attrs:
            delattr(self, a)

    def _is_bubble(self) -> bool:
        """Return true if the graph forms a bubble, i.e., has a single source and sink node.

        Assumes that the node ids have been compacted into a contiguous range.
        """
        assert self._graph.is_optimized(), "Graph node space is not compacted"
        incoming = Counter()
        outgoing = Counter()

        def increment(counter, node_id):
            counter[node_id] += 1

        def count_edges(node):
            node_id = self._graph.get_id(node)
            self._graph.follow_edges(node, False, lambda _: increment(outgoing, node_id))
            self._graph.follow_edges(node, True, lambda _: increment(incoming, node_id))

        self._graph.for_each_handle(count_edges)

        return (
            incoming[1] == 0
            and outgoing[self.max_node_id] == 0
            and all(incoming[i] > 0 for i in range(2, self.max_node_id + 1))
            and all(outgoing[i] > 0 for i in range(1, self.max_node_id))
        )

    def as_handle(self, node: HandleOrIntType) -> odgi.handle:
        if isinstance(node, int):
            return self._graph.get_handle(node)
        return node

    def as_id(self, node: HandleOrIntType) -> int:
        if isinstance(node, int):
            return node
        return self._graph.get_id(node)

    @cached_property
    def min_node_id(self) -> int:
        return self._graph.min_node_id()

    @cached_property
    def max_node_id(self) -> int:
        return self._graph.max_node_id()

    @cached_property
    def ref_nodes(self) -> set[int]:
        nodes = set()
        self._graph.for_each_step_in_path(
            self._graph.get_path_handle(self.region.contig),
            lambda s: nodes.add(self._graph.get_id(self._graph.get_handle_of_step(s)))
        )
        return nodes

    def nodes_on_path(self, path_name: str) -> Iterable[int]:
        nodes = []
        self._graph.for_each_step_in_path(
            self._graph.get_path_handle(path_name),
            lambda s: nodes.append(self._graph.get_id(self._graph.get_handle_of_step(s))),
        )
        return nodes

    def first_handle(self, path_name: str) -> odgi.handle:
        path = self._graph.get_path_handle(path_name)
        return self._graph.get_handle_of_step(self._graph.path_begin(path))

    def last_handle(self, path_name: str) -> odgi.handle:
        path = self._graph.get_path_handle(path_name)
        return self._graph.get_handle_of_step(self._graph.path_back(path))

    @cached_property
    def variant_paths(self) -> set[str]:
        paths = set()
        self._graph.for_each_path_handle(
            lambda p: (
                paths.add(self._graph.get_path_name(p)) if self._graph.get_path_name(p).startswith("_alt") else None
            )
        )
        return paths

    @cached_property
    def path_nodes(self) -> dict[str, set[int]]:
        path_node_sets = {}

        def extract_nodes(path):
            nodes = set()
            self._graph.for_each_step_in_path(
                path, lambda s: nodes.add(self._graph.get_id(self._graph.get_handle_of_step(s)))
            )
            path_node_sets[self._graph.get_path_name(path)] = nodes

        self._graph.for_each_path_handle(extract_nodes)
        return path_node_sets

    @cached_property
    def node_paths(self) -> dict[int, list[str]]:
        node_path_lists: dict[int, list[str]] = {}
        for path_name, nodes in self.path_nodes.items():
            for node in nodes:
                node_path_lists.setdefault(node, []).append(path_name)
        return node_path_lists

    def has_path(self, name: str) -> bool:
        return self._graph.has_path(name)

    def sequence(self, nodes: Iterable[int]) -> str:
        seq = ""
        for node in nodes:
            node_seq = self._graph.get_sequence(self._graph.get_handle(node))
            if node_seq != "*":  # Used for "deletion" and "insertion" edges
                seq += node_seq
        return seq

    def sequence_length(self, nodes: Iterable[int]) -> int:
        length = 0
        for node in nodes:
            node_length = self._graph.get_length(self._graph.get_handle(node))
            if node_length == 1 and self._graph.get_sequence(self._graph.get_handle(node)) == "*":
                # "deletion" and "insertion" edges w/ "*" have length 0
                continue
            length += node_length
        return length

    def _from_source(self, free_nodes: set[int], start_id = None, end_id = None):
        if start_id is None:
            start_id = self.min_node_id
        if end_id is None:
            end_id = self.max_node_id

        # TODO: Shift to partial list that only includes the subset of nodes
        length = [sys.maxsize] * (self.max_node_id + 1)
        prev = [None] * (self.max_node_id + 1)

        ref_nodes = self.path_nodes[self.region.contig]

        length[start_id] = 0
        for node in range(start_id, end_id + 1):
            if node in free_nodes:
                pass  # Equivalent to += 0
            elif node in ref_nodes:
                length[node] = sys.maxsize if length[node] == sys.maxsize else length[node] + len(self.sequence([node]))
            else:
                length[node] = sys.maxsize

            # Propagate score "along" edges
            next_nodes = []
            self._graph.follow_edges(
                self._graph.get_handle(node), False, lambda n: next_nodes.append(self._graph.get_id(n))
            )
            op = operator.le if node in free_nodes else operator.lt # Use "free node" as tiebreaker
            for next_node in next_nodes:
                if op(length[node], length[next_node]):
                    length[next_node] = length[node]
                    prev[next_node] = node

        return length, prev

    def _to_sink(self, free_nodes: set[int]):
        length = [sys.maxsize] * (self._graph.max_node_id() + 1)
        prev = [None] * (self._graph.max_node_id() + 1)

        ref_nodes = self.path_nodes[self.region.contig]

        length[self._graph.max_node_id()] = 0
        for node in range(self._graph.max_node_id(), self._graph.min_node_id() - 1, -1):
            if node in free_nodes:
                pass
            elif node in ref_nodes:
                length[node] = sys.maxsize if length[node] == sys.maxsize else length[node] + len(self.sequence([node]))
            else:
                length[node] = sys.maxsize

            # Propagate score backward along edges
            prev_nodes = []
            self._graph.follow_edges(
                self._graph.get_handle(node), True, lambda n: prev_nodes.append(self._graph.get_id(n))
            )
            for prev_node in prev_nodes:
                if length[node] < length[prev_node]:
                    length[prev_node] = length[node]
                    prev[prev_node] = node

        return length, prev

    def shortest_path(self, base_path_prefix: str) -> list[int]:
        """Find a path through graph 'bubble' graph using paths matching base_path_prefix as the backbone.

        Assumes nodes are in topological order

        Args:
            base_path_prefix (str): Prefix of backbone path(s)

        Returns:
            list[int]: Path of node ids
        """
        # TODO: Future optimization, if prefix exactly matches end-to-end path, we can skip
        # the shortest path and just return the path
        free_nodes = set()
        for path, nodes in self.path_nodes.items():
            if path.startswith(base_path_prefix):
                free_nodes.update(nodes)

        _, prev = self._from_source(free_nodes, self.min_node_id, self.max_node_id)

        # Reconstruct the path
        return _path_from_prev(prev, self.max_node_id)


    def all_haplotypes(
        self, inference_vcf: str, base_path_prefix: str, region: Range
    ) -> list["InferenceHaplotype"]:
        """Enumerate all possible haplotypes containing alleles in region of inference_vcf using base_path_prefix as backbone"""
        # Extract variant paths from the inference VCF
        inference_alleles = {}
        inference_paths_ordered = []
        with pysam.VariantFile(inference_vcf, drop_samples=True) as vcf_file:
            for record in vcf_file.fetch(**region.pysam_fetch):
                variant = Variant.from_pysam(record)
                if variant_path_name(variant.vg_variant_id, 0) not in self.path_nodes:
                    continue  # Variant "fell outside the graph"
                assert record.alts is not None

                inference_alleles[variant.vg_variant_id] = variant.alt_allele_indices
                inference_paths_ordered.extend(variant_path_name(variant.vg_variant_id, allele_idx) for allele_idx in variant.allele_indices)

        inference_paths = set(inference_paths_ordered)

        free_nodes = set()
        for path, nodes in self.path_nodes.items():
            if path.startswith(base_path_prefix):
                free_nodes.update(nodes)

        _, source_prev = self._from_source(free_nodes)
        _, sink_prev = self._to_sink(free_nodes)

        # Mark nodes to include in the all paths enumeration. Start with the base path so one of the reported
        # haplotypes matches the "true" haplotype.
        include_nodes = set(_path_from_prev(source_prev, self.max_node_id))
        for variant_id, alt_allele_indices in inference_alleles.items():
            path = variant_path_name(variant_id, 0)

            path_start = self.as_id(self.first_handle(path))
            path_end = self.as_id(self.last_handle(path))

            # Add nodes for path into and from variant
            include_nodes.update(path_into := _path_from_prev(source_prev, path_start, drop_start=True))
            include_nodes.update(path_from := _path_from_prev(sink_prev, path_end, reverse=True, drop_start=True))

            # Add nodes for alternate alleles
            include_nodes.update(variant_nodes := set().union(*(self.path_nodes[variant_path_name(variant_id, allele_idx)] for allele_idx in alt_allele_indices)))

            # The absence of the variant might conflict with the base path. Thus we explicitly include the base path above,
            # and here include the reference path. This ensures we have the backbone (i.e., the "true" path) and the explicit
            # reference path for each variant at the cost of increase the number of haplotypes.
            _, interior_prev = self._from_source(free_nodes - variant_nodes, path_into[-1], path_from[0])
            include_nodes.update(_path_from_prev(interior_prev, path_from[0]))

        # DFS to enumerate all possible haplotypes
        assert self.min_node_id in include_nodes and self.max_node_id in include_nodes

        haplotypes = []

        def _generate_all_paths(node: odgi.handle, path: list[int]):
            path = [*path, self._graph.get_id(node)]

            while True:
                next_nodes = []
                self._graph.follow_edges(node, False, lambda n: next_nodes.append(n))
                if len(next_nodes) == 0:
                    # Reached a tip/terminus
                    haplotypes.append(
                        InferenceHaplotype(
                            self,
                            path,
                            set(itertools.chain.from_iterable(self.node_paths[node] for node in path))
                            & inference_paths,
                        )
                    )
                    return

                recurse_nodes = [n for n in next_nodes if self._graph.get_id(n) in include_nodes]
                if len(recurse_nodes) == 1:
                    # Optimize for the common case with no branching
                    node = recurse_nodes[0]
                    path.append(self._graph.get_id(node))
                else:
                    break
            for next_node in recurse_nodes:
                _generate_all_paths(next_node, path)

        _generate_all_paths(self._graph.get_handle(self.min_node_id), [])

        # Label haplotypes with reference paths if they don't have any of the alternate alleles for that variant
        for haplotype in haplotypes:
            for variant_id, alt_allele_indices in inference_alleles.items():
                if haplotype.paths.isdisjoint({variant_path_name(variant_id, a) for a in alt_allele_indices}):
                    haplotype.paths.add(variant_path_name(variant_id, 0))

        # Sort the paths in the order of the VCF records and alleles (leveraging that
        # inference paths are listed in allele order)
        def sort_key(haplotype: InferenceHaplotype):
            return tuple(path in haplotype.paths for path in inference_paths_ordered)
        haplotypes.sort(key=sort_key, reverse=True)

        return haplotypes

    def is_bubble_path(self, path_name: str) -> bool:
        into_outof_nodes = []
        self._graph.follow_edges(
            self.first_handle(path_name), True, lambda n: into_outof_nodes.append(n)
        )
        self._graph.follow_edges(
            self.last_handle(path_name), False, lambda n: into_outof_nodes.append(n)
        )
        return len(into_outof_nodes) == 0

    @classmethod
    def from_vcf(cls, reference_fasta: str, background_vcf: str, region: Range, inference_vcf: str | None = None):
        """Construct graph from VCF

        Args:
            reference_fasta (str): Reference fasta
            background_vcf (str): VCF with background variants, i.e. calls in this individual
            region (Range): Construct graph for variants in this region
            inference_vcf (Optional[str], optional): Additional variants into include graph. Defaults to None.

        Returns:
            _type_: _description_
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Merge a separate inference VCF into the haplotype VCF, matching the samples if needed
            if inference_vcf and inference_vcf != background_vcf:
                with pysam.VariantFile(background_vcf) as background_vcf_file:
                    background_header = background_vcf_file.header
                    background_samples = list(background_header.samples)
                with pysam.VariantFile(inference_vcf) as inference_vcf_file:
                    inference_header = inference_vcf_file.header
                    inference_samples = list(inference_header.samples)

                matched_inference_vcf = os.path.join(temp_dir, "matched_inference.vcf.gz")
                if background_samples == inference_samples:
                    matched_inference_vcf = inference_vcf
                elif set(background_samples).issubset(set(inference_samples)):
                    bcftools.view(
                        "-s",
                        ",".join(background_samples),
                        "-r",
                        str(region),
                        "-Oz",
                        "-o",
                        matched_inference_vcf,
                        inference_vcf,
                    )
                    index_variant_file(matched_inference_vcf)
                else:
                    # Samples to keep from the inference VCF
                    keep_samples = set(background_samples) & set(inference_samples)

                    with pysam.VariantFile(inference_vcf) as src_vcf_file:
                        src_vcf_file.subset_samples(list(keep_samples))

                        src_header = src_vcf_file.header
                        dst_header = pysam.VariantHeader()

                        # Copy existing header fields and make sure GT is present
                        for header_record in src_header.records:
                            if header_record.type in VCF_HEADER_TYPES_TO_COPY and (
                                header_record.type != "INFO" or header_record["ID"] in VCF_INFO_FIELDS_TO_COPY
                            ):
                                dst_header.add_record(header_record)
                        dst_header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')

                        for sample in background_samples:
                            dst_header.add_sample(sample)

                        with pysam.VariantFile(matched_inference_vcf, mode="wz", header=dst_header) as dst_vcf_file:
                            for record in src_vcf_file.fetch(**region.pysam_fetch):
                                # Copy variants, adding no-call samples as needed
                                existing_calls = [{"GT": call["GT"]} for call in record.samples.itervalues()]
                                dst_record = dst_header.new_record(
                                    contig=record.contig,
                                    start=record.start,
                                    stop=record.stop,
                                    alleles=record.alleles,
                                    id=record.id,
                                    info={
                                        key: value
                                        for key, value in record.info.iteritems()
                                        if key in VCF_INFO_FIELDS_TO_COPY
                                    },
                                    samples=existing_calls
                                    + [{"GT": None} for _ in range(len(background_samples) - len(existing_calls))],
                                )
                                dst_vcf_file.write(dst_record)
                    index_variant_file(matched_inference_vcf)

                # When combining the VCFs we can get incompatible records, e.g., an SV overlapping an SNV. We are assuming the background
                # VCF is the true source of genotypes and so should prepare it accordingly.

                merged_graph_vcf = os.path.join(temp_dir, "merged_graph.vcf.gz")
                bcftools.concat(
                    "--allow-overlaps",
                    "--remove-duplicates",
                    "-r",
                    str(region),
                    "-O",
                    "z",
                    "-o",
                    merged_graph_vcf,
                    background_vcf,
                    matched_inference_vcf,
                    catch_stdout=False,
                )
                index_variant_file(merged_graph_vcf)
            else:
                merged_graph_vcf = background_vcf

            # Do we start or end on a variant (such that the graph does not form a bubble)? If so, we we want to extend the flanks
            # to ensure valid and starting ending nodes.
            new_start, new_end = region.start, region.end
            with pysam.VariantFile(merged_graph_vcf, drop_samples=True) as vcf_file:
                for record in vcf_file.fetch(contig=region.contig, start=new_start, end=new_start + 1):
                    if record.start == new_start:
                        new_start = record.start - 1
                for record in vcf_file.fetch(contig=region.contig, start=new_end - 1, end=new_end):
                    if record.stop >= new_end:
                        new_end = record.stop + 1
            if new_start != region.start or new_end != region.end:
                return cls.from_vcf(
                    reference_fasta, background_vcf, Range(region.contig, new_start, new_end), inference_vcf
                )

            # Construct graph (via GFA file)
            gfa_path = os.path.join(temp_dir, "graph.gfa")

            constructor = GraphConstructor(region, merged_graph_vcf)
            constructor.to_gfa(reference_fasta, gfa_path)

            # Append haplotype paths to GFA before loading
            # with open(gfa_path, "a") as gfa_file:
            #     for name, strand, nodes in vcf_to_paths(gfa_path, background_vcf, region):
            #         print("P", name, ",".join(f"{n}{strand}" for n in nodes), "*", sep="\t", file=gfa_file)


            # Construct graph object from GFA file
            graph = cls(gfa_path, region)
            assert graph._graph.is_optimized(), "Graph node space is not compacted"
            return graph

    def test_kmers(self, k: int):
        kmers = []
        partial_kmers = defaultdict(list)

        def kmerize_node(h):
            seq = self._graph.get_sequence(h)
            if seq == "*":
                return

            curr_id = self._graph.get_id(h)
            for i in range(len(seq) - k + 1):
                kmers.append(GraphKmer(self._graph, seq[i : i + k], [curr_id]))

            next_nodes = []
            self._graph.follow_edges(h, False, lambda n: next_nodes.append(n))
            for next_node in next_nodes:
                next_id = self._graph.get_id(next_node)
                partial_kmers[next_id].extend(
                    GraphKmer(self._graph, seq[i:], [curr_id]) for i in range(-min(k - 1, len(seq)), 0)
                )

        self._graph.for_each_handle(kmerize_node)

        # Extend partial kmers until we reach specified length, or a tip
        print(len(kmers), len(partial_kmers))

        # print(partial_kmers)
        next_id, kmers_to_extend = partial_kmers.popitem()
        seq = self._graph.get_sequence(self._graph.get_handle(next_id))
        print(next_id, kmers_to_extend, seq)

        for kmer in kmers_to_extend:
            kmer.sequence += seq[: k - len(kmer.sequence)]
            if len(kmer.sequence) == k:
                kmers.append(kmer)
            else:
                # Need to extend further, find follow-on nodes
                pass
        print(kmers_to_extend)

        #     print(kmer.sequence)

        return kmers


@dataclass
class InferenceHaplotype:
    graph: Graph
    nodes: list[int]
    paths: set[str]

    def sequence(self) -> str:
        return self.graph.sequence(self.nodes)


@dataclass
class GraphKmer:
    graph: Graph
    sequence: str
    node_ids: list[int]

def _path_from_prev(prev, start: int, reverse=False, drop_start=False) -> list[int]:
    path = [start]
    while True:
        if (next_node := prev[path[-1]]) is None:
            break
        path.append(next_node)
    if drop_start:
        path = path[1:]
    return path if reverse else path[::-1]
