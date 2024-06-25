import itertools
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict, Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from shlex import quote
from typing import Dict, List, Optional, Set, Tuple

import odgi
import pysam
from pysam import bcftools

from npsv3 import variant
from npsv3.util.range import Range
from npsv3.util.vcf import index_variant_file

# Length of SHA1 hex digest used for variants IDs by vg
VARIANT_ID_LENGTH = 40

VCF_HEADER_TYPES_TO_COPY = frozenset(["GENERIC", "STRUCTURED", "INFO", "FILTER", "CONTIG"])
VCF_INFO_FIELDS_TO_COPY = frozenset(["SVTYPE", "SVLEN", "END", "CIPOS", "CIEND"])

SAMPLE_PATH_REGEX = re.compile(r"^(?P<sample>[^#]+)#(?P<allele>\d)#(?P<contig>[^#]+)#(?P<count>\d+)$")


def variant_path_name(variant_id: str, allele: int) -> str:
    return f"_alt_{variant_id}_{allele}"


def extract_variant_id(path_name: str) -> str:
    return path_name[5 : VARIANT_ID_LENGTH + 5]


def extract_variant_allele(path_name: str) -> int:
    return int(path_name[VARIANT_ID_LENGTH + 6 :])


def _alt_path_key(path_name: str) -> Tuple[str, int]:
    return (extract_variant_id(path_name), extract_variant_allele(path_name))


def _get_genotype(vcf_path: str, region: Range, variant_id: str, sample: str) -> Tuple[int, ...]:
    with pysam.VariantFile(vcf_path) as vcf_file:
        vcf_file.subset_samples([sample])
        for record in vcf_file.fetch(**region.pysam_fetch):
            if variant.vg_variant_id(record) == variant_id:
                return record.samples[sample]["GT"]
    raise ValueError("Variant not found in VCF")


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
    def max_node_id(self) -> int:
        return self._graph.max_node_id()

    @cached_property
    def variant_paths(self) -> Set[str]:
        paths = set()
        self._graph.for_each_path_handle(
            lambda p: (
                paths.add(self._graph.get_path_name(p)) if self._graph.get_path_name(p).startswith("_alt") else None
            )
        )
        return paths

    @cached_property
    def path_nodes(self) -> Dict[str, Set[int]]:
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
    def node_paths(self) -> Dict[int, List[str]]:
        node_path_lists: Dict[int, List[str]] = {}
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

    def shortest_path(self, base_path_prefix: str):
        """_summary_

        Assumes nodes are in topological order

        Args:
            base_path_prefix (str): _description_

        Returns:
            _type_: _description_
        """
        length = [sys.maxsize] * (self._graph.max_node_id() + 1)
        prev = [None] * (self._graph.max_node_id() + 1)

        free_nodes = set()
        for path, nodes in self.path_nodes.items():
            if path.startswith(base_path_prefix):
                free_nodes.update(nodes)

        ref_nodes = self.path_nodes[self.region.contig]

        length[self._graph.min_node_id()] = 0
        for node in range(self._graph.min_node_id(), self._graph.max_node_id() + 1):
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
            for next_node in next_nodes:
                if length[node] < length[next_node]:
                    length[next_node] = length[node]
                    prev[next_node] = node

        # Reconstruct the path
        path = [self._graph.max_node_id()]
        while True:
            if prev[path[-1]] is None:
                break
            path.append(prev[path[-1]])

        return path[::-1]

    def generate_possible_haplotypes(
        self, inference_vcf: str, base_path_name: str, region: Range
    ) -> List["InferenceHaplotype"]:
        # Extract variant paths from the inference VCF
        inference_paths_ordered = []
        with pysam.VariantFile(inference_vcf, drop_samples=True) as vcf_file:
            for record in vcf_file.fetch(**region.pysam_fetch):
                variant_id = variant.vg_variant_id(record)
                if variant_path_name(variant_id, 0) not in self.path_nodes:
                    continue  # Variant "fell outside the graph"
                assert record.alts is not None
                for allele_idx in range(len(record.alts) + 1):
                    inference_paths_ordered.append(variant_path_name(variant_id, allele_idx))

        inference_paths = set(inference_paths_ordered)

        inference_nodes = set()
        for inference_path in inference_paths:
            inference_nodes.update(self.path_nodes[inference_path])

        # Other variants (not part of inference) are removable
        removable_paths = self.variant_paths - inference_paths

        # Identify nodes on the base path so we can follow those if not specifically exploring an inference variant
        base_path_nodes = self.path_nodes[base_path_name]

        excluded_paths: set[str] = set()
        for _, paths_iter in itertools.groupby(sorted(removable_paths, key=extract_variant_id), extract_variant_id):
            paths = list(paths_iter)

            # fmt: off
            retain_path = next(
                (path for path in paths if self.path_nodes[path] & base_path_nodes), None
            ) or next(
                (path for path in paths if self.path_nodes[path] & inference_nodes), None
            )
            # fmt: on
            excluded_paths.update(path for path in paths if path != retain_path)

        excluded_nodes = set()
        for path in excluded_paths:
            excluded_nodes.update(self.path_nodes[path])

        # Depth-First-Search to enumerate all possible haplotypes for inference variants
        haplotypes = []

        def _generate_all_paths(node, path):
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

                recurse_nodes = [n for n in next_nodes if self._graph.get_id(n) not in excluded_nodes]
                if len(recurse_nodes) == 1:
                    # Optimize for the common case with no branching
                    node = recurse_nodes[0]
                    path.append(self._graph.get_id(node))
                else:
                    break
            for next_node in recurse_nodes:
                _generate_all_paths(next_node, path)

        _generate_all_paths(self.first_handle(base_path_name), [])

        # Sort the paths in the order of the VCF records and alleles (leveraging that
        # inference paths are listed in allele order)
        def sort_key(haplotype: InferenceHaplotype):
            return tuple(path in haplotype.paths for path in inference_paths_ordered)

        haplotypes.sort(key=sort_key, reverse=True)

        return haplotypes

    def generate_haplotype(self, base_path_name: str, inference_paths) -> "InferenceHaplotype":
        inference_nodes = set(itertools.chain.from_iterable(self.path_nodes[path] for path in inference_paths))
        base_path_nodes = self.path_nodes[base_path_name]

        node_id = self._graph.get_id(self.first_handle(base_path_name))
        path = []
        while True:
            path.append(node_id)

            next_nodes = []
            self._graph.follow_edges(self._graph.get_handle(node_id), False, lambda n: next_nodes.append(n))
            if len(next_nodes) == 0:
                # Reached a tip/terminus
                traversed_inference_paths = set(
                    itertools.chain.from_iterable(self.node_paths[node] for node in path)
                ) & set(inference_paths)
                if len(traversed_inference_paths) != len(inference_paths):
                    msg = "Unable to traverse the specified inference path"
                    raise ValueError(msg)
                return InferenceHaplotype(
                    self,
                    path,
                    traversed_inference_paths,
                )

            next_nodes_ids = {self._graph.get_id(next_node) for next_node in next_nodes}
            if len(inference_next_nodes := next_nodes_ids & inference_nodes) > 1:
                msg = "Ambiguous query, multiple possible paths"
                raise ValueError(msg)
            elif len(inference_next_nodes) == 1:
                node_id = next(iter(inference_next_nodes))
            elif len(base_path_next_nodes := next_nodes_ids & base_path_nodes) != 1:
                msg = "Ambiguous query, multiple possible paths"
                raise ValueError(msg)
            else:
                node_id = next(iter(base_path_next_nodes))

    def is_bubble_path(self, path_name: str) -> bool:
        source_nodes, _, _, sink_nodes = self._get_source_and_sink(path_name)
        return len(source_nodes) == 0 and len(sink_nodes) == 0

    def _get_source_and_sink(self, path_name: str, added_nodes: Set[int] = None):
        path = self._graph.get_path_handle(path_name)

        # True to traverse "left" or  "upstream" edges to find source nodes
        source_nodes = []
        start_node = self._graph.get_handle_of_step(self._graph.path_begin(path))
        self._graph.follow_edges(start_node, True, lambda n: source_nodes.append(n))

        # If one of the source nodes is an added "empty" sequence node, then traverse to its sources
        while added_nodes is not None:
            for i, source in enumerate(source_nodes):
                if self._graph.get_id(source) in added_nodes:
                    source_nodes = source_nodes[:i] + source_nodes[i + 1 :]
                    self._graph.follow_edges(source, True, lambda n: source_nodes.append(n))
                    break
            else:
                break  # No added nodes found

        # False to traverse "right" or "downstream" edges to find sink nodes
        sink_nodes = []
        end_node = self._graph.get_handle_of_step(self._graph.path_back(path))
        self._graph.follow_edges(end_node, False, lambda n: sink_nodes.append(n))

        # If one of the sink nodes is an added "empty" sequence node, then traverse to its sinks
        while added_nodes is not None:
            for i, sink in enumerate(sink_nodes):
                if self._graph.get_id(sink) in added_nodes:
                    sink_nodes = sink_nodes[:i] + sink_nodes[i + 1 :]
                    self._graph.follow_edges(sink, False, lambda n: sink_nodes.append(n))
                    break
            else:
                break  # No added nodes found

        return source_nodes, start_node, end_node, sink_nodes

    def _rewrite_edge_with_alt_path(
        self, source_nodes: Iterable[odgi.handle], sink_nodes: Iterable[odgi.handle], new_path_name: str
    ):
        # Identify node pairs that are actually connected by an edge
        filtered_edges = [edge for edge in itertools.product(source_nodes, sink_nodes) if self._graph.has_edge(*edge)]
        assert len(filtered_edges) > 0

        # Create a new node to represent the edge
        new_node = self._graph.create_handle("*")
        for source, sink in filtered_edges:
            if not self._graph.has_edge(source, new_node):
                self._graph.create_edge(source, new_node)
            if not self._graph.has_edge(new_node, new_node):
                self._graph.create_edge(new_node, sink)

        # Create a new path for variant allele
        new_path = self._graph.create_path_handle(new_path_name)
        self._graph.append_step(new_path, new_node)

        # Update paths that traversed the original edge
        for source in source_nodes:
            steps = []
            self._graph.for_each_step_on_handle(source, lambda s: steps.append(s))  # noqa: B023
            for step in steps:
                next_step = self._graph.get_next_step(step)
                if self._graph.is_path_end(next_step):
                    continue
                next_node = self._graph.get_handle_of_step(next_step)
                for sink in sink_nodes:
                    if self._graph.get_id(sink) == self._graph.get_id(next_node):
                        # NOTE: This currently requires a modified version of odgi to work (the current main
                        # leaves deleted steps attached to the node)
                        self._graph.rewrite_segment(step, next_step, [source, new_node, sink])

        # Remove the original edge(s)
        for source, sink in filtered_edges:
            self._graph.destroy_edge(source, sink)

        return new_node

    @classmethod
    def from_vcf(cls, reference_fasta: str, background_vcf: str, region: Range, inference_vcf: str | None = None):
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
                    if record.stop == new_end:
                        new_end = record.stop + 1
            if new_start != region.start or new_end != region.end:
                return cls.from_vcf(
                    reference_fasta, background_vcf, Range(region.contig, new_start, new_end), inference_vcf
                )

            # Use VG to convert VCF to GFA with paths from haplotype VCF

            # 1. Construct vg graph from haplotype VCF, including all alt paths
            graph_path = os.path.join(temp_dir, "graph.vg")
            gbwt_path = os.path.join(temp_dir, "graph.gbwt")
            gfa_path = os.path.join(temp_dir, "graph.gfa")

            # This side effect of keeping of the leading indel bases is the potential for broken haplotypes
            # so instead we add "empty" nodes back into the graph after initial construction

            with open(graph_path, "w") as vg_file:
                construct_command = f"vg construct \
                    --reference {quote(reference_fasta)} \
                    --alt-paths \
                    --handle-sv \
                    --flat-alts 1024 \
                    --node-max 1024 \
                    --region {quote(str(region))} \
                    --vcf {quote(merged_graph_vcf)}"
                subprocess.run(construct_command, shell=True, stdout=vg_file, check=True)

            # 2. Construct GBWT file with haplotype paths
            gbwt_command = f"vg gbwt \
                --xg-name {quote(graph_path)} \
                --vcf-input {quote(background_vcf)} \
                --vcf-region {quote(str(region))} \
                --ignore-missing \
                --output {quote(gbwt_path)}"
            subprocess.run(gbwt_command, shell=True, check=True)

            # 3. Construct GFA file with haplotype paths from VG and GBWT files
            with open(gfa_path, "w") as gfa_file:
                gfa_command = f"vg view {quote(graph_path)}"
                subprocess.run(gfa_command, shell=True, stdout=gfa_file, check=True)

                # Attempt to stream paths from GBWT
                threads_command = f"vg paths --extract-gaf --xg {quote(graph_path)} --gbwt {quote(gbwt_path)}"
                with subprocess.Popen(threads_command, shell=True, stdout=subprocess.PIPE, text=True) as threads:
                    assert threads.stdout is not None
                    while True:
                        line = threads.stdout.readline()
                        if not line and threads.poll() is not None:
                            break

                        path_name, length, _, _, strand, nodes, *_ = line.split("\t", 6)
                        # odgi doesn't seem to support walks, so emit the haplotypes as paths instead
                        # If it did support walks:
                        #   print("W", *path_name.split("#"), length, nodes, sep="\t", file=gfa_file)
                        if int(length) > 0:
                            print(
                                "P",
                                path_name,
                                nodes[1:].replace(">", f"{strand},") + strand,
                                "*",
                                sep="\t",
                                file=gfa_file,
                            )

            # 4. Construct graph object from GFA file
            graph = cls(gfa_path, region)

            # 5. Add additional nodes to the graph for otherwise "empty" deletion and insertion alleles, detecting
            # co-located variants (where we have already made modifications)
            with pysam.VariantFile(merged_graph_vcf, drop_samples=True) as vcf_file:
                added_edges = {}
                added_nodes = set()
                for record in vcf_file.fetch(**region.pysam_fetch):
                    assert record.ref is not None and record.alts is not None and len(record.alts) >= 1
                    ref_allele = record.ref
                    if all(
                        len(alt_allele) > len(ref_allele) and alt_allele.startswith(ref_allele)
                        for alt_allele in record.alts
                    ):
                        # Insertions
                        variant_id = variant.vg_variant_id(record)

                        if not graph.has_path(variant_path_name(variant_id, 1)):
                            continue  # Variant "fell off the end of the graph"
                        assert not graph.has_path(f"_alt_{variant_id}_{0}")

                        source_nodes, _, _, sink_nodes = graph._get_source_and_sink(
                            f"_alt_{variant_id}_{1}", added_nodes
                        )
                        if len(source_nodes) == 0 or len(sink_nodes) == 0:
                            continue  # Variant likely at the end of the graph, such that we can't rewrite

                        key = (
                            frozenset([graph._graph.get_id(n) for n in source_nodes]),
                            frozenset([graph._graph.get_id(n) for n in sink_nodes]),
                        )

                        if new_node := added_edges.get(key):
                            # Node already added (i.e., multiple insertions starting at same position), just add the path to
                            # the already inserted node
                            new_path = graph._graph.create_path_handle(f"_alt_{variant_id}_{0}")
                            graph._graph.append_step(new_path, new_node)
                        else:
                            # Create new node along "reference" path for insertion
                            new_node = graph._rewrite_edge_with_alt_path(
                                source_nodes, sink_nodes, f"_alt_{variant_id}_{0}"
                            )
                            added_edges[key] = new_node
                            added_nodes.add(graph._graph.get_id(new_node))
                    else:
                        for allele_idx, alt_allele in enumerate(record.alts, 1):
                            if len(ref_allele) > len(alt_allele) and ref_allele.startswith(alt_allele):
                                # Deletion
                                variant_id = variant.vg_variant_id(record)

                                if not graph.has_path(variant_path_name(variant_id, 0)):
                                    continue  # Variant "fell off the end of the graph"
                                assert not graph.has_path(f"_alt_{variant_id}_{allele_idx}")

                                source_nodes, _, _, sink_nodes = graph._get_source_and_sink(
                                    f"_alt_{variant_id}_{0}", added_nodes
                                )
                                if len(source_nodes) == 0 or len(sink_nodes) == 0:
                                    continue  # Variant likely at the end of the graph, such that we can't rewrite

                                # We want to rewrite edges that directly connect the source and sink nodes
                                new_node = graph._rewrite_edge_with_alt_path(
                                    source_nodes, sink_nodes, f"_alt_{variant_id}_{allele_idx}"
                                )
                                added_nodes.add(graph._graph.get_id(new_node))

                                # Incomplete paths might end on the source of the deletion. Extend the path where relevant.
                                for source_node in source_nodes:
                                    paths = graph.node_paths[source_node_id := graph._graph.get_id(source_node)]
                                    for path in paths:
                                        match = SAMPLE_PATH_REGEX.match(path)
                                        if not match:
                                            continue
                                        if graph._graph.get_id(graph.last_handle(path)) != source_node_id:
                                            continue
                                        match_gt = _get_genotype(
                                            merged_graph_vcf,
                                            Range(record.contig, record.start, record.stop),
                                            variant_id,
                                            match.group("sample"),
                                        )
                                        if not match_gt or match_gt[int(match.group("allele"))] != allele_idx:
                                            continue

                                        graph._graph.append_step(graph._graph.get_path_handle(path), new_node)

                                # We should not encounter multiple "pure" deletions with the same source and sink sets

            # 6. Sort and compact the graph so nodes are in topological order (and compacted)
            graph._sort_and_compact()
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
    nodes: List[int]
    paths: Set[str]

    def sequence(self) -> str:
        return self.graph.sequence(self.nodes)


@dataclass
class GraphKmer:
    graph: Graph
    sequence: str
    node_ids: List[int]
