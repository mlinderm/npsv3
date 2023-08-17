import itertools
import os
import subprocess
import tempfile
from dataclasses import dataclass
from functools import cached_property
from shlex import quote
from typing import Dict, Iterable, List, Optional, Set, Tuple

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


def variant_path_name(variant_id: str, allele: int) -> str:
    return f"_alt_{variant_id}_{allele}"


def extract_variant_id(path_name: str) -> str:
    return path_name[5 : VARIANT_ID_LENGTH + 5]


def extract_variant_allele(path_name: str) -> int:
    return int(path_name[VARIANT_ID_LENGTH + 6 :])


def _alt_path_key(path_name: str) -> Tuple[str, int]:
    return (extract_variant_id(path_name), extract_variant_allele(path_name))


class Graph:
    def __init__(self, gfa_path: str, region: Range):
        with tempfile.TemporaryDirectory() as temp_dir:
            og_path = os.path.join(temp_dir, "graph.og")
            # Build graph with odgi executable
            build_command = f"odgi build -g {quote(gfa_path)} -o {quote(og_path)}"
            subprocess.run(build_command, shell=True, check=True)

            # Load graph into Python object
            self._graph = odgi.graph()
            self._graph.load(og_path)

        self.region = region

    def nodes_on_path(self, path_name: str) -> Iterable[int]:
        nodes = []
        self._graph.for_each_step_in_path(
            self._graph.get_path_handle(path_name),
            lambda s: nodes.append(self._graph.get_id(self._graph.get_handle_of_step(s))),
        )
        return nodes

    @cached_property
    def variant_paths(self) -> Set[str]:
        paths = set()
        self._graph.for_each_path_handle(
            lambda p: paths.add(self._graph.get_path_name(p))
            if self._graph.get_path_name(p).startswith("_alt")
            else None
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

    def first_handle(self, path_name: str) -> odgi.handle:
        path = self._graph.get_path_handle(path_name)
        return self._graph.get_handle_of_step(self._graph.path_begin(path))

    def sequence(self, nodes: Iterable[int]) -> str:
        seq = ""
        for node in nodes:
            node_seq = self._graph.get_sequence(self._graph.get_handle(node))
            if node_seq != "*":  # Used for "deletion" and "insertion" edges
                seq += node_seq
        return seq

    def generate_possible_haplotypes(self, inference_vcf: str, base_path_name: str, region: Range) -> List["InferenceHaplotype"]:
        # Extract variant paths from the inference VCF
        inference_paths_ordered = []
        with pysam.VariantFile(inference_vcf, drop_samples=True) as vcf_file:
            for record in vcf_file.fetch(**region.pysam_fetch):
                variant_id = variant.vg_variant_id(record)
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

        excluded_paths: Set[str] = set()
        for _, paths_iter in itertools.groupby(sorted(removable_paths, key=extract_variant_id), extract_variant_id):
            paths = list(paths_iter)
            assert len(paths) >= 2  # noqa: PLR2004
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

            next_nodes = []
            self._graph.follow_edges(node, False, lambda n: next_nodes.append(n))
            if len(next_nodes) == 0:
                # Reached a tip/terminus
                haplotypes.append(
                    InferenceHaplotype(
                        self,
                        path,
                        set(itertools.chain.from_iterable(self.node_paths[node] for node in path)) & inference_paths,
                    )
                )
                return

            for next_node in next_nodes:
                next_node_id = self._graph.get_id(next_node)
                if next_node_id not in excluded_nodes:
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

    def _get_source_and_sink(self, path_name: str):
        path = self._graph.get_path_handle(path_name)

        # True to traverse "left" or  "upstream" edges to find source nodes
        source_nodes = []
        start_node = self._graph.get_handle_of_step(self._graph.path_begin(path))
        self._graph.follow_edges(start_node, True, lambda n: source_nodes.append(n))

        # False to traverse "right" or "downstream" edges to find sink nodes
        sink_nodes = []
        end_node = self._graph.get_handle_of_step(self._graph.path_back(path))
        self._graph.follow_edges(end_node, False, lambda n: sink_nodes.append(n))

        return source_nodes, start_node, end_node, sink_nodes

    def _rewrite_edge_with_alt_path(
        self, source_nodes: Iterable[odgi.handle], sink_nodes: Iterable[odgi.handle], new_path_name: str
    ):
        new_node = self._graph.create_handle("*")
        for source in source_nodes:
            self._graph.create_edge(source, new_node)
        for sink in sink_nodes:
            self._graph.create_edge(new_node, sink)

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
        for source, sink in itertools.product(source_nodes, sink_nodes):
            if self._graph.has_edge(source, sink):
                self._graph.destroy_edge(source, sink)

    @classmethod
    def from_vcf(cls, reference_fasta: str, background_vcf: str, region: Range, inference_vcf: Optional[str] = None):
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

            # Use VG to convert VCF to GFA with paths from haplotype VCF

            # 1. Construct vg graph from haplotype VCF, including all alt paths
            graph_path = os.path.join(temp_dir, "graph.vg")
            gbwt_path = os.path.join(temp_dir, "graph.gbwt")
            gfa_path = os.path.join(temp_dir, "graph.gfa")

            # --no-trim-indels \
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

            # 5. Add additional nodes to the graph for otherwise "empty" deletion and insertion alleles
            with pysam.VariantFile(merged_graph_vcf, drop_samples=True) as vcf_file:
                for record in vcf_file.fetch(**region.pysam_fetch):
                    assert record.ref is not None and record.alts is not None
                    ref_allele = record.ref
                    if all(
                        len(alt_allele) > len(ref_allele) and alt_allele.startswith(ref_allele)
                        for alt_allele in record.alts
                    ):
                        # Insertions
                        variant_id = variant.vg_variant_id(record)

                        assert not graph.has_path(f"_alt_{variant_id}_{0}")
                        assert len(record.alts) >= 1

                        source_nodes, _, _, sink_nodes = graph._get_source_and_sink(f"_alt_{variant_id}_{1}")

                        graph._rewrite_edge_with_alt_path(source_nodes, sink_nodes, f"_alt_{variant_id}_{0}")
                    else:
                        for allele_idx, alt_allele in enumerate(record.alts, 1):
                            if len(ref_allele) > len(alt_allele) and ref_allele.startswith(alt_allele):
                                # Deletion
                                variant_id = variant.vg_variant_id(record)

                                assert not graph.has_path(f"_alt_{variant_id}_{allele_idx}")

                                source_nodes, _, _, sink_nodes = graph._get_source_and_sink(f"_alt_{variant_id}_{0}")
                                graph._rewrite_edge_with_alt_path(
                                    source_nodes, sink_nodes, f"_alt_{variant_id}_{allele_idx}"
                                )

            return graph


@dataclass
class InferenceHaplotype:
    graph: Graph
    nodes: List[int]
    paths: Set[str]

    def sequence(self) -> str:
        return self.graph.sequence(self.nodes)
