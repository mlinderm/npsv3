"""Capture the haplotypes, alleles and flanks that `make_graph_example` uses for one region, to a pickle.

Run from the repo root (see SKILL.md); this is a pytest file only to reuse the `tests` fixtures/helpers
(graph shard cache, Sample fixtures). Configured entirely by environment variables:

  REGION          required, e.g. chr17:940789-941433
  OUT             required, output pickle path
  SAMPLE_FIXTURE  Sample fixture from tests/conftest.py (default hg00096_sample)
  CFG_OVERRIDES   whitespace-separated hydra overrides (default: the HGSVC3/HPRC PAV population setup)
"""

import os
import pickle
import shlex

import hydra
import pytest
import webdataset as wds

import npsv3.graphs.haplotype as hp
import npsv3.images.example as ex

try:
    import npsv3.images.haplotypes as hs
except ImportError:  # older layout: haplotype preparation lived in npsv3.images.example
    hs = None
try:
    from npsv3.graphs.sharding import decode_graph_record
except ImportError:  # older layout: graph shard members were not individually compressed
    def decode_graph_record(record):
        return record
from npsv3.util.range import Range
from npsv3.util.variant import VariantFileReader
from tests import HG38_REF_FASTA, cache_graph_and_filter_kmc_database

_RES = "/storage/mlinderman/projects/sv/npsv3-experiments/resources"
DEFAULT_OVERRIDES = [
    f"reference={HG38_REF_FASTA}",
    f"input={_RES}/hgsvc3-hprc-2024-02-23-mc-chm13/hgsvc3-hprc-2024-02-23-mc-chm13.pav.population.passing.training.hg38.vcf.gz",
    f"reads={_RES}/sequence/HG00096.final.cram",
    "graph.ref_kmer_counts_kmc_prefix=" + _RES + "/Homo_sapiens_assembly38.non_unique.k${graph.kmer_size}",
    "graph.population_prior=True",
    "haplotype.missing_are_ref=True",
    "pileup=unphased_graph_kmer",
    "simulation.replicates=0",  # no simulation needed: we only need haplotypes and the k-mer index inputs
]


def test_capture(hydra_setup, request):  # noqa: ARG001
    region = Range(os.environ["REGION"])
    sample = request.getfixturevalue(os.environ.get("SAMPLE_FIXTURE", "hg00096_sample"))
    overrides = shlex.split(os.environ["CFG_OVERRIDES"]) if "CFG_OVERRIDES" in os.environ else DEFAULT_OVERRIDES
    cfg = hydra.compose(config_name="config", overrides=overrides)

    shards, filtered_kmer_path = cache_graph_and_filter_kmc_database(cfg, sample, cfg.input, region)

    cap = {"region": str(region), "sample": sample.name, "reads": str(cfg.reads), "reference": str(cfg.reference),
           "input": str(cfg.input), "kmer_size": int(cfg.pileup.evidence_kmer_size), "kmer_set": str(cfg.pileup.evidence_kmer_set),
           "ref_kmer_counts": str(cfg.graph.ref_kmer_counts_kmc_prefix), "graph_kmer_size": int(cfg.graph.kmer_size)}

    # Wrap two steps of make_graph_example to observe their inputs/outputs without modifying the source. Haplotype
    # preparation moved from `example` into the `SampledHaplotypeStrategy` (npsv3.images.haplotypes); patch the module
    # that actually calls it. Only the sampled strategy calls `prepare_genotyping_haplotypes`.
    prep_mod = hs if hs is not None and hasattr(hs, "prepare_genotyping_haplotypes") else ex
    orig_prep = prep_mod.prepare_genotyping_haplotypes

    def prep(graph, *args, **kwargs):
        result = orig_prep(graph, *args, **kwargs)
        cap["haps"] = [graph.path_sequence(h) for h in result[0]]
        cap["alleles"] = result[1]
        cap["true"] = result[2]
        return result

    orig_build = ex.build_kmer_evidence_index

    def build(cfg_, graph, sampler, unique_kmers, haplotypes, left_flank, right_flank):
        cap["left_flank"], cap["right_flank"] = left_flank, right_flank
        return orig_build(cfg_, graph, sampler, unique_kmers, haplotypes, left_flank, right_flank)

    # `TrueHaplotypeStrategy` instead calls `true_haplotype_neighbors` (same (graph, ...) -> Haplotypes contract)
    orig_neighbors = getattr(hs, "true_haplotype_neighbors", None)

    def neighbors(graph, *args, **kwargs):
        result = orig_neighbors(graph, *args, **kwargs)
        cap["haps"] = [graph.path_sequence(h) for h in result[0]]
        cap["alleles"] = result[1]
        cap["true"] = result[2]
        return result

    prep_mod.prepare_genotyping_haplotypes, ex.build_kmer_evidence_index = prep, build
    if orig_neighbors is not None:
        hs.true_haplotype_neighbors = neighbors
    try:
        with VariantFileReader.open(cfg.input) as vcf:
            sample_idx = vcf.samples().index(sample.name)
            for record in wds.WebDataset(list(shards), shardshuffle=False):
                if Range(record["region.txt"].decode()) != region:
                    continue
                variants = ex._filter_variants(list(vcf.fetch(region)), 50)
                members = decode_graph_record(record)
                graph = hp.Graph.load_bytes(members["graph.bytes"])
                unique_kmers = hp.UniqueKmersOverlay(graph, members["unique_kmer_overlay.bytes"])
                prior_bytes = members.get("haplotype_prior_overlay.bytes")
                prior = hp.HaplotypePriorOverlay(graph, prior_bytes) if prior_bytes else None
                sampler = hp.HaplotypeSamplerOverlay(
                    graph, unique_kmers, cfg.input, region, 50, prior=prior,
                    params=hydra.utils.instantiate(cfg.haplotype.sampler_params),
                )
                ex.make_graph_example(
                    cfg, graph, sampler, unique_kmers, region, sample_idx, sample, cfg.reads, filtered_kmer_path, variants
                )
                cap["variants"] = [(v.variant_id, str(v).rstrip("\n").split("\t")) for v in variants]
                # The sample's own assembled haplotype paths in the graph (None if absent). `prepare_genotyping_haplotypes` matches
                # these to sampled haplotypes using only the >=50bp analysis variants, so the haplotype used as the "true" one can
                # carry small alleles that the sample's own path does not (e.g. alleles borrowed from the population prior).
                cap["graph_ref_path"] = graph.path_sequence(graph.path_nodes(region.contig))  # hg38 over the graph's region
                cap["graph_region"] = str(graph.region) if hasattr(graph, "region") else None
                cap["true_paths"] = [
                    graph.path_sequence(graph.path_nodes(f"{sample.name}#{i}#{region.contig}#0"))
                    if graph.has_path(f"{sample.name}#{i}#{region.contig}#0") else None
                    for i in range(2)
                ]
                break
            else:
                pytest.fail(f"Region {region} not found in graph shards")
    finally:
        prep_mod.prepare_genotyping_haplotypes, ex.build_kmer_evidence_index = orig_prep, orig_build
        if orig_neighbors is not None:
            hs.true_haplotype_neighbors = orig_neighbors

    with open(os.environ["OUT"], "wb") as f:
        pickle.dump(cap, f)
