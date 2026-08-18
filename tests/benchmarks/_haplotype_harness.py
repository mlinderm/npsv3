"""Shared setup for the haplotype-sampling benchmarks in this directory.

Builds sampler + k-mer classifier directly via `_create_graph_and_sampler` (one graph
for the whole literal `Range`), the same direct-construction path used by e.g.
`test_graph_construction_errors`/`test_overlapping_alleles` in tests/graphs/test_haplotype.py
for this exact "large" region -- not `serialize_graph_and_unique_kmers`, which instead
splits a region into separate per-cluster graphs via `overlapping_variants()`. That
matters here: the "large" benchmark region deliberately spans one dense 3,786-variant
cluster meant to become a single, worst-case merged graph/automaton (as documented in
`src/npsv3/native/PERFORMANCE_NOTES.md`), but `overlapping_variants()` actually splits
it into ~21 independent smaller clusters, which is the wrong shape for that benchmark.
No Ray needed for any of this -- both `_create_graph_and_sampler` and the KMC filtering
below are local/subprocess calls.
"""
import os

from npsv3._native_graph import KmerClassify, KmerCounts
from npsv3.graphs.haplotype import HaplotypeSamplerOverlay, _create_graph_and_sampler
from npsv3.util.range import Range
from npsv3.util.sample import Sample

from .. import RESOURCES_DIR, _first_existing, cache_filter_kmc_database

# Two fixed regions from the HG00733 population VCF, chosen as a
# representative-scale ("small") and worst-case dense-variant-cluster ("large") target.
REGIONS = {
    "small": Range("chr1:148538120-148538280"),
    "large": Range("chr1:148531170-148577610"),
}

HG00733_POPULATION_VCF = os.path.join(
    RESOURCES_DIR, "HG00733.hgsvc3-hprc-2024-02-23.dipcall.population.passing.hg38.vcf.gz"
)

def hg00733_sample() -> Sample:
    """Construct the HG00733 `Sample`, mirroring the `hg00733_sample` fixture in tests/conftest.py.

    Kept as a plain function (rather than reusing the fixture) so the standalone CLI
    script, which doesn't run under pytest, can build the same sample.
    """
    sample = Sample(
        "HG00733",
        mean_coverage=31.41,
        mean_insert_size=454.57,
        std_insert_size=106.63,
        sequencer="HSXn",
        read_length=150,
        kmer_coverage=14,
    )
    kmc_prefix_path = _first_existing(
        os.path.join(RESOURCES_DIR, "sequence", "HG00733.final.k31.kmc_pre"),
    )
    if kmc_prefix_path is not None:
        sample.kmc_prefix, *_ = os.path.splitext(kmc_prefix_path)
    return sample


def missing_inputs(cfg, sample: Sample) -> list[str]:
    """Return human-readable reasons required inputs are missing, empty if all present."""
    reasons = []
    if not cfg.reference or not os.path.exists(cfg.reference):
        reasons.append(f"reference FASTA not found: {cfg.reference}")
    if not os.path.exists(HG00733_POPULATION_VCF):
        reasons.append(f"HG00733 population VCF not found: {HG00733_POPULATION_VCF}")
    ref_counts_prefix = str(cfg.kmer.ref_kmer_counts_kmc_prefix)
    if not os.path.exists(f"{ref_counts_prefix}.kmc_pre"):
        reasons.append(f"reference k-mer counts not found: {ref_counts_prefix}")
    if not sample.kmc_prefix:
        reasons.append(f"KMC database for {sample.name} not found")
    return reasons


def resolve_region(region_name: str) -> Range:
    """Resolve `region_name` to a `Range`: a preset key in `REGIONS` if it matches one,
    otherwise parsed directly as a literal region string (e.g. `chr1:31431661-31432319`).

    Lets callers (the CLI script, `scan_dense_clusters.py`'s output, ad hoc investigation)
    target any region -- not just the two fixed presets -- without a code change here.
    """
    return REGIONS.get(region_name, Range(region_name))


def build_sampler(cfg, sample: Sample, region_name: str, tmp_path) -> tuple[HaplotypeSamplerOverlay, KmerClassify]:
    """Construct the sampler and k-mer classifier for `region_name`.

    `region_name` may be a preset key in `REGIONS` (`"small"`/`"large"`) or an arbitrary
    literal region string -- see `resolve_region`.

    The filtered k-mer database is cached under tests/results/ (via
    `cache_filter_kmc_database`, keyed by region + VCF hash + sample), so repeat runs
    only pay the (fast, subprocess-based) `kmc_tools` cost once. Graph/sampler
    construction itself is not cached -- per PERFORMANCE_NOTES.md's fix 6 benchmark,
    it only takes ~2-3s even for the large worst-case region, well under the cost of
    the KMC filtering step it would otherwise need to be cached alongside.
    """
    region = resolve_region(region_name)
    ref_kmer_counts = KmerCounts(str(cfg.kmer.ref_kmer_counts_kmc_prefix))
    _graph, unique_kmers, sampler = _create_graph_and_sampler(
        cfg.reference,
        HG00733_POPULATION_VCF,
        region,
        k=cfg.kmer.kmer_size,
        ref_kmer_counts=ref_kmer_counts,
    )
    filtered_kmer_path = cache_filter_kmc_database(cfg, sample, HG00733_POPULATION_VCF, region, unique_kmers, tmp_path=tmp_path)
    counts = KmerClassify(filtered_kmer_path, sample.kmer_coverage)
    return sampler, counts


def sample_once(sampler, counts, *, max_haplotypes: int, max_diplotypes: int):
    """One full sampling iteration: reset scores, then sample haplotypes/diplotypes.

    Resetting scores every iteration is required for repeatable timing. `UpdateScores`'s
    homozygous_discount/het_adjustment can drift across repeated calls on the same sampler
    otherwise.
    """
    sampler.initialize_scores(counts)
    haplotypes = sampler.sample_haplotypes(n=max_haplotypes)
    diplotypes = sampler.sample_diplotypes(haplotypes, n=max_diplotypes)
    assert len(haplotypes) > 0, "No haplotypes sampled"
    return haplotypes, diplotypes
