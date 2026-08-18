"""Statistical wall-clock benchmarks for `HaplotypeSamplerOverlay` sampling, via pytest-benchmark.

    uv run pytest tests/benchmarks --benchmark-only

pytest-benchmark calibrates its own round/iteration count per test, so these are more
statistically robust than a hand-rolled loop. Save and compare against a baseline:

    uv run pytest tests/benchmarks --benchmark-only --benchmark-autosave
    uv run pytest tests/benchmarks --benchmark-only --benchmark-compare=0001 --benchmark-compare-fail=mean:10%

For flamegraphs/leaf-frame profiling or memory profiling, use
`bench_haplotype_sampling.py` instead (see README.md's "Benchmarking" section).
pytest-benchmark's calibration re-runs the target an unpredictable number of times,
which makes profiler output harder to interpret than a fixed iteration count.
"""
import os

import pytest

from .. import HG38_REF_FASTA, RESOURCES_DIR
from . import _haplotype_harness as _harness

pytestmark = pytest.mark.skipif(not HG38_REF_FASTA, reason="HG38 reference FASTA not found")


@pytest.mark.cfg_overrides(
    f"reference={HG38_REF_FASTA}",
    f"kmer.ref_kmer_counts_kmc_prefix={os.path.join(RESOURCES_DIR,'Homo_sapiens_assembly38.non_unique.k${kmer.kmer_size}')}",
)
@pytest.mark.parametrize("region_name", sorted(_harness.REGIONS))
def test_sample_haplotypes_and_diplotypes(benchmark, cfg, hg00733_sample, region_name, tmp_path):
    missing = _harness.missing_inputs(cfg, hg00733_sample)
    if missing:
        pytest.skip("; ".join(missing))

    sampler, counts = _harness.build_sampler(cfg, hg00733_sample, region_name, tmp_path)

    benchmark.extra_info["region"] = region_name
    benchmark.extra_info["num_kmers"] = sampler.num_kmers()

    # `benchmark.pedantic` (fixed rounds/iterations) instead of plain `benchmark()`
    # (auto-calibrated) to avoid very long runs due to automatically increasing duration.
    rounds = 100 if region_name == "small" else 2
    benchmark.pedantic(
        _harness.sample_once,
        args=(sampler, counts),
        kwargs={"max_haplotypes": cfg.kmer.max_haplotypes, "max_diplotypes": cfg.kmer.max_diplotypes},
        rounds=rounds,
        iterations=1,
        warmup_rounds=1,
    )
