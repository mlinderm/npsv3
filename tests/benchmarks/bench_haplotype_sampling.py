"""Repeatable, external-tool-friendly benchmark for `HaplotypeSamplerOverlay` sampling.

Builds the sampler directly (graph/sampler construction, plus a cached filtered k-mer
DB), with no Ray/pytest/cProfile in the timed loop -- the same isolation method used to
produce every number in `src/npsv3/native/PERFORMANCE_NOTES.md`. Run standalone for a
quick timing, or point an external profiler at it, e.g.:

    # CPU flamegraph (Python + native frames)
    py-spy record --native --rate 100 -f speedscope -o profile.json -- \\
        python3 -m tests.benchmarks.bench_haplotype_sampling --region large --iterations 3

    # Allocations (Python + native frames)
    memray run --native -o memray-small.bin -m tests.benchmarks.bench_haplotype_sampling \\
        --region small --iterations 60

    # Native heap profile (slow -- small region only)
    valgrind --tool=massif --massif-out-file=massif.out \\
        python3 -m tests.benchmarks.bench_haplotype_sampling --region small --iterations 5

    # Statistical wall-clock A/B across commits
    hyperfine --warmup 1 \\
        'python3 -m tests.benchmarks.bench_haplotype_sampling --region small --iterations 20'

See README.md's "Benchmarking" section for tool-specific notes (RelWithDebInfo builds
for symbol resolution, jemalloc/valgrind interaction, etc).

Must be run as a module (`python3 -m tests.benchmarks.bench_haplotype_sampling ...`) from
the repository root so the relative imports in this package resolve.
"""
# ruff: noqa: T201
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

import hydra

import npsv3

from .. import HG38_REF_FASTA, RESOURCES_DIR
from . import _haplotype_harness as _harness


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _check_load_average() -> None:
    load1, _load5, _load15 = os.getloadavg()
    cpu_count = os.cpu_count() or 1
    if load1 > 0.5 * cpu_count:
        print(
            f"WARNING: 1-minute load average ({load1:.1f}) is high relative to CPU count ({cpu_count})",
            file=sys.stderr,
        )

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region", choices=sorted(_harness.REGIONS), default="small")
    parser.add_argument("--reference", default=None, help="Reference FASTA (default: autodetected test reference)")
    parser.add_argument("--iterations", type=int, default=None, help="Timed iterations (default: 100 for small, 2 for large)")
    parser.add_argument("--warmup", type=int, default=2, help="Untimed warmup iterations")
    parser.add_argument("--max-haplotypes", type=int, default=None, help="Default: cfg.kmer.max_haplotypes")
    parser.add_argument("--max-diplotypes", type=int, default=None, help="Default: cfg.kmer.max_diplotypes")
    parser.add_argument("--json", metavar="PATH", help="Write per-iteration timings and summary stats as JSON to PATH")
    parser.add_argument("--no-check-load", action="store_true", help="Skip the load-average warning")
    args = parser.parse_args()

    if not args.no_check_load:
        _check_load_average()

    reference = args.reference or HG38_REF_FASTA
    if not reference:
        print("No reference FASTA found; pass --reference explicitly.", file=sys.stderr)
        return 1

    # Load Hydra configuration
    try:
        hydra.initialize_config_dir(config_dir= os.path.join(os.path.dirname(npsv3.__file__), "conf"), version_base=None)
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"reference={reference}",
                f"kmer.ref_kmer_counts_kmc_prefix={os.path.join(RESOURCES_DIR,'Homo_sapiens_assembly38.non_unique.k${kmer.kmer_size}')}",
            ],
        )
    finally:
            hydra.core.global_hydra.GlobalHydra.instance().clear()  # type: ignore

    sample = _harness.hg00733_sample()
    missing = _harness.missing_inputs(cfg, sample)
    if missing:
        print("Missing required inputs:", file=sys.stderr)
        for reason in missing:
            print(f"  - {reason}", file=sys.stderr)
        return 1

    max_haplotypes = args.max_haplotypes or cfg.kmer.max_haplotypes
    max_diplotypes = args.max_diplotypes or cfg.kmer.max_diplotypes
    iterations = args.iterations or (100 if args.region == "small" else 2)

    with tempfile.TemporaryDirectory() as tmp_dir:
        sampler, counts = _harness.build_sampler(cfg, sample, args.region, tmp_dir)
    print(f"region={args.region} num_kmers={sampler.num_kmers()} max_haplotypes={max_haplotypes} max_diplotypes={max_diplotypes}")

    for _ in range(args.warmup):
        _harness.sample_once(sampler, counts, max_haplotypes=max_haplotypes, max_diplotypes=max_diplotypes)

    durations_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        _harness.sample_once(sampler, counts, max_haplotypes=max_haplotypes, max_diplotypes=max_diplotypes)
        durations_ms.append((time.perf_counter() - start) * 1000)

    mean_ms = statistics.mean(durations_ms)
    stdev_ms = statistics.stdev(durations_ms) if len(durations_ms) > 1 else 0.0
    print(
        f"iterations={iterations} mean={mean_ms:.2f}ms stdev={stdev_ms:.2f}ms "
        f"min={min(durations_ms):.2f}ms max={max(durations_ms):.2f}ms"
    )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "region": args.region,
                    "num_kmers": sampler.num_kmers(),
                    "iterations": iterations,
                    "warmup": args.warmup,
                    "max_haplotypes": max_haplotypes,
                    "max_diplotypes": max_diplotypes,
                    "mean_ms": mean_ms,
                    "stdev_ms": stdev_ms,
                    "min_ms": min(durations_ms),
                    "max_ms": max(durations_ms),
                    "durations_ms": durations_ms,
                    "git_commit": _git_commit(),
                },
                f,
                indent=2,
            )
        print(f"Wrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
