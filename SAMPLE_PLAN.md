# Implementation Plan: `top_k_genotype` command

## Key design decisions from the codebase

- **Sample path naming:** Haplotype paths in the graph are named `{sample}#{haplotype_idx}#{contig}#{segment_idx}` (graph.cpp:1090). A haplotype may span multiple segments, so getting the true diplotype means finding all path segments for each haplotype copy and concatenating their nodes.
- **Variant allele bitsets:** The graph's `node_variant_paths_` maps each node to the set of variant allele paths (inference paths) it belongs to. `HaplotypeSamplerOverlay` already uses this internally (via `inference_node_mask_` + `inference_path_mask_`) to compute `covered_paths` during DP. We need to expose this as a callable method.
- **True genotype comparison:** Rather than comparing raw `NodeIdSeq`s (which may differ due to segmentation), compare the variant allele `PathIdSet` derived from the true haplotype paths against those of each sampled diplotype — unordered (since phase doesn't matter for aggregate stats).
- **`SampleDiplotypes` does not expose covered alleles:** It returns `NodeIdSeq` pairs only. To compare against the true diplotype we need a separate `CoveredAlleles(NodeIdSeq)` method on `HaplotypeSamplerOverlay`.

---

## Step 1 — C++: add `CoveredAlleles` method to `HaplotypeSamplerOverlay`

**Files:** `src/npsv3/native/haplotype.hpp`, `src/npsv3/native/haplotype.cpp`

Add a public method:
```cpp
/// Returns the set of inference allele paths covered by the given NodeIdSeq,
/// using the active inference node/path masks. Empty bitset if no inference VCF
/// was provided at construction.
Graph::PathIdSet CoveredAlleles(const Graph::NodeIdSeq& path) const;
```

Implementation: iterate over nodes in the path, test each against `inference_node_mask_`, OR together `node_variant_paths_[nid] & inference_path_mask_`. This is the same logic as the `covered_paths` accumulation in `FindBestPaths`.

---

## Step 2 — C++: add `Graph::HaplotypePaths` helper

**Files:** `src/npsv3/native/graph.hpp`, `src/npsv3/native/graph.cpp`

Add a method:
```cpp
/// Enumerate all path names with the given prefix and return a concatenated NodeIdSeq.
/// Used to gather all segments of one sample haplotype (e.g., prefix = "SAMPLE#0#chr1").
NodeIdSeq HaplotypePaths(const std::string& prefix) const;
```

This is needed because sample haplotypes span multiple path segments (`SAMPLE#0#chr1#0`, `SAMPLE#0#chr1#1`, ...) and we need to merge them in order.

---

## Step 3 — Python bindings: expose `KmerCounts` and `HaplotypeSamplerOverlay`

**File:** `src/npsv3/native/graph_bindings.cpp`

Add nanobind bindings for:

**`KmerCounts`:**
- Constructor: `KmerCounts(db_path: str, coverage: float = 0.0)` — wraps the existing C++ constructor with default params (absent/het/homo fractions all have defaults)

**`HaplotypeSamplerOverlay`:**
- Constructor 1 (no VCF filter): `(graph, k, max_edge, counts)`
- Constructor 2 (VCF-aware): `(graph, k, max_edge, counts, inference_vcf, region, min_size=50)`
- `sample_haplotypes(n: int) -> list[list[int]]`
- `sample_diplotypes(candidates: list[list[int]], max_diplotypes: int = 1) -> list[tuple[list[int], list[int]]]`
- `covered_alleles(path: list[int]) -> list[bool]` — returns the `PathIdSet` bitset as a `list[bool]`

**`Graph.haplotype_paths(prefix: str) -> list[int]`** — add to existing Graph binding.

---

## Step 4 — Python: implement `top_k_genotype` logic

**New file:** `src/npsv3/topk.py`

```python
def top_k_genotype(cfg, vcf_path, sample_kmc_map, output_path):
    """
    For each variant region:
    1. Build graph
    2. For each sample with a KMC file:
       - Get true diplotype from graph sample paths
       - Sample haplotypes with HaplotypeSamplerOverlay (VCF-aware)
       - Score all diplotype pairs with SampleDiplotypes
       - Find rank of true diplotype in sorted list
    3. Accumulate and output aggregate statistics
    """
```

`sample_kmc_map`: `dict[str, str]` mapping sample names to KMC db paths (without extension).

**Core per-region loop:**
```
for region, variants in overlapping_variants(vcf_file):
    graph = Graph(reference, vcf_path, region)

    for sample, kmc_path in sample_kmc_map.items():
        # True genotype: get haplotype paths from graph
        hap0 = graph.haplotype_paths(f"{sample}#0#{region.contig}")
        hap1 = graph.haplotype_paths(f"{sample}#1#{region.contig}")

        counts = KmerCounts(kmc_path, coverage=cfg.coverage)
        sampler = HaplotypeSamplerOverlay(graph, cfg.kmer_size, cfg.max_edge,
                                          counts, vcf_path, region, cfg.min_sv_size)

        candidates = sampler.sample_haplotypes(cfg.n_haplotypes)
        diplotypes = sampler.sample_diplotypes(candidates, cfg.max_diplotypes)

        # Convert true haplotypes to allele bitset
        true_alleles = (sampler.covered_alleles(hap0), sampler.covered_alleles(hap1))

        # Find rank of true diplotype (unordered comparison)
        rank = _find_rank(true_alleles, diplotypes, sampler)

        # Accumulate stats
        stats.record(region, sample, rank, len(diplotypes))
```

`_find_rank`: converts each diplotype's pair of `NodeIdSeq` to allele bitsets via `covered_alleles`, then checks for unordered equality with `true_alleles`. Returns 1-based rank or `None` if not found.

**Aggregate statistics output (JSON):**
```json
{
  "total_variant_regions": 100,
  "total_sample_regions": 400,
  "rank_counts": {"1": 280, "2": 40, "...": "...", "not_found": 30},
  "recall_at_k": {"1": 0.70, "5": 0.92, "10": 0.95}
}
```

---

## Step 5 — Hydra config

**New file:** `src/npsv3/conf/command/top_k_genotype.yaml`
```yaml
# @package _global_
command: top_k_genotype

defaults:
  - kmc: ???          # Must be provided: a kmc config group file, or omitted in favour of kmc= on CLI

n_haplotypes: 10      # Haplotypes to sample before diplotype scoring
max_diplotypes: 10    # Top-k diplotypes to keep
kmer_size: 31
max_edge: 3
min_sv_size: 50
coverage: 0.0         # 0.0 = auto-estimate from KMC histogram
```

### KMC file specification

KMC paths are provided via Hydra config groups, matching the pattern already used in this project for `pileup`, `model`, etc. The `kmc` config group lives under `src/npsv3/conf/kmc/`.

**Simple case — single sample, specified directly on the command line:**

No config file needed. Use a Hydra dict override:
```
npsv3 command=top_k_genotype ... 'kmc={SAMPLE1: /path/to/sample1}'
```

Or with `++` to set a single key:
```
npsv3 command=top_k_genotype ... ++kmc.SAMPLE1=/path/to/sample1
```

`cfg.kmc` becomes a single-entry `DictConfig`; the handler iterates it like any other case. No special-casing required.

**Multi-sample case — config group file:**

Create a YAML file for the cohort/experiment:
```yaml
# src/npsv3/conf/kmc/my_cohort.yaml
# @package kmc
SAMPLE1: /path/to/sample1
SAMPLE2: /path/to/sample2
```

Then select it on the command line:
```
npsv3 command=top_k_genotype ... +kmc=my_cohort
```

Config group files are version-controlled alongside the experiment, and Hydra logs the resolved config automatically for reproducibility.

In both cases `cfg.kmc` is a `DictConfig` that the handler iterates with `.items()` — no detection logic needed.

---

## Step 6 — Wire into `main.py`

```python
elif cfg.command == "top_k_genotype":
    from npsv3.topk import top_k_genotype

    _make_paths_absolute(cfg, ["reference"])
    output = "top_k_stats.json" if OmegaConf.is_missing(cfg, "output") else \
        hydra.utils.to_absolute_path(cfg.output)

    # cfg.kmc is a DictConfig {sample_name: kmc_path} in all cases
    sample_kmc_map = OmegaConf.to_container(cfg.kmc)

    top_k_genotype(cfg, hydra.utils.to_absolute_path(cfg.input), sample_kmc_map, output)
```

---

## Files to create/modify

| File | Change |
|------|--------|
| `src/npsv3/native/haplotype.hpp` | Add `CoveredAlleles()` declaration |
| `src/npsv3/native/haplotype.cpp` | Implement `CoveredAlleles()` |
| `src/npsv3/native/graph.hpp` | Add `HaplotypePaths()` declaration |
| `src/npsv3/native/graph.cpp` | Implement `HaplotypePaths()` |
| `src/npsv3/native/graph_bindings.cpp` | Bind `KmerCounts`, `HaplotypeSamplerOverlay`, `Graph.haplotype_paths` |
| `src/npsv3/topk.py` | New — core command logic |
| `src/npsv3/conf/command/top_k_genotype.yaml` | New — Hydra config |
| `src/npsv3/conf/kmc/` | New config group directory; one YAML per cohort for multi-sample use |
| `src/npsv3/main.py` | Add `top_k_genotype` branch |
