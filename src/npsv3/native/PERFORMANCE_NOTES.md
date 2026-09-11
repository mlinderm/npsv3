# HaplotypeSamplerOverlay / KmerClassify performance notes

Profiling target: `HaplotypeSamplerOverlay::SampleHaplotypes` (`haplotype.cpp`/`haplotype.hpp`) and its
`KmerClassify::InitializeScores` dependency (`kmer.cpp`/`kmer.hpp`). This file documents the **current**
architecture and hot-spot profile, plus a condensed record of what's already been tried so it isn't
re-attempted. For open/unimplemented ideas, see `OPTIMIZATION_PROPOSALS.md`.

Consolidated 2026-08-19 from a much longer, session-by-session investigation log (fixes 1-9, Proposal 2,
Stage 0-2). Updated same day with Fix 9 (`SortAndTrimBacktrack`'s single-pass dedup+trim, replacing its
former double-sort).

## Current architecture (as of 2026-08-19)

The DP is a forward pass over (graph node) × (Aho-Corasick automaton state) that produces up to `n` best-
scoring haplotype paths per call. Current design decisions, each shipped after the profiling that motivated
it (see "Condensed history" below for dates/numbers):

- **`SampleHaplotypes(n)` runs one fixed-width `PropagateBestPathState(samples.size()+2)` pass per
  haplotype draw** — no adaptive beam-doubling, no `ComputeScoreToGo` backward sweep. That machinery
  (`ComputeScoreToGo`, `PropagateBestPathStateAdaptively`, `FindBestPathsFixedWidth`, etc.) was deleted from
  the tree entirely, not just bypassed, once validated exact-equivalent to the old certified path across
  450+ synthetic-adversarial and real-KMC-data trials.
- **Every `(node, automaton_state)` backpointer pool — root, sink, and pending/mid-k-mer-match alike — is
  width-trimmed to `n` in `SortAndTrimBacktrack`.** Previously only root/sink pools were capped; pending
  pools were only exact-duplicate-merged, which could grow unboundedly on dense real regions (confirmed
  >64GB OOM at `chr1:31431661-31432319`, 521 variants/659bp — real HG00733 KMC coverage, not synthetic).
  Validated exact via a same-pool score-invariance proof (every entry in a pool shares the same
  `score_to_go`, so trimming by current score can't discard the eventual winner) plus 240+ trials.
- **`SortAndTrimBacktrack` dedups and trims in one linear pass against a small `kept` buffer (capacity
  `n`), not two full comparison sorts.** The result's order is otherwise unspecified — descending-score
  order is only restored (one cheap sort of the ≤n survivors) at the single call site whose consumer needs
  it (`SampleHaplotypes`'s greedy per-draw walk / `FindBestPaths`'s sorted-order contract, both fed only by
  the final sink-merge pool). Every other call site's result only feeds further DP propagation via stable
  stored indices (`Backpointer::pred_path_idx`), which don't depend on pool order.
- **`AutomatonState::goto_` is a hand-rolled `GotoMap`** (`haplotype.hpp`) — a fixed inline 4-slot array
  with linear scan, falling back to `unordered_map` only past that capacity (matches ~99% of automaton
  states having ≤2 `goto_` children; only `kAutomatonRoot`-like wide states spill over).
- **`AutomatonState::output_kmer_indices_` is a sparse `vector<uint32_t>`**, not a dense bitset — median
  popcount across states is 1. A `trie_symbol_mask_` bitset short-circuits `AutomatonGoto` straight to
  root for node ids that never appear anywhere in the trie (~40%+ of graph nodes at dense regions).
- **`Backpointer::covered_paths` is an immutable, refcounted `SharedPathIdSet`**
  (`boost::intrusive_ptr<const PathIdSetHolder>`), not a per-transition deep copy. A `contributes_paths_mask_`
  bitset lets the DP skip the allocate+OR+hash cost entirely for transitions into nodes that add no new
  path bits (reference-only stretches between variants — the common case genome-wide).
- **`KmerClassify` keeps one `CKMCFile` handle open for its lifetime** (`db_`, opened once at
  construction), reused via `RestartListing()` instead of reopening/reparsing the DB's ~32MB `.kmc_pre` LUT
  on every `ClassifySorted` call.
- **`ClassifySorted` dispatches per call between a linear full-DB scan and per-k-mer random-access
  lookup** — `sequences.size() < db_.KmerCount()` routes to random access (`ra_db_`, opened lazily on first
  use). Matters in production's pooled-filtered-DB path (`pool_kmers=True`), where one shared DB spanning a
  whole shard is queried once per region with far fewer k-mers than the DB holds.

## Current hot spots (2026-08-19)

### Whole-pipeline picture: `initialize_scores` vs. the DP

Before Fix 7/8, `initialize_scores` (`KmerClassify::ClassifySorted`'s full on-disk KMC database rescan) was
the dominant cost of `sample_once` at typical/small scale — up to **84.4% of total wall time** — because
every call reopened and reparsed the DB's ~32MB `.kmc_pre` LUT from scratch, a fixed ~26-28ms floor
independent of region size. Fix 7 (cache the open `CKMCFile` handle, reuse via `RestartListing()`) and Fix 8
Stage A (adaptive scan-vs-random-access dispatch) together removed that floor:

| region (single-region filtered DB, as cached for benchmarking) | `initialize_scores` before → after | share of `sample_once` total before → after |
|---|---|---|
| small (1,469 k-mers) | 25.95 ms → **0.097 ms** (~267x) | 84.4% → **2.1%** |
| large worst-case (30,099 k-mers) | 28.38 ms → **1.01 ms** (~28x) | 3.2% → **0.1%** |

Net effect on `sample_once` (`initialize_scores` + `sample_haplotypes` + `sample_diplotypes`, unprofiled
`time.perf_counter()`, same benchmark harness): small region 16.06 ms/iter → **4.72 ms/iter** (~3.4x); large
region 823.92 ms/iter → **750.63 ms/iter** (~9.7% reduction, smaller because the DP itself already dominates
at this scale). **At both benchmark regions, `initialize_scores` is no longer a meaningful contributor to
`sample_once` at all** — confirmed independently by the "Current hot spots" DP profile below, whose
excluded-sample analysis at the small region (the case where `initialize_scores` would show up most, being
the largest fraction of a fast call) found the non-`SampleHaplotypes` time dominated entirely by
Python-loop/GC/import overhead and `SampleDiplotypes`, with no `initialize_scores`-attributable leaf frame
at all. It was **not separately re-measured at the dense `chr1:31431661-31432319` region** in this
benchmark-region form; given the fix operates per-region (RestartListing's scan cost scales with the DB's
own k-mer count, not a fixed reparse floor) and that region's 15,248 k-mers sit between the two measured
points, its post-fix cost is expected to be of the same sub-millisecond-to-low-single-digit-millisecond
order, but this is an inference, not a measurement — flagged as a gap below.

**Production's actual regime is different and more favorable to Fix 8 specifically:** the benchmark table
above uses a filtered DB built for exactly one region (`sequences.size() ≈ db_.KmerCount()`, so
`ClassifySorted` stays on the scan path even post-fix). Production's `pool_kmers=True` path
(`_diplotypes_in_topk_shard`, `haplotype.py`) instead pools k-mers across a whole shard into *one* filtered
DB shared by every region in it — exactly the regime `sequences.size() < db_.KmerCount()` targets, routing
to Fix 8's random-access dispatch. Measured directly against a real pooled shard DB (5.67M k-mers, HG00733,
`+pop=solo`):

| region | k-mers | pre-Fix-7/8 linear scan | post-Fix-8 (shipped) | speedup |
|---|---|---|---|---|
| `chr18:48997367-48997559` | 82 | 493.7 ms | 0.0094 ms | ~52,500x |
| `chr9:130429590-130429866` | 159 | 495.1 ms | 0.0135 ms | ~36,700x |
| `chr7:61260523-61381488` | 32,409 | 626.2 ms | 4.31 ms | ~145x |

At 2,562 regions in one real shard, the pre-fix scan cost alone was tens of minutes of pure overhead per
shard — now sub-second, aggregate. This pooled-DB regime, not the single-region benchmark form above, is
the one that matters for real genome-wide runs.

### `SampleHaplotypes` internals (DP forward pass)

`py-spy --native` leaf-frame profile, filtered to samples whose stack contains `SampleHaplotypes` (so the
now-negligible `initialize_scores` cost above, plus Python-loop overhead, doesn't confound the DP's own
shape). Two regions profiled at leaf-frame granularity this round (small/representative and the classic
large worst-case dense cluster); the third (`chr1:31431661-31432319`, the Stage-2-motivating region) was
re-timed but not re-profiled at leaf-frame granularity this round — see note below.

| region | k-mers | mean ms/iter | speedup vs. pre-Fix-9 |
|---|---|---|---|
| small (`chr1:148538120-148538280`) | 1,469 | 3.62 (n=100, unprofiled) | 4.66 → 3.62 ms, **~22%** |
| large worst-case (`chr1:148531170-148577610`, dense 3,786-variant cluster) | 30,099 | 699.81 (n=10, unprofiled) | 897.20 → 699.81 ms, **~22%** |
| `chr1:31431661-31432319` (Stage-2-motivating, 521 variants/659bp) | 15,248 | 190.77 (n=30, unprofiled, `ulimit -v 64000000`) | 263.60 → 190.77 ms, **~28%** |

Leaf-frame self-time by category (small/large only; py-spy's own `--native` sampling overhead inflates
absolute per-iteration time in these profiled runs, so use the unprofiled table above for wall-clock, this
one only for relative category share):

| leaf-frame category | small | large worst-case |
|---|---|---|
| `covered_paths` hashing (`PathIdSetHolder`'s `boost::hash_value` — `hash_mix_impl`/`hash_range`) | 4.71% | **43.65%** |
| allocator internals (jemalloc: `cache_bin_alloc/dalloc_easy`, `imalloc_fastpath`, `edata_*`, `phn_*`, `madvise`, etc.) | 19.36% | 12.74% |
| `Backpointer`/`SortAndTrimBacktrack` pool machinery (ctor/assign, `SameCoveredPathsClass`, the single linear-scan loop) | 20.44% | 4.40% |
| `memmove` (`covered_paths` bitset copy) | 7.78% | 11.24% |
| `covered_paths` refcounting (`intrusive_ptr` add_ref/release/ctor/dtor/get/swap) | 4.58% | 1.73% |
| `memcmp` (`covered_paths` equality, hash-collision fallback) | 0.17% | 1.41% |
| `dynamic_bitset` ops (`accumulate_covered_paths` mask/OR/compare) | 1.38% | 8.38% |
| `AutomatonGoto`/`GotoMap` lookup | 5.75% | 1.73% |
| `KmerSetScoreDelta` own body | 5.32% | 0.70% |
| everything else individually small (diffuse tail, inflated here by py-spy's own overhead relative to the now-much-shorter per-iteration DP time) | ~30.5% | ~14.0% |

Rollups, with the prior (pre-Fix-9) profile's numbers alongside for direct comparison:

| rollup | small (before → after) | large worst-case (before → after) |
|---|---|---|
| **`covered_paths` copy-compare-hash machinery** (hashing + memmove + memcmp + refcounting + `dynamic_bitset` ops) | 23.28% → 18.62% | 63.58% → **66.41%** |
| **beam-trim pool machinery** (`Backpointer`/`SortAndTrimBacktrack`) | **38.41% → 20.44%** | 8.24% → 4.40% |

**Reading this:**

- **Fix 9 (below) worked as designed: beam-trim pool machinery's share roughly halved at both regions**
  (small 38.4%→20.4%, large 8.2%→4.4%, both ≈47% relative reduction) — consistent with the wall-clock win
  above and with removing one of the two full comparison sorts everywhere plus removing the second sort
  entirely at every call site except the final sink merge.
- **`covered_paths` copy-compare-hash machinery is now unambiguously the dominant remaining cost**,
  especially at the large worst-case region (66.4%, up from 63.6% simply because the total shrank around
  it, not because this machinery itself got more expensive — it's untouched by Fix 9). This is
  `OPTIMIZATION_PROPOSALS.md`'s open item 1 (incremental/rolling `covered_paths` hash) and is now an even
  clearer #1 target than before.
- **`AutomatonGoto` remains a minor, stable cost** (1.7-5.8%), unaffected by this change as expected.
- Allocator/jemalloc share is noisy in this comparison (16.1%→19.4% small, 15.6%→12.7% large) — Fix 9
  reduces `resize()` calls (one bounded `reserve(n)` per call instead of the old sort/unique/`resize`
  sequence) but the *reported* share moved in opposite directions at the two regions, most likely because
  the profiled-run absolute times are now short enough (single-digit ms at the small region) that py-spy's
  own per-sample overhead and general sampling noise dominate any small remaining signal here — not a
  finding to act on without a cleaner (e.g. `memray`-based) allocation-count comparison.
- The Stage-2-motivating region's wall-clock improved by the largest margin of the three (~28%) but wasn't
  re-profiled at leaf-frame granularity this round — flagged in "Known gaps" below.

Current top candidates for further work — see `OPTIMIZATION_PROPOSALS.md`.

## Method (for future profiling sessions)

- `cProfile` can't see into native code. Use `py-spy record -f raw --native --rate 100 -- python3 <script>`
  and aggregate leaf-frame self-time (no root needed on this box).
- The default `Release` build (plain `uv pip install -e .`) is fully stripped. Rebuild once with
  `--config-settings=cmake.build-type="RelWithDebInfo"` for symbol resolution before profiling (~7 min),
  then rebuild back to plain `Release` afterward — confirm via `file` on the built `.so` (or matching
  `BuildID`) that the revert was clean.
- `tests/benchmarks/bench_haplotype_sampling.py` + `_haplotype_harness.py` already build a sampler directly
  against a cached graph shard + filtered-kmer DB (no `ray.init()`, no `kmc_tools intersect` needed) —
  `--region small|large` or any literal region string (e.g. `chr1:31431661-31432319`).
- Known benchmark regions: **small** = `chr1:148538120-148538280` (~1.5-1.7K k-mers); **large worst-case**
  = `chr1:148531170-148577610` (~30K k-mers, dense 3,786-variant cluster; splits into 21 sub-clusters under
  the current `overlapping_variants()` — pick the largest sub-cluster's shard, `chr1:148544977-148574123`,
  for continuity with most historical numbers); **dense/Stage-2-motivating** =
  `chr1:31431661-31432319` (521 variants/659bp — wrap in `ulimit -v 64000000` per `CLAUDE.md` if profiling
  any pre-Stage-2 code path against this region, since that's exactly the region that used to OOM).
- Reset scores (`initialize_scores`) between benchmark-loop iterations — `UpdateScores`'s zygosity discounts
  otherwise drift call-over-call and silently change beam-search behavior, corrupting the timing signal.
- `VmRSS`/`VmHWM` figures can be inflated by `KmerCounts`' ~18.7GB `mmap`'d whole-genome reference k-mer
  database (clean, read-only, file-backed — reclaimable by the kernel under pressure, a materially
  different and lower OOM risk than the DP's own anonymous heap memory). Always read a pre-call
  `stage=sizes`-style baseline and report deltas, not raw peaks, when comparing memory across
  regions/sessions.
- Verify any change with `ctest --test-dir $EXT_BUILD_DIR` (should be ~92/92, 1 pre-existing disabled,
  unrelated) and `pytest tests/graphs/test_haplotype.py -k "not test_sampling_performance"` (~21/21) before
  trusting a benchmark number.

## Condensed history

| date | change | result |
|---|---|---|
| 2026-07-30 | Fix 1: skip `KmerSetScoreDelta`'s bitset scan when a state has no output k-mers | ~9.3x combined with the dedup fix below; superseded by Fix 4's sparse representation |
| 2026-07-30 | Fix 2: cache `covered_paths`' hash, compare hash before full bitset compare in dedup | superseded by Fix 6's refcounted sharing |
| 2026-07-31 | Fix 3: `trie_symbol_mask_` short-circuits `AutomatonGoto`/`ComputeScoreToGo` for node ids absent from the trie (full/lazy DFA completion investigated first, ruled out — see "Dead ends") | ~15-20% at the worst-case region; kept |
| 2026-07-31 | Fix 4: dense `output_kmers_` bitset → sparse `output_kmer_indices_` vector (median popcount 1; `sdsl-lite`/Roaring investigated, ruled out — see "Dead ends") | ~2.4x at the worst-case region; kept |
| 2026-07-31 | Fix 5: hand-rolled `GotoMap` replaces `unordered_map` for `goto_` (two `flat_map` attempts failed first — see "Dead ends") | ~1.3-4.8% depending on scale; kept |
| 2026-08-11 | Fix 6: `covered_paths` → immutable refcounted `SharedPathIdSet` + `contributes_paths_mask_` skip for non-contributing transitions | ~1.55-2.3x depending on scale; kept |
| 2026-08-13 | Instrumentation + driver decomposition: `ComputeScoreToGo` (paid once per haplotype draw) + widening-retry waste ≈ 67% of wall time at dense regions; discarded widening attempts also compound peak RSS to ~7.8x baseline | motivated Proposal 2 |
| 2026-08-14 | **Proposal 2 shipped**: drop adaptive widening/`ComputeScoreToGo` entirely, one fixed-width `PropagateBestPathState(n)` pass per draw. Validated exact-equivalent (450+ trials) before shipping | ~2.5x/~28.4x faster, ~1.10x/~5.7x lower peak memory (small/large region) |
| 2026-08-18 | Investigation: real dense coverage at `chr1:31431661-31432319` (521 variants/659bp) confirmed >64GB OOM from unbounded pending (non-root) backpointer pools | motivated Stage 0-2 |
| 2026-08-18 | **Stage 2 shipped**: `SortAndTrimBacktrack` unconditionally width-trims every pool, not just root/sink. Validated exact via same-pool score-invariance proof + 240+ trials | confirmed-OOM region: never completing (>64GB) → 260.57ms, flat memory |
| 2026-08-19 | Post-Stage-2 profiling: `KmerClassify::ClassifySorted`'s full-DB rescan is now the dominant cost at small/typical scale (~26-28ms fixed floor per `initialize_scores` call, ~84% of small-region wall time) | motivated Fix 7/8 |
| 2026-08-19 | **Fix 7 shipped**: cache the open `CKMCFile` handle, `RestartListing()` instead of reopening/reparsing the LUT | ~267x/~28x faster `initialize_scores` (small/large region) |
| 2026-08-19 | **Fix 8 Stage A shipped**: adaptive scan-vs-random-access dispatch in `ClassifySorted` | ~52,500x/~145x faster at real pooled-DB regions with few k-mers per region relative to DB size |
| 2026-08-19 | Fix 8 Stage B (exploit k-mer sort order to cache LUT-prefix ranges across queries): investigated, **rejected** | 6-14% slower than per-query lookup — this project's real DBs' LUT is already tuned so `BinarySearch` is ~O(1); reverted, not shipped |
| 2026-08-19 | Post-Stage-2/Fix-7/8 profile, isolated to `SampleHaplotypes` alone, 3 regions | found beam-trim pool machinery still ~38.4%/8.2% (small/large) of DP time |
| 2026-08-19 | **Fix 9 shipped**: `SortAndTrimBacktrack` replaces two full comparison sorts with one O(m·n) single-pass dedup+bounded-top-n reduction against a small `kept` buffer; sorts only the final sink-merge result (the one order-sensitive call site) | ~22-28% faster wall time across all three regions; beam-trim leaf-frame share roughly halved (small 38.4%→20.4%, large 8.2%→4.4%) |
| 2026-08-19 | **Proposal 1 shipped**: `PathIdSetHolder`'s hash replaced with an order-independent per-bit ("Zobrist-style") hash — `boost::hash_value` over the full bitset dropped in favor of XOR-folding a cheap `PathHash(idx)` finalizer over set bit indices, letting each contributing DP transition update the parent's cached hash by XOR-ing in only the newly-set bits' `PathHash` (found via the sparse per-node `graph_.node_variant_paths_[next_node]` mask) instead of rehashing the whole `covered_paths` bitset. Valid because `accumulate_covered_paths` only ever sets bits, never clears them, so a pure per-set-bit hash is structurally guaranteed to agree with any from-scratch recomputation | small/large regions: 4.62→3.35ms (**~27%**), 753.9→355.2ms (**~53%**) wall time (`tests/benchmarks`, `--benchmark-only`, same build, before/after via `git stash`); full native (`ctest`, 92/92) and Python (`pytest tests/graphs/test_haplotype.py`) suites pass unchanged |

**Fix 9's tie-breaking note:** the old `std::sort`/`std::partial_sort`-based dedup+trim was already
unstable on exact ties (both within a `covered_paths` class and at the top-n score boundary) — Stage 2's
own shipping notes record a real test failure traced to exactly this. Fix 9's single-pass reduction breaks
ties differently (first-encountered-in-iteration-order wins, instead of whatever introsort produced) but
does not change which *value* is selected (best score per class, best n classes by score — the same
same-pool score-invariance proof Stage 2 relied on still applies, since Fix 9 doesn't touch what's being
compared, only how). Validated via the existing tie-sensitive test suite —
`HaplotypeSamplerOptimalityTest` (brute-force ground truth), `PendingPoolWidthLimitTest` (brute-force
ground truth on the pending-pool-heavy fixture), `SampleHaplotypesSanityTest` (determinism),
`RealRegionSanityTest` (the sorted-order contract Fix 9's design is built around), and the full real-data
exact-score pytest suite (`test_downstream_small_del`, `test_overlapping_alleles`,
`test_correct_diplotype_homalt`, `test_correct_diplotype_het`) — all passed unchanged, with no test
assertion needing adjustment this time (unlike Stage 2's shipping).

## Dead ends (don't retry without new evidence)

- **`flat_map`/`small_vector` for `goto_`** (two attempts, plain `vector`-backed and `small_vector`-backed)
  — 10-22% *slower* than `unordered_map`; Boost.Container's generic sorted-vector-map/small_vector
  abstraction overhead (binary search, pointer-indirection for inline-vs-heap storage) outweighed avoiding
  the hash lookup. Superseded by the hand-rolled `GotoMap` (Fix 5).
- **Full or lazily-memoized DFA completion for the Aho-Corasick automaton** (to make `AutomatonGoto` never
  chase `fail` links) — memory-infeasible: completing every state's transitions against the real
  `next_node` alphabet is ~240M entries / multiple GB at worst-case-region scale, and lazy memoization
  converges to nearly the same size within one `SampleHaplotypes` call.
- **`sdsl-lite`/Roaring bitmaps for `output_kmers_`** — wrong tool at this cardinality (median popcount 1
  across states); both are built to amortize over large sparse-or-compressible universes, and their
  per-instance overhead would exceed a plain array's cost here. Also would add a GPLv3 dependency
  (`sdsl-lite`) for no benefit over the ~10-line sparse-vector fix actually shipped (Fix 4).
- **Fix 8 Stage B** (k-mer sort-order exploitation for `ClassifySorted`'s random-access path) — see history
  table above.

## Known gaps

- Fix 9's leaf-frame profile covers only the small and large worst-case regions; the dense
  Stage-2-motivating region (`chr1:31431661-31432319`) was re-timed (wall clock improved ~28%, the largest
  of the three) but not re-profiled at leaf-frame granularity. Worth doing if that region's specific
  hot-spot shape ever needs attribution beyond the wall-clock number.
- `initialize_scores`' post-Fix-7/8 cost was never separately measured at the dense
  `chr1:31431661-31432319` region (only inferred by interpolation — see "Whole-pipeline picture" above) or
  under the real pooled-DB regime at a region this dense. Worth a direct measurement if that region's
  end-to-end cost ever needs precise attribution rather than the DP-dominant assumption used here.
- A genome-wide search for a real region denser than `chr1:31431661-31432319` (the confirmed worst case) was
  time-boxed and left incomplete on 2026-08-18: `tests/benchmarks/scan_dense_clusters.py` ranks candidate
  clusters cheaply (no graph construction) across chr1/chr6/chr9/chr16/chrX; 7 candidates were identified but
  none was measured to real-KMC-coverage completion (each needs a ~4-6 min `kmc_tools intersect` against the
  36.7GB whole-genome KMC DB). Not evidence the current bound (Stage 2) is sufficient everywhere — just
  unmeasured. Whoever picks this up next should start from that candidate list rather than re-deriving it.
