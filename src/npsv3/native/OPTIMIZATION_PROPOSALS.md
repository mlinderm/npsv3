# Haplotype sampling: optimization proposals

**Status:** the proposals this document originally tracked — dropping adaptive beam-widening/
`ComputeScoreToGo` (Proposal 2) and bounding pending-pool memory growth (Stages 0-2) — have both shipped.
See `PERFORMANCE_NOTES.md`'s "Condensed history" table for what shipped, when, and the measured result;
its "Current hot spots" section is the current profile against which the open proposals below should be
judged. This document now tracks only ideas that are **not yet implemented**.

Consolidated 2026-08-19 from a longer investigation log (proposals A-F, 1-3, Stages 0-3, plus the empirical
validation work that justified shipping Proposal 2/Stage 2).

## Open proposals

### 2. Branch-and-bound / A* single-pass redesign (larger reframing)

The DP is, more literally than it first appears, longest-path-in-a-DAG: `PropagateBestPathState` is a
forward pass over (graph node) × (automaton state), itself a DAG. Today a beam width is guessed and the
whole forward pass either accepts or (formerly, pre-Proposal-2) retries at 2x width — width is chosen
up-front, not derived from the DP's own structure. The standard technique once an exact score-to-go bound
exists is to use it as an A*/branch-and-bound admissible heuristic *during* a single forward pass: at each
node, only expand/keep candidates whose `current score + score-to-go bound` could still beat the current
n-th-best complete path found so far — determining the necessary effective width dynamically, branch by
branch, in one pass.

- **Pro:** the largest potential structural win of anything in this document — would remove the current
  fixed-width beam's risk of both being too narrow (missed candidates, though Proposal 2/Stage 2's
  validation found no cases where this happened in practice) and too wide (wasted work at nodes where a
  narrower beam would have sufficed) at once.
- **Con — real implementation risk, not a tuning change:** replaces the current fixed-width array-based
  beam (which fixes 1-8 spent significant effort making cache-friendly and allocation-cheap) with
  priority-queue-driven relaxation, carrying its own constant-factor overhead that could regress the
  "useful forward work" baseline even while removing other waste. Must preserve the existing
  settlement-point/dedup semantics (`covered_paths`-equivalence-class distinctness, not literal node-
  sequence distinctness — a non-standard variant of k-shortest-paths). Would need its own dedicated
  `ComputeScoreToGo`-equivalent bound computed once up front (that function was deleted with Proposal 2;
  reviving some form of it is a prerequisite, not a side effect). This is a multi-day redesign-and-
  reverify project, not a patch — treat as a separate follow-on, not a next step.

### 3. Upstream density cap (different owner, defense in depth)

Independent of the DP itself: if the variant-clustering/graph-building step feeding
`HaplotypeSamplerOverlay` flagged or split clusters above some variants/bp density threshold *before*
graph construction, the DP would never see input as pathological as `chr1:31431661-31432319` (521
variants/659bp) again, regardless of what the DP's own bounding logic does. Trades off joint-inference
accuracy at the split boundary; needs domain input on an acceptable threshold. Out of scope for
`haplotype.cpp`/`haplotype.hpp` — noted here so it isn't lost, not designed or evaluated.

This is also the "proportionate, no-retraining" alternative to the k-mer allele-evidence plan
(`REALIGNER_PERFORMANCE_NOTES.md`, and the full plan at
`as-observed-in-realigner-performance-not-dazzling-key.md`) if realigner timeouts turn out rare in
practice. A genome-wide production run (full training callset, HG00096) completed 2026-09-10 in under 9
minutes with 0 timeouts observed for most of its run (exact final count wasn't captured — see
`REALIGNER_PERFORMANCE_NOTES.md`'s "0c production run" sections) — a preliminary but encouraging signal that
few or no regions actually hit this in practice, which would favor this item over the k-mer plan. A second
pass with full profiling (`NPSV3_REALIGN_PROFILE=1`) was launched immediately after to get a clean final
count plus the k-mer plan's divergence-hypothesis data either way; once it completes,
`grep -c "Timeout error for region" main.log` in its output directory gives the number that actually decides
between this item and the k-mer plan shipping. The same log is also worth checking for item 4's
`grep -c "Skipping region"` count at the same time, since both come from one run.

### 4. Scope `SamePopulationState`'s GBWT-interval equivalence to inference-relevant distinctions

Root-caused via a real production failure: `SampleHaplotypes` returning only 1 haplotype (the reference)
for a hom-ref individual at `chr3:20967930-20972107`, even though the population VCF has 4 other
individuals carrying a heterozygous ALT deletion there (`AC=4`) — see
`InsufficientHaplotypesError`/`add_population_haplotypes` in `src/npsv3/graphs/haplotype.py`, the
Python-level workaround that was shipped instead of a native fix.

When the population prior is active, *every* per-node beam-pool trim (`SortAndTrimBacktrack`, not just the
final sink dedup) collapses candidates via `SamePoolClass = SameCoveredPathsClass AND
SamePopulationState`. `SamePopulationState` compares a raw GBWT `[lo,hi)` `PanelState` interval that is
deliberately unmasked by `inference_path_mask_` (see the existing comment at `haplotype.hpp:465-469` —
`HaplotypePriorOverlay` is built with no knowledge of any inference VCF). A rare/sub-panel ALT branch's
score, compounded across thousands of nodes, can fail to survive *any* node's top-width pool even at
generous width (confirmed via `NPSV3_HAPLOTYPE_PROFILE=1` on the region above: the search exhausted its
full candidate set at width=80 without hitting the "search-width ceiling" condition — not a budget
problem, the candidate was provably absent from the enumerated set).

Masking this axis the way `covered_paths`/`contributes_paths_mask_` is already masked (see
`accumulate_covered_paths`, `haplotype.cpp:994-1000`, and the other `inference_path_mask_`-masked call
sites) would directly fix the blind spot at the source, but has **no existing precedent** —
`population_distinguishing_mask_` is explicitly built and documented as staying unfiltered by
`inference_node_mask_`/`inference_path_mask_` on purpose — and would be real design+validation work in
this codebase's most heavily profiled/tested hot path (its own equivalence proof, adversarial-trial
validation per this doc's established bar; no existing native test currently combines active-prior +
inference-VCF-filtering + real beam-width stress, so there's no regression harness to build from either).
Deferred in favor of the Python-level top-up; revisit if `grep -c "Skipping region" <job log>` on a real
population run shows the safety net still firing often (i.e. no population-panel carrier available either).

Cheaper, blunter lever in the same neighborhood (not pursued as primary): softening
`haplotype_sampler_params.haplotype_prior_weight` / `panel_fallback_penalty` / `transition_prior_alpha` in
`src/npsv3/conf/graph/base.yaml` back toward less extreme values reduces how often any locus hits this
blind spot, but doesn't fix the structural gap and undoes tuning that was deliberately dialed in.

## Shipped since this document's last consolidation

- **`SortAndTrimBacktrack`'s double-sort** — was an open item here as of 2026-08-19; shipped later the same
  day as Fix 9 (single-pass dedup+bounded-top-n reduction replacing both sorts, sorting only the final
  sink-merge result — the one call site whose order is load-bearing for `SampleHaplotypes`/`FindBestPaths`).
  See `PERFORMANCE_NOTES.md`'s history table and that entry's "Tie-breaking impact" note for the design and
  the reasoning on why it doesn't regress correctness despite changing tie-break outcomes in rare exact-tie
  cases.
- **Cheaper `covered_paths` hashing** — was open item 1 as of 2026-08-19; shipped later the same day.
  `PathIdSetHolder`'s `boost::hash_value(bits)` (rehashed the whole bitset on every contributing transition)
  replaced with an order-independent per-bit ("Zobrist-style") hash: `hash(S) = XOR over i in S of
  PathHash(i)`, a pure function of the set of indices (so equal bitsets provably hash equal regardless of
  construction history, unlike trying to incrementally replicate `boost::hash_value`'s block-sequential
  mixer). Since `accumulate_covered_paths` only ever sets bits, each contributing transition now updates the
  parent's cached hash by XOR-ing in `PathHash` for just the newly-set bits (found via the sparse per-node
  `graph_.node_variant_paths_[next_node]` mask) instead of rehashing the whole bitset. See
  `PERFORMANCE_NOTES.md`'s history table for the measured ~27-53% wall-time win and validation.

## Rejected (evaluated, not pursued)

- **Fix 8 Stage B** — caching LUT-prefix ranges to exploit `ClassifySorted`'s sorted k-mer input, instead
  of a per-query `CheckKmer` lookup. Measured 6-14% *slower*: this project's real filtered-kmer DBs have
  their LUT prefix length tuned so `CheckKmer`'s `BinarySearch` is already near-O(1) per query, leaving
  little redundant search for a coordinated sorted merge to eliminate, while the manual per-character
  prefix computation this approach needs costs strictly more. See `PERFORMANCE_NOTES.md`'s history table.
- **Amortizing `ComputeScoreToGo` across a whole `SampleHaplotypes` call** (originally "Proposal A"), and
  the related cheapen/memoize variants (proposals B/C/D) — moot: Proposal 2 removed `ComputeScoreToGo`
  from the production path entirely rather than reducing how often it's called.
- Flat-map/DFA-completion/sparse-bitset dead ends for the DP's own hot loop — see `PERFORMANCE_NOTES.md`'s
  "Dead ends" section (these predate and motivated the currently-shipped `GotoMap`/sparse-vector designs,
  not independent open proposals).

## Before implementing anything above

Re-verify with `ctest --test-dir $EXT_BUILD_DIR` and
`pytest tests/graphs/test_haplotype.py -k "not test_sampling_performance"` first so any regression is
attributable to the new change, not pre-existing drift. Proposal 1 above changes *implementation* only
(same guarantee, cheaper computation) and should be checked against the existing correctness suite;
proposal 2 changes the DP's *structure* and would need new equivalence tests of the same rigor Proposal
2/Stage 2 got (synthetic-adversarial trials with brute-forced ground truth + real worst-case-region trials)
before shipping. Proposal 3 is a different subsystem (variant clustering, not `haplotype.cpp`/`.hpp`) and
isn't subject to this suite at all.
