# FragmentRealigner performance notes

Profiling target: `FragmentRealigner::RealignReadPair` (`realigner.cpp`/`realigner.hpp`), reached from Python
via `realign_fragment` (`src/npsv3/realigner.py`), called once per overlapping read-pair fragment per
candidate haplotype allele inside `ImageGenerator._generate`'s per-read pileup loop
(`src/npsv3/images/generator.py`). This file documents a single investigation session (2026-09-08/09) into
why one specific region times out during image generation, what was actually shipped (instrumentation + a
tunable, not a fix), and what remains open. No prior performance-notes doc existed for this subsystem before
this session; see `PERFORMANCE_NOTES.md`/`OPTIMIZATION_PROPOSALS.md` for the separate (and much more mature)
`HaplotypeSamplerOverlay`/DP-sampler investigation — this file does not overlap with that one except where
noted.

## Symptom

`tests/images/test_example.py::TestPopulationGraphExamples::test_error_regions[chr4:49101919-49110988]`
(HG00096, population VCF, `graph.population_prior=True`) times out during image generation. The region is
~9.07kb wide with 770 variants in the population VCF (dense, though not as densely-packed as
`PERFORMANCE_NOTES.md`'s `chr1:31431661-31432319` pathological case for the DP sampler). Default
`cfg.timeout` is 1800s (`config.yaml`); graph/k-mer construction for this region succeeds quickly, and the
native haplotype sampler DP itself is fast here too (confirmed below) — the timeout is entirely inside
`make_graph_example`'s image-generation phase.

## Method (reproducible recipe)

- Prerequisites beyond a plain `uv pip install -e .`: `module load npsv3/main` (bundles `libjemalloc`, `bwa`,
  `art`, `samtools`, `samblaster`, `sambamba` — all required by `simulate_variant_sequencing`/`synthBAM`; see
  `CLAUDE.md`). Load the BWA index into shared memory first (`bwa shm <reference>`) for a realistic run —
  otherwise every `synthBAM` invocation reloads the ~3GB hg38 index from disk, which dominates wall time and
  masks the signal investigated here.
- `vcf_to_graph_examples` uses Ray (`ray.util.ActorPool` over `GraphWriter` actors). For `cProfile` to see
  inside actor method calls, initialize Ray with `local_mode=True` so actor code runs synchronously in the
  driver process instead of a separate worker process.
- Mirror `main.py`'s `_check_shared_reference` manually: `cfg.shared_reference = bwa_index_loaded(cfg.reference)`.
  **Gap, not yet fixed**: `test_error_regions` itself only calls `bwa_index_loaded(cfg.reference)` as a
  skip-guard (`pytest.skip` if not loaded) — it never sets `cfg.shared_reference`, so as currently written the
  test requires the shared-memory index but doesn't actually benefit from it. Worth fixing in the test if
  someone revisits it.
- For bounded interactive profiling, override `timeout=90` (or similar) in the Hydra overrides instead of the
  1800s default. `GraphWriter.from_shard` catches the resulting `TimeoutError` internally and just skips the
  region (logs and continues) — `vcf_to_graph_examples` still returns normally, so `cProfile` gets a complete,
  uninterrupted profile of whatever ran inside that window instead of an exception escaping mid-profile.
- Reuse the on-disk graph-shard/filtered-kmer cache under `tests/results/<region>.<vcf-hash>.<sample>.k<k>/`
  (built by `tests/__init__.py`'s `cache_graph_and_filter_kmc_database`) to skip graph construction and focus
  profiling time on image generation.
- Native-level instrumentation, both opt-in via env var and negligible cost when unset:
  - `NPSV3_HAPLOTYPE_PROFILE=1` (pre-existing, see `PERFORMANCE_NOTES.md`) — per-draw DP timing/memory. Used
    here only to *rule out* the sampler, not to investigate it further.
  - `NPSV3_REALIGN_PROFILE=1` (**added this session**, see below) — per-allele-realignment timing/counts from
    `FragmentRealigner::RealignReadPair`.
- Comparison regions used throughout: `chr7:136717247-136717522` (276bp, fine), `chr3:20967930-20972107`
  (4.18kb, fine — needed the population-panel top-up in `add_population_haplotypes`, see `example.py`),
  `chr4:49101919-49110988` (9.07kb, 770 variants, times out).

## Findings

### `cProfile`: `realign_fragment` dominates, and disproportionately so

With `graph.population_prior=True`, shared reference loaded, `timeout=90`: 90.5s total profiled wall time at
chr4. `realign_fragment` (`realigner.py:29`, wrapping the native `realign_read_pair` call) is essentially the
entire cost:

| | chr3 (fine) | chr4 (times out) |
|---|---|---|
| `realign_fragment` calls | 1,424 | 3,330 |
| self time | 0.306s | 84.669s |
| self time / call | 0.215ms | 25.4ms (**~118x**) |
| share of total profiled wall time | small | **94% self / 96% cumulative** |

Everything else (`fetch_reads`, `pileup.add_read`, `_refine_base_alignment`, OmegaConf lookups — see the
`vcf_to_graph_examples` profiling turns earlier in this investigation) is noise by comparison at chr4. The
118x per-fragment jump is far more than region-width scaling (~2.2x wider) or haplotype-sequence-length
scaling (`realigner_flank=1000` on each side: ~6.18kb chr3 haplotype sequences vs ~11.07kb chr4) would predict
on their own — motivating the native-level breakdown below.

### `NPSV3_REALIGN_PROFILE`: two compounding, roughly-independent factors

Added this session (`realigner.cpp`, gated on `NPSV3_REALIGN_PROFILE=1`, mirrors `NPSV3_HAPLOTYPE_PROFILE`'s
pattern): one `REALIGN_PROFILE` line per allele realignment (reference + each alt) inside
`RealignReadPair`, printing `ms`, `read1_alignments`, `read2_alignments`, `read_pairs` (=
`read1_alignments × read2_alignments`, the `RealignedFragment` pairwise-scoring cost), plus one per-fragment
`stage=total` line. Required adding two accessors to `RealignedFragment`
(`NumRead1Alignments()`/`NumRead2Alignments()`) and linking `fmt::fmt` into the `_native_realign` CMake
target (not previously linked there).

| | chr3 | chr4 | ratio |
|---|---|---|---|
| alleles realigned per fragment (ref+alts) | 2 | 11 | 5.5x |
| fragments realigned (in profiled window) | 1,439 | 3,295 | 2.3x |
| **mean ms / single allele-realignment** | **0.100ms** | **2.324ms** | **23.2x** |
| max ms / single allele-realignment | 1.900ms | 29.605ms | |
| mean `read_pairs` | 0.58 | 30.0 | |
| max `read_pairs` | 2 | 182 | |
| mean `read1_alignments` | 0.62 | 5.37 | |
| max `read1_alignments` | 2 | 14 | |
| correlation(ms, `read_pairs`) at chr4 | — | 0.28 | |
| correlation(ms, `read1_alignments+read2_alignments`) at chr4 | — | 0.32 | |

Two compounding, largely independent factors:

1. **Call-count multiplication (~13x)**: chr4's native haplotype sampler finds the full diversity it's
   allowed (11 total alleles — reference plus up to `max_haplotypes=8` sampled plus true-haplotype additions
   from `prepare_genotyping_haplotypes`; no population-panel top-up needed, unlike chr3 which only reached 2)
   over more overlapping fragments (3,295 vs 1,439, from the wider region). `RealignReadPair` runs once per
   (fragment, allele) pair, so total calls scale as their product: 36,245 vs 2,878.
2. **Per-call cost inflation (~23x, the larger single factor)**: even a *single* allele's realignment costs
   23x more at chr4 than chr3 — far beyond the ~1.8x haplotype-sequence-length difference. The `read_pairs`/
   `read1_alignments` correlation with `ms` is only moderate (r≈0.28-0.32): real and responsible for the
   *extreme tail* (individual allele-realignments up to 29.6ms, fragment totals up to 172ms, driven by reads
   hitting BWA's secondary-alignment cap on *both* mates simultaneously — 3.98% of chr4's allele-realignments
   did), but it does not explain the elevated *baseline* (mean 2.324ms holds even restricting to
   low-`read_pairs` cases). The baseline inflation's root cause was not pinned down further this session —
   working hypothesis is that chr4's haplotype sequences (or specific reads against them) are more repetitive
   for BWA's own seed-and-extend search, but this is unconfirmed (see "Known gaps").

### Source-level mechanism: what the two BWA parameters actually control

`IndexedSequence::AlignSequence` (`realigner.cpp`) previously hardcoded
`bwa_.AlignSequence(seq, name, alignments, false, 0.9, 10)`. Reading SeqLib's own implementation
(`lib/seqlib/src/BWAWrapper.cpp:255`, `BWAWrapper::AlignSequence`) directly:

- `mem_align1()` — BWA-MEM's full seed/chain/extend search — runs **unconditionally**, before either
  parameter is consulted.
- `mem_reg2aln()` (full CIGAR/alignment reconstruction) is then called for **every** hit `mem_align1`
  reported, including every secondary, **except** when `keep_sec_with_frac_of_primary_score` is outside
  `[0, 1]` — that specific case skips secondary hits *before* `mem_reg2aln`, the one place these parameters
  can avoid real work rather than merely filter already-computed results.
- Only after that does the code filter by `keep_sec_with_frac_of_primary_score`/`max_secondary` to decide
  what's actually returned in `vec` (i.e., what becomes `read1_alignments`/`read2_alignments`).

**Conclusion**: these are a post-hoc output filter over an already-fully-computed hit list, not a
search-pruning knob for BWA's own internal cost — with the single exception of the out-of-range-fraction case
above, which is the maximal lever available through this interface.

## Shipped this session

Not a fix — instrumentation plus a previously-hardcoded pair turned into a tunable, defaults unchanged:

- `realigner.hpp`/`realigner.cpp`: `keep_sec_with_frac_of_primary_score` (default `0.9`) and `max_secondary`
  (default `10`) threaded through as `FragmentRealigner` constructor kwargs (mirrors the existing
  `num_alts`/`iupac_fasta_path`/`alt_alignment_paths` kwargs pattern) instead of being hardcoded at the
  `IndexedSequence::AlignSequence` call site. No caller in the production path (`example.py`/`realigner.py`)
  passes these kwargs, so behavior is unchanged unless a caller explicitly opts in.
- `realigner.hpp`: `RealignedFragment::NumRead1Alignments()`/`NumRead2Alignments()` accessors.
- `realigner.cpp`: `NPSV3_REALIGN_PROFILE=1` opt-in instrumentation (see above), following the same
  cached-`getenv` pattern as `NPSV3_HAPLOTYPE_PROFILE` — negligible (one `getenv` call) when unset.
- `CMakeLists.txt`: linked `fmt::fmt` into the `_native_realign` target (needed for `fmt::print`; the target
  didn't link it before).

Verified: `cmake --build $EXT_BUILD_DIR -t _native_realign` compiles clean; `uv pip install -e .` succeeds;
`pytest tests/test_realigner.py` (3/3) passes with defaults unchanged.

## Experiment: `keep_sec_with_frac_of_primary_score=-1`

The maximal lever available via these parameters (drop all secondary alignments before `mem_reg2aln`, per
the mechanism above), tested against chr4 (same conditions, `timeout=90`):

| | baseline | `frac=-1` | change |
|---|---|---|---|
| mean ms / allele-realignment | 2.324 | 1.599 | **-31%** |
| max ms / allele-realignment | 29.605 | 23.107 | -22% |
| mean `read_pairs` | 30.0 (max 182) | 1.09 (max 8) | -96% |
| mean `read1_alignments` | 5.37 (max 14) | 1.13 (max 4) | -79% |
| fragments completed in the 90s window | 3,295 | 4,734 | **+44%** |
| mean ms / fragment (`stage=total`) | 25.6 | 17.66 | -31% |
| max ms / fragment | 171.966 | 132.097 | -23% |

**Reading this**: the O(`read1_alignments × read2_alignments`) pairwise-combination blowup and its extreme
tail are essentially eliminated (`read_pairs` max 182→8). But the mean per-allele cost only drops 31%, not
the ~20x+ that would actually resolve the timeout — confirming the source-level prediction above: the
irreducible cost is `mem_align1`'s always-on search, once per read per allele, against each ~11kb
per-haplotype BWA index, and this parameter doesn't touch it. **`keep_sec_with_frac_of_primary_score=-1` (or
any tightening of these two parameters) is not a fix for this region's timeout** — it's a real, measured,
secondary throughput win (31%/44%) with no further headroom available through this interface.

**Correctness caveat — not evaluated, not free**: dropping secondary alignments changes what
`RealignedFragment::TotalLogProb` marginalizes over for reads that genuinely map ambiguously (plausible here,
given how many secondaries BWA was reporting before this change) — it changes the realignment quality/allele-
assignment computation itself in repetitive stretches, not just its speed. `keep_sec_with_frac_of_primary_score=-1`
was **not** validated against genotype/allele-assignment accuracy and should not be treated as a safe default
change without that validation.

## Known gaps / open items

- **The dominant driver — why chr4's per-call BWA cost is ~23x chr3's, independent of the secondary-alignment
  count — was not pinned down.** The working hypothesis (chr4's haplotype sequences are more repetitive/
  duplicated, making `mem_align1`'s own seed-and-extend search slower even when it ultimately reports few
  hits) is unconfirmed. Would need instrumentation inside BWA itself (out of scope — "we can't change BWA")
  or an independent characterization of the haplotype FASTA sequences' repeat content (e.g., self-alignment
  or a k-mer-uniqueness scan of the `realign.fasta`/`simulation.fasta` sequences for this region) to confirm
  or refute.
- **The actual fix has to reduce the (fragments × alleles) BWA-call volume itself** — not explored this
  session. Candidate directions, none implemented or evaluated: capping how many alleles get realigned per
  fragment (vs. today's every-fragment-against-every-allele), narrowing `pileup.realigner_flank`/
  `pileup.fetch_flank` (currently 1000bp each) to shrink both the per-haplotype BWA index and the overlapping-
  fragment count, or an upstream density cap on variant clusters before graph construction (this exact idea
  is already open item 3 in `OPTIMIZATION_PROPOSALS.md` for the DP sampler side — the same lever would help
  here too, for the same reason: it keeps the realigner from ever seeing a region this dense).
  Also note `possible_genotypes` scales as `C(haplotypes+ploidy-1, ploidy)` — up to `max_diplotypes=36` at
  `max_haplotypes=8` — so the timeout risk compounds further once synthetic per-genotype images are reached;
  this session's profiling only got partway through the *first* (real-sample) image within the 90s window,
  never as far as the synthetic-genotype images.
- `test_error_regions`'s `bwa_index_loaded` skip-guard not actually wiring `cfg.shared_reference` (noted
  under "Method" above) means the test's own stated prerequisite doesn't do what it implies. Low-risk,
  low-effort fix if anyone touches that test next.
- No regression/benchmark harness exists for this subsystem analogous to
  `tests/benchmarks/bench_haplotype_sampling.py` — every number in this document came from ad hoc scripted
  profiling runs (Ray `local_mode`, cProfile, shortened `timeout`), not a checked-in, rerunnable benchmark.
  Worth building one (parametrized by region, mirroring the DP sampler's harness) if this investigation
  continues.

## Session 2026-09-10: Stage 0 of the k-mer allele-evidence plan

Executed Stage 0 of `as-observed-in-realigner-performance-not-dazzling-key.md` (two real bugs, the 0c
instrumentation extension, two cheap BWA-path guards). Stage 1 (the k-mer evidence engine itself) and the 0c
production timeout-count run were deliberately deferred to a later session — this entry covers Stage 0 only.

### 0a. Base-mismatch bug in `_read_realignment` — fixed

`realigner.py:22-26` normalized base-10 log-probability `scores` with `scipy.special.logsumexp`, which computes
`ln(sum(e^x))` (natural log), not `log10(sum(10^x))`. Replaced with an explicit base-10 normalizer
(`np.log10(np.sum(np.power(10.0, scores)))`). Verified against the plan's worked examples: equal scores
`(-1, -1)` now give `Q = 3.010` (was `0.98`); `(-5, -9)` saturates at the `40` cap (matches). Channel 6 (native
`LogSumPow`, already base-10-correct) is unaffected — this only changes channel 7 / per-read `AlleleRealignment`
qualities.

### 0b. `read_scores` zero-sentinel bug in `RealignReadPair` — fixed

`realigner.cpp:499` initialized `read_scores(4, 0)`; since these are log10 probabilities, `0` decodes as
`P = 1` (maximal support) rather than "no alignment for this read/allele". Changed the sentinel to `NaN`.
`generator.py`'s existing `_allele_pixel` guard (`math.isnan(...)`) already renders NaN as pixel 0 ("no
information"), so no change was needed on the Python rendering side — verified directly that
`_read_realignment([nan, nan], ...)` now propagates NaN through to `AlleleRealignment` rather than a
maximal-quality artifact.

### 0c. Instrumentation only (production run deferred)

Extended `NPSV3_REALIGN_PROFILE` (opt-in, negligible cost when unset) to also report, per allele-realignment
line: the best-pair alignment score (log10 P(data|alignment)) and edit distance (`NM`, summed over the pair's
reads, from BWA's own tag; `-1` if there's no best pair or no `NM` tag). This is what Stage 0c's "test the
divergence hypothesis" measurement will read from — it lets a profiling run distinguish "slow because many BWA
secondary hits" (already visible via `read_pairs`/`read1_alignments`/`read2_alignments`) from "slow because
this read is divergent from this allele" (high `NM`, low score), which bears on the open question from the
prior session ("Known gaps": chr4's ~23x per-call cost inflation, only weakly correlated with reported
alignment counts, r≈0.28-0.32). **Not yet run against production or chr4** — that measurement, plus the
region-timeout-count production run, are still open and are what Stage 0c's "what it decides" language in the
plan hinges on before Stage 1 ships.

### 0d. Two cheap guards on the BWA path — implemented, independent of 0a-0c

- **Subsample before realigning** (`generator.py`, `CoverageImageGenerator._generate`): fragments overlapping
  the region are now collected first, `max_reads = (region.length * image_height) // sample.read_length` is
  computed up front, and the fragment list is subsampled to `max_reads` *before* the per-fragment realignment
  loop runs — rather than realigning every overlapping fragment and only then subsampling reads for rendering.
  `pileup.overlapping_reads(region, max_reads=...)` still runs afterward as the final authority (a fragment can
  contribute up to 2 reads, so the fragment-level subsample alone doesn't exactly hit the read-level target),
  so this changes cost, not the rendered image.
- **Memoize `realign_fragment` on `(read1_seq, read1_qual, read2_seq, read2_qual)`**: added an optional `cache`
  dict parameter to `realign_fragment` (`realigner.py`), and `example.py` now creates one `realign_cache = {}`
  per region, threaded through every `generator.generate(...)` call for that region — the real image plus all
  `combinations_with_replacement(haplotypes, ploidy)` synthetic-genotype images. This targets a real source of
  duplicate work: `replicate_bam_splits[f"{h}_{i}"]` (one simulated BAM per haplotype/ploidy-copy) is reused
  byte-identical across every genotype combination containing that `(h, i)` pair, so without the cache the same
  fragment gets realigned once per genotype it appears in (up to ~11 times at chr4's 11-haplotype case) instead
  of once.

### Verification

`cmake --build $EXT_BUILD_DIR -t _native_realign`/`-t graph_test` compile clean; `uv pip install -e .`
succeeds. `ctest --test-dir $EXT_BUILD_DIR` 116/116. `pytest tests/test_realigner.py` 3/3,
`pytest tests/images/test_example.py` 4/4 (needed `bwa`/`samtools`/`samblaster`/`sambamba`/`art_illumina` on
`PATH` — `module load npsv3/main` did not put them on `PATH` in this session's shell for unknown reasons, so
they were added directly from their `/home/mlinderman/modules/...` install directories as a workaround),
`pytest tests/graphs/test_haplotype.py -k "not test_sampling_performance"` 41/41. Full `pytest tests`
(excluding that same performance test and two modules — `tests/models/test_dvae.py`,
`tests/test_genotype.py` — that fail to *collect* on `main`/this branch already, unrelated to this session:
stale imports of `vcf_to_variant_examples`/`npsv3.graphs.graph_constructor`, neither of which exists) gives
126 passed / 10 skipped, with the same single pre-existing failure
(`test_loaders.py::TestNPSV2Examples::test_visualize_converted_images`, missing `cfg.reference` hydra value)
and the same 4 pre-existing benchmark-fixture errors reproducing identically against unmodified `main` — all
confirmed pre-existing by re-running against a stash of this session's changes, not introduced by Stage 0.

### Not done this session (deferred by explicit scope decision, not forgotten)

- **0c's production run and divergence-hypothesis measurement.** The instrumentation is ready; the run itself
  needs a real production launch (Ray, full VCF, cluster resources) that was out of scope for this pass.
- **Stage 1** (the k-mer allele-evidence engine) and everything downstream of it (agreement harness, ablations,
  retraining) — not started.

## Follow-up: per-Ray-actor `REALIGN_PROFILE` output files

Motivation: the driver-stdout-capture approach floated for the production run (redirect the whole `npsv3`
invocation's stdout/stderr to one file, relying on Ray's `log_to_driver=True` streaming) doesn't scale cleanly
to many parallel actors and depends on Ray's private per-run temp dir surviving long enough to be read, which
it isn't designed to do (`vcf_to_graph_examples` uses a self-cleaning `tempfile.TemporaryDirectory`). Replaced
with a per-actor file written directly alongside that actor's webdataset shard, no shared stdout/temp-dir
dependency at all.

- `realigner.cpp`: added `RealignProfileStream()`, a function-local-static-cached `FILE*` resolved once per
  process from `NPSV3_REALIGN_PROFILE_PATH` (append mode; falls back to `stderr`, the prior behavior, if unset
  or unopenable). All three `REALIGN_PROFILE`/`stage=total` `fmt::print` call sites now target this stream
  instead of hardcoded `stderr`. Added an `fflush` after each fragment's `stage=total` line (not after every
  allele line) so an actor kill/crash loses at most one fragment's buffered output, without paying a flush per
  allele-realignment.
- `example.py`: `ExampleActor.__init__` sets `os.environ["NPSV3_REALIGN_PROFILE_PATH"]` to
  `<output_dir>/images-{index:04d}.realign_profile.log` — mirroring the existing
  `images-{index:04d}.tar.gz` shard naming — but only when `NPSV3_REALIGN_PROFILE` is already set, so nothing
  changes (no file created, no env var touched) for a normal non-profiling run. Placed in the base class
  (rather than `GraphWriter`) since it only needs `index`/`output_dir`, both already available there; correct
  ordering relies on `__init__` always completing before `from_shard`/`from_graph` (and thus any realignment)
  runs in that actor's process — true by construction.

**Known caveat, not fixed**: under `ray.init(local_mode=True)` with more than one actor, all actors share the
driver's single process, so `os.environ[...]` assignments from each actor's `__init__` race, and because the
C++ side caches the resolved `FILE*` in a function-local static on *first use only*, whichever actor happens to
realign first silently wins the path for the rest of the run. Irrelevant for a real distributed production run
(the actual target here — each actor is a separate OS process there) and only affects `local_mode` interactive
profiling with multiple actors, which the prior session's recipe didn't use (single actor / driver-only
profiling). Not worth the complexity of re-resolving the path on every `FragmentRealigner` construction to fix
a debug-only edge case.

**Verified**: `cmake --build $EXT_BUILD_DIR -t _native_realign`/`-t graph_test` compile clean;
`ctest --test-dir $EXT_BUILD_DIR` 116/116; `pytest tests/test_realigner.py` 3/3 and
`pytest tests/images/test_example.py` 4/4 both pass unchanged (profiling disabled in those runs, confirming no
behavior change on the default path). Direct check with `NPSV3_REALIGN_PROFILE=1` +
`NPSV3_REALIGN_PROFILE_PATH` set against a single `FragmentRealigner` call confirmed lines land in the target
file, not stderr. End-to-end check via `test_vcf_to_dataset` with `NPSV3_REALIGN_PROFILE=1` (no `_PATH`, so the
actor sets it itself) confirmed a real Ray-actor run produces `images-0000.realign_profile.log` next to
`images-0000.tar.gz` in the output directory, containing 2,415 lines (483 `RealignReadPair` calls x ~5 lines
each) of real score/NM data.

## 0c production run — launched (user-initiated, outside this session's control)

The user launched the actual production run separately (not via a command this session issued). Recorded here
for continuity since it directly informs the plan's "Status" section and Stage 1's ship/no-ship gate.

- **Command** (from `/proc/<pid>/cmdline` at launch): `npsv3 -m command=images threads=16
  reference=.../Homo_sapiens_assembly38.fasta generator=coverage sample=HG00096
  input=".../hgsvc3-hprc-2024-02-23.dipcall.population.passing.training.hg38.vcf.gz"
  reads=".../sequence/${sample}.final.cram" stats_path=".../sequence/${sample}.final.stats.json"
  graph_shards="hgsvc3-hprc-2024-02-23.dipcall.population.passing.training.hg38.graphs/graphs-*-*.tar.gz"
  filtered_kmer_path=".../${sample}/filtered_kmers" simulation.replicates=1 load_reference=true
  pileup=unphased_graph graph.population_prior=true graph.missing_are_ref=false
  hydra/launcher=submitit_slurm hydra.launcher.partition="long,himem-long" hydra.launcher.cpus_per_task=16
  hydra.launcher.mem_gb=44 hydra.launcher.timeout_min=10080`, run from
  `/storage/mlinderman/projects/sv/npsv3-experiments/training` (an existing `Makefile`-driven pipeline with
  many prior `*.images` runs for other VCF/sample combinations — this is real production infrastructure, not
  ad hoc).
- **Scale**: genome-wide (no `region=` restriction), full training callset, 140 pre-built graph shards, 16
  parallel `GraphWriter` actors (`threads=16`) via SLURM (`submitit_slurm`, partition `long`/`himem-long`,
  `cpus_per_task=16`, `mem_gb=44`, `timeout_min=10080` = 7 days ceiling; user estimates ~24h actual).
- **Dispatch**: `hydra/launcher=submitit_slurm` submits to SLURM; observed as job `78085` on `node009`,
  partition `long`, submitted 2026-09-10 13:06:17 EDT. `squeue -u <user>` to check status.
- **Output**: `/storage/mlinderman/projects/sv/npsv3-experiments/training/hgsvc3-hprc-2024-02-23.dipcall.population.passing.training.hg38.images/HG00096/generator=coverage,graph.missing_are_ref=False,graph.population_prior=True,pileup=unphased_graph,simulation.replicates=1/`
  — `main.log` (driver log, includes `logging.exception("Timeout error for region ...")` on timeout) and 16
  `images-NNNN.tar.gz` shards. At the first check (~2.5 min after launch): BWA shared-memory index loaded
  successfully, "Using 140 pre-generated graph shards", 0 timeouts logged yet, 16 shards already receiving
  data (111MB written). This confirms the run is using this session's Stage 0 code (built and
  `uv pip install -e .`-installed earlier in the same session, before this run was launched) — i.e. the
  timeout count it eventually produces reflects the fixed/faster code, not pre-Stage-0 behavior.
- **`NPSV3_REALIGN_PROFILE` was not set** for this launch (confirmed via `/proc/<pid>/environ`) — this run
  will yield the region-timeout count (0c's first measurement) but not the per-fragment score/NM data for the
  divergence hypothesis (0c's second measurement, which needs the instrumentation from the section above). A
  separate, much smaller follow-up run with `NPSV3_REALIGN_PROFILE=1` set — targeted at the known dense
  regions rather than genome-wide — is still needed for that second measurement if it's still wanted.
- **Read the timeout count with**: `grep -c "Timeout error for region" main.log` in the output directory
  above (valid mid-run, not just at completion — `from_shard` logs and continues past a timeout).

## 0c production run — pass 1 completed fast; pass 2 (with profiling) launched

Update to the above, same day. The user restarted the run with `NPSV3_REALIGN_PROFILE=1` set, closing the
gap noted above; recorded here since both passes' details are load-bearing for whoever reads this next.

- **Pass 1 (job `78085`) completed**: `sacct -j 78085` shows `COMPLETED`, exit `0:0`, elapsed **8m42s** for
  the entire genome-wide run (140 shards, full training callset, 16 actors) — far under the user's initial
  ~24h estimate. A mid-run check (4m42s in) showed 0 timeouts logged; `main.log` still showed 0 timeouts / 0
  `Skipping region` (insufficient-haplotypes) entries in the moments before it was overwritten by pass 2 (see
  below). **The exact final count from pass 1 was not captured** — pass 2 was launched against the identical
  `hydra.sweep.dir`/`subdir` (same override set, only the env var differs) about 30s after pass 1 finished,
  and Hydra's job logging overwrites `main.log` per run, so pass 1's precise final numbers are gone. What
  survives is directional: a full genome-wide pass finishing in under 9 minutes with 0 timeouts observed
  almost to the end is meaningful evidence toward "few or no regions time out in practice" for this
  VCF/sample, but it is not a substitute for pass 2's clean final count, and it says nothing about whether
  the original pathological `chr4:49101919-49110988` case specifically is even among this training VCF's 140
  shards.
- **Operational lesson**: when rerunning against the same `hydra.sweep.dir`/`subdir`, copy `main.log` (and
  any `*.realign_profile.log` files) out before relaunching, or point the sweep dir somewhere new — otherwise
  the previous run's log is silently clobbered with no way to recover it.
- **Pass 2 (job `78087`) launched** 2026-09-10 13:15:32 EDT, `NPSV3_REALIGN_PROFILE=1` set. Confirmed working
  end-to-end under real distributed SLURM conditions (not just the local `test_vcf_to_dataset` check from
  earlier this session): all 16 `images-{0000..0015}.realign_profile.log` files present in the output
  directory, together already over **1.3 million lines** within the first ~2.5 minutes, e.g.:
  ```
  REALIGN_PROFILE call=0 allele=ref ms=0.107 read1_alignments=1 read2_alignments=1 read_pairs=1 score=-6.736 nm=0
  REALIGN_PROFILE call=0 allele=alt_0 ms=0.085 read1_alignments=1 read2_alignments=1 read_pairs=1 score=-6.736 nm=0
  REALIGN_PROFILE call=0 stage=total ms=1.494 num_alt_alleles=1
  ```
  This run should yield both 0c measurements (timeout count + divergence-hypothesis score/NM data) from one
  pass once it completes; expect somewhat slower than pass 1's 8m42s given the added per-fragment
  `fmt::print`/`fflush` cost, but still likely well under the original 24h estimate.
- **Unrelated, noted only to avoid confusion**: SLURM job `78086` (`node006`) is a different pipeline step
  under `.../resources/sequence/.submitit/...` (different working directory) — not part of this run.

## Stage 1 implementation — per-base k-mer allele evidence (session 2026-09-10, continued)

Started while 0c's pass-2 SLURM job (`78087`) was still running (per the plan, Stage 1 doesn't depend on
0c's results to *start*, only to *ship*). Implemented the full mechanism described in the plan's Stage 1
section; the dual-run agreement harness / ablation study (plan "Verification") is **not** part of this
session's work -- that's a substantial separate measurement effort, tracked as remaining work below.

**What shipped:**
- `native/kmer_evidence.{hpp,cpp}`: `KmerEvidenceIndex`, a flat open-addressing hash table over packed
  2-bit-encoded k-mer keys (k <= 32) -> allele bitmask (up to 16 alleles). Two constructors match the
  `evidence_kmer_set` config axis: one over raw haplotype sequences (`all`/`local_unique`, the latter
  dropping keys that repeat within a single haplotype), one over a pre-enumerated k-mer set + per-k-mer
  masks (`graph_unique`, fed from `HaplotypeSamplerOverlay::kmers_on_path`). `Scan()` accumulates a
  per-allele difference array over one forward pass of a read (no reverse-complementing -- reads are
  already reference-oriented per plan constraint 7), then reduces to ref-vs-best-alt vote arrays exactly
  as the BWA path's argmax collapse does. Added to the `graph` static library and bound via `_native_graph`
  (`graph_bindings.cpp`). 10 new GoogleTest cases in `tests/native/test_kmer_evidence.cpp`, including a
  hand-derived tent-shape check (k=4 worked example) that matches the plan's 1a worked example exactly.
  Full native suite: 126/126 (116 prior + these 10).
- Python wiring: `graphs/haplotype.py:build_kmer_evidence_index` builds the index per region from the
  already-sampled/genotyping-prepared haplotypes, dispatching on `pileup.evidence_kmer_set`.
  `images/example.py:make_graph_example` now gates the (expensive) `FragmentRealigner` construction and the
  (cheap) `KmerEvidenceIndex` construction independently off `pileup.allele_evidence` (`bwa`/`kmer`/`both`)
  -- `kmer` mode skips the per-(fragment, allele) BWA cost entirely, which is the actual point of this
  plan. `pileup.py`/`images/generator.py` carry per-read `(ref_votes, alt_votes)` arrays through
  `PileupRead`/`PileupInsert` alongside the existing scalar `allele`/`read_allele`, and render them into a
  new channel 8 per the plan 1b pixel encoding (`ImageGenerator._kmer_evidence_pixel`). `MAX_NUM_CHANNELS`
  bumped 8 -> 9. New `conf/pileup/unphased_graph_kmer.yaml` variant (channels `[0,1,2,3,4,8]`,
  `allele_evidence: kmer`). Deleted the dead, already-broken `conf/pileup/npsv2.yaml` (referenced a channel
  8 that would `IndexError` before this change and would have silently started returning k-mer data instead
  after it, with zero actual callers anywhere in the codebase -- see plan 1e).
- Resize (plan constraint 3): `ImageGenerator._resize` now splits the `compress=True` resize by channel
  group -- unchanged bilinear resize for every existing channel, and a separate
  `_pool_by_evidence_strength` for channel 8 that picks, per output cell, the input pixel with the largest
  `|value - center|` (strongest ref/alt contrast in either direction), leaving a window 0 only if every
  pixel in it was 0. Note this is deliberately *not* `torch.nn.functional.adaptive_max_pool2d` on raw pixel
  values directly -- that would silently and systematically bias every downsampled region toward the ALT
  allele, since alt-favoring pixels sit numerically higher than ref-favoring ones around the 128 center.
  Implemented via `adaptive_max_pool2d(..., return_indices=True)` on a signed-distance proxy tensor, then
  gathering the original pixel values at the winning indices.

**End-to-end validation this session** (real HG00733/HG00096 data, `module load npsv3/main` +
`LD_PRELOAD=libjemalloc`, `ulimit -v 64000000`):
- `bwa` vs `kmer` vs `both` modes all run to completion and produce non-degenerate images.
  `chr7:136717247-136717522` (HG00096, population prior): BWA channels 6/7 populate 11173/10432 of 27648
  pixels; the k-mer channel 8 populates 5220 (`local_unique`/`all`, identical on this region) to 5652
  (`graph_unique`) -- lower than BWA's count as expected (discordant-pair-only fragments have no k-mer
  evidence by construction, plan's own predicted weak class), not zero or saturated.
- **Found and diagnosed what looked like a bug, wasn't one**: on `TestHG00733GraphExamples`'s own test
  region (`chr1:1134644-1134867`), channel 8 came back **entirely empty** (0/27648 nonzero) regardless of
  `evidence_kmer_set`. Traced to the region itself: both of that VCF's insertion alleles are tandem
  duplications of nearly the same ~59bp repeat unit (`ACATCTGAGGCTGCAGCCCCTCTCAGGAGGGGGCACCGTGAGGGGTGTGGCTTCCTCGC`
  vs. a ~2x tandem copy of a near-identical unit), so essentially every k-mer the sampled haplotypes
  produce is shared across alleles and gets pruned -- exactly the tandem-repeat blind spot plan constraint
  2 already documents ("every internal k-mer is shared... graceful loss of resolution, not a wrong
  answer"), not a defect in the index or scan logic. Confirmed by building the index directly and checking
  its retained-key count (8, vs. thousands expected at a non-repetitive locus of similar size) and by
  reproducing normal, non-empty channel-8 output on the non-repetitive `chr7`/`chr3` regions above. Kept as
  a live example of the documented limitation rather than something to fix.
- Full `pytest` suite (module loaded, no `ulimit` -- a whole-suite ulimit run cross-contaminates memory
  budget with the ~20GB KMC database mmap used elsewhere in the same process and produces a spurious
  "cannot open database" failure unrelated to any code change; that guard is meant per-test/per-target, not
  suite-wide): **126 passed / 10 skipped**, matching the exact Stage-0 baseline above. The only two
  remaining failures are pre-existing and unrelated to this session's changes (confirmed by reading, not
  stashing, the pre-session code): `tests/models/test_loaders.py::test_visualize_converted_images` is
  missing a `reference` config override it never had; `tests/benchmarks/test_haplotype_sampling_bench.py`
  overrides a `kmer.*` config key that has never existed (should be `graph.*`).

### Direct chr4:49101919-49110988 profiling comparison — BWA vs. k-mer (same session)

Re-ran the `REALIGNER_PERFORMANCE_NOTES.md` "Method" recipe above, unchanged (shared-memory BWA index,
cached graph shard, `local`/direct `make_graph_example` call so `cProfile` sees inside it, HG00096,
`graph.population_prior=True`, `pileup=unphased_graph`, `simulation.replicates=1` so the full 67-image
set — 1 real + `combinations_with_replacement(11, 2)` = 66 simulated genotypes — is generated, matching
the plan's own accounting), this time with the working tree's actual `allele_evidence=bwa` vs.
`allele_evidence=kmer` config, not a synthetic microbenchmark. Full script:
`/home/mlinderman/research/npsv3-native/tests/results/` cache reused across both runs (graph/filtered-kmer
construction identical either way, so profiled time is 100% image generation).

**`allele_evidence=bwa`, `timeout=90` (bounded — a full run was not let complete; see extrapolation
below):**
- **Timed out** as expected -- did not even finish realigning the *first* (real) image's fragments within
  90s: 2,957 of an expected ~3,295 `realign_fragment` calls completed, 86.18s cumulative self time =
  **29.1ms/fragment** (matches the historical ~2.324ms/allele x ~11 alleles = ~25.6ms/fragment baseline in
  "Findings" above almost exactly -- confirms Stage 0's fixes changed correctness/guards, not BWA's
  per-call cost, exactly as that section predicted).
- **Extrapolated full-run time** from this measurement: ~3,295 fragments/image x 29.1ms x 67 images =
  **~6,430s (~107 min)**, corroborating (same order of magnitude as, slightly above) the plan's earlier
  ~5,600s estimate. Not run to actual completion -- ~1-2 hours of otherwise-idle profiling for a result
  already unambiguous at 90s was not a good use of session time.

**`allele_evidence=kmer`, `timeout=1800` (default), run to completion:**
- **COMPLETED in 183.4s** -- the *entire* 67-image set (1 real + 66 simulated genotypes), well inside the
  1,800s timeout (currently ~90% headroom), directly confirming the plan's opening-section performance
  claim on the actual pathological region it was written about. This was the one specific confirmation
  still open after the initial Stage 1 implementation session (see the now-superseded "remaining work" note
  below).
- **Where the 183.4s actually goes** (cProfile, cumulative):
  - `_generate` (all 67 images): **145.5s** (2.17s/image average) -- the direct analog of BWA's ~84s/image.
  - `_scan_kmer_evidence` itself (283,634 calls, i.e. every read in every one of the 67 images): **5.7s
    total** -- matches the plan's own prediction ("~50-100ms per image, ~3-7s for all 67 passes") almost
    exactly.
  - `simulate_variant_sequencing` (ART + BWA-aligning the synthetic reads that back the 66 simulated
    images): **35.2s** -- a fixed cost identical under either `allele_evidence` setting, not part of what
    this plan optimizes.
  - **The new bottleneck, now that realignment is gone, is generic per-base pileup rendering** --
    `read_columns` (637,067 calls, 39.7s cumulative), `_refine_base_alignment` (39.99M calls, 29.6s),
    `_align_pixel` (362,699 calls, 20.0s), assorted numpy `clip`/enum-hashing overhead -- none of it new or
    specific to the k-mer channel, all of it pre-existing Python-level per-base coloring cost that was
    simply invisible before, dwarfed by realignment. Not urgent given 183s is already ~10x under budget,
    but the next thing to look at if this ever needs to be faster still.
- **Net result at this region**: **~35x** faster end-to-end (6,430s extrapolated BWA vs. 183.4s measured
  k-mer, including the ~35s simulation step common to both paths), and **~1,000x** faster on the
  realignment/evidence computation specifically (3,295 fragments/image x 29.1ms x 67 ≈ 6,080s of
  BWA `realign_fragment` time vs. 5.7s of `_scan_kmer_evidence` time) -- matching the plan's own "~1000x
  headline" phrasing in "Consequence of that scope decision" almost exactly.

### Dual-run agreement harness (same session, after the chr4 profiling comparison)

Implemented the harness described in the plan's "Dual-run agreement harness" subsection. New module
`images/agreement.py`; hooks added at the plan's specified single call site
(`ImageGenerator.generate`/`_generate` in `generator.py`, via a new `agreement_sink` parameter) rather than
by diffing rendered images (which the plan correctly notes would lose per-fragment traceability to the
pileup subsample and resize).

**What shipped:**
- `AgreementSink`: collects one row per (fragment, engine) as `_generate`'s existing `allele_evidence=both`
  computation produces both engines' results for a fragment -- `region, image_index, fragment_qname,
  engine, ref_quality, alt_quality, assignment, read_scores`, plus (needed for the plan's stratification
  asks, not literally in its minimal row tuple) `fragment_class`, `variant_class`, `min_baseq`,
  `num_alleles`, `multi_occurrence_hits`.
- `classify_fragment`: soft_clip / breakpoint_spanning / discordant_pair_only / both_mates_internal, from
  CIGAR ops and overlap with `analysis_variants` -- a best-effort proxy classifier (documented as such),
  not exact breakpoint truth.
- `nearest_variant_class`: INS/DEL/MIXED/SNV_OR_OTHER from the nearest analysis variant's
  `length_change()`. "Repeat-mediated" classification (also named in the plan) was **not** attempted --
  no reliable signal for it was readily available; left as a manual/curated axis for a future session.
- `native/kmer_evidence.{hpp,cpp}`: added `KmerEvidenceScan::multi_occurrence_hits` (a new
  `multi_flag_` table parallel to `keys_`/`masks_`, always populated by the haplotype-sequence constructor
  regardless of `exclude_local_multi`, so `all`-mode scans can report it even though `local_unique`
  drops those keys and can never report it, and `graph_unique` never flags anything -- documented in the
  header). `scan()`'s Python return grew from a 2-tuple to a 3-tuple accordingly. One new native test
  (`MultiOccurrenceHitsReportedUnderAllNotLocalUnique`); full native suite now 127/127.
- Analysis functions matching the plan's list: `confusion_matrix` (REF/ALT/AMB, any delta),
  `quality_correlations` (Pearson/Spearman on `alt_quality - ref_quality`), `bland_altman_data`,
  `stratified_disagreement` (by fragment_class/variant_class/min_baseq/num_alleles/multi_occurrence/region),
  `pixel_correlation` (pre-resize and at any candidate output width, reusing `ImageGenerator._resize` so it
  matches production rendering exactly), and `format_report` (sweeps `assign_delta` in {0.5, 1, 2} per the
  plan, to separate genuine disagreement from threshold-boundary noise). 21 new unit tests
  (`tests/images/test_agreement.py`), synthetic data plus hand-built `pysam.AlignedSegment`s for the
  classifiers -- full suite now 147 passed / 10 skipped (was 126; +21 agreement tests), same two
  pre-existing unrelated failures as before.
- `images/example.py:make_graph_example` gained an `agreement_sink` parameter: when given, tags it with
  the region/variants/allele-count context and captures the pre-resize, all-channels tensor (needed for
  pixel correlation) alongside the real image. **Scoped to the real image only, not the 66 simulated
  genotypes**, to bound harness cost -- a deliberate, documented simplification, not an oversight.

**A real bug found and fixed while building this**: the first version of `AgreementSink`'s k-mer
reduction summed the *entire* per-base cumulative vote array (`sum(ref_votes)`) to get a per-fragment
"quality". That integrates the tent's *area* (inflated by roughly `k * window_count`, observed values in
the thousands) rather than reporting a magnitude comparable to BWA's bounded [0,40] quality scale --
completely swamping any delta threshold and making every informative fragment "confidently" classified
regardless of actual evidence strength. Fixed to take the *peak* height reached (`max(ref_votes)`,
`max(alt_votes)`, summed across the fragment's one or two reads) -- bounded ~[0, 2k], the same
"apex height" quantity the pixel encoding (plan 1b) already treats as the informative signal. Caught by
inspecting raw joined rows directly rather than trusting the first report's numbers at face value; a
regression test (`test_kmer_reduction_uses_peak_height_not_area`) pins this down.

**Demo run** (`chr7:136717247-136717522` and `chr3:20967930-20972107`, HG00096, `population_prior=True`,
`allele_evidence=both` -- **not** `chr4:49101919-49110988`: BWA alone extrapolates to ~6,430s there per the
profiling comparison above, making a `both`-mode run on it impractical for an interactive session; that
region needs either a long-running batch job or a sampled approach in a follow-up):

- 1,240 rows (620 fragments x 2 engines) collected and analyzed end-to-end in under 3s total (both regions
  combined) -- the harness mechanism itself is cheap, as expected.
- **chr7** (2 alleles, a real het/hom-alt-carrying region): engines agree 85% of the time overall; BWA's 5
  confident-ALT calls that k-mer misses are (4 of 5) exactly the plan's own predicted weak class --
  `discordant_pair_only` fragments whose BWA signal comes entirely from insert size (`read_scores`
  confirms both reads score *identically* between ref/alt, i.e. only the fragment-level insert-size term
  discriminates) and so have no k-mer evidence by construction (plan 1c). This is the harness validating a
  specific, falsifiable prediction from the plan, not a surprise.
- **chr3** (2 alleles, but the alt haplotype came from population-panel top-up -- i.e. likely not actually
  carried by HG00096): overall agreement only 7.7%, entirely concentrated in one pattern, stable across the
  full delta sweep (not threshold noise): BWA reports ~0/~0 (both engines' own scale, landing on AMB) for
  508/522 fragments where k-mer reports near-maximal ref-only support (median peak height 42 = 2 x k,
  i.e. both reads in the pair saturate). Neither engine ever calls ALT here, so the two engines do **not**
  disagree on *direction* -- this is a confidence-calibration difference (k-mer is far more decisive than
  BWA's own scoring for hom-ref reads in this population-topped-up region), not the "disagreement on the
  hom-ref/hom-alt extremes" red flag the plan specifically warns to watch for (that would be one engine
  saying REF and the other saying ALT, which does not happen here). Worth deeper investigation before
  trusting confidence calibration across the two channels, but not evidence of a k-mer-side correctness bug.
- Pixel correlation (channel 6/7 vs. 8, informative pixels only) is weak-to-moderate and inconsistent in
  sign across regions/channels (e.g. chr3 pre-resize ch7 r=0.05, post-resize-288 ch6 r=-0.585) -- ranges too
  small (2 regions, one dominated by the calibration pattern above) to draw a real conclusion; needs the
  full region sweep below before it means anything.
- **fragment_class stratification is the clearest, most useful signal**: `both_mates_internal` (background,
  away from any variant) agrees 100% of the time (45/45) -- the two engines never manufacture disagreement
  out of nothing. `breakpoint_spanning` (the class nearest the actual variant) is where nearly all the
  measured disagreement concentrates (12%, n=551) -- expected, since that's exactly the region where the
  two engines' fundamentally different evidence models (local exact k-mer match vs. global realignment
  score) diverge most; the two regions' `variant_class` happens to be `DEL` for both, so INS/repeat
  comparisons need the fuller region set below.

**Remaining work, still not attempted** (all under plan "Verification", explicitly a larger, separate
effort -- the dual-run agreement harness itself and the chr4 timing confirmation, previously listed here as
open, are now done; see the two sections above):
- **Region curation**: the plan calls for >=3 insertion regions, >=3 repeat-mediated regions, and >=3
  discordant-pair-only-deletion regions specifically (beyond the 3 already-known regions used above). None
  of that curation has been done -- the demo run above used only the 2 fast pre-existing regions, both
  `variant_class=DEL`. Needs either domain knowledge or a script that scans the population VCF for
  candidates matching each criterion.
- **Repeat-mediated classification**: `nearest_variant_class` deliberately does not attempt this (no
  reliable, readily-available signal was found this session) -- it's a gap in the harness itself, not just
  in region curation.
- **The chr4-scale run**: `allele_evidence=both` on `chr4:49101919-49110988` specifically would need BWA to
  actually complete there (~6,430s extrapolated), which is impractical interactively -- needs a batch job,
  or a bounded/sampled variant (e.g. cap the number of fragments BWA-realigns for harness purposes only).
- **Simulated-genotype coverage**: the harness currently only instruments the real image, not the 66
  simulated genotypes per region (a deliberate scope decision to bound cost, see above) -- extending it
  would need threading `agreement_sink`/`return_raw` through the simulated-genotype loop in
  `make_graph_example` too.
- **The confidence-calibration pattern found at chr3** (see above) is worth a closer look before it's
  treated as settled: is BWA's own near-zero/near-zero scoring for hom-ref reads in population-topped-up
  regions expected/correct, or itself a separate pre-existing quirk? Not investigated further this session.
- The `evidence_kmer_set`/k sweep ablations, the informative-depth-channel ablation, the baseq-gate
  ablation, and the real-vs-simulated novel-k-mer domain-gap check -- all still unstarted.
- Model retraining/accuracy validation (explicitly out of the agreed validation scope per the plan, but
  should be scheduled before channel 8 replaces 6/7 in production).
