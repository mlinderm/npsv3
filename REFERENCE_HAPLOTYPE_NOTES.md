# Reference haplotype / background mismatch: notes

Status: brainstorm plus two diagnostic runs: A1/A2 on the chr4 region, and dataset-wide co-positive triage of a
`TrueHaplotypeStrategy` model (see "Co-positive triage results"). A prototype exists for the Truvari comparison. The
`TrueHaplotypeStrategy` (true paths plus/minus analysis-variant alleles) now addresses option 1 for training. No
loss/label changes have been made. The label-construction mechanism was confirmed by running the region. The original
"background mismatch causes the high loss" hypothesis was **not** supported.

## Problem

Region chr4:109850391-109850582 (microsatellite expansion), sample HG00096, PAV population VCF. Sampled haplotypes
(2 total; label genotype idx 2 = (1, 1)):

| idx | vs hg38 | content |
|---|---|---|
| 0 | +35bp | 35bp INS (1bp + 34bp blocks) in the left tract, no SV |
| 1 | +87bp | hap0 background + 52bp INS SV (hap1 vs hap0 is a single 52bp insert at hap0[131]) |
| assembled path | +54bp | two 1bp deletions + 56bp INS at the SV site (not in the sampled set) |

The sample's assembled haplotype is **not** a 2bp INS as originally written: it has two single-base deletions (-2bp)
and a 56bp INS (+56bp). It matches no sampled haplotype (closest is hap1; they differ by a 33bp deletion in the left tract).

The real image is more similar to the 0/0 simulation (two copies of haplotype 0) than to the simulated homozygous true
genotype, even though the sample has the SV. This example was derived from a small population sample and so could not take
advantage of correlation information in the background during sampling. Thus its sampling is entirely dependent on k-mer signal.

**New information (Oxford Nanopore).** ONT reads show insertions of roughly 18-50bp in this region: a true insertion of
*uncertain length*. From an interpretation perspective a 35bp INS (hap0) and the 52bp SV (hap1) would be equally
acceptable. So the "no-SV" label on hap0 is itself questionable, and the real question is whether a sample with an
insertion of roughly the right size, in the right place, should count as having the SV (see Goals).

## Mechanism

`prepare_genotyping_haplotypes` ([haplotype.py:484](src/npsv3/graphs/haplotype.py#L484)) matches the true haplotype to
a sampled one by comparing allele sets over `analysis_variants` (>=50bp) only. The small variants (the deletions, and the
35bp INS carried by hap0/hap1) are not analysis variants:

- True haplotype reduces to `{SV: 1}`; sampled haplotype 1 also reduces to `{SV: 1}`.
- They match, so haplotype 1 is reported as the truth and the exact true path is never added. (Confirmed: capture shows
  the assembled path is "NOT identical to any sampled haplotype".)

The simulation labeled "true" therefore has a different background from the sample's own path. Total insertion relative to hg38:

| | insertion | vs. assembled path |
|---|---|---|
| Assembled path | -2 + 56 = +54bp | |
| Haplotype 0 (no SV) | +35bp | 19bp off |
| Haplotype 1 (labeled true) | +87bp | 33bp off |

The original hypothesis was that this 33bp background mismatch is what makes 0/0 look more similar to the real image than
the nominal truth. **The A1/A2 diagnostic below does not support that**; see "Diagnostic results".

Note: `exact_reference=True` ([haplotype.py:458](src/npsv3/graphs/haplotype.py#L458)) may help in this particular instance
by removing SV-like variants from the reference, making it more "different", but would not address the incorrect background
for the SV haplotype. The "reference" haplotype (hap0) is also not hg38 here: it is the first sampled haplotype with all
>=50bp alleles REF, and carries the +35bp INS. Real reads are mapped to hg38.

## Diagnostic results (A1 / A2)

Script: [evidence_distance.py](.claude/skills/diagnose-kmer-evidence/evidence_distance.py) (step 3b of the
`diagnose-kmer-evidence` skill), run on a capture of this region. Re-running the integrated version reproduces these numbers
(noise-floor replicates vary: self deviance 88-311). Evidence is canonical
21-mer counts over the 116 k-mers whose multiplicity differs between candidates, scored by Poisson deviance. Per-copy depth
is calibrated separately for simulated (15.2) and real (31.5 both haplotypes) data from 596 unique flank k-mers. Candidates:
hg38, hap0, hap1, asm (assembled path), and **asmNoSV** (the assembled path with the 56bp SV removed: a background-matched
counterfactual twin).

**A2, noise floor.** Replicate synthBAM simulations (4 per candidate) scored against their own genotype give deviance
86-236. The nearest *wrong* genotype is 740-1400 away. In this k-mer evidence space no pair of candidates is statistically
indistinguishable.

**A1, model distance between hom genotypes** (deviance, expected data from row scored against column):

| | hg38 | hap0 | hap1 | asm | asmNoSV |
|---|---|---|---|---|---|
| hap0/hap0 | 7730 | - | 6341 | 14405 | 8695 |
| hap1/hap1 | 15995 | 7118 | - | 8536 | 16833 |
| asm/asm | 13214 | 12865 | 6308 | - | 7601 |

**Real reads** (lower is better; noise floor is about 100-250):

| genotype | deviance |
|---|---|
| hg38/hap0 | 1743 (best) |
| hg38/asmNoSV | 1960 |
| hg38/hg38 | 2093 |
| asmNoSV/asmNoSV | 2671 |
| hap0/hap0 | 2926 |
| asm/asm | 8213 |
| hap1/hap1 (the label) | 8584 (worst) |

Takeaways:

1. The "real closer to 0/0 than to the label" behavior reproduces (2926 vs 8584).
2. A correct background does not fix it: asm/asm (8213) is rejected about as badly as hap1/hap1 (8584), and the twin
   comparison on real reads is asm/asm 8213 vs asmNoSV/asmNoSV 2671. Options 1-3 would give a cleaner label but would not
   change what the reads show. The real reads do not carry the 56bp expansion's exact sequence.
3. The best real fit is hg38/hap0, i.e., the reads support a roughly 35bp insertion (hap0-private k-mers 19/19 seen, 360
   hits; hap1-private 1/23). This is **consistent with the ONT range of 18-50bp**: the short reads carry a real insertion,
   just not the assembly's 52-56bp representation. By the "equally satisfied with 35bp or 52bp" criterion, hap0 is
   arguably a correct description of this sample, yet it is labeled non-SV.
4. Even the best real fit (1743) is about 10x the noise floor, so the Poisson model is also misspecified (no overdispersion,
   stutter or GC bias; deviance dominated by high-multiplicity CTTT repeat k-mers; repeat reads can be soft-clipped or
   mapped elsewhere). Absolute thresholds from this model are not yet trustworthy.

Caveats: one region, k-mer space only (not the image channels), unfiltered k-mer counting of reads in region +-500bp.

**Reframing.** The failure here is not "two near-identical haplotypes, one labeled correct". It is an **equivalence-class
labeling problem**: hap0 and hap1 both represent "an insertion in this tract" at different lengths, the data and ONT cannot
cleanly choose a length, and the label forces a binary SV/no-SV split along a length boundary that the evidence does not
resolve.

## Goals

- SV vs. non-SV: recognize the SV when present (avoid collision).
- Between similar SVs (including same-site insertions of different lengths): "indistinguishable", if meaningfully defined,
  is sufficient. Treat same-site INS of similar length as one equivalence class (e.g., 35bp and 52bp here).
- A sample with a nearby INS but no SV is labeled as not having the SV. (Tension with the previous bullet: needs a length
  or reciprocal-overlap tolerance, otherwise a 35bp INS is "no SV" while a 52bp INS is "SV".)

## Options

### Label and construction (original options)

1. **Identify the true haplotype by full path, not analysis-variant alleles.**
   - Compare `hap_path` to sampled haplotypes by node sequence, or by alleles over all variants in the region.
   - During training if the exact path was not sampled, append it (the existing `None` branch already does this) or replace
     the matching SV path with the exact path.
   - Fixes label construction (truth known). Does not help inference, where the sampler would still choose the 35bp background.
   - Per the diagnostic, would not by itself fix this region's loss; useful mainly as the exact-truth ingredient for the
     options below.
2. **Factorial construction (background x SV allele).**
   - Split each sampled haplotype into background (non-analysis variants) and SV allele; for each background emit both a
     no-SV and an SV form (hap0/hap1 are already one such pair). Include the assembled background with and without the SV.
   - Guarantees a background-matched no-SV twin for each SV allele. Cost: up to 2x haplotypes per background; cap backgrounds.
   - The asmNoSV candidate above is this twin; it scores well against real reads (2671 vs 8213 for asm/asm), so twins make
     "reads lack this exact SV" visible.
3. **Make the sampler SV-aware.** Select backgrounds by k-mer support independent of the SV allele; check whether the
   assembled background is reachable at all.

### Brainstormed approaches (similarity as an evidence-space question)

**A. Measure similarity in evidence space**
- **A1 Expected-evidence distance.** Distance between two haplotype paths = informative k-mers that differ, weighted by
  uniqueness and expected coverage (Poisson deviance in the prototype). Cheap, no read simulation. Prototype works but
  repeat-dominated; needs weights that discount high-multiplicity repeat k-mers.
- **A2 Noise-calibrated "indistinguishable".** Replicate simulations give a noise floor; two haplotypes whose distance is
  below the floor are indistinguishable. Gives a principled epsilon. In this region everything is above the floor, so a
  distance threshold alone does *not* merge the 35bp and 52bp INS; an additional *semantic* equivalence (same site, similar
  length, per the ONT-informed goal) is needed. Likewise A2 can flag examples where the data fit no candidate (real vs best
  candidate >> floor).
- **A3 Real-vs-sim gap as an example-quality diagnostic.** Real-to-best-candidate deviance far above the floor flags
  unsupported or misspecified labels for down-weighting or exclusion.

**B. Change the label, not just the sampler**
- **B4 Dosage-level label with haplotypes latent.** Target is "insertion in this class" dosage (0/1/2); each dosage's score
  is a logsumexp over candidate haplotype pairs in the class. Background and exact length become marginalized nuisance.
  Natural fit for the "35bp or 52bp, equally fine" goal: hap0 and hap1 would both belong to the INS class.
- **B5 Soft labels from distance (plus semantic equivalence).** Target mass over candidates proportional to exp(-d/tau),
  where d combines evidence distance and a same-site/length-tolerance term.
- **B6 Multi-positive / false-negative masking.** Every candidate in the equivalence class of the truth is a positive, or is
  masked out of the negatives (debiased / supervised contrastive).
- **B7 Adaptive margin or temperature per pair.** Required separation scales with distance, so indistinguishable or
  equivalent pairs are not pushed apart. This is the RINCE-style question below, with A1 supplying the metric.

**C. Isolate the SV signal**
- **C8 Counterfactual twins.** For each true background build bg+SV and bg-SV; the SV effect is sim(bg+SV) - sim(bg-SV).
  Train on those pairs, or score real data by d(real, noSV twin) - d(real, SV twin).
- **C9 Background from the assembly during training** (third bigger question below), with background-noise augmentation
  at the rate the inference-time background estimator errs, to avoid a train/inference gap.
- **C10 Auxiliary background head** supervised on non-analysis variants; at inference feeds a posterior over backgrounds (B4).

**D. Background at inference**
- **D11 Read-level linkage.** Reads spanning both the background variant and the SV give direct phase evidence, no
  population LD needed.
- **D12 Report equivalence classes** when the data cannot separate them; evaluate at class level so the model is not
  penalized for ambiguity the data cannot resolve.

### Suggested combination
A1+A2 (distance with a calibrated floor) feeding B4+B6 (class-level label and multi-positive loss), with C8 twins to ensure
background-matched comparisons. Add a same-site length-tolerance rule for the semantic equivalence that A2 cannot supply.

### Truvari-style comparison (implemented, prototype)

Two modules in `src/npsv3/graphs/` (aligner extras: `uv pip install ".[harmonize]"` or `edlib pyabpoa pywfa`):

- `sequence_harmonize.py`: the `truvari phab` step. Aligns the reference and haplotypes together (abPOA default, WFA optional) and
  re-calls each haplotype's variants from the one alignment. Port of Truvari 5.3.0, checked against it.
- `sequence_compare.py`: the `truvari bench` step as used by `truvari refine`. `compare_haplotypes` / `haplotypes_differ` answer
  "would Truvari consider these haplotypes different?"; `Comparison`/`MatchResult` give per-pair sequence, size, overlap and
  distance values and, for near misses, the threshold each criterion would need (`MatchResult.required`, `explain()`).
  `compare_haplotype_set` does all pairs (pairwise alignment per pair, the most faithful; `joint=True` aligns once).

Thresholds are `ComparisonParams`, defaulting to the criteria used in past runs (`refdist` 20, `pctseq` 0.7, `pctsize` 0.7,
`pctovl` 0.6, `sizefilt` 30, `sizemin` 30); `ComparisonParams.truvari_defaults()` gives stock Truvari. Rolling/unrolling for
repeats follows the `bench` CLI default. Validation: `tests/graphs/test_sequence_harmonize.py` and `test_sequence_compare.py` compare
against oracle data generated by real Truvari (`tests/graphs/data/make_truvari_*_oracle.py`; 12 harmonization cases and 140
`bench` runs across 5 parameter sets all agree on TP/FN/FP and on Truvari's annotated similarity values).

Results with the project criteria (pairwise alignment):

| region | pair | Truvari would differ? | why |
|---|---|---|---|
| chr7 | hap1, hap2 (INS-171 / INS-172), asm | no (all pairs) | same INS at one site, sizesim 0.99 |
| chr4 | hap0 vs hap1 | **yes**, robustly | hap1 has an extra 52bp INS at the SV site; relaxing `refdist`, `pctsize`, `pctovl` or `sizemin` does not flip it |
| chr4 | hap0 vs asm | no | the MSA places asm's expansion in hap0's tract as one ~35bp INS |
| chr4 | hap1 vs asm | yes at `sizemin` 30, no at 40+ | the SV matches (52bp at the same site); hap1's extra 35bp left-tract INS has no partner |

Caveats: results depend on the alignment partner (asm's expansion is one 35bp INS beside hap0 but a 52bp INS at the SV site
beside hap1), exactly as with `phab`; `sizemin` matters (stock 50 would hide the 35bp background insertion); abPOA is documented
as non-deterministic across machines.

### Bootstrapping co-positives (continuous labels from the stage-0 model)

Idea: train with the single positive as now (M0). Its high-loss mass should include groups where the label genotype has
near-identical neighbours (e.g. same-site INS a few bases apart). Use M0's own notion of similarity to derive continuous
co-positive weights and retrain (stage 2). This makes B5/B6/B7 concrete, with M0 supplying the metric instead of (or alongside) A1.

**Why the single-positive loss hurts here.** A group has one real query and R simulated replicates for each of G candidate
genotypes. `InfoNCE` scores each replicate of the label against all negatives, excluding the other positives from the
denominator. If k negatives are truly indistinguishable from the label, the loss cannot go below **log(1 + kR)**. The real cost
is the gradient: it keeps pushing look-alikes away from the query, which can only be done with simulation noise or artifacts.

**Terms.** Everything is per group and per *genotype* (an index into `possible_genotypes`, an unordered pair of sampled
haplotypes), not per haplotype.
- **g\***: the labeled genotype (`example["label"]`).
- **g′**: any other candidate genotype in the group, i.e. any current negative. In the flagging rule, g′ is specifically the
  real query's top alternative.

**Three signals, three roles.** In short: M0's sim-vs-sim confusion decides which look-alike genotypes to stop separating, the
Truvari prior decides which differences don't matter, and the real query only flags examples to check.

| Signal | Measures | Role |
|---|---|---|
| **conf**: sim-vs-sim confusion from M0. Hold out replicate r of g\* as the query and softmax over the remaining supports. | Whether the evidence *can* separate g\* and g′, even with no real data | Sets the co-positive weight |
| **sem**: Truvari equivalence of g′'s haplotypes to the sample's **true assembled paths** (not to the matched sampled haplotype) | Whether the difference *matters* | Sets the co-positive weight |
| **p_real**: M0's posterior on the real image | What the reads show, *including* real-vs-sim gaps and label errors | Never sets the weight; only flags or down-weights the example |

Using p_real as the weight would be plain self-distillation. It would absorb the chr4 failure (put mass on hap0/hap0) rather
than expose it.

```python
# Per group, from M0: S[r, g] support embeddings (R x G), q real-query embedding
def genotype_scores(query, S, exclude=None):            # softmax over genotypes, replicates pooled by log-mean-exp
    logits = cos(query, S) / tau                         # (R, G)
    if exclude: logits[exclude] = -inf
    return softmax(logmeanexp(logits, axis=0))

# 1. Sim-vs-sim confusion: can M0 tell g* from g' when nothing is real?
P = mean(genotype_scores(S[r, g_star], S, exclude=(r, g_star)) for r in range(R))
conf = clip(P / P[g_star], 0, 1)                         # symmetrized in practice: also query with replicates of g'

# 2. Semantic equivalence against the sample's true paths (c, d)
geno_equiv(g) = (equiv(a, c) and equiv(b, d)) or (equiv(a, d) and equiv(b, c))   # (a, b) = haplotypes of g
sem = [geno_equiv(g) for g in genotypes]                 # equiv = not haplotypes_differ(...)

# 3. The only quantity the stage-2 loss sees
w = maximum(conf, sem); w[g_star] = 1

# 4. Real query: flag, don't relabel
g_hat = argmax(genotype_scores(q, S))
if g_hat != g_star and w[g_hat] < 0.5 and p_real[g_hat] > 0.8:
    flag(example); example_weight = 0.1                  # A3: reads prefer a genotype the simulator CAN separate
```

**Worked example: chr4 (this region).** Genotypes 0=(0,0), 1=(0,1), 2=(1,1). The label is g\*=2. Assume both assembled paths look like `asm`.

| g′ | conf | sem (vs asm, asm) | w | p_real |
|---|---|---|---|---|
| 0 = (0,0) | ≈0 (A2: separable) | **1** (Truvari: hap0 ≈ asm) | **1** | top |
| 1 = (0,1) | ≈0 | 0 | 0 | low |
| 2 = (1,1) = g\* | — | 0 (hap1 vs asm differ at `sizemin` 30) | 1 | lowest |

- **Confusion contributes nothing:** M0 bootstrapping alone cannot fix this region.
- **The semantic term does the work:** compared with the true path, hap0 is the Truvari-equivalent of the truth, so arguably the label is wrong.
- **Without the semantic term, the flag fires:** g_hat=0, w[0]=0, high p_real.

This is also why sem must compare to the true paths. Against the matched sampled haplotype (hap1), hap0 "differs robustly"
(see the Truvari table).

**Worked example: same-site INS-171 / INS-172 (illustrative numbers).** Haplotypes A (ref), B (INS-171), C (INS-172); g\*=(B,B).
- (B,C): conf 0.8, sem 1, so w=1. (C,C): conf 0.6, sem 1, so w=1. Both stop being pushed away.
- (A,B), (A,C), (A,A): conf ≈0, so they remain full negatives (dosage differs).
- Without the weights, about 2R negatives in this group cannot be separated, so the loss floor is about log(1+2R).
- If Truvari said "differ" but conf were high, still stop separating them. At inference they should show up as an ambiguous call.

**Loss variants given w** (w_{g\*}=1):

| Variant | Form | Character |
|---|---|---|
| Soft negative masking (debiased / false-negative cancellation) | −log e^p / (e^p + Σ (1−w_g) e^{s_g}) | Stops pushing look-alikes away, never pulls toward them. Equals the current code at w ∈ {0,1}. **Start here.** |
| Soft-target CE / KL (distillation, X-CLR, y-Aware InfoNCE) | −Σ_g q_g log softmax(s)_g, q ∝ w | Requires probability ratios to match w. Only if w is calibrated. |
| RINCE with ranks from binned w | already implemented | Cheapest experiment. Replace today's "any non-ref = rank 3" with similarity ranks. |
| Pair-specific additive margin (B7) | z_g = s_g + m(1−w_g) for negatives | Required separation scales with dissimilarity; argmax stays defined. |
| Rank-N-Contrast | denominator holds only candidates ranked farther than j | Needs a trusted full ordering; probably more than needed. |

Answer to the RINCE open question below: yes, but the rank, weight or temperature should come from conf/sem, not from p_real.

**Pitfalls.**
- **Cross-fit:** M0 scoring groups it trained on will under-report confusion. Use K fold-specific teachers, or freshly simulated replicates.
- **Merging can feed on itself** (merge → less separation → more confusion → more merging). Mitigations:
  - compute w once;
  - use a sharp teacher temperature;
  - prefer masking to soft targets;
  - cap w < 1 unless sem agrees.
- **Different SV dosage:** look-alikes across a dosage change are uncallable, not equivalent. Mask, but report them as uncertain.
- **Inference consistency:** co-positives tie, so argmax among them is arbitrary. Support-vs-support similarity needs no truth,
  so the same embedding can cluster candidates into classes at inference (B4/D12). Concordance needs a class-level variant.

**Diagnostic: is stage 2 worth building?**
[copositive.py](.claude/skills/diagnose-kmer-evidence/copositive.py) runs M0 over a split (same overrides as
`command=loss`). It hooks the encoder to record, per group:
- the model loss and an fp32 InfoNCE recomputation;
- `sim_loss` (InfoNCE with a held-out simulated replicate of g\* as the query: the loss the evidence allows with no sim/real gap);
- the full G×G sim confusion matrix, `conf`, `k_lookalike` and `floor` = log(1+kR);
- `p_real`.

`semantic` computes sem from `capture.py` pickles. `classify` splits the high-loss groups by the real query's top alternative g′:

| Category | Rule | Meaning |
|---|---|---|
| a_lookalike | conf[g′] ≥ 0.5 | Stage 2 helps |
| a_semantic | separable, sem[g′] = 1 | The label is the problem; g′ is arguably correct |
| b_flag | separable, not equivalent, p_real[g′] ≥ p_real[g\*] | Label or simulation problem (chr4 type); stage 2 should not absorb it |
| c_model | otherwise | Model uncertain on the real image |

It reports the share of groups and of loss mass per category, the share where g′ changes SV presence, and loss vs. log(1+kR)
by k. If look-alikes explain the loss, real InfoNCE sits near the floor. Verified so far:
- the InfoNCE recomputation matches `paired.InfoNCE` (1e-8 in fp32; within bf16 precision against the trained model's own loss);
- a planted look-alike pair gives conf=1 and a loss at log(1+R);
- the end-to-end smoke run uses the `test_paired.py` fixture.

**Needs >= 2 simulated replicates.** With one replicate (the default) there is no held-out replicate of g\*, so `conf` is a
noise-free lower bound and `sim_loss` is undefined. Either generate with `simulation.replicates >= 2`, or, with a
deterministic haplotype strategy, pair two image sets of the same regions with
[copositive_pairs.py](.claude/skills/diagnose-kmer-evidence/copositive_pairs.py) (used for the results below).

### Co-positive triage results (2026-10-09)

Model `+train=0011,data.batch_size=512/epoch=19-step=25980.ckpt` (`TrueHaplotypeStrategy`, 9 samples), evaluated on
HG00096 (48,980 groups). The full analysis is re-runnable in
[copositive_analysis.ipynb](.claude/skills/diagnose-kmer-evidence/copositive_analysis.ipynb). Outputs are in
`npsv3-experiments/training/test_outputs/copositive-0011-HG00096/`.

**Setup.** `TrueHaplotypeStrategy` is deterministic, so the `+repeat=0` (training) and `+repeat=1` (held-out) image sets
of HG00096 share each region's candidates, genotypes and label. Their sims were stacked as R=2: replicate 0 is
trained-on, replicate 1 is held out and supplies the real query. Scoring replicate 1 alone reproduces `command=loss`
(max |diff| 0.02, half precision). Caveats:
- HG00096's real images were in the training set, so the real query is in-sample.
- `conf` comes from a model trained on replicate 0 (the cross-fit pitfall). The mean loss against trained-on sims is
  only modestly lower than against held-out sims (0.75 vs 0.84).
- The real image differs between the two sets in 6.5% of groups (random read subsampling where depth exceeds image
  height, `generator.py`).

**Category split.** The high-loss cutoff is loss ≥ 2.87 (top 10%).

| category | all groups (%) | all loss (%) | high-loss groups (%) | high-loss mass (%) |
|---|---|---|---|---|
| a_lookalike | 29.4 | 61.2 | 56.4 | 51.2 |
| b_flag | 8.9 | 32.1 | 43.0 | 48.2 |
| c_model | 61.7 | 6.7 | 0.6 | 0.5 |

Exact accuracy is 75%. 36-genotype regions (18% of groups) carry 53% of the loss.

**1. Look-alikes explain most of the loss, and they are real.**
- **The loss sits on the floor.** For every k from 0 to 35, the median real InfoNCE equals log(1+kR) within ±0.1, and
  so does `sim_loss`. The model ties exactly the k look-alikes with the label and separates everything else.
- **Many negatives are identical.** 58k negative genotypes have conf > 0.99.
- **The equivalence is at the haplotype level.** Haplotypes were grouped when their homozygous genotypes are
  look-alikes; that grouping explains 74% of look-alike genotypes. Of 8,899 regions with 8 haplotypes, only 88 have
  all 8 separable, 451 have 1 class and 2,870 have 2.
- **The images are nearly identical; this is not an encoder failure.** Over 400 sampled groups, the median per-channel
  pixel difference between the label's and a look-alike's (conf ≥ 0.9) simulations is 1.02-1.12x the
  replicate-to-replicate difference. For separable (conf < 0.1) genotypes it is 1.13-1.71x. The k-mer channel ratios
  are 1.06 and 1.41.
- **Most look-alikes differ only in which SV haplotype.** Same dosage, other SV haplotype accounts for 62% of
  look-alike loss.
- **Some look-alikes change SV presence.** "Label has no SV, g′ has an SV" accounts for 16%. Overall, 13% of all
  groups have a look-alike that changes SV presence. These are uncallable, not equivalent, and masking them would hide
  a genuine failure to detect the SV.
- 64% of mispredictions are a look-alike of the label. Class-level accuracy (prediction in label + look-alikes) is 91%,
  versus 75% exact.

**2. Upper bound for stage 2.** These figures re-score the recorded R=2 logits; they are not from a retrained model.

| | real InfoNCE | change |
|---|---|---|
| unmasked | 51.4k | |
| soft masking (w = conf) | 31.6k | −38% |
| hard masking (conf ≥ 0.5) | 28.9k | −44% |

The remainder is dominated by b_flag (about 14k under either masking).

**3. Most b_flag groups are uncertain, not confidently wrong.** `classify` flags whenever p_real[g′] ≥ p_real[g\*].

| p_real[g′] | groups | share of all loss | presence differs | median sim_loss |
|---|---|---|---|---|
| < 0.5 | 2,629 | 20.3% | 47% | 2.49 |
| 0.5-0.8 | 1,114 | 5.7% | 88% | 0.51 |
| ≥ 0.8 | 610 | 6.1% | 96% | 0.23 |

The 610 confident flags (the notes' flag rule) are almost all presence disagreements that simulation separates easily:
416 have an SV in the label but look like no SV, and 170 the reverse. The model could not fit them even with these real
images in its training set, which points to label or simulation problems. These are the regions to inspect with steps
2-3b of the skill.

**4. The semantic term matters for some b_flags.** A random sample of 16 of the 922 high-loss b_flag groups where both
g\* and g′ carry an SV was captured and run through `semantic`. 4 became a_semantic: same-site INS/DEL of similar size
(e.g. INS 85 vs 103bp, DEL 115 vs 92bp, INS 132 vs 184bp).

**5. The chr4:109850259-109850582 label is fixed, but the region is still a b_flag (loss 5.0).**
- Under `TrueHaplotypeStrategy` the label haplotype (1) is exactly the assembled path (+54bp), and haplotype 0 is the
  background-matched no-SV twin (+2bp, i.e. asmNoSV).
- The other candidates add same-tract INS of 59, 90 and 114bp.
- The model predicts (3,3), a 90bp INS at 109850354 on the true background (p_real 0.71; then (0,0) at 0.09).
  Simulation separates it cleanly from the label (conf 0.01, sim_loss 0.36).
- Truvari treats only haplotype 2 (59bp INS) as equivalent to the assembled path, not haplotype 3.
- So with a correct background and label, the real image still matches none of the candidates' exact sequence, and the
  model picks a longer insertion than either the assembly or the ONT range (18-50bp). This is consistent with point (b)
  under "Root problem".

**Takeaways.**
- The "are look-alikes the problem?" question has a clear yes. About half the high-loss mass is the k-way tie, at
  exactly the information-theoretic floor, and the simulated images themselves are not separable.
- Stage 2 (masking) would remove that mass. But it must exclude, or report as uncallable, the presence-changing
  look-alikes.
- The other half is b_flag. It is mostly diffuse rather than confident. Its confident core is presence disagreements,
  and a fraction is semantic equivalence that a Truvari term would absorb.

## Root problem and bigger questions

A root problem is deciding which differences are important enough to separate. Revised view after the diagnostic: the issue
here is less "background mismatch" than **(a) a label that splits an equivalence class (35bp vs 52bp INS) the evidence and
ONT cannot resolve, and (b) the assembly's exact SV representation not being the sequence the short reads carry.** Many similar
SVs where only one is labeled correct may confound training if the short reads carry no information to differentiate them.

Open questions:
* Are there efficient mechanisms for identifying "too similar" a priori? (A1 is cheap; needs repeat-aware weights, and the
  statistical floor alone is not the full definition, per A2 results.)
* How should the same-site, different-length insertion equivalence be defined (length ratio, reciprocal overlap, SV type,
  breakpoint distance), and should it come from the VCF, from ONT support, or from both?
* If we have a similarity metric during training, could we use RINCE (or similar) to adaptively set the temperature for
  different "non-matching" examples to avoid suppressing indistinguishable ones?
* If true haplotypes/background are known during training (full assemblies), could we "sample" presence/absence of analysis
  variants from the true background to isolate the SV signal? What would we do at inference where that information is
  missing but we still want to exploit the background?

## Next steps

1. Run the A1/A2 script on a region where the label is right but the model fails, to see whether it separates
   "indistinguishable" from "unsupported by reads".
2. Make the deviance repeat-aware (downweight high-multiplicity k-mers, add overdispersion) so absolute thresholds are meaningful.
3. Decide the same-site equivalence definition (see open questions) and whether hap0 here should be relabeled as SV-present.
4. Prototype B4/B6 on the paired loss using the A1 distance, with C8 twins in `prepare_genotyping_haplotypes`.
5. ~~Run `copositive.py record`/`classify` with the current M0~~ Done (see "Co-positive triage results"). a_lookalike
   and b_flag carry about equal shares of the high-loss mass. Follow-ups:
   - Prototype soft negative masking restricted to look-alikes that keep SV presence, reporting presence-changing
     look-alikes as uncallable.
   - Cross-fit the teacher: train on a subset of samples and score held-out samples, so neither the real query nor the
     sims are in-sample.
   - Inspect the 610 confident b_flag groups (presence disagreements) as label/simulation problems with steps 2-3b.
   - Size a_semantic on more than 16 regions (the notebook's `N_SEMANTIC`). 4/16 suggests Truvari equivalence would
     absorb roughly a quarter of the SV-vs-SV b_flags.
