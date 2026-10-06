# Reference haplotype / background mismatch: notes

Status: brainstorm plus one diagnostic run (A1/A2 below). Nothing implemented. The label-construction mechanism was
confirmed by running the region. The original "background mismatch causes the high loss" hypothesis was **not** supported.

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

Three modules in `src/npsv3/graphs/` (aligner extras: `uv pip install ".[harmonize]"` or `edlib pyabpoa pywfa`):

- `similarity.py`: cheap tier. Pairwise-to-reference edits (`site_similarity`) and alignment-free k-mer `evidence_distance`.
- `harmonize.py`: the `truvari phab` step. Aligns the reference and haplotypes together (abPOA default, WFA optional) and
  re-calls each haplotype's variants from the one alignment. Port of Truvari 5.3.0, checked against it.
- `truvari_compare.py`: the `truvari bench` step as used by `truvari refine`. `compare_haplotypes` / `truvari_differ` answer
  "would Truvari consider these haplotypes different?"; `Comparison`/`MatchResult` give per-pair sequence, size, overlap and
  distance values and, for near misses, the threshold each criterion would need (`MatchResult.required`, `explain()`).
  `compare_haplotype_set` does all pairs (pairwise alignment per pair, the most faithful; `joint=True` aligns once).

Thresholds are `TruvariParams`, defaulting to the criteria used in past runs (`refdist` 20, `pctseq` 0.7, `pctsize` 0.7,
`pctovl` 0.6, `sizefilt` 30, `sizemin` 30); `TruvariParams.truvari_defaults()` gives stock Truvari. Rolling/unrolling for
repeats follows the `bench` CLI default. Validation: `tests/graphs/test_harmonize.py` and `test_truvari_compare.py` compare
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
