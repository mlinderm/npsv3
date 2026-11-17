# End-to-end genotyping plan

Status as of 2026-11-09: **steps 1–6 of 7 done** (input preparation, genotyping-mode graphs and images, the energon
predict mode, the predict runner, the VCF writer and the orchestrator). See [Status](#status).

## Goal

Replace `src/npsv3/genotype.py`, which uses an obsolete per-variant path (`make_graph_example_from_region`,
`graphs.graph_constructor`), with an end-to-end genotyper (`command=genotype`). The genotyper reuses the same
components as training-data generation, so genotyping can't drift from how the model was trained.

**Inputs**
- A phased population VCF.
- A preprocessed BAM/CRAM. Its stats JSON (`command=preprocess`) and KMC k-mer database (`Sample.kmc_prefix`) already
  exist.
- An optional inference VCF of variants to genotype. Without one, all of the population's analysis variants are
  genotyped.
- A model checkpoint.

**Output**
- A VCF with every variant in each genotyped region.
- Phased genotypes, with each region as one phase block.
- Quality scores derived from the predicted similarity to the simulated genotypes.
- Usable with Truvari and other VCF comparison tools.

**Requirements**
- Image generation runs in parallel (Ray, CPU).
- Inference runs on the CPU or on one or more GPUs.
- Graphs and images can optionally be saved, so new models can be evaluated without regenerating images.

## Design decisions (user-approved; do not re-litigate)

1. **The inference VCF may contain novel variants.**
   - Its *sites* are merged into the population VCF with `bcftools merge -m none`. Each inference ALT is tagged with
     `INFERENCE_ALLELE` (Number=A, `<vg_variant_id>_<allele index>`) and `INFERENCE_QUAL` (Number=A), so every merged
     record maps back to its inference alleles. `-m none` never combines records within a file, but does merge a
     population and inference record at the same POS that share any ALT (after normalizing REF padding), into a record
     with the union of their ALTs. (Revised 2026-11-11.)
   - Since each record merges at most once, an allele can still be in two records at the same POS. Those with an
     inference allele are resolved so it is in one record: kept in the record(s) whose genotypes refer to it (else the
     first), removed from the others (with the tags moved), and records left without ALTs dropped. So an inference
     record may be contained in a larger record, or split across records, but is never reported twice.
   - The inference VCF's samples are never read, so it acts as an excluded sample and has no effect on the
     population prior.
   - Variants present only in the population VCF may also be reported.
2. **If the sample is in the population VCF, it is excluded.** Its columns are dropped before graph construction
   (leave-one-out). With an inference VCF, the alleles only it carried are also dropped. Without one, the population is
   likely the truth set, so its private variants and alleles are kept as candidates (revised 2026-11-11). Otherwise its own haplotype paths would be added as candidates (`prepare_genotyping_haplotypes`)
   and would inform the prior and top-up, which leaks the truth. A consequence is that graphs are sample-specific
   whenever the sample is in the population.
3. **The prior should penalize novel alleles; no special handling is needed.**
   - No panel haplotype passes through a novel allele's nodes. The sampler's GBWT extension therefore comes up empty,
     it pays `panel_fallback_penalty` once, and it continues under the Markov model (`ApplyPopulationEdge` in
     `native/haplotype.cpp`).
   - Caveat: the Markov edge into the novel node is also scored with `transition_prior_alpha`-smoothed zero counts,
     which adds to the penalty. Add a test that a well-supported novel allele can still be sampled.
   - The prior is only used if the graphs were built with `graph.population_prior=true`.
4. **Population top-up is disabled** (`haplotype.population_haplotype_topup=false`). No truth haplotypes are added,
   because the sample is excluded and so has no paths in the graph.
5. **Single-haplotype regions.** If the sampler finds only the reference haplotype, `InsufficientHaplotypesError` is
   raised. In genotyping such a region is a call: emit `0|0`, set the qualities (GQ, DQ, PL) to `.`, and add the INFO
   flag `SINGLE_HAPLOTYPE`. Revisit this if it hurts accuracy.
6. **Pipeline: materialize the images, then predict.** Ray actors write an energon image dataset, to a temp dir or a
   kept one. Inference then reads it with `EnergonPackedImageDataModule`, the same loader and packing as training. A
   streaming design was rejected.
7. **Multiple GPUs: one spawned process per device, not Lightning DDP.** DDP re-launches the Hydra entry point, which
   would redo stages 1–3, and prediction doesn't need a process group.
8. **Output scope: all variants in each genotyped region,** not just the analysis (≥ `min_variant_size`) variants.
   Their alleles are decoded from the called haplotype paths.
9. **Quality scores** use a softmax at the training temperature (`model.loss.temperature`, 0.07). That is sufficient
   for v1; calibration is a possible later follow-up.
10. **Variants excluded from the graph.** `Graph::Graph` (`native/graph.cpp:278-284`) skips a record in only two
    cases: its reference span isn't *strictly* inside the graph region, or every ALT is `*`. Overlapping variants
    are not excluded; they become separate bubbles, and overlap only affects how panel paths are threaded.
    - Analysis variants are never at a region boundary, since regions are the analysis-variant spans plus padding.
    - Small variants that straddle a boundary are therefore never reported, unless they are inference records, which
      get `./.` with FILTER `NotInGraph`.

## Architecture

```
inputs ──► 1. prepare VCF ──► 2. graphs ──► 3. images (Ray, CPU) ──► 4. inference (CPU / N GPUs) ──► 5. VCF writer
            union VCF,        graphs_ds/    energon dataset          per-region (R,G) metrics       phased VCF
            sample excluded   unique_kmers  + failures sidecar        predictions-rank*.pkl          + PS/GQ/...
```

Working-directory layout:
```
work/
  inputs/genotyping.vcf.gz      # union VCF without the sample, plus provenance.yaml (input digests, sample, options)
  graphs_ds/  unique_kmers.*    # reusable via graph_shards= / unique_kmer_path= (only with the same provenance)
  filtered_kmers.*
  images/     dataset.yaml, meta_ds/, imgs_ds/, failures-*.jsonl, generation.yaml
  predictions/predictions-rank*.pkl, rejected.jsonl, prediction.yaml
  genotypes.vcf.gz(.tbi)
```

### Stage 1: Prepare the genotyping VCF (`genotyping/inputs.py`), DONE

- `prepare_genotyping_vcf(population_vcf, output_dir, *, sample_name, inference_vcf=None, threads=1)` returns
  `GenotypingInputs(vcf_path, inference_sites)`.
  - Without an inference VCF, the population VCF is used unchanged (nothing is written). The sample is excluded from
    the population prior (`graph.population_excluded_samples`) and its own paths from its candidate haplotypes
    (`haplotype.add_true_haplotypes=false`) by `genotyping_config`.
  - With one, `bcftools view -s ^SAMPLE -a --min-ac 1` also removes the ALTs only it carried and any records left
    without an ALT. Trimming changes the IDs of those records, which is harmless since every later stage reads this VCF.
  - The inference sites (with the `INFERENCE_*` tags) are written with pysam (`drop_samples=True`), and merged with
    `-m none`. Inference-only records come out of the merge with no genotype fields, which graph construction (and the
    native `Variant::Genotypes` and `Variant::genotype`) treats as `./.` for every sample.
  - One pysam pass over the merged output (`_resolve_shared_alleles`) resolves shared alleles and writes both
    `genotyping.vcf.gz` and `inference.sites.vcf.gz` (the records with an inference allele, sites only), which the
    region selection and the VCF writer read instead of the whole genotyping VCF. On chr20 (1.1M records, 102 samples,
    98k inference records) this takes 23.5s vs. 20.6s for the previous bcftools-only preparation.
  - If nothing changes (the sample isn't in the population and there's no inference VCF), the population VCF path is
    returned as-is, to avoid rewriting a whole-genome file.
- `genotyping_regions(vcf_path, *, flank, min_variant_size, inference_sites=None, region=None)` finds regions with
  `overlapping_variants` on the merged VCF, exactly as in training. With `inference_sites`, it keeps only regions with an
  analysis variant (merged record) that has an inference allele. `region_selection: all` genotypes every region instead
  (revised 2026-11-11; small inference variants no longer select a region on their own).
- `file_provenance`, `write_provenance` and `check_provenance` identify inputs by path, size and mtime, because
  hashing whole-genome inputs is too slow.
- Tests: `tests/genotyping/test_inputs.py`, including every relationship of population and inference records
  (identical, different REF padding, subset, superset, partial overlap, disjoint, and split vs. multi-allelic either way).

### Stages 2 and 3: Graphs and images (`genotyping/images.py`), DONE

`generate_genotyping_images(cfg, read_path, sample, inputs, regions, work_dir, *, graph_shards=None,
unique_kmer_path=None, filtered_kmer_path=None, images_dir=None, ploidy=2, min_variant_size=50)` runs both stages and
returns `GenotypingImages(dataset_path, images_dir, graph_shards, unique_kmer_path, filtered_kmer_path, counts,
failures)`. It reuses training-data generation rather than a separate path.

- **Ray** is initialized (with `images.example._init_ray`) only for the duration of the call. Unless the caller already
  started Ray, it is shut down on return, before stage 4.
- **Stage 2, graphs.** `serialize_graph_and_unique_kmers` has a new `regions: Sequence[Range] | None` parameter (it
  can't be combined with `region`). The genotyper uses it to build exactly the `genotyping_regions` into
  `work/graphs_ds`, pooling the unique k-mers into `work/unique_kmers`. The sample's k-mers are then filtered into
  `work/filtered_kmers` with `kmc_filter`. Graphs passed in are reused. Those need either `unique_kmer_path` or
  `filtered_kmer_path`.
- **Genotyping-mode settings** (`genotyping_config(cfg, sample_name)`) returns a copy of the config with
  `FORCED_SETTINGS` applied: `skip_unlabeled=false`, `hom_ref_downsample.enabled=false`,
  `genotype_subset.enabled=false`, `compare.enabled=false` and `haplotype.population_haplotype_topup=false`. It also
  appends the sample to `graph.population_excluded_samples`. It raises `ValueError` unless `haplotype.strategy` is a
  `SampledHaplotypeStrategy` (or a subclass), so `TrueHaplotypeStrategy` is rejected.
- **Sample absent from the VCF.**
  - `RegionActor.iter_regions` sets `GraphRegion.sample_idx=None` when the sample isn't in the VCF.
  - `generate_haplotypes` normalizes empty true indices to `[None]*ploidy` and skips label resolution when
    `sample_idx is None`.
  - The examples are therefore unlabeled (`label`/`label_rank` are null in `meta_ds`).
- **All-variant alleles in the haplotype record.**
  - `GraphRegion.variants` holds every VCF record overlapping the region.
  - With `variant_alleles=True` (a `RegionActor`/`GraphWriter`/`vcf_to_graph_examples` kwarg, not a config key, so it
    doesn't change `haplotype_options`), `haplotype_record` adds two fields:
    - `variant_ids`: the IDs of those records
    - `variant_alleles`: for each candidate path, the allele of each record, or `None` if the record isn't in the graph
  - `images.haplotypes.decode_variant_alleles` implements this in Python, from the `_alt_{id}_{allele}` paths, and
    mirrors `Graph::node_variant_paths_`. ALT allele *a* is identified by its path's nodes that aren't on the REF path
    (its ALT node; deletions have a zero-length `*` node). REF is identified by the reference nodes that any ALT
    replaces. A path through neither (spanned by an overlapping deletion) gets the record's first `*` allele if it has
    one, and REF otherwise (as `Graph::ApplyReferenceFallback` does).
  - Tests check that it matches the sampler's own `haplotype_alleles` for the analysis variants.
  - With `variant_alleles`, `GraphWriter` always generates the haplotypes itself before calling `make_graph_example`.
- **Failures sidecar** (`images.example.FailureWriter`, enabled with `record_failures=True`).
  - Each actor writes `failures-NNNN.jsonl` to the image directory, one row per region without an example:
    `{key, region, sample, reason, message}`.
  - The reasons are `InsufficientHaplotypes`, `Timeout`, `PrecomputedSkipped`, `Unlabeled` and `Downsampled`. The last
    two can't occur in genotyping mode.
  - `InsufficientHaplotypesError` now has a `haplotypes` attribute, the `haplotype_record` (with `variant_alleles`)
    of the single haplotype generated. The row records it as `haplotypes`.
  - `read_failures(dir)` reads every row back.
  - `GraphWriter.counts()` now also counts `insufficient_haplotypes` and `timeout`.
- **Dataset path and counts.** `vcf_to_graph_examples` returns `(dataset_path, counts)`. With `allow_empty=True` (set
  by the genotyper), a run with no examples, e.g. every region single-haplotype, returns `dataset_path=None` instead of
  failing in `prepare_energon_dataset`.
- **Generation settings.** The image directory gets a `generation.yaml` (see `generation_settings`, `read_generation`)
  with these fields:
  - `ploidy` and `min_variant_size`
  - `image: {channels, height, width, replicates}`, for stage 4 to check against `encoder.num_channels`, the image
    geometry and `r`
  - the resolved `pileup`, `haplotype`, `simulation` and `graph` config groups
- **Tests:** `tests/genotyping/test_images.py`, 10 tests (17 in `tests/genotyping` in total), all passing. The
  `replicates=2` variant of the HG00733 test skips unless the bwa-mem3 index is loaded in shared memory. It passed on
  a node with the index loaded. The tests cover:
  - variant-allele decoding against the sample paths, including a `*` allele under a deletion and a variant that
    straddles the region boundary
  - agreement with the sampler's decoding
  - unlabeled and single-haplotype records
  - `genotyping_config`
  - `regions=` graph construction
  - a reads-free single-haplotype run, where every region fails and there is no dataset
  - an end-to-end HG00733 run with the sample excluded from a two-sample population, with `replicates=0` and `2`

**Notes for later stages**
- The single haplotype in a `SINGLE_HAPLOTYPE` failure has every analysis variant's REF allele, but it can carry small
  non-analysis ALT alleles in `variant_alleles`. Decision 5 says to emit `0|0`. The record lets stage 5 emit
  homozygous-for-that-haplotype instead, if that turns out to be better.
- An analysis variant that isn't in the graph (its span isn't strictly inside the region) is never on any haplotype.
  `prepare_genotyping_haplotypes` therefore never finds a match for the reference allele set and always adds the
  reference path. This can't happen for an analysis variant that defines its region. It can happen for another
  analysis variant that straddles the region boundary. Such a region is still genotyped, but that variant gets
  `None` in `variant_alleles`, and so `NotInGraph` if it's an inference record.
- The custom `InsufficientHaplotypesError.__init__` means a pickled copy (e.g. across Ray) loses `haplotypes`. It's only
  read inside the actor, so this doesn't matter now.

### Stage 4: Inference

**Energon predict mode (`models/energon.py`), DONE:**
- `PackedGroupTaskEncoder(require_label=False)` is the predict mode; `EnergonPackedImageDataModule(predict_unlabeled=True)`
  applies it to `predict_dataloader` only (train/val/test still need labels for the loss).
  - Unlabeled groups are accepted. They're packed with an all-zero `label_rank` and `genotype_label` equal to
    `LABEL_PADDING_VALUE` (-100, the existing "no label" convention, rather than -1; consumers test `< 0`).
    `positive_indices()` skips them.
  - Each group's `meta` is `(key, region)`, where `key` is the `energon_key` (now `GroupMeta.key`). Only in predict
    mode: labeled consumers (`RecordLossCallback`, existing tests) still get the region string.
  - Predict mode rejects `genotype_subset` and `copositive`, so every stored genotype is scored.
- **No silent drops.**
  - At construction (both the encoder and the data module), `max_group_cost` checks that the largest group within the
    limits, `(1 + r_max·g_max)·ceil(w_max / w_bucket)`, fits in `batch_size`. `g_max` and `r_max` are required (and
    `w_max` if `w_bucket` is set); a violation is a `ValueError`.
  - The reject rules are now one function, `group_rejection`, returning a reason (`NoSimulations`, `TooManyGenotypes`,
    `TooManyReplicates`, `TooWide`, `OverBudget`, or `Unlabeled` outside predict mode). `preencode_sample` and
    `_group_accepted` both use it.
  - Rejections are found up front from the metadata rather than recorded by the workers: `predict_rejections(dataset,
    ...)` (or `dm.predict_rejections()`) streams the metadata jsonl, as `estimate_epoch_batches` does, and returns one
    `{key, region, sample, reason}` row per group the predict loader skips. Every other group is predicted exactly once.
    This avoids any IPC from energon's worker processes, and the shared `group_rejection` keeps them in agreement. These
    rows become FILTER `Rejected` in stage 5.
- Predict uses `get_val_dataset`, which doesn't shuffle. The packing buffer still reorders regions, but everything is
  keyed by `key`, so order doesn't matter.
- **Tests:** `tests/models/test_energon_loaders.py`, `TestPredictMode` and `TestPredictUnlabeled` (6 tests, 7 with
  parametrization): rejection reasons, the budget check, packing unlabeled groups, and an unlabeled dataset with two
  oversized groups read with 0 and 2 workers. Each group is either predicted once, with the right pixels, region and
  `num_genotypes`, or listed as rejected. All of `tests/models` (156) and the `test_vcf_to_energon_dataset` image tests
  pass.

**Runner (`genotyping/predict.py`), DONE:**
- `predict_genotypes(cfg, dataset_path, predictions_dir, *, devices="cpu", num_workers=None, precision=None)` returns
  `GenotypingPredictions(predictions_dir, paths, rejected)`.
  - `check_generation` compares `generation.yaml` with the config first: `pileup.image_channels`, the image height and
    width, `simulation.replicates` and `model.encoder.num_channels`. It also requires at least one replicate.
  - `predict_data_config` builds the data module config from `cfg.data` (batch size, caching, buffering), in predict
    mode over just the image dataset. `g_max` is the recorded `haplotype.max_diplotypes`, `r_max` the replicates, and
    `w_bucket`/`w_max` the width. Filters, genotype subsets and co-positive targets are cleared.
  - The batch size is checked against the largest possible region (`max_group_cost`) before anything starts, so a
    too-small `data.batch_size` is a plain `ValueError`.
  - The rejections are found once, in the parent (`dm.predict_rejections()`), and written to `rejected.jsonl`.
    `prediction.yaml` records the checkpoint's provenance, the dataset, the devices and the world size.
  - Earlier `predictions-rank*.pkl` files are removed first, so a rerun with fewer processes can't leave stale ones.
- **Devices** (`resolve_devices`): `"cpu"`/None, `"auto"` (every visible GPU, else the CPU), a GPU count, or a list
  of GPU indices.
  - CPU: one in-process run with `torch.set_num_threads(cfg.threads)` and `32-true` precision.
  - One GPU: in-process. Several GPUs: `torch.multiprocessing.spawn`, one process per device, each a single-device
    `Trainer` with `cfg.trainer.precision` (`16-mixed`) and `float32_matmul_precision("high")`.
  - Each process passes `rank`/`world_size` to the data module (a new `EnergonPackedImageDataModule` override of
    `_worker_config`), so energon gives it its own slice of the shards, and gets its own cache pool directory.
  - Each `Trainer` uses `LightningEnvironment`. Otherwise Lightning infers a cluster from the SLURM job: it rejects
    `--ntasks` (as in the VS Code tunnel job), and would give every process rank 0.
  - Spawned processes register the OmegaConf resolvers themselves (`_ensure_resolvers`).
- **Output:** `PredictionWriterCallback` writes `predictions-rank{i}.pkl` atomically, even if the slice is empty. It is
  a pickled DataFrame with one row per region: `key`, `region`, `num_genotypes`, `num_replicates`, and `metric`, the
  (R, G) float32 cosine similarities. **Pickle, not parquet:** pyarrow isn't in the environment, and this matches
  `RecordLossCallback`. Switching is local to the callback and `read_predictions` if pyarrow is added.
- `read_predictions(dir)` concatenates the ranks into one DataFrame indexed by `key`, and raises if a region was
  predicted twice. `read_rejections(dir)` reads `rejected.jsonl`.
- **Tests:** `tests/genotyping/test_predict.py`, 10 tests (CPU only; no GPU on the dev node). They train a one-batch
  checkpoint and use an unlabeled 12-region dataset with one region over `max_diplotypes`, and cover:
  - device resolution
  - `check_generation` mismatches
  - `predict_data_config`
  - the batch-size check
  - a CPU run where every region is predicted once or rejected, with the right shapes, and a rerun replaces the files
  - two ranks run one after the other, which partition the regions and match the single-process similarities
  - the `torch.multiprocessing.spawn` entry point with two CPU processes
  - an empty slice
- **Not yet checked:** the multi-GPU path by hand on `node002`.

### Stage 5: Scoring and the VCF writer (`genotyping/vcf.py`), DONE

**Scoring for each region (configurable via `scoring.*`):**
- `s_g = mean_r metric[r, g]` and `p_g = softmax(s / T)`, with T taken from `model.loss.temperature` and overridable.
- The called diplotype is `argmax p`. Map it through `genotype_idxs` to the pair (hᵢ, hⱼ) of
  `combinations_with_replacement(range(num_haplotypes), 2)`.
- The model's own `MaximizingPredictor` takes the argmax over all R·G supports. Averaging over replicates first is the
  default here, and a config switch (`scoring.aggregate`) selects either.

**Fields for each variant in the region:**
- **GT:** `a|b`, the alleles on hᵢ and hⱼ, taken from `variant_alleles`.
- **PS:** the region's first POS. Each region is one phase block, since phase across regions is unknown. Hom calls
  are also written phased.
- **GQ:** for each unphased genotype x of the variant, `P(x) = Σ p_g` over the diplotypes that produce x.
  `GQ = min(99, -10·log10(1 - P(called)))`.
- **PL:** Phred-scaled and normalized from the same marginals (biallelic and multiallelic).
- **DQ:** `-10·log10(1 - p_called)`, the region-level diplotype quality.
- **MT:** the called diplotype's similarity. **MG:** the margin between the best and second-best `s_g`.
- **QUAL:** `-10·log10(P(0/0))`. The inference records' QUALs are in `INFERENCE_QUAL`.

**Which records appear:**
- Every merged-VCF record inside a genotyped region.
- Every merged-VCF record with an inference allele, even when it wasn't genotyped. Those get `./.` and a FILTER:
  - `NoRegion`: too small, or outside any region
  - `InsufficientHaplotypes`, `Timeout` or `Rejected`: the region wasn't scored
  - `NotInGraph`: excluded by the graph-inclusion rule
- `SINGLE_HAPLOTYPE` regions: `0|0`, qualities `.`, plus the INFO flag.
- Records are written as merged (alleles, ID such as `id1;id2`, INFO including `INFERENCE_ALLELE`). Genotypes for a
  subset of a record's alleles (e.g., just an inference record contained in it) aren't calculated.

**Header and output:** copy the merged VCF's header, add the new FORMAT, INFO and FILTER lines, and add the
checkpoint path and digest. Then `bcftools sort` and index.

**As implemented:** `score_diplotypes`, `genotype_variant` and `write_genotypes_vcf`. Where this section left details open:
- A failure row *with* a haplotype record is a `SINGLE_HAPLOTYPE` call. `InsufficientHaplotypes` is only the FILTER for a
  row without one.
- The default temperature comes from the instantiated loss, since InfoNCE's is a constructor default rather than a
  config key. A loss without a scalar temperature (RINCE) requires `scoring.temperature`.
- Probabilities are kept in log space. Phred values are capped at `MAX_PHRED` (999), because a genotype that no diplotype
  produces has a probability of zero. It sets the PL of unreachable genotypes and the QUAL if no candidate has REF.
- Because the call is the best *diplotype*, another unphased genotype can have a higher marginal, and so the zero PL.
- PS is the POS of the region's first reported record. Genotyped records are FILTER `PASS` whatever the input's FILTER was.
- All of the source's FORMAT lines are replaced, since population VCFs often define GQ or PS differently.
- Records in a timed-out or rejected region that aren't in the inference VCF aren't reported.
- Any region without a prediction, failure or rejection is a `ValueError`, as is a failure reason that can't occur.
- Truvari, like any input to it, needs multi-allelic records split first (`bcftools norm -m-any`).
- Tests: `tests/genotyping/test_vcf.py`. They include reading the output with bcftools and Truvari (`truvari/5.3.0` module).

### Stage 6: Orchestration and config (`genotype.py`), DONE

- `genotype.py` is rewritten as `genotype(cfg, reads, sample, population_vcf, output, *, inference_vcf=None,
  work_dir=None, images_dir=None, graph_shards=None, unique_kmer_path=None)`.
- `conf/command/genotype.yaml` gets:
  - `population` and an optional `inference`, replacing `input`/`background`
  - `devices` (`cpu`, or a GPU count or list)
  - `images_dir`, `work_dir`
  - `scoring.{temperature, aggregate, max_gq}`
  - the genotyping-mode overrides from stage 3
- `main.py`'s `genotype` branch drops `background` and makes the new paths absolute.
- Resuming: if `images_dir` is given, stages 1–3 are skipped after checking provenance. If `graph_shards` is given,
  stage 2 is skipped.

**As implemented:**
- `images_dir` is reused only if it is complete, i.e., has `generation.yaml`. Otherwise images are generated there. Its
  prepared VCF and `provenance.yaml` are in `images_dir/inputs`, so it can be reused without the original working directory.
- Images are never added to a non-empty directory (a `ValueError`), so stale shards or failures can't be mixed in.
- Reused graphs need the `provenance.yaml` the run that built them wrote to `graphs_ds`, which is compared without the
  reads. Graphs from `command=graphs` can't be used, since they aren't built from the merged, sample-excluded VCF.
- A specified `work_dir` is always kept. Otherwise a temporary directory is used.
- The YAML's genotyping-mode overrides duplicate `FORCED_SETTINGS`, which remain the enforcement. The exception is
  `haplotype.population_haplotype_topup`, which is only forced by `genotyping_config`: `genotype` is the default command, so
  setting it in the YAML changes the default for everything that composes the default configuration.
- Tests: `tests/test_genotype.py`. The orchestration tests replace stages 2 and 3 with a stub that writes real image shards,
  and run real input preparation, inference and output. `test_genotype_small_region` runs every real stage, and skips unless
  the bwa-mem3 index is in shared memory.

## Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Input preparation: merge, sample exclusion, regions, provenance (`genotyping/inputs.py`) | **Done**, 7 tests passing |
| 2 | Genotyping mode in haplotypes/images: no truth, forced settings, `variant_alleles`, failures sidecar, `regions=` for graph construction (`genotyping/images.py`) | **Done**, 10 tests passing (the `replicates=2` test needs the bwa-mem3 shm index) |
| 3 | Energon predict mode for unlabeled groups, `(key, region)` meta, no silent drops (`models/energon.py`) | **Done**, 7 new tests passing, not committed |
| 4 | Multi-device predict runner and per-rank predictions (`genotyping/predict.py`) | **Done**, 10 tests passing (CPU; multi-GPU not yet checked by hand), not committed |
| 5 | Scoring and VCF writer: marginalization, GQ/PL/DQ, PS, FILTERs, `SINGLE_HAPLOTYPE` | **Done**, tests passing, not committed |
| 6 | Orchestrator, Hydra config, `main.py`; rewrite `tests/test_genotype.py` as an end-to-end small-region test | **Done**, tests passing, including the all-real-stages test (needs the bwa-mem3 shm index), not committed |
| 7 | Validation: genotype a held-out sample (e.g. HG002 or HGSVC) and compare with Truvari against assembly-derived calls | Next |

### Testing notes

- Tests for each step go in `tests/genotyping/`.
- The image tests use a synthetic chr1 VCF (needs the hg38 FASTA) and an HG00733 region with the sample excluded from
  a two-sample population (needs the HG00733 CRAM and KMC database).
- The `replicates>0` image tests also need the bwa-mem3 index in shared memory. Don't load it from a 32GB SLURM job:
  it is a ~23GB segment.
- Stage 5 unit tests cover marginalization, GQ/PL, PS and FILTER assignment, plus a check that bcftools and Truvari
  can read the output.
- Run tests with `libjemalloc` preloaded. Wrap anything that builds graphs or images in `ulimit -v` (see
  `CLAUDE.md`).
- The GPU path is checked by hand on `node002`; CI covers the CPU path.

### Open items and follow-ups

- Test that a well-supported novel allele can still be sampled despite the fallback penalty plus the smoothed
  transition penalty (decision 3).
- Optionally fit the softmax temperature on held-out labeled data (from `command=loss` output) for better calibrated
  GQ.
- Revisit calling GT from the best diplotype when another unphased genotype has a higher marginal probability (and so
  the zero PL). GT comes from the single most probable diplotype, while GQ and PL sum over every diplotype that gives the
  variant the same genotype, so an allele carried by several similar candidate haplotypes has its probability split.
  - Example: with candidates H0 (REF), H1 and H2 (both ALT), and diplotype probabilities H0/H0 0.30, H0/H1 0.25, H0/H2 0.25,
    H1/H1 0.05, H1/H2 0.10 and H2/H2 0.05, the call is `0|0`, but the marginals are 0/0 0.30, 0/1 0.50 and 1/1 0.20, giving
    `PL=2,0,4` and `GQ=2`.
  - Possible effects: a bias against alleles with several candidates (likely favoring REF), GQ calibration curves that mix
    wrong genotypes with correct alleles lost to a split, and disagreement with tools that derive a genotype from PL.
  - How often this happens is unknown. In step 7, count the records where GT isn't the zero-PL genotype, and compare
    accuracy against the truth for both choices at those sites.
  - If the zero-PL genotype is noticeably more accurate, the options are to merge candidates that match at the analysis
    variants before choosing the diplotype (keeps one consistent diplotype, so try this first), or to choose GT per
    variant from the marginals (simpler, but variants in a region may no longer come from one pair of haplotypes, which
    the phasing relies on).
- Revisit the `SINGLE_HAPLOTYPE` `0|0` handling if it hurts accuracy. The alternative is to build ALT candidates from
  the graph with `set_variant_allele` on the reference haplotype.
