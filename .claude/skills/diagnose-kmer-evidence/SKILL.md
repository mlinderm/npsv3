---
name: diagnose-kmer-evidence
description: Explain why a region has a high loss or a blank/missing k-mer evidence channel in the real image despite the simulated true genotype showing evidence. Use when asked to investigate a high-loss or mispredicted region (chr:start-end), compare simulated vs real k-mer evidence, or decode what a genotype index means.
---

# Diagnose k-mer evidence for a region

Pipeline facts this relies on (verify against `src/npsv3/images/example.py` and `graphs/haplotype.py` if they seem stale):

- Genotype index `i` is the `i`th pair of `itertools.combinations_with_replacement(range(n_haplotypes), 2)`. Haplotype 0 is
  always reference; the rest come from the sampler, plus the sample's "true" haplotypes. Haplotypes, not alleles: 4 haplotypes
  = 10 genotypes. Labels come from the VCF GT for the sample; `missing_are_ref=True` makes `.` into REF.
- The k-mer evidence channel is `build_kmer_evidence_index` over `left_flank + haplotype + right_flank` (flanks are
  **reference** sequence, `pileup.realigner_flank`). With `evidence_kmer_set=local_unique`, a k-mer is dropped if it
  recurs inside any one haplotype, or is present in every haplotype. Only exact k-mer matches of `evidence_kmer_size`
  (not graph.kmer_size) light up. Simulated reads are drawn from the haplotype sequences, so they always match; real reads
  only match if the sample really carries that exact sequence.
- Consequences to check: (1) alleles built from repeat units (VNTR/tandem repeats) have no unique k-mers; (2) a
  haplotype's evidence is concentrated in the few junction k-mers; (3) variants outside the region only affect the
  (reference) flank and are shared by all haplotypes, so cannot create evidence.

## Steps

1. **Get the loss record** (optional). Losses tables are `losses.pkl.gz` pandas DataFrames with columns `region, loss,
   genotype (label idx), prediction, correct, num_genotypes, label_rank, genotype_metric...`.
   `--losses` in step 3 prints it and takes the prediction index from it.
2. **Capture** haplotypes/alleles/flanks for the region by running the pipeline for it with a thin pytest wrapper. From the
   repo root, in the project environment (`conda activate npsv3-native`, `module load npsv3/main`, jemalloc preloaded per
   CLAUDE.md), and with the memory bound:
   ```sh
   S=.claude/skills/diagnose-kmer-evidence
   ( ulimit -v 64000000; REGION=chr17:940789-941433 OUT=$SCRATCH/cap.pkl \
     exec python3 -m pytest $S/capture.py -p tests.conftest --rootdir=. -q )
   ```
   Defaults are the HGSVC3/HPRC PAV population VCF and sample `hg00096_sample`. Override with `SAMPLE_FIXTURE=<fixture in
   tests/conftest.py>` and `CFG_OVERRIDES="..."` (mirror the `cfg_overrides` marker on the corresponding test in
   `tests/images/test_example.py`, keeping `simulation.replicates=0`). The first run for a region builds graph shards and a
   filtered KMC database, cached under `tests/results/<region>.<vcfhash>.<sample>.k31/` and reused by the real tests.
   Needs read, reference and KMC inputs under `/storage/mlinderman/projects/sv/npsv3-experiments/resources`; if
   a file is missing the capture skips/fails rather than guessing.
3. **Analyze**:
   ```sh
   python3 $S/analyze.py $SCRATCH/cap.pkl --losses <losses.pkl.gz> [--pred IDX] [--flank-del POS:REFLEN]
   ```
   Output sections: repeat period and how many 31-mers recur in hg38 (canonical count >= 2; see the reference-DB caveat below),
   haplotype to variant mapping with what the label and
   predicted indices mean, retained k-mers per haplotype, the k-mers each haplotype adds over the reference haplotype and how
   many appear in real reads, the k-mers separating label from prediction, whether the sample's own called variants near the
   region (density, SNV alt-allele read fraction) are present in the reads, and a read depth/MAPQ profile. `--flank-del` checks
   whether a downstream variant in the flank changes the index.
3b. **Distinguishability (A1/A2)** (optional; use when the question is "are these haplotypes too similar to separate?" or "which
   genotype do the real reads actually support?"). Needs the same environment as step 2 plus `art_illumina`/`bwa`/`samtools` on
   `PATH` (`module load npsv3/main`), since it runs replicate read simulations:
   ```sh
   ( ulimit -v 64000000; exec python3 $S/evidence_distance.py $SCRATCH/cap.pkl --replicates 4 [--out res.pkl] )
   ```
   Candidates are hg38, each sampled haplotype, the sample's assembled path (`asm`) and `asmNoSV` (the assembled path with its
   largest insertion removed: a background-matched counterfactual twin). It prints (A1) the Poisson-deviance distance between
   homozygous genotypes over the informative k-mers, (A2) a replicate-simulation noise floor and nearest wrong genotype, and the
   real reads' deviance against every genotype. Read it as: a pair whose A1 distance is near the A2 floor is indistinguishable;
   a best real fit many times the floor means the data fit no candidate (unsupported or misspecified label); comparing
   `asm/asm` with `asmNoSV/asmNoSV` on real reads separates "wrong background" from "the reads lack this SV". Caveats: Poisson
   (no overdispersion/stutter/GC bias), deviance dominated by high-multiplicity repeat k-mers, k-mer space only (not the image
   channels), so use floor ratios qualitatively. `--coverage/--insert-mean/--insert-std/--sequencer/--read-length` default to
   HG00096; override for other samples. Example chr4:109850391-109850582: real best fit hg38/hap0 (1743), label hap1/hap1 worst
   (8584), floor ~230; the assembled-background twin comparison (8213 vs 2671) shows the reads lack the exact SV, not that the
   background was wrong.
4. **Interpret**, in this order. Two very different outcomes have appeared so far:
   - **Evidence exists in the index but not in reads** ("label over prediction" k-mers > 0, seen in real reads = 0).
     Then ask *why the reads lack the allele's sequence*:
     - **Repeat/segdup region** (self-similarity at a short period, most of the region's 31-mers recurring in hg38 (e.g. 91% chr17, 86% chr4 vs 0% chr20), MAPQ-0 or soft-clipped
       reads, depth far from the sample mean): alleles built from repeat units have no unique k-mers, so the k-mer
       channel is blind by construction. Example chr17:940789-941433 (57/171 bp VNTR).
     - **Unique region, normal depth and MAPQ, 0% non-unique** but the sample's called variants are absent from the reads
       (SNV alt fraction ~0 where the VCF says het/hom-alt, no >=20bp I/D or clips in CIGARs, depth unchanged inside a "deleted"
       interval): the label is not supported by the short reads at all. Suspect an assembly (PAV) artifact: a divergent or
       mis-aligned contig segment (very dense small variants, e.g. 1 per ~10bp over >1kb, `QRY_STRAND=-`, called in one
       sample only). The k-mer channel isn't hiding anything; the model's prediction (typically hom-ref) agrees with the
       reads and the label is noise. Example chr20:31640012-31640510.
   - **Short tandem repeat (microsatellite) expansion whose label is real** (e.g. LRS supports the insertion): the inserted
     sequence is made of repeat units, so pure-unit k-mers are shared by all haplotypes or recur within one and are dropped;
     only the imperfect ends can discriminate. Use the "hap vs hap0" diff (is the INS pure units plus an odd tail? does the
     true-path haplotype carry extra nearby alleles, e.g. a cluster of overlapping INS records?) and the cluster table: if
     reads contain the *reference* boundary k-mers (hundreds of hits) but almost none of the assembled haplotype's boundary
     k-mers, and some sub-tract matches neither, the short reads carry a real expansion whose exact sequence differs from the
     assembly's representation (imperfect ends, stutter, nearby alleles), and reads lying wholly inside the tract are
     multi-mapped or soft-clipped away. Short-read k-mer evidence cannot resolve it. Example chr4:109850391-109850582
     (INS-52 = 13 x CTTT units in a ~64bp CTTT/CTTC tract; 19/19 ref right-junction k-mers seen with 360 hits, 1/23 alt).
   - **Predicted genotype's haplotypes retain no k-mers** (typical for repeat-unit insertions): a blank real image is
     consistent with it, and the model must be using other channels.
   Report what was verified directly (k-mer sets, read scans) separately from inference; this tool does not render the
   image channels, and the read scan uses raw CRAM reads (the generator may downsample or filter).
   For a suspected assembly artifact, optionally rebuild the sample's h1 allele over the dense run (apply the GT[0]==1 records
   to the reference), count 21-mers absent from the reference, check they are absent from reads, and `bwa mem -a` it against
   the reference to see whether it aligns elsewhere (paralog) or only to this locus with heavy mismatches.

## Notes

- **Reference k-mer DB is canonical; `KmerCounts.count()` is not.** `Homo_sapiens_assembly38.non_unique.k31` is a KMC database with
  `both strands: yes` and ~2.5G k-mers, i.e. *every* hg38 31-mer with its occurrence count (the name "non_unique" is a misnomer;
  count 1 = unique). `count()` does not canonicalize, so looking up a forward-orientation k-mer returns 0 for ~50% of k-mers.
  Always query `min(kmer, revcomp(kmer))` (analyze.py does). The pipeline itself currently does not
  (`UniqueKmersOverlay` + `graph.canonicalize: false`, see kmer.cpp), so its reference filter misses ~half of recurring k-mers:
  33% of chr17 and 50% of chr4 graph-unique k-mers recur in hg38. This does not affect the `local_unique` evidence channel (it never
  reads that DB) but does affect haplotype sampling and `graph_unique` evidence.

- Graph haplotypes DO carry small variants (the graph is built from the whole VCF; only the analysis/matching step below uses the
  >=50bp subset). Don't assume the simulation lacks the nearby small calls: in chr20:31640012-31640510 the sampled alt haplotypes
  had 12-30 edit blocks vs hg38, so the simulated alt reads carried many small alleles the real reads lacked.
- The haplotype used as the sample's "true" one is a *sampled* haplotype matched to the sample's genotype using only the >=50bp
  analysis variants (`prepare_genotyping_haplotypes`); the sample's own assembled graph path is only added if no sampled haplotype
  matches. So the simulated true-genotype image can carry small alleles the sample lacks (chr4:109850391-109850582: sampled hap1
  had a +20bp left sub-tract insertion where the assembled path has a 58bp deletion). `analyze.py` reports whether the
  assembled path (`true_paths` in the capture) equals a sampled haplotype and how many of its own k-mers the reads contain.
- The "reference" haplotype (hap0) is likewise the first *sampled* haplotype whose >=50bp alleles are all REF, not necessarily hg38;
  it only falls back to the true reference path if no sampled haplotype qualifies. Real reads are mapped to hg38, so where small
  variants cluster (STRs) hap0 can disagree with the reads while hg38 agrees. chr4:109850391-109850582: hap0 is +52bp vs hg38
  (insertions at 109850418/424), hap1 +120bp, the sample's assembled path +54bp (all of it the 56bp INS-52); the reads match hg38 in
  the left tract (17/53 hg38-only k-mers seen vs 0/11 hap0-only). `analyze.py` prints each haplotype's edits vs hg38.
- Haplotypes are anchored at the graph region's start and end, which equal the requested region; the flanks are hg38 outside it
  (verified: left flank + graph reference path + right flank is contiguous hg38). In a tandem repeat the flank's last bases and the
  haplotype's first bases are the same periodic unit, so a "suffix == prefix" overlap is expected by periodicity and is NOT a
  duplication; don't report one.
- Region must be exactly the graph region string used when serializing (`chrN:start-end`, 1-based inclusive as in `loss` tables).
- Don't edit pipeline code for this; `capture.py` wraps `prepare_genotyping_haplotypes` and `build_kmer_evidence_index`
  on the `npsv3.images.example` module instead. If those functions change signature, update the wrappers.
- Worktrees: apply the sitecustomize redirect from CLAUDE.md first, or you will analyze the main checkout's code.
