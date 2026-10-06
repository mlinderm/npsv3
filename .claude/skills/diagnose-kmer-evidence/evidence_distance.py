#!/usr/bin/env python3
"""A1/A2 haplotype-distinguishability diagnostic from a `capture.py` pickle.

  python evidence_distance.py CAP.pkl [--replicates 4] [--out RESULT.pkl] [--coverage 32.29 ...]

Question answered: are the candidate haplotypes (hg38, the sampled haplotypes, the sample's assembled path, and a
counterfactual "assembled path without its largest insertion/SV" twin) statistically distinguishable from k-mer evidence, and
which genotype do the REAL reads support?

  A1  model distance between genotypes: Poisson deviance of the expected k-mer counts of genotype g scored against genotype h,
      over the "informative" k-mers (multiplicity differs between at least two candidates).
  A2  noise floor: replicate read simulations (synthBAM, so needs art_illumina/bwa/samtools on PATH, i.e. `module load
      npsv3/main`) of each homozygous candidate, scored against every genotype. A pair whose A1 distance is near the floor
      is statistically indistinguishable.
  Real reads (CRAM, region +-500bp, soft-clipped bases and all primary reads included) are scored against every genotype.
      A best real fit far above the floor means the data fit no candidate (unsupported/misspecified label).

Per-copy depth is calibrated separately for simulated and real data from unique flank k-mers. The model is Poisson with no
overdispersion/stutter/GC bias and the deviance is dominated by high-multiplicity repeat k-mers, so compare real vs floor
qualitatively, not against an absolute threshold. `--sample-*` defaults describe HG00096; override for other samples.
"""
import argparse
import collections
import difflib
import itertools
import os
import pickle
import tempfile

import numpy as np
import pysam

from npsv3.simulation import simulate_variant_sequencing
from npsv3.util.sample import Sample

ap = argparse.ArgumentParser()
ap.add_argument("cap")
ap.add_argument("--replicates", type=int, default=4, help="simulation replicates per candidate (>=2)")
ap.add_argument("--out", help="optionally pickle the distance/deviance tables here")
ap.add_argument("--kmer-size", type=int, help="default: the capture's evidence k-mer size")
ap.add_argument("--eps", type=float, default=0.3, help="additive error floor on expected counts")
ap.add_argument("--coverage", type=float, default=32.29, help="sample mean coverage (both haplotypes)")
ap.add_argument("--insert-mean", type=float, default=437.41)
ap.add_argument("--insert-std", type=float, default=102.4)
ap.add_argument("--sequencer", default="HSXn")
ap.add_argument("--read-length", type=int, default=150)
a = ap.parse_args()
assert a.replicates >= 2, "need at least 2 replicates to form replicate pairs"

cap = pickle.load(open(a.cap, "rb"))
K = a.kmer_size or cap["kmer_size"]
R = a.replicates
lf, rf = cap["left_flank"], cap["right_flank"]

# --- Candidates: hg38, every sampled haplotype, the sample's assembled path and its SV-removed twin
cands = {"hg38": cap["graph_ref_path"]}
for i, h in enumerate(cap["haps"]):
    cands[f"hap{i}"] = h
asm = next((t for t in cap.get("true_paths", []) if t), None)
if asm is None:
    print("no assembled path in the capture; skipping asm/asmNoSV candidates")
else:
    cands["asm"] = asm
    ops = [o for o in difflib.SequenceMatcher(None, cands["hg38"], asm, autojunk=False).get_opcodes() if o[0] != "equal"]
    ins = [o for o in ops if o[0] == "insert" or o[4] - o[3] > o[2] - o[1]]
    if ins:
        big = max(ins, key=lambda o: (o[4] - o[3]) - (o[2] - o[1]))
        print(f"asm largest insertion-like block vs hg38: {big[0]} hg38[{big[1]}:{big[2]}] -> asm[{big[3]}:{big[4]}] ({big[4] - big[3]}bp)")
        # Counterfactual twin: assembled background, SV removed (replaces the block with the hg38 sequence it displaced)
        cands["asmNoSV"] = asm[: big[3]] + cands["hg38"][big[1] : big[2]] + asm[big[4] :]
        print(f"asmNoSV vs hg38: {len(cands['asmNoSV']) - len(cands['hg38']):+d}bp")
names = list(cands)
seqs = {n: lf + s + rf for n, s in cands.items()}

comp = str.maketrans("ACGT", "TGCA")


def canon(s):
    r = s[::-1].translate(comp)
    return s if s <= r else r


def kcounts(s):
    return collections.Counter(canon(s[i : i + K]) for i in range(len(s) - K + 1) if "N" not in s[i : i + K])


mult = {n: kcounts(s) for n, s in seqs.items()}
info = sorted(k for k in set().union(*mult.values()) if len({mult[n].get(k, 0) for n in names}) > 1)
idx = {k: i for i, k in enumerate(info)}
M = np.array([[mult[n].get(k, 0) for k in info] for n in names], float)  # candidate x informative k-mer
print(f"candidates: {names}; informative {K}-mers: {len(info)}")

# Calibration k-mers: occur once in every candidate (unique), 100-400bp either side of the region
region_len = len(cands["hg38"])
offsets = list(range(len(lf) - 400, len(lf) - 100)) + list(range(len(lf) + region_len + 100, len(lf) + region_len + 400))
calib = {canon(seqs["hg38"][o : o + K]) for o in offsets}
calib = {k for k in calib if all(mult[n].get(k, 0) == 1 for n in names)}
cidx = {k: i for i, k in enumerate(sorted(calib))}
print("calibration k-mers:", len(cidx))


def vec(read_seqs, index):
    v = np.zeros(len(index))
    for s in read_seqs:
        for i in range(len(s) - K + 1):
            j = index.get(canon(s[i : i + K]))
            if j is not None:
                v[j] += 1
    return v


sample = Sample("sim", mean_coverage=a.coverage, mean_insert_size=a.insert_mean, std_insert_size=a.insert_std,
                sequencer=a.sequencer, read_length=a.read_length, kmer_coverage=int(a.coverage))
HAPCOV = sample.mean_coverage / 2

# --- Simulate replicates per candidate (one haplotype copy at HAPCOV)
sim = {n: [] for n in names}
simcal = []
with tempfile.TemporaryDirectory() as d:
    for n in names:
        fa = os.path.join(d, f"{n}.fa")
        with open(fa, "w") as f:
            f.write(f">seq-0-0\n{seqs[n]}\n")
        for _ in range(R):
            bam = simulate_variant_sequencing(fa, HAPCOV, sample, reference=cap["reference"], dir=d, aligner="bwa")
            with pysam.AlignmentFile(bam) as b:
                rs = [r.query_sequence for r in b.fetch(until_eof=True) if r.query_sequence]
            sim[n].append(vec(rs, idx))
            simcal.append(vec(rs, cidx).mean())
LAM_SIM = float(np.mean(simcal))

# --- Real reads over the region +-500bp (all primary reads, soft-clipped bases included)
contig, rng = cap["region"].split(":")
s0, e0 = (int(x) for x in rng.split("-"))
with pysam.AlignmentFile(cap["reads"], reference_filename=cap["reference"]) as cram:
    real_reads = [r.query_sequence for r in cram.fetch(contig, s0 - 500, e0 + 500) if r.query_sequence and not r.is_secondary]
real = vec(real_reads, idx)
LAM_REAL = vec(real_reads, cidx).mean() / 2
print(f"per-haplotype-copy k-mer depth: sim {LAM_SIM:.1f}, real {LAM_REAL:.1f}; real informative k-mer hits {int(real.sum())}")

# --- Genotypes = unordered pairs of candidates
gts = list(itertools.combinations_with_replacement(range(len(names)), 2))
gname = [f"{names[x]}/{names[y]}" for x, y in gts]
hom = [g for g, (x, y) in enumerate(gts) if x == y]


def expected(g, lam):
    x, y = gts[g]
    return lam * (M[x] + M[y])


def deviance(obs, mu):
    mu = mu + a.eps
    return 2 * np.sum(np.where(obs > 0, obs * np.log(np.maximum(obs, 1e-9) / mu), 0) - (obs - mu))


def dev_vs_all(obs, lam):
    return np.array([deviance(obs, expected(g, lam)) for g in range(len(gts))])


print("\n== A1: expected deviance, hom genotype (row, truth) scored against hom genotype (column)")
A1 = np.array([[deviance(expected(g, LAM_SIM), expected(h, LAM_SIM)) for h in range(len(gts))] for g in range(len(gts))])
print(f"{'':22s}" + "".join(f"{names[gts[h][0]]:>9s}" for h in hom))
for g in hom:
    print(f"{g:2d} {gname[g]:18s}" + "".join(f"{A1[g, h]:9.0f}" for h in hom))

print("\n== A2: simulated hom genotype (row, mean over replicate pairs) scored against hom genotypes (columns)")
simdev = {}
for x, n in enumerate(names):
    g = gts.index((x, x))
    simdev[n] = np.array([dev_vs_all(sim[n][r] + sim[n][(r + 1) % R], LAM_SIM) for r in range(R)])
    print(f"{g:2d} {gname[g]:18s}" + "".join(f"{simdev[n].mean(0)[h]:9.0f}" for h in hom) + f"   self sd={simdev[n][:, g].std():.0f}")

print("\n== Noise floor: self deviance (mean+-sd) and nearest wrong genotype")
for x, n in enumerate(names):
    g = gts.index((x, x))
    others = [h for h in range(len(gts)) if h != g]
    m = simdev[n].mean(0)[others]
    print(f"  {n}/{n}: self {simdev[n][:, g].mean():.0f}+-{simdev[n][:, g].std():.0f}; nearest wrong {gname[others[int(m.argmin())]]} at {m.min():.0f}")

print("\n== Real reads scored against each genotype (deviance, lower is better)")
rd = dev_vs_all(real, LAM_REAL)
floor = np.mean([simdev[n][:, gts.index((x, x))].mean() for x, n in enumerate(names)])
for g in np.argsort(rd):
    print(f"{g:2d} {gname[g]:22s} {rd[g]:8.0f}")
print(f"best real fit {rd.min():.0f} vs mean self-deviance floor {floor:.0f} ({rd.min() / max(floor, 1):.1f}x)")

if a.out:
    with open(a.out, "wb") as f:
        pickle.dump(dict(names=names, gname=gname, A1=A1, simdev=simdev, real=rd, M=M, info=info), f)
