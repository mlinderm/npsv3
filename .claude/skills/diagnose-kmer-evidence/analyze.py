#!/usr/bin/env python3
"""Explain a (high-loss) region's k-mer evidence from a `capture.py` pickle.

  python analyze.py CAP.pkl [--pred IDX] [--losses losses.pkl.gz] [--flank-del POS:REFLEN ...]

Reports, in order: the loss record (if --losses), repeat structure + reference non-unique k-mer fraction, what each
haplotype is (variants) and what label/predicted genotype indices mean, which k-mers the evidence index retains per
haplotype, whether the real reads contain them, and a read depth/MAPQ profile. --flank-del tests whether a
flank-only variant (outside the region, so reference in the simulation) changes the index.
"""
import argparse
import collections
import gzip
import itertools
import pickle
import sys

import pysam

ap = argparse.ArgumentParser()
ap.add_argument("cap")
ap.add_argument("--pred", type=int, help="predicted genotype index (from the loss table)")
ap.add_argument("--losses", help="losses.pkl.gz to look up the region's loss record")
ap.add_argument("--flank-del", action="append", default=[], help="POS:REFLEN deletion in the right flank to apply (VCF POS, REF length)")
ap.add_argument("--min-support", type=int, default=3, help="min reads for a k-mer to count as 'seen' in real data")
a = ap.parse_args()

cap = pickle.load(open(a.cap, "rb"))
K = cap["kmer_size"]
haps, alleles, lf, rf = cap["haps"], cap["alleles"], cap["left_flank"], cap["right_flank"]
contig, rng = cap["region"].split(":")
start, end = (int(x) for x in rng.split("-"))
nh = len(haps)
gts = list(itertools.combinations_with_replacement(range(nh), 2))
fa = pysam.FastaFile(cap["reference"])
cram = pysam.AlignmentFile(cap["reads"], reference_filename=cap["reference"])


def rc(s):
    return s[::-1].translate(str.maketrans("ACGT", "TGCA"))


def kcount(s, k=K):
    return collections.Counter(s[i : i + k] for i in range(len(s) - k + 1))


print(f"== {cap['region']} sample={cap['sample']} k={K} kmer_set={cap['kmer_set']} haplotypes={nh} flank={len(lf)}/{len(rf)}")

# --- Loss record
if a.losses:
    df = pickle.load(gzip.open(a.losses))
    row = df[df.region == cap["region"]]
    if len(row):
        r = row.iloc[0]
        print(f"loss={r.loss:.3f} (rank {int((df.loss > r.loss).sum()) + 1}/{len(df)}) label={r.genotype} pred={r.prediction} correct={r.correct}")
        a.pred = a.pred if a.pred is not None else int(r.prediction)

# --- Repeat structure of the reference region and reference non-unique k-mer fraction
ref_region = fa.fetch(contig, start - 1, end).upper()
best = sorted(((sum(x == y for x, y in zip(ref_region, ref_region[p:])) / (len(ref_region) - p), p) for p in range(5, min(400, len(ref_region) // 2))), reverse=True)[:3]
print("self-similarity by shift (identity, period):", [(round(i, 2), p) for i, p in best])
from npsv3._native_graph import KmerCounts  # noqa: E402

kc = KmerCounts(cap["ref_kmer_counts"])
gk = cap["graph_kmer_size"]
for label, (s0, e0) in {"region": (start - 1, end), "left 1kb": (start - 1001, start - 1), "right 1kb": (end, end + 1000)}.items():
    seq = fa.fetch(contig, s0, e0).upper()
    # The reference DB is CANONICAL (kmc "both strands: yes", ~2.5G k-mers = every hg38 31-mer, counts = occurrences on both strands), and
    # KmerCounts.count() does not canonicalize: a forward-orientation lookup misses ~50% of k-mers (returns 0). Always query min(kmer, rc).
    cs = [kc.count(min(x, rc(x))) for x in (seq[i : i + gk] for i in range(len(seq) - gk + 1)) if "N" not in x]
    n2, n5 = sum(c >= 2 for c in cs), sum(c >= 5 for c in cs)  # count 1 == occurs once in hg38 (unique); >=2 recurs in hg38
    print(f"  {label}: {n2}/{len(cs)} {gk}-mers recur in hg38 (canonical count>=2, {n2 / max(len(cs), 1):.0%}); count>=5: {n5 / max(len(cs), 1):.0%}")
cnt_region = kcount(ref_region)
print(f"  {K}-mers in the reference region that recur within it: {sum(v > 1 for v in cnt_region.values())}/{len(cnt_region)} distinct")

# --- Haplotypes and genotype indices
vinfo = {}
for vid, f in cap["variants"]:
    vinfo[vid] = f"{f[2]} (pos {f[1]}, ref {len(f[3])}bp, alt {','.join(str(len(x)) for x in f[4].split(','))}bp, GT {f[9:]})"
print("haplotypes:")
for i, (h, al) in enumerate(zip(haps, alleles)):
    desc = ", ".join(f"{vinfo[v].split(' ')[0]}:allele{n}" for v, n in sorted(al) if n)
    print(f"  hap{i}: {len(h)}bp  {desc or 'reference'}")
for v in vinfo.values():
    print("  variant", v)
true = cap["true"]
label_idx = gts.index(tuple(sorted(true))) if true and None not in true else None
if label_idx is not None:
    print(f"label genotype idx {label_idx} = haplotypes {gts[label_idx]}")
if a.pred is not None:
    print(f"predicted genotype idx {a.pred} = haplotypes {gts[a.pred]}")


# --- Is the "reference" haplotype (hap0) actually hg38? `prepare_genotyping_haplotypes` moves the first sampled haplotype whose
# >=50bp alleles are all REF to index 0, so hap0 can carry population small-variant alleles (it is hg38 only if no sampled
# haplotype qualified). Compare every haplotype with the graph's own reference path (hg38 over exactly the graph region); both
# are anchored at the region start/end, so these edits are unambiguous. (Do NOT infer coordinates by searching a haplotype in
# hg38: in tandem repeats the anchor lands in the wrong copy.)
import difflib  # noqa: E402

gref = cap.get("graph_ref_path")
if gref:
    assert gref == fa.fetch(contig, start - 1, end).upper(), "graph reference path != hg38 over the region"
    for _name, _h in [(f"hap{i}", h) for i, h in enumerate(haps)] + [(f"assembled path {i}", t) for i, t in enumerate(cap.get("true_paths", [])[:1]) if t]:
        _ops = [o for o in difflib.SequenceMatcher(None, gref, _h, autojunk=False).get_opcodes() if o[0] != "equal"]
        print(f"{_name} vs hg38 over the region ({len(_h) - len(gref):+d}bp): " + ("identical" if not _ops else f"{len(_ops)} edit blocks"))
        for tag, i1, i2, j1, j2 in _ops[:8]:
            print(f"    {tag} hg38 {start + i1}-{start + i2 - 1} {gref[i1:i2][:30]!r} -> {_h[j1:j2][:60]!r} ({j2 - j1}bp)")
    if haps[0] != gref:
        print("  NOTE: hap0 (the simulated 'reference') is not hg38 here; real reads are mapped to hg38.")

# --- How each haplotype actually differs from the reference haplotype (can include more than the VCF variant: the graph's
# "true" path may carry nearby small alleles, and an assembly's representation of a repeat expansion may include imperfect ends)
for i in range(1, nh):
    ops = [o for o in difflib.SequenceMatcher(None, haps[0], haps[i], autojunk=False).get_opcodes() if o[0] != "equal"]
    print(f"hap{i} vs hap0 ({len(haps[i]) - len(haps[0]):+d}bp net), {len(ops)} edit blocks:")
    for tag, i1, i2, j1, j2 in ops[:12]:
        print(f"    {tag} hap0[{i1}:{i2}]={haps[0][i1:i2][:40]!r} -> hap{i}[{j1}:{j2}]={haps[i][j1:j2][:60]!r}")


# --- Evidence index contents
def retained(seqs):
    cnts = [kcount(s) for s in seqs]
    ret = {}
    for k in set().union(*cnts):
        m = [k in c for c in cnts]
        # local_unique drops k-mers recurring within any haplotype; shared-by-all k-mers carry no information
        if (cap["kmer_set"] == "local_unique" and any(c.get(k, 0) > 1 for c in cnts)) or all(m):
            continue
        ret[k] = m
    return ret


seqs = [lf + h + rf for h in haps]
ret = retained(seqs)
print(f"evidence index retains {len(ret)} k-mers")
for i in range(nh):
    mine = [k for k, m in ret.items() if m[i]]
    private = [k for k in mine if sum(ret[k]) == 1]
    print(f"  hap{i}: {len(mine)} retained ({len(private)} private to this haplotype) of {len(seqs[i]) - K + 1} windows")

# --- Is the haplotype used as "true" really the sample's assembled path? (matching is by >=50bp alleles only)
tps = [t for t in cap.get("true_paths", []) if t]
if tps:
    for i, t in enumerate(tps[:1] if len(set(tps)) == 1 else tps):
        same = [j for j, h in enumerate(haps) if h == t]
        print(f"sample assembled path {i} ({len(t)}bp): identical to haplotype {same}" if same else f"sample assembled path {i} ({len(t)}bp): NOT identical to any sampled haplotype")
        if not same:
            best = min(range(nh), key=lambda j: sum(max(o[2] - o[1], o[4] - o[3]) for o in difflib.SequenceMatcher(None, haps[j], t, autojunk=False).get_opcodes() if o[0] != "equal"))  # fewest changed bases
            print(f"  closest is hap{best}; assembled path vs hap{best}:")
            for tag, i1, i2, j1, j2 in [o for o in difflib.SequenceMatcher(None, haps[best], t, autojunk=False).get_opcodes() if o[0] != "equal"][:8]:
                print(f"    {tag} hap{best}[{i1}:{i2}]={haps[best][i1:i2][:40]!r} -> path[{j1}:{j2}]={t[j1:j2][:60]!r}")
            print("  (the sampled haplotype was matched to the sample's genotype by >=50bp variants only, so it may carry small alleles the sample lacks)")
    TRUE_PATH = tps[0]
else:
    TRUE_PATH = None

# --- Real reads
reg = (contig, start - 1 - 500, end + 500)
reads = [r for r in cram.fetch(*reg) if r.query_sequence and not r.is_secondary and not r.is_supplementary]
support = collections.Counter()
for r in reads:
    s = r.query_sequence
    seen = {s[i : i + K] for i in range(len(s) - K + 1)}
    for k in seen & ret.keys():
        support[k] += 1
print(f"real reads (±500bp): {len(reads)}; retained k-mers seen in >=1 read: {len(support)}/{len(ret)}; in >={a.min_support}: {sum(v >= a.min_support for v in support.values())}")
for i in range(nh):
    mine = [k for k, m in ret.items() if m[i] and sum(m) == 1]
    if mine:
        print(f"  hap{i}-private k-mers seen in real reads: {sum(k in support for k in mine)}/{len(mine)}")


def hap_set(g):
    return set(g)


def discriminating(gt_a, gt_b):
    """Retained k-mers carried by genotype A's haplotypes but not genotype B's (the evidence favoring A over B)."""
    return [k for k, m in ret.items() if any(m[h] for h in gt_a) and not any(m[h] for h in gt_b)]


print("k-mers each haplotype adds relative to the reference haplotype (retained, absent from hap0), seen in real reads:")
for i in range(1, nh):
    add = [k for k, m in ret.items() if m[i] and not m[0]]
    print(f"  hap{i}: {len(add)} k-mers, {sum(k in support for k in add)} seen in real reads")


def clusters(offsets, gap=3):
    out = []
    for o in sorted(offsets):
        if out and o - out[-1][-1] <= gap:
            out[-1].append(o)
        else:
            out.append([o])
    return out


# Where (in region coordinates) do the k-mers that exist only in one haplotype sit, and do reads contain them? Reads agreeing with
# one boundary of an allele but not the rest show the real allele differs from the assembled haplotype at that location.
print("clusters of haplotype-specific k-mers (start offset from region start; seen = distinct k-mers / total read hits):")
for i in range(nh):
    # hap0: k-mers on no other haplotype; hap i>0: k-mers absent from the reference haplotype (may be shared with other alts)
    mine = [k for k, m in ret.items() if (m[0] and sum(m) == 1)] if i == 0 else [k for k, m in ret.items() if m[i] and not m[0]]
    base = len(lf)
    pos = {k: seqs[i].find(k) - base for k in mine}
    for g in clusters(pos.values()):
        ks = [k for k in mine if g[0] <= pos[k] <= g[-1]]
        print(f"  hap{i} offsets {g[0]}..{g[-1]}: {len(ks)} k-mers, seen {sum(k in support for k in ks)}/{len(ks)}, read hits {sum(support[k] for k in ks)}")

if label_idx is not None and a.pred is not None:
    for name, x, y in (("label over prediction", gts[label_idx], gts[a.pred]), ("prediction over label", gts[a.pred], gts[label_idx])):
        d = discriminating(x, y)
        print(f"  evidence for {name} {x} vs {y}: {len(d)} k-mers, {sum(k in support for k in d)} seen in real reads")
if label_idx is not None:
    print("  (label haplotype private k-mers are the ones the simulated true-genotype image should show; zero here means real reads lack the label allele's sequence)")

# --- Does the sample's own call set look real? Density of its alt-allele variants and their support in the reads.
# (Sampled haplotypes carry these small variants too, so the simulated alt reads contain them; a dense run of calls whose alt
# alleles never appear in the real reads means the assembly haplotype is divergent from, or mis-aligned to, the reference here.)
vcf = pysam.VariantFile(cap["input"])
sidx = cap["sample"]
win = (start - 1 - 1000, end + 1000)
calls = [r for r in vcf.fetch(contig, *win) if any(g not in (0, None) for g in r.samples[sidx]["GT"])]
snvs = [r for r in calls if len(r.ref) == 1 and len(r.alts[0]) == 1]
print(f"sample variants (non-ref GT) within +-1kb: {len(calls)} ({len(snvs)} SNVs, {len(calls) - len(snvs)} indel/SV) = 1 per {(win[1] - win[0]) / max(len(calls), 1):.0f}bp")
alt_n = tot_n = 0
for r in snvs:
    cnt = collections.Counter()
    for col in cram.pileup(contig, r.start, r.start + 1, truncate=True, stepper="all", min_base_quality=20):
        cnt.update(x.alignment.query_sequence[x.query_position] for x in col.pileups if x.query_position is not None)
    alt_n += cnt.get(r.alts[0], 0)
    tot_n += sum(cnt.values())
if snvs:
    gts = collections.Counter(tuple(r.samples[sidx]["GT"]) for r in snvs)
    print(f"SNV alt-allele reads: {alt_n}/{tot_n} = {alt_n / max(tot_n, 1):.3f} (VCF genotypes {dict(gts)}; ~0.5 expected for het, ~1 for hom-alt)")

if TRUE_PATH and TRUE_PATH not in haps:
    tk = kcount(lf + TRUE_PATH + rf)
    base = kcount(seqs[0])
    mine = [k for k, v in tk.items() if v == 1 and k not in base]
    seen = collections.Counter()
    for r in reads:
        q = r.query_sequence
        for k in {q[i : i + K] for i in range(len(q) - K + 1)} & set(mine):
            seen[k] += 1
    print(f"sample assembled path k-mers absent from reference haplotype: {len(mine)}, seen in real reads {len(seen)} (hits {sum(seen.values())})")

# --- Depth / MAPQ profile (non-dup primary reads)
lo, hi = start - 1 - 1000, end + 1500
bins = (hi - lo) // 200
cover = [[0, 0, 0] for _ in range(bins)]
for r in cram.fetch(contig, lo, hi):
    if r.is_secondary or r.is_supplementary or r.is_duplicate:
        continue
    s, e = max(r.reference_start, lo), min(r.reference_end, hi)
    for b in range((s - lo) // 200, min(bins - 1, (e - 1 - lo) // 200) + 1):
        ov = min(e, lo + (b + 1) * 200) - max(s, lo + b * 200)
        if ov > 0:
            cover[b][0] += ov
            if r.mapping_quality >= 20:
                cover[b][1] += ov
            if r.mapping_quality == 0:
                cover[b][2] += ov
print("depth profile (200bp bins): start  depth  depth(MAPQ>=20)  depth(MAPQ0)")
for b, (t, q, z) in enumerate(cover):
    flag = " <- region" if start - 1 <= lo + b * 200 < end else ""
    print(f"  {lo + b * 200}  {t / 200:5.1f}  {q / 200:5.1f}  {z / 200:5.1f}{flag}")

# --- Optional: does a flank-only deletion change the index?
for spec in a.flank_del:
    pos, reflen = (int(x) for x in spec.split(":"))
    off = pos - (end + 1) + 1
    if not 0 <= off < len(rf):
        sys.exit(f"--flank-del {spec} is outside the right flank")
    rf2 = rf[:off] + rf[off + reflen - 1 :]
    ret2 = retained([lf + h + rf2 for h in haps])
    print(f"flank-del {spec}: retained {len(ret)} -> {len(ret2)} (lost {len(set(ret) - set(ret2))}, gained {len(set(ret2) - set(ret))})")
