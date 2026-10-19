# ruff: noqa: INP001, B905  (standalone script run under Truvari's Python 3.9, which has no `zip(strict=)`)
"""Generate `truvari_harmonize_oracle.json`: what Truvari itself produces for a set of haplotype-harmonization cases.

Run with Truvari's own Python (it is not a dependency of npsv3), from the repo root:

    /home/mlinderman/modules/truvari/5.3.0/bin/python3 tests/graphs/data/make_truvari_harmonize_oracle.py

For every case the oracle records the MSA that `truvari.phab.run_poa` / `run_wfa` produce and the variants
`truvari.msa2vcf` calls from it, so `tests/graphs/test_sequence_harmonize.py` can check our port against Truvari without
needing it. Truvari's haplotype keys are `<sample>_<1|2>_<region>`; each haplotype here is a "sample" with one (first)
haplotype.
"""

import json
import os
import random
import sys

import truvari
from truvari import phab

HERE = os.path.dirname(os.path.abspath(__file__))


def random_sequence(length, seed):
    """Reproducible random ACGT sequence of `length` bases."""
    return "".join(random.Random(seed).choices("ACGT", k=length))


def synthetic_cases():
    """Small hand-built harmonization cases covering the variant shapes and alignment edge cases of interest.

    Covers a SNV, an insertion, a deletion, an in-place MNV, indels at the start of the region, expansions and shifted
    insertions in a tandem repeat (where placement is ambiguous), duplicate and reference-identical haplotypes, and
    multiple variants per haplotype.

    Returns:
        Case name -> (reference sequence, haplotype name -> haplotype sequence).
    """
    ref = random_sequence(240, seed=1)
    tract = "CTTT" * 10
    repeat_ref = random_sequence(80, seed=2) + tract + random_sequence(80, seed=3)
    repeat_at = 80 + len(tract)
    return {
        "snv": (ref, {"a": ref[:100] + ("A" if ref[100] != "A" else "C") + ref[101:]}),
        "insertion": (ref, {"a": ref[:120] + random_sequence(60, seed=4) + ref[120:]}),
        "deletion": (ref, {"a": ref[:90] + ref[150:]}),
        "mnv_in_place": (ref, {"a": ref[:100] + ("GGG" if ref[100:103] != "GGG" else "CCC") + ref[103:]}),
        "delete_at_start": (ref, {"a": ref[5:]}),
        "insert_at_start": (ref, {"a": "ACGTACGTAC" + ref}),
        "repeat_expansions": (
            repeat_ref,
            {
                "a": repeat_ref[:repeat_at] + "CTTT" * 3 + repeat_ref[repeat_at:],
                "b": repeat_ref[:repeat_at] + "CTTT" * 5 + repeat_ref[repeat_at:],
                "c": repeat_ref[:80] + "CTTT" * 3 + repeat_ref[80:],
            },
        ),
        "shifted_repeat_insertions": (
            repeat_ref,
            {
                "left": repeat_ref[:80] + "CTTT" * 4 + repeat_ref[80:],
                "right": repeat_ref[:repeat_at] + "CTTT" * 4 + repeat_ref[repeat_at:],
            },
        ),
        "identical_haplotypes": (ref, {"a": ref[:50] + "AAAA" + ref[50:], "b": ref[:50] + "AAAA" + ref[50:], "c": ref}),
        "multiple_variants": (
            ref,
            {
                "a": ref[:30] + random_sequence(25, seed=5) + ref[30:200] + ref[210:],
                "b": ref[:60] + ref[70:],
            },
        ),
    }


def run_case(reference, haplotypes, aligner_name):
    """Harmonize one case with Truvari and record its alignment and called variants.

    Haplotypes are passed to Truvari as the first haplotype of a sample of the same name, and keyed by that name in the
    output.

    Args:
        reference: Reference sequence of the region.
        haplotypes: Haplotype name -> sequence.
        aligner_name: `"poa"` (`phab.run_poa`) or `"wfa"` (`phab.run_wfa`).

    Returns:
        Dict with `"msa"`, the aligned sequences keyed by haplotype name (and the `ref_` reference key), with pairwise
        alignments as [gapped reference, gapped haplotype] lists, and `"variants"`, the distinct variants as dicts with
        `pos`, `ref`, `alt` and the sorted names of the haplotypes carrying them (`carriers`).
    """
    region = f"ref_region:1-{len(reference)}"
    inputs = {region: reference} | {f"{name}_1_r": seq for name, seq in haplotypes.items()}
    msa = {"poa": phab.run_poa, "wfa": phab.run_wfa}[aligner_name](dict(inputs))
    vcf_text = truvari.msa2vcf(msa)
    variants = {}
    for line in vcf_text.strip().split("\n"):
        if not line:
            continue
        fields = line.split("\t")
        pos, ref, alt = int(fields[1]), fields[3], fields[4]
        carriers = [name for name, gt in zip(sorted(haplotypes), fields[9:]) if gt.split("|")[0] == "1"]
        variants.setdefault(name_key(pos, ref, alt), {"pos": pos, "ref": ref, "alt": alt, "carriers": carriers})
    return {
        "msa": {
            (k.removesuffix("_1_r") if not k.startswith("ref_") else k): (list(v) if isinstance(v, tuple) else v)
            for k, v in msa.items()
        },
        "variants": list(variants.values()),
    }


def name_key(pos, ref, alt):
    """Key identifying a variant by position and alleles, used to de-duplicate Truvari's VCF records."""
    return f"{pos}:{ref}:{alt}"


def main():
    """Run every synthetic and real case (`harmonize_real_cases.json`) with both aligners and write the oracle."""
    cases = {name: {"reference": ref, "haplotypes": haps} for name, (ref, haps) in synthetic_cases().items()}
    with open(os.path.join(HERE, "harmonize_real_cases.json")) as f:
        cases.update(json.load(f))

    oracle = {"truvari_version": truvari.__version__, "cases": {}}
    for name, case in cases.items():
        oracle["cases"][name] = {
            "reference": case["reference"],
            "haplotypes": case["haplotypes"],
            "poa": run_case(case["reference"], case["haplotypes"], "poa"),
            "wfa": run_case(case["reference"], case["haplotypes"], "wfa"),
        }
    with open(os.path.join(HERE, "truvari_harmonize_oracle.json"), "w") as f:
        json.dump(oracle, f, indent=1)
    sys.stdout.write(f"Wrote {len(oracle['cases'])} cases from Truvari {truvari.__version__}\n")


if __name__ == "__main__":
    main()
