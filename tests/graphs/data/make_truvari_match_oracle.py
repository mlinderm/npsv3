# ruff: noqa: INP001  (standalone script run under Truvari's Python 3.9, which has no `zip(strict=)`)
"""Generate `truvari_match_oracle.json`: the matches Truvari `bench` itself finds between harmonized haplotype variants.

Run with Truvari's own Python (not a dependency of npsv3), from the repo root, after `make_truvari_harmonize_oracle.py`:

    /home/mlinderman/modules/truvari/5.3.0/bin/python3 tests/graphs/data/make_truvari_match_oracle.py

For every harmonization case in `truvari_harmonize_oracle.json` (poa variants), every ordered pair of haplotypes and several
parameter sets, this writes a two-sample VCF (base sample, comparison sample), runs `truvari.Bench` the way `truvari refine`
does (`no_ref='a'`, `pick='single'`), and records the TP/FN/FP variants and the similarity values Truvari annotates, so
`tests/graphs/test_sequence_compare.py` can check our port against Truvari without needing it.

`no_roll=True` below is Truvari's CLI default (`--no-roll` is a `store_false` option), which *enables* rolling/unrolling.
"""

import itertools
import json
import os
import sys
import tempfile

import truvari

HERE = os.path.dirname(os.path.abspath(__file__))

# `bench` parameter sets to record: the criteria used in this project, Truvari's defaults, stricter and looser variants
# of these, and one with rolling/unrolling of repeats disabled
PARAM_SETS = {
    "project": {"refdist": 20, "pctseq": 0.7, "pctsize": 0.7, "pctovl": 0.6, "sizemin": 30, "sizefilt": 30},
    "truvari_defaults": {"refdist": 500, "pctseq": 0.7, "pctsize": 0.7, "pctovl": 0.0, "sizemin": 50, "sizefilt": 30},
    "strict": {"refdist": 5, "pctseq": 0.9, "pctsize": 0.9, "pctovl": 0.8, "sizemin": 20, "sizefilt": 10},
    "loose": {"refdist": 100, "pctseq": 0.5, "pctsize": 0.5, "pctovl": 0.0, "sizemin": 10, "sizefilt": 5},
    "no_roll": {
        "refdist": 100,
        "pctseq": 0.7,
        "pctsize": 0.5,
        "pctovl": 0.0,
        "sizemin": 20,
        "sizefilt": 10,
        "roll": False,
    },
}


def write_vcf(path, contig_len, variants, base, comp):
    """Write a bgzipped, indexed two-sample VCF of the variants carried by `base` and/or `comp`.

    Each haplotype becomes a sample with genotype `1|0` where it carries the variant and `0|0` otherwise. Variants
    carried by neither haplotype are omitted.

    Args:
        path: Output path, ending in `.gz`. The uncompressed VCF is written alongside it (without `.gz`).
        contig_len: Length of the single `region` contig (the reference length).
        variants: Variants as dicts with `pos`, `ref`, `alt` and `carriers` (haplotype names), as in the harmonize
            oracle.
        base: Haplotype name to use as the base sample.
        comp: Haplotype name to use as the comparison sample.
    """
    lines = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID=region,length={contig_len}>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{base}\t{comp}",
    ]
    for v in sorted(variants, key=lambda v: (v["pos"], v["ref"], v["alt"])):
        if base not in v["carriers"] and comp not in v["carriers"]:
            continue
        gts = ["1|0" if name in v["carriers"] else "0|0" for name in (base, comp)]
        lines.append("\t".join(["region", str(v["pos"]), ".", v["ref"], v["alt"], ".", ".", ".", "GT", *gts]))
    with open(path[: -len(".gz")], "w") as f:
        f.write("\n".join(lines) + "\n")
    truvari.compress_index_vcf(path[: -len(".gz")], path)


def read_vcf(path, *, with_info=False):
    """Read the variants in one of `truvari bench`'s output VCFs.

    Args:
        path: VCF path. A missing file is treated as having no variants.
        with_info: Also record the similarity annotations Truvari adds to matched calls (`PctSeqSimilarity`, `TruScore`,
            etc.) under `"info"`.

    Returns:
        Variants as dicts with `pos`, `ref`, `alt` and, with `with_info`, `info`.
    """
    out = []
    if not os.path.exists(path):
        return out
    for entry in truvari.VariantFile(path):
        item = {"pos": entry.pos, "ref": entry.ref, "alt": entry.alts[0]}
        if with_info:
            info = entry.info
            item["info"] = {
                k: info[k]
                for k in info
                if k
                in (
                    "PctSeqSimilarity",
                    "PctSizeSimilarity",
                    "PctRecOverlap",
                    "SizeDiff",
                    "StartDistance",
                    "EndDistance",
                    "TruScore",
                )
            }
        out.append(item)
    return out


def run_bench(workdir, contig_len, variants, base, comp, params):
    """Benchmark `comp`'s variants against `base`'s with `truvari.Bench`, configured as `truvari refine` does.

    Args:
        workdir: Empty scratch directory for the input VCF and Truvari's output.
        contig_len: Length of the region (the reference length).
        variants: Harmonized variants for all haplotypes, as dicts with `pos`, `ref`, `alt` and `carriers`.
        base: Haplotype name to benchmark against.
        comp: Haplotype name to benchmark.
        params: `bench` parameters, one of `PARAM_SETS`. `roll` is translated to Truvari's `no_roll`.

    Returns:
        Dict with the variants of each outcome: `"tp_base"` and `"tp_comp"` (with Truvari's annotations under `info`),
        `"fn"` and `"fp"`.
    """
    vcf = os.path.join(workdir, "in.vcf.gz")
    write_vcf(vcf, contig_len, variants, base, comp)
    kwargs = dict(params)
    kwargs["no_roll"] = kwargs.pop("roll", True)  # see module docstring
    var_params = truvari.VariantParams(bSample=base, cSample=comp, no_ref="a", **kwargs)
    outdir = os.path.join(workdir, "out")
    truvari.Bench(params=var_params, base_vcf=vcf, comp_vcf=vcf, outdir=outdir).run()
    return {
        "tp_base": read_vcf(os.path.join(outdir, "tp-base.vcf.gz"), with_info=True),
        "tp_comp": read_vcf(os.path.join(outdir, "tp-comp.vcf.gz"), with_info=True),
        "fn": read_vcf(os.path.join(outdir, "fn.vcf.gz")),
        "fp": read_vcf(os.path.join(outdir, "fp.vcf.gz")),
    }


def main():
    """Benchmark each ordered pair of haplotypes in every harmonize-oracle case with each `PARAM_SETS` entry."""
    with open(os.path.join(HERE, "truvari_harmonize_oracle.json")) as f:
        harmonize_oracle = json.load(f)

    oracle = {"truvari_version": truvari.__version__, "param_sets": PARAM_SETS, "cases": {}}
    for case_name, case in harmonize_oracle["cases"].items():
        names = sorted(case["haplotypes"])
        poa = case["poa"]
        results = {}
        for base, comp in itertools.permutations(names, 2):
            for set_name, params in PARAM_SETS.items():
                with tempfile.TemporaryDirectory() as workdir:
                    results[f"{base}>{comp}|{set_name}"] = run_bench(
                        workdir, len(case["reference"]), poa["variants"], base, comp, params
                    )
        oracle["cases"][case_name] = {"variants": poa["variants"], "results": results}
    with open(os.path.join(HERE, "truvari_match_oracle.json"), "w") as f:
        json.dump(oracle, f, indent=1)
    sys.stdout.write(
        f"Wrote {sum(len(c['results']) for c in oracle['cases'].values())} benchmark runs from Truvari {truvari.__version__}\n"
    )


if __name__ == "__main__":
    main()
