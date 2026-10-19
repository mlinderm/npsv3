"""Lightweight, dependency-free constants shared across packages (no torch, native extension, etc. imports)."""

REAL_SUFFIX = "real.image.bin.gz"
SIMS_SUFFIX = "sims.image.bin.gz"
"""Pixel-shard tar member suffixes for the polylithic energon layout: `{key}.{REAL_SUFFIX}` /
`{key}.{SIMS_SUFFIX}`. Shared between `images.example.EnergonExampleWriter` (writer) and
`models.energon` (reader) so the two cannot drift apart."""

HAPLOTYPES_SUFFIX = "haplotypes.json.gz"
"""Pixel-shard tar member suffix, `{key}.{HAPLOTYPES_SUFFIX}`, for an example's candidate haplotypes (see
`images.haplotypes.haplotype_record`). Optional: absent from datasets generated before it was introduced."""

COMPARISON_SUFFIX = "comparison.json.gz"
COMPARISONS_AUX = "comparisons"
COMPARISONS_DIR = "compare_ds"
"""`{key}.{COMPARISON_SUFFIX}` members of the optional `comparisons` aux tree, a dataset's `COMPARISONS_DIR`, comparing an
example's candidate genotypes with its label (see `images.sequence_compare`)."""
