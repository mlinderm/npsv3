"""Lightweight, dependency-free constants shared across packages (no torch, native extension, etc. imports)."""

REAL_SUFFIX = "real.image.chw.zst"
SIMS_SUFFIX = "sims.image.chw.zst"
"""Pixel-shard tar member suffixes for the polylithic energon layout: `{key}.{REAL_SUFFIX}` /
`{key}.{SIMS_SUFFIX}`. Shared between `images.example.EnergonExampleWriter` (writer) and
`models.energon` (reader) so the two cannot drift apart. Pixels are channel-first, i.e., `(..., C, H, W)`, and
zstd-compressed, as recorded by the metadata's `layout` (`CHW_LAYOUT`)."""

LEGACY_REAL_SUFFIX = "real.image.bin.gz"
LEGACY_SIMS_SUFFIX = "sims.image.bin.gz"
"""Pixel-shard tar member suffixes for channel-last, i.e., `(..., H, W, C)`, gzip-compressed pixels, written by default
before the metadata included `layout` (and still readable)."""

CHW_LAYOUT = "chw"
HWC_LAYOUT = "hwc"
"""Values of a metadata row's `layout`, the pixel members' layout (and suffixes). Rows without `layout` are `HWC_LAYOUT`."""

HAPLOTYPES_SUFFIX = "haplotypes.json.gz"
"""Pixel-shard tar member suffix, `{key}.{HAPLOTYPES_SUFFIX}`, for an example's candidate haplotypes (see
`images.haplotypes.haplotype_record`). Optional: absent from datasets generated before it was introduced."""

HAPLOTYPES_DIR = "haplotypes_ds"
"""Precomputed `{key}.{HAPLOTYPES_SUFFIX}` members in a `command=haplotypes` output directory (see `images.haplotype_tree`)"""

COMPARISON_SUFFIX = "comparison.json.gz"
COMPARISONS_AUX = "comparisons"
COMPARISONS_DIR = "compare_ds"
"""`{key}.{COMPARISON_SUFFIX}` members of the optional `comparisons` aux tree, a dataset's `COMPARISONS_DIR`, comparing an
example's candidate genotypes with its label (see `images.sequence_compare`)."""
