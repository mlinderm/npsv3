"""Lightweight, dependency-free constants shared across packages (no torch, native extension, etc. imports)."""

REAL_SUFFIX = "real.image.bin.gz"
SIMS_SUFFIX = "sims.image.bin.gz"
"""Pixel-shard tar member suffixes for the polylithic energon layout: `{key}.{REAL_SUFFIX}` /
`{key}.{SIMS_SUFFIX}`. Shared between `images.example.EnergonExampleWriter` (writer) and
`models.energon` (reader) so the two cannot drift apart."""
