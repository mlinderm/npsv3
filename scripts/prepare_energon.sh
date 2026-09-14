#!/bin/bash
# Prepare the polylithic energon trees written by `npsv3 command=images pileup.example_format=energon`.
#
# Runs `energon prepare` on both trees and writes the metadataset that binds them:
#
#   <dir>/meta_ds/     the primary dataset, prepared as CRUDE so conversion happens in our cooker
#                      (npsv3.models.energon.cook_group) rather than in a generated sample loader
#   <dir>/pixel_ds/    prepared too, because an `aux` reference must be a prepared energon dataset --
#                      that is what gives key-addressed random access within a shard
#   <dir>/metadataset.yaml
#
# Usage: prepare_energon.sh <images_dir> [split_ratio] [num_workers]
#   split_ratio  train,val,test ratio passed to `energon prepare`. Default "0.9,0.1,0.0".

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <images_dir> [split_ratio] [num_workers]" >&2
    exit 2
fi

IMAGES_DIR=$(cd "$1" && pwd)
SPLIT_RATIO=${2:-0.9,0.1,0.0}
NUM_WORKERS=${3:-16}

META_DS="$IMAGES_DIR/meta_ds"
PIXEL_DS="$IMAGES_DIR/pixel_ds"

for tree in "$META_DS" "$PIXEL_DS"; do
    if [ ! -d "$tree" ]; then
        echo "Missing $tree -- was this generated with pileup.example_format=energon?" >&2
        exit 1
    fi
done

# Both trees are prepared crude: the primary because our cooker does the conversion, the aux because
# it is only ever addressed by key (a sample loader would never run on it).
for tree in "$META_DS" "$PIXEL_DS"; do
    echo "==> energon prepare $tree"
    energon prepare "$tree" \
        --non-interactive \
        --split-ratio "$SPLIT_RATIO" \
        --sample-type CrudeWebdataset \
        --num-workers "$NUM_WORKERS" \
        --force-overwrite
done

METADATASET="$IMAGES_DIR/metadataset.yaml"
echo "==> writing $METADATASET"
cat > "$METADATASET" <<'YAML'
__module__: megatron.energon
__class__: MetadatasetV2
splits:
  train:
    path: ./meta_ds
    split_part: train
    aux:
      # Pixels live beside the metadata so the packing buffer can decide a pack from `g`, `r` and
      # the geometry alone, and only then fetch the bytes for the groups it actually selected.
      px: ./pixel_ds
  val:
    path: ./meta_ds
    split_part: val
    aux:
      px: ./pixel_ds
YAML

# `energon lint` and `energon info` (7.4.1) cannot validate a CRUDE dataset: lint builds a dataset
# with the default task encoder and asserts it has cookers, and info reads a `sample_type` that a
# crude dataset.yaml does not carry. So validate the way this dataset is actually consumed --
# through our cooker and task encoder -- which checks strictly more: that the aux keys resolve, the
# pixels reshape, and packs come out at the expected geometry.
echo "==> validating $METADATASET through the task encoder"
python3 - "$METADATASET" <<'PY'
import sys

from npsv3.models.energon import EnergonPackedImageDataModule

dm = EnergonPackedImageDataModule(metadataset_path=sys.argv[1], num_workers=0, packing_buffer_size=32)
for index, batch in enumerate(dm.train_dataloader()):
    print(
        f"  batch {index}: images={tuple(batch.images.shape)} {batch.images.dtype} "
        f"groups={batch.num_groups} occupancy={batch.num_images / batch.images.shape[0]:.2f}"
    )
    if index >= 1:
        break
dm.teardown()
print("  OK")
PY

echo
echo "Prepared. Train with:"
echo "  npsv3 command=train data=energon_packed_images data.metadataset_path=$METADATASET"
