#!/usr/bin/env bash
# Clone beatrice-trainer (MIT) without LFS blobs and fetch only what the bench needs.
set -euo pipefail
WORK="${BEATRICE_WORK:-$(cd "$(dirname "$0")" && pwd)/work}"
mkdir -p "$WORK"
cd "$WORK"
if [ ! -d beatrice-trainer/.git ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://huggingface.co/fierce-cats/beatrice-trainer
fi
cd beatrice-trainer
for p in assets/pretrained/104_3_checkpoint_00300000.pt \
         assets/pretrained/122_checkpoint_03000000.pt \
         assets/pretrained/151_checkpoint_libritts_r_200_02750000.pt.gz \
         assets/test/common_voice_ja_38843402_16k.wav; do
  # LFS pointer files are ~130 bytes; download the real blob if not already present
  if [ ! -s "$p" ] || [ "$(wc -c < "$p")" -lt 1000 ]; then
    curl -sSfL --retry 3 -o "$p" "https://huggingface.co/fierce-cats/beatrice-trainer/resolve/main/$p"
  fi
  echo "$p $(wc -c < "$p") bytes"
done
