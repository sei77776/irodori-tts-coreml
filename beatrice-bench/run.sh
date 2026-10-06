#!/usr/bin/env bash
# Usage: ./run.sh [results_dir] [extra bench.py args...]   (e.g. ./run.sh results --convert-only)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export BEATRICE_WORK="${BEATRICE_WORK:-$HERE/work}"
OUT="${1:-$HERE/results}"; shift || true
bash "$HERE/fetch_assets.sh"
python -I "$HERE/bench.py" --out "$OUT" "$@"
