#!/usr/bin/env bash
# Run ncu on provided executable and compare with baseline CSV using tools/compare_ncu.py
set -euo pipefail

EXE=${1:-}
BASELINE=${2:-ci/baselines/baseline.csv}
METRICS=${3:-}

if [ -z "$EXE" ]; then
  echo "Usage: $0 <exe> [baseline.csv] [metrics]"
  exit 2
fi

NCU=$(python - <<'PY'
from src.discover import find_ncu_path
print(find_ncu_path() or 'ncu')
PY
)

if [ ! -x "$NCU" ] && ! command -v "$NCU" >/dev/null 2>&1; then
  echo "ncu not found: $NCU"
  exit 2
fi

OUT="ncu_ci_$(date +%Y%m%d_%H%M%S)"
CMD=("$NCU")
if [ -n "$METRICS" ]; then
  CMD+=(--metrics "$METRICS")
fi
CMD+=(--csv --output "$OUT" "$EXE")

echo "Running: ${CMD[*]}"

if ! "${CMD[@]}"; then
  echo "ncu run failed"
  exit 1
fi

CSV="${OUT}.csv"
if [ ! -f "$CSV" ]; then
  echo "ncu did not produce CSV: $CSV"
  exit 2
fi

if python tools/compare_ncu.py "$BASELINE" "$CSV"; then
  echo "NCU check: OK"
else
  status=$?
  echo "NCU check: Regression detected or error"
  exit "$status"
fi
