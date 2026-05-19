#!/usr/bin/env bash
set -euo pipefail

EXE=${1:-}
BASELINE=${2:-ci/baselines/baseline.csv}
METRICS=${3:-}
COL=${4:-}
if [ -z "$EXE" ] || [ -z "$BASELINE" ]; then
  echo "Usage: $0 path/to/exe baseline.csv [metrics] [column]"
  exit 2
fi

NCU_BIN=$(which ncu || true)
if [ -z "$NCU_BIN" ]; then
  echo "ncu not found in PATH"
  exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
OUT="ncu_current_${TS}.csv"
echo "Running ncu to produce ${OUT}..."

CMD=("$NCU_BIN")
if [ -n "$METRICS" ]; then
  CMD+=("--metrics" "$METRICS")
fi
CMD+=("--csv" "--output" "${OUT%.*}" "$EXE")

if ! "${CMD[@]}" > ncu_run.log 2>&1; then
  echo "ncu run failed; see ncu_run.log"
  cat ncu_run.log
  exit 1
fi

COMPARE_ARGS=("$BASELINE" "$OUT")
if [ -n "$COL" ]; then
  COMPARE_ARGS+=("-c" "$COL")
fi

if python3 tools/compare_ncu.py "${COMPARE_ARGS[@]}"; then
  echo "NCU check: OK"
else
  status=$?
  echo "NCU check: Regression detected or error"
  exit "$status"
fi
