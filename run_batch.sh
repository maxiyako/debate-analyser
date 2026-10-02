#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=".venv/bin/python"
run_one() {
  local url="$1"
  local date="$2"
  local id="${url##*/}"
  echo "=== Starting $id ($date) at $(date) ==="
  "$PY" main.py --url "$url" --debate-date "$date" 2>&1 | tee "data/reports/run_${id}.log"
  echo "=== Finished $id at $(date) ==="
}
run_one "https://www.stvr.sk/televizia/archiv/14036/592879" "2026-04-19"
run_one "https://www.stvr.sk/televizia/archiv/14036/594102" "2026-04-26"
run_one "https://www.stvr.sk/televizia/archiv/14036/589139" "2026-03-29"
run_one "https://www.stvr.sk/televizia/archiv/14036/587825" "2026-03-22"