#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/../.."
source scripts/efficiency/env.sh
R="$(pwd)/outputs/efficiency/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$R"/{setup,source,fixtures,runs,metrics,comparison}
printf '%s\n' "$R" > outputs/efficiency/latest_run.txt
date -Iseconds > "$R/setup/start.txt"
python scripts/efficiency/download_assets.py > "$R/setup/download.log" 2>&1
python scripts/efficiency/prepare_fixture.py > "$R/setup/fixture.log" 2>&1
python scripts/efficiency/run_benchmark.py --repeats 3 | tee "$R/metrics/benchmark.log"
python scripts/efficiency/run_benchmark.py --only E1_full_request --repeats 3 | tee "$R/metrics/planner.log"
python scripts/efficiency/validate_render.py --case baseline > "$R/comparison/baseline_render.log" 2>&1
for D in "$R"/runs/*; do
  C=$(basename "$D")
  EXTRA=()
  [[ "$C" == *_r1 ]] && EXTRA=(--release)
  python scripts/efficiency/validate_render.py --case "$C" "${EXTRA[@]}" > "$R/comparison/${C}_render.log" 2>&1
done
python scripts/efficiency/summarize.py > "$R/metrics/summary.log"
date -Iseconds > "$R/setup/end.txt"
echo "Results: $R/metrics/summary.json"
