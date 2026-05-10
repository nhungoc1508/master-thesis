#!/usr/bin/env bash
# Run run_clean.py for every dataset defined in config/raw_files.json.
# Fields per entry:
#   source     - cleaner name (required)
#   input      - path to raw data (required)
#   output     - path to write canonical parquet output (required)
#   sample     - number of trajectories for grid sampling
#   grid_rows  - grid rows for sampling (default 20)
#   grid_cols  - grid cols for sampling (default 20)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/config/raw_files.json"

n=$(python3 -c "import json,sys; d=json.load(open('$CONFIG')); print(len(d['datasets']))")

for i in $(seq 0 $((n - 1))); do
    get() {
        python3 -c "
import json, sys
d = json.load(open('$CONFIG'))['datasets'][$i]
val = d.get('$1')
if val is None: print('')
else: print(val)
"
    }

    SOURCE=$(get source)
    INPUT=$(get input)
    OUTPUT=$(get output)
    SAMPLE=$(get sample)
    GRID_ROWS=$(get grid_rows)
    GRID_COLS=$(get grid_cols)

    if [[ -z "$SOURCE" || -z "$INPUT" || -z "$OUTPUT" ]]; then
        echo "Skipping entry $i: source, input, or output is empty"
        continue
    fi

    CMD=(python3 "$SCRIPT_DIR/run_clean.py" "$SOURCE" "$INPUT" "$OUTPUT")

    [[ -n "$SAMPLE" ]] && CMD+=(--sample "$SAMPLE")
    [[ -n "$GRID_ROWS" ]] && CMD+=(--grid-rows "$GRID_ROWS")
    [[ -n "$GRID_COLS" ]] && CMD+=(--grid-cols "$GRID_COLS")

    echo ">>> [$((i+1))/$n] ${CMD[*]}"
    "${CMD[@]}"
done

echo "All datasets cleaned."
