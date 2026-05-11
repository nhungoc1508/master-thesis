#!/usr/bin/env bash
# Run run_describe for every dataset defined in config/to_describe_files.json
# Fields per entry:
#   source      - dataset name (required)
#   input       - path to enriched data to be described (required)
#   output      - path to directory to write described .parquet in (optional)
#                   data/described/ by default if not specified

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/config/to_describe_files.json"

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

    if [[ -z "$SOURCE" || -z "$INPUT" ]]; then
        echo "Skipping entry $i: source or input is empty"
        continue
    fi

    echo ""
    echo "Describing dataset $SOURCE: $INPUT"
    CMD=(python3 "$SCRIPT_DIR/run_describe.py" "$INPUT")

    [[ -n "$OUTPUT" ]]        && CMD+=(--output-dir "$OUTPUT")

    echo ">>> [$((i+1))/$n] ${CMD[*]}"
    "${CMD[@]}"
done

echo "All datasets described."