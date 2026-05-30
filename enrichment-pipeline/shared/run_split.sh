#!/usr/bin/env bash
# Run run_split for every dataset defined in config/to_split_files.json
# Fields per entry:
#   source      - dataset name (required)
#   input       - path to enriched/canonical data to be split (required)
#   output      - path to directory to write splits in (optional)
#                   data/splits/ by default if not specified
#   train       - train set ratio (optional)
#                   0.7 by default if not specified
#   val         - val set ratio (optional)
#                   0.15 by default if not specified
#   test        - test set ratio (optional)
#                   0.15 by default if not specified

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/config/to_split_files.json"

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
    TRAIN=$(get train)
    VAL=$(get val)
    TEST=$(get test)

    if [[ -z "$SOURCE" || -z "$INPUT" ]]; then
        echo "Skipping entry $i: source or input is empty"
        continue
    fi

    echo ""
    echo "Splitting dataset $SOURCE: $INPUT"
    CMD=(python3 "$SCRIPT_DIR/run_split.py" "$INPUT")

    [[ -n "$OUTPUT" ]]        && CMD+=(--output-dir "$OUTPUT")
    [[ -n "$TRAIN" ]]         && CMD+=(--train "$TRAIN")
    [[ -n "$VAL" ]]           && CMD+=(--val "$VAL")
    [[ -n "$TEST" ]]          && CMD+=(--test "$TEST")

    echo ">>> [$((i+1))/$n] ${CMD[*]}"
    "${CMD[@]}"
done

echo "All datasets split."