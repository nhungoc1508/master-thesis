#!/usr/bin/env bash
# Run run_enrich for every dataset defined in config/canonical_files.json
# Fields per entry:
#   source      - dataset name (required)
#   input       - path to canonical data (required)
#   stages      - space-separated list of enrichment stages to run (optional)
#                   run all stages by default if not specified
#   output      - path to directory to write output in (optional)
#                   data/enriched/ by default if not specified
#   cache       - path to cache directory (optional)
#                   data/ (root dir) by default if not specified
#   config      - path to config file (optional)
#                   config/enrichment_config.yaml by default if not specified

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/config/canonical_files.json"

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
    STAGES=$(get stages)
    OUTPUT=$(get output)
    CACHE=$(get cache)
    ENRICH_CONFIG=$(get config)

    if [[ -z "$SOURCE" || -z "$INPUT" ]]; then
        echo "Skipping entry $i: source or input is empty"
        continue
    fi

    echo ""
    echo "Enriching dataset $SOURCE"
    CMD=(python3 "$SCRIPT_DIR/run_enrich.py" "$INPUT")

    [[ -n "$STAGES" ]]        && CMD+=(--stages "$STAGES")
    [[ -n "$OUTPUT" ]]        && CMD+=(--output-dir "$OUTPUT")
    [[ -n "$CACHE" ]]         && CMD+=(--cache-dir "$CACHE")
    [[ -n "$ENRICH_CONFIG" ]] && CMD+=(--config "$ENRICH_CONFIG")

    echo ">>> [$((i+1))/$n] ${CMD[*]}"
    "${CMD[@]}"
done

echo "All datasets enriched."