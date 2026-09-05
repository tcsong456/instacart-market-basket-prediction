#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SOURCE=raw

for PERIOD in initial t1 t2; do
    echo "Building base training data for source=$SOURCE period=$PERIOD"

    for MODE in train validation evaluation; do
        "$ROOT_DIR/scripts/feature/create_user_product_count_data.sh" "$SOURCE" "$PERIOD" "$MODE"
    done
done


for MODE in train validation; do
    "$ROOT_DIR/scripts/feature/create_user_product_count_data.sh" "$SOURCE" "stacking_train" "$MODE"
done