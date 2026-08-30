#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

build_features() {
    local source=$1
    local snapshot_period=$2
    local mode=$3

    "$ROOT_DIR/scripts/feature/create_user_data.sh" \
        "$source" "$snapshot_period" "$mode"

    "$ROOT_DIR/scripts/feature/create_user_product_count_data.sh" \
        "$source" "$snapshot_period" "$mode"

    "$ROOT_DIR/scripts/feature/create_product_history_data.sh" \
        "$source" "$snapshot_period" "$mode"

    "$ROOT_DIR/scripts/feature/create_aisle_history_data.sh" \
        "$source" "$snapshot_period" "$mode"

    "$ROOT_DIR/scripts/feature/create_product_training_data.sh" \
        "$source" "$snapshot_period" "$mode"

    "$ROOT_DIR/scripts/feature/create_aisle_training_data.sh" \
        "$source" "$snapshot_period" "$mode"

    "$ROOT_DIR/scripts/feature/create_reorder_size_training_data.sh" \
        "$source" "$snapshot_period" "$mode"
}

SOURCE=raw
# for PERIOD in initial t1 t2; do
#     echo "Building base training data for source=$SOURCE period=$PERIOD"

#     "$ROOT_DIR/scripts/simulation/create_order_role_split_data.sh" \
#         "$SOURCE" "$PERIOD"

#     "$ROOT_DIR/scripts/feature/create_order_products_data.sh" \
#         "$SOURCE" "$PERIOD"

#     "$ROOT_DIR/scripts/simulation/create_order_simulation_split_data.sh" \
#         "$SOURCE" "$PERIOD" "base_train"

#     for MODE in train validation evaluation; do
#         build_features "$SOURCE" "$PERIOD" "$MODE"
#     done
# done

STACKING_INPUT_PERIOD="t2"

echo "Building stacking training data for source=$SOURCE"

"$ROOT_DIR/scripts/simulation/create_order_simulation_split_data.sh" \
    "$SOURCE" "$STACKING_INPUT_PERIOD" "stacking_train"

for MODE in train validation; do
    build_features "$SOURCE" "stacking_train" "$MODE"
done

echo "All training data builds completed."