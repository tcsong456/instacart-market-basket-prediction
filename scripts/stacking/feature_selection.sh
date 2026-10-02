set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <mode> <timeline>" >&2
    exit 1
fi

mode=$1
timeline=$2

stack_dir=$(python -m instacart_rnn.stacking.folds \
  --runs-root gs://instacart-runs-fc45ebb3 \
  --mode "$mode" \
  --timeline "$timeline" \
  --n-folds 5)

python -m instacart_rnn.stacking.feature_selection \
  --stack-dir "$stack_dir"
