stack_dir=$(python -m instacart_rnn.stacking.folds \
  --runs-root gs://instacart-runs-fc45ebb3 \
  --mode sample \
  --timeline initial \
  --n-folds 5)

python -m instacart_rnn.stacking.feature_selection \
  --stack-dir "$stack_dir"