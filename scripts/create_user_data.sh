#!/usr/bin/env bash

set -euo pipefail

file=$1

if [[ "$file" == "raw" ]]; then
    folder="curated"
else
    folder="sample"
fi

(
    cd src
    python -m zipfile -c ../src.zip instacart_etl_rnn
)

gcloud dataproc jobs submit pyspark \
    src/instacart_etl_rnn/cli/build_user_dataset.py \
    --cluster=instacart-dataproc-cluster-fc45ebb3 \
    --region=europe-west1 \
    --py-files=src.zip \
    -- \
    --path="gs://instacart-silver-fc45ebb3/$folder" \
    --contract-path="gs://instacart-raw-fc45ebb3/contracts" \
    --validation