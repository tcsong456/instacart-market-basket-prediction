#!/usr/bin/env bash

set -euo pipefail

file=$1

if [[ "$file" == "raw" ]]; then
    folder="curated"
else
    folder="sample"
fi

contract_path=gs://instacart-raw-fc45ebb3/contracts
gcloud storage rsync -r src/instacart_etl_rnn/contracts $contract_path

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
    --contract-path=$contract_path