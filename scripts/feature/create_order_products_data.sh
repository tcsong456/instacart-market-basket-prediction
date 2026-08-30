#!/usr/bin/env bash

set -euo pipefail

file=$1
period=$2

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
    src/instacart_etl_rnn/cli/build_order_products_dataset.py \
    --cluster=instacart-dataproc-cluster-fc45ebb3 \
    --region=europe-west1 \
    --py-files=src.zip \
    -- \
    --input-path="gs://instacart-bronze-fc45ebb3/$file" \
    --order-path="gs://instacart-bronze-fc45ebb3/simulation/$file/$period" \
    --output-path="gs://instacart-silver-fc45ebb3/$folder/$period" \
    --contract-path="gs://instacart-raw-fc45ebb3/contracts"