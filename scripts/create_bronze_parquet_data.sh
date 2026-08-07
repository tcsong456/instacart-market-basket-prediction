#!/usr/bin/env bash

set -euo pipefail

gcloud storage rsync -r src/instacart_etl_rnn/contracts gs://instacart-raw-fc45ebb3/contracts
gcloud storage ls gs://instacart-raw-fc45ebb3/contracts

file=$1

(
    cd src
    python -m zipfile -c ../src.zip instacart_etl_rnn
)

gcloud dataproc jobs submit pyspark \
    src/instacart_etl_rnn/cli/build_bronze_parquet_dataset.py \
    --cluster=instacart-dataproc-cluster-fc45ebb3 \
    --region=europe-west1 \
    --py-files=src.zip \
    -- \
    --csv-path="gs://instacart-raw-fc45ebb3/$1" \
    --parquet-path="gs://instacart-bronze-fc45ebb3/$1" \
    --contract-path="gs://instacart-raw-fc45ebb3/contracts"