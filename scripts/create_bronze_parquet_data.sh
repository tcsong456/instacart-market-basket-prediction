#!/usr/bin/env bash

set -euo pipefail

gcloud storage rsync -r src/instacart_etl_rnn/contracts gs://instacart-raw-fc45ebb3/contracts
gcloud storage ls gs://instacart-raw-fc45ebb3/contracts

build_bronze \
    --csv-path gs://instacart-raw-fc45ebb3/sample \
    --parquet-path gs://instacart-bronze-fc45ebb3/sample \
    --contract-path gs://instacart-raw-fc45ebb3/contracts

build_bronze \
    --csv-path gs://instacart-raw-fc45ebb3/raw \
    --parquet-path gs://instacart-bronze-fc45ebb3/raw \
    --contract-path gs://instacart-raw-fc45ebb3/contracts