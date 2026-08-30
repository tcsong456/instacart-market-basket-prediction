#!/usr/bin/env bash

set -euo pipefail

mkdir -p data/sample

python -m src.instacart_etl.sample.build_sample_dataset \
    --input-dir data/raw \
    --output-dir data/sample \
    --chunk-size 1000000 \
    --seed 7872 \
    --sample-n 500

gcloud storage rsync -r data/raw gs://instacart-raw-fc45ebb3/raw
gcloud storage rsync -r data/sample gs://instacart-raw-fc45ebb3/sample

gcloud storage ls gs://instacart-raw-fc45ebb3/raw
gcloud storage ls gs://instacart-raw-fc45ebb3/sample