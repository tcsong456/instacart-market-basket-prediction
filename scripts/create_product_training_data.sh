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
    src/instacart_etl_rnn/cli/build_product_training_dataset.py \
    --cluster=instacart-dataproc-cluster-fc45ebb3 \
    --region=europe-west1 \
    --py-files=src.zip \
    -- \
    --input-path="gs://instacart-gold-fc45ebb3/$folder" \
    --raw-path="gs://instacart-bronze-fc45ebb3/$1" \
    --output-path="gs://instacart-gold-fc45ebb3/$folder" \
    --contract-path="gs://instacart-raw-fc45ebb3/contracts" \
    --min-word-freq=10 \
    --product-name-length=30 \
    --encode-length=100