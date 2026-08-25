set -euo pipefail

file=$1
period=$2

contract_path=gs://instacart-raw-fc45ebb3/contracts
gcloud storage rsync -r src/instacart_etl_rnn/contracts $contract_path

(
    cd src
    python -m zipfile -c ../src.zip instacart_etl_rnn
)

gcloud dataproc jobs submit pyspark \
    src/instacart_etl_rnn/cli/build_user_order_split_dataset.py \
    --cluster=instacart-dataproc-cluster-fc45ebb3 \
    --region=europe-west1 \
    --py-files=src.zip \
    -- \
    --input-path="gs://instacart-bronze-fc45ebb3/$file" \
    --output-path="gs://instacart-bronze-fc45ebb3/simulation/$file/$period" \
    --contract-path="gs://instacart-raw-fc45ebb3/contracts" \
    --period=$period
