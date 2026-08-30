set -euo pipefail

file=$1
period=$2
mode=$3

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
    src/instacart_etl_rnn/cli/build_aisle_training_dataset.py \
    --cluster=instacart-dataproc-cluster-fc45ebb3 \
    --region=europe-west1 \
    --py-files=src.zip \
    -- \
    --input-path="gs://instacart-gold-fc45ebb3/snapshots/$folder/$period" \
    --output-path="gs://instacart-gold-fc45ebb3/training/$folder/$period" \
    --contract-path="gs://instacart-raw-fc45ebb3/contracts" \
    --pad-length=100 \
    --mode=$mode