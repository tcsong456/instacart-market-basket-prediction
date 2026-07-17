mkdir -p data/sample
python -m src.instacart_etl.sample.build_sample_dataset \
    --input-dir data/raw \
    --output-dir data/sample \
    --chunk-size 1000000 \
    --seed 7872 \
    --sample-n 500
