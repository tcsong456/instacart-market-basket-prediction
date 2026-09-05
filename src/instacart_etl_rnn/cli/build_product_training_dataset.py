import argparse
import logging

from instacart_etl_rnn.common.setup_logging import configure_logging
from instacart_etl_rnn.common.spark import create_spark_session
from instacart_etl_rnn.jobs.create_product_training_data_job import (
    run_product_training_data_job,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--contract-path", required=True)
    parser.add_argument("--min-word-freq", default=5, type=int)
    parser.add_argument("--product-name-length", default=30, type=int)
    parser.add_argument("--encode-length", default=100, type=int)
    parser.add_argument(
        "--mode", choices=["train", "validation", "evaluation"], required=True
    )

    return parser.parse_args()


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    args = parse_args()

    spark = create_spark_session("build_product_training_data")

    try:
        run_product_training_data_job(
            spark=spark,
            input_path=args.input_path,
            raw_path=args.raw_path,
            output_path=args.output_path,
            contract_path=args.contract_path,
            min_word_freq=args.min_word_freq,
            product_name_length=args.product_name_length,
            encode_length=args.encode_length,
            mode=args.mode,
        )
    except Exception:
        logger.exception("Build product training data failed!")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":  # pragma: no cover
    main()
