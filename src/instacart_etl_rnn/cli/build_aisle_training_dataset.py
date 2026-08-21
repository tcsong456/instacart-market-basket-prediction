import argparse
import logging

from instacart_etl_rnn.common.setup_logging import configure_logging
from instacart_etl_rnn.common.spark import create_spark_session
from instacart_etl_rnn.jobs.create_aisle_training_data_job import (
    run_aisle_training_data_job,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--contract-path", required=True)
    parser.add_argument("--pad-length", default=30, type=int)

    return parser.parse_args()


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    args = parse_args()

    spark = create_spark_session("build_aisle_training_data")

    try:
        run_aisle_training_data_job(
            spark=spark,
            input_path=args.input_path,
            output_path=args.output_path,
            contract_path=args.contract_path,
            pad_length=args.pad_length,
        )
    except Exception:
        logger.exception("Build aisle training data failed!")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":  # pragma: no cover
    main()
