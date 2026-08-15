import argparse
import logging

from instacart_etl_rnn.common.setup_logging import configure_logging
from instacart_etl_rnn.common.spark import create_spark_session
from instacart_etl_rnn.jobs.create_product_history_data_job import (
    run_product_history_job,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--contract-path", required=True)

    return parser.parse_args()


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    args = parse_args()

    spark = create_spark_session("build_product_history_data")

    try:
        run_product_history_job(
            spark=spark,
            input_path=args.input_path,
            data_path=args.data_path,
            output_path=args.output_path,
            contract_path=args.contract_path,
        )
    except Exception:
        logger.exception("Build product history data failed!")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":  # pragma: no cover
    main()
