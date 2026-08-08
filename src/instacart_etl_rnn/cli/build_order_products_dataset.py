import argparse
import logging

from instacart_etl_rnn.common.setup_logging import configure_logging
from instacart_etl_rnn.common.spark import create_spark_session
from instacart_etl_rnn.silver.create_order_products import build_order_products


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--contract-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    configure_logging()
    logger = logging.getLogger(__name__)

    spark = create_spark_session("build_order_products")

    try:
        build_order_products(
            spark=spark,
            input_path=args.input_path,
            contract_path=args.contract_path,
            output_path=args.output_path,
            validation=args.validation,
        )
    except Exception:
        logger.exception("Building order_proudcts dataset failed")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
