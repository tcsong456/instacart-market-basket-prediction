import argparse
import logging

from instacart_etl_rnn.common.setup_logging import configure_logging
from instacart_etl_rnn.common.spark import create_spark_session
from instacart_etl_rnn.jobs.create_user_data_job import run_user_data_job


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--contract-path", required=True)
    parser.add_argument("--mode", required=True)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    args = parse_args()

    spark = create_spark_session("build_user_data")

    try:
        run_user_data_job(
            spark=spark,
            path=args.path,
            contract_path=args.contract_path,
            mode=args.mode,
        )
    except Exception:
        logger.exception("Build user data failed!")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":  # pragma: no cover
    main()
