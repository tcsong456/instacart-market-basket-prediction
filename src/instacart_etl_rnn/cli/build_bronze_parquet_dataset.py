import argparse
import logging
from pathlib import Path

from instacart_etl_rnn.common.setup_logging import setup_logging
from instacart_etl_rnn.common.spark import create_spark_session
from instacart_etl_rnn.jobs.create_bronze_dataset_job import run_bronze_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build bronze Parquet datasets from raw CSV files."
    )

    parser.add_argument(
        "--csv-path",
        required=True,
        help="Raw CSV directory or GCS prefix.",
    )
    parser.add_argument(
        "--parquet-path",
        required=True,
        help="Bronze Parquet output directory or GCS prefix.",
    )
    parser.add_argument(
        "--contract-path",
        required=True,
        type=Path,
        help="Directory containing dataset contracts.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    spark = create_spark_session("build_bronze_datasets")

    try:
        run_bronze_job(
            spark,
            csv_path=args.csv_path,
            parquet_path=args.parquet_path,
            contract_path=args.contract_path,
        )
    except Exception:
        logger.exception("Bronze dataset job failed")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
