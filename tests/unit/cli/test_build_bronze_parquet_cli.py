import pytest

from instacart_etl_rnn.cli.build_bronze_parquet_dataset import main


def test_main_runs_bronze_job_and_stops_spark(mocker):
    args = mocker.Mock(
        csv_path="gs://raw-bucket/raw",
        parquet_path="gs://bronze-bucket/raw",
        contract_path="gs://raw-bucket/contracts",
    )

    spark = mocker.Mock()

    mocked_parse_args = mocker.patch(
        "instacart_etl_rnn.cli.build_bronze_parquet_dataset.parse_args",
        return_value=args,
    )

    mocked_configure_logging = mocker.patch(
        "instacart_etl_rnn.cli.build_bronze_parquet_dataset.configure_logging",
    )

    mocked_create_spark = mocker.patch(
        "instacart_etl_rnn.cli.build_bronze_parquet_dataset.create_spark_session",
        return_value=spark,
    )

    mocked_run_job = mocker.patch(
        "instacart_etl_rnn.cli.build_bronze_parquet_dataset.run_bronze_job",
    )

    result = main()

    assert result is None

    mocked_parse_args.assert_called_once_with()
    mocked_configure_logging.assert_called_once_with()

    mocked_create_spark.assert_called_once_with("build_bronze_datasets")

    mocked_run_job.assert_called_once_with(
        spark,
        csv_path="gs://raw-bucket/raw",
        parquet_path="gs://bronze-bucket/raw",
        contract_path="gs://raw-bucket/contracts",
    )

    spark.stop.assert_called_once_with()


def test_main_logs_reraises_and_stops_spark_when_job_fails(
    mocker,
):
    args = mocker.Mock(
        csv_path="gs://raw-bucket/raw",
        parquet_path="gs://bronze-bucket/raw",
        contract_path="gs://raw-bucket/contracts",
    )

    spark = mocker.Mock()
    job_error = RuntimeError("bronze job failed")

    mocker.patch(
        "instacart_etl_rnn.cli.build_bronze_parquet_dataset.parse_args",
        return_value=args,
    )

    mocker.patch(
        "instacart_etl_rnn.cli.build_bronze_parquet_dataset.configure_logging",
    )

    mocker.patch(
        "instacart_etl_rnn.cli.build_bronze_parquet_dataset.create_spark_session",
        return_value=spark,
    )

    mocker.patch(
        "instacart_etl_rnn.cli.build_bronze_parquet_dataset.run_bronze_job",
        side_effect=job_error,
    )

    mocked_logger = mocker.Mock()

    mocker.patch(
        "instacart_etl_rnn.cli.build_bronze_parquet_dataset.logging.getLogger",
        return_value=mocked_logger,
    )

    with pytest.raises(RuntimeError) as exc_info:
        main()

    assert exc_info.value is job_error

    mocked_logger.exception.assert_called_once_with("Bronze dataset job failed")

    spark.stop.assert_called_once_with()
