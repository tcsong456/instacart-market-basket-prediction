from pathlib import Path
from unittest.mock import call

from instacart_etl_rnn.jobs.create_bronze_dataset_job import run_bronze_job


def test_run_bronze_job_runs_bronze_builders(
    mocker,
):
    spark = mocker.sentinel.spark

    mocked_independent = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.build_independent_bronze_datasets"
    )

    mocked_dependent = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.build_dependent_bronze_datasets"
    )

    manager = mocker.Mock()
    manager.attach_mock(
        mocked_independent,
        "independent",
    )
    manager.attach_mock(
        mocked_dependent,
        "dependent",
    )

    run_bronze_job(
        spark,
        csv_path="raw",
        parquet_path="bronze",
        contract_path=Path("contracts"),
    )

    assert manager.mock_calls == [
        call.independent(
            spark,
            input_path="raw",
            output_path="bronze",
            contract_path=Path("contracts"),
        ),
        call.dependent(
            spark,
            csv_path="raw",
            parquet_path="bronze",
            contract_path=Path("contracts"),
        ),
    ]
