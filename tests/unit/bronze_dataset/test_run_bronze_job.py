from pathlib import Path
from unittest.mock import call

import pytest

from instacart_etl_rnn.jobs.create_bronze_dataset_job import run_bronze_job
from instacart_etl_rnn.validation.exceptions import DataValidationError
from instacart_etl_rnn.validation.models import ValidationReport


def test_run_bronze_job_runs_bronze_builders(
    mocker,
):
    spark = mocker.sentinel.spark

    mocked_independent = mocker.patch(
        "instacart_etl_rnn.jobs.create_bronze_dataset_job.build_independent_bronze_datasets"
    )

    mocked_dependent = mocker.patch(
        "instacart_etl_rnn.jobs.create_bronze_dataset_job.build_dependent_bronze_datasets"
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


def test_run_bronze_job_does_not_build_dependent_datasets_when_independent_fails(
    mocker,
):
    spark = mocker.sentinel.spark

    report = ValidationReport(dataset_name="orders", results=[])
    independent_error = DataValidationError(report)
    mocked_independent = mocker.patch(
        "instacart_etl_rnn.jobs.create_bronze_dataset_job.build_independent_bronze_datasets",
        side_effect=independent_error,
    )

    mocked_dependent = mocker.patch(
        "instacart_etl_rnn.jobs.create_bronze_dataset_job.build_dependent_bronze_datasets"
    )

    with pytest.raises(DataValidationError) as exc_info:
        run_bronze_job(
            spark,
            csv_path="raw",
            parquet_path="bronze",
            contract_path=Path("contracts"),
        )

    assert exc_info.value is independent_error

    mocked_independent.assert_called_once_with(
        spark,
        input_path="raw",
        output_path="bronze",
        contract_path=Path("contracts"),
    )

    mocked_dependent.assert_not_called()
