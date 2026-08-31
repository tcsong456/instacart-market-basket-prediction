import pytest
from pyspark import StorageLevel
from pyspark.sql import DataFrame

from instacart_etl_rnn.jobs.create_aisle_training_data_job import (
    run_aisle_training_data_job,
)
from instacart_etl_rnn.validation.exceptions import DataValidationError
from instacart_etl_rnn.validation.models import ValidationReport


def test_run_aisle_training_data_job_unpersists_when_validation_fails(
    spark,
    mocker,
):
    aisle_history_data = mocker.sentinel.aisle_history_data
    contract = mocker.sentinel.contract
    aisle_training_data = mocker.Mock(spec=DataFrame)

    mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_training_data_job.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_training_data_job.read_parquet",
        return_value=aisle_history_data,
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_training_data_job.parse_aisle_seq_data",
        return_value=aisle_training_data,
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_training_data_job.load_contract",
        return_value=contract,
    )

    report = ValidationReport(dataset_name="aisle_training_data", results=[])
    validation_error = DataValidationError(report)
    mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_training_data_job.validate_dataset",
        side_effect=validation_error,
    )

    mocked_write = mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_training_data_job.write_parquet",
    )

    with pytest.raises(DataValidationError):
        run_aisle_training_data_job(
            spark=spark,
            input_path="gold",
            output_path="training",
            contract_path="contracts",
            pad_length=40,
            mode="train",
        )

    aisle_training_data.persist.assert_called_once_with(StorageLevel.MEMORY_AND_DISK)

    mocked_write.assert_not_called()

    aisle_training_data.unpersist.assert_called_once_with()
