import pytest
from pyspark import StorageLevel
from pyspark.sql import DataFrame

from instacart_etl_rnn.jobs.create_reorder_size_training_data_job import (
    run_reorder_size_training_data,
)
from instacart_etl_rnn.validation.exceptions import DataValidationError
from instacart_etl_rnn.validation.models import ValidationReport


def test_run_reorder_size_training_data_unpersists_when_validation_fails(
    spark,
    mocker,
):
    user_data = mocker.sentinel.user_data
    transformed_data = mocker.sentinel.transformed_data
    contract = mocker.sentinel.contract

    reorder_size_data = mocker.Mock(spec=DataFrame)

    mocker.patch(
        "instacart_etl_rnn.jobs.create_reorder_size_training_data_job.read_parquet",
        return_value=user_data,
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_reorder_size_training_data_job."
        "transform_reorder_size_data",
        return_value=transformed_data,
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_reorder_size_training_data_job."
        "pad_column_arrays",
        return_value=reorder_size_data,
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_reorder_size_training_data_job.load_contract",
        return_value=contract,
    )

    report = ValidationReport(dataset_name="reorder_size", results=[])
    validation_error = DataValidationError(report)

    mocked_validate = mocker.patch(
        "instacart_etl_rnn.jobs.create_reorder_size_training_data_job.validate_dataset",
        side_effect=validation_error,
    )

    mocked_write = mocker.patch(
        "instacart_etl_rnn.jobs.create_reorder_size_training_data_job.write_parquet",
    )

    with pytest.raises(DataValidationError):
        run_reorder_size_training_data(
            spark=spark,
            input_path="silver",
            output_path="gold",
            contract_path="contracts",
            pad_length=5,
            mode="train",
        )

    reorder_size_data.persist.assert_called_once_with(StorageLevel.MEMORY_AND_DISK)

    mocked_validate.assert_called_once_with(
        reorder_size_data,
        contract=contract,
    )

    mocked_write.assert_not_called()

    reorder_size_data.unpersist.assert_called_once_with()
