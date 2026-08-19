from unittest.mock import call

import pytest
from pyspark import StorageLevel
from pyspark.sql import DataFrame

from instacart_etl_rnn.jobs.create_reorder_size_training_data_job import (
    run_reorder_size_training_data,
)
from instacart_etl_rnn.validation.exceptions import DataValidationError
from instacart_etl_rnn.validation.models import ValidationReport


def test_run_reorder_size_training_data_runs_pipeline(
    spark,
    mocker,
):
    user_data = mocker.sentinel.user_data
    transformed_data = mocker.sentinel.transformed_data
    contract = mocker.sentinel.contract

    reorder_size_data = mocker.Mock(spec=DataFrame)

    mocked_join = mocked_read = mocker.patch(
        "instacart_etl_rnn.jobs.create_reorder_size_training_data_job.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )

    mocked_read = mocker.patch(
        "instacart_etl_rnn.jobs.create_reorder_size_training_data_job.read_parquet",
        return_value=user_data,
    )

    mocked_transform = mocker.patch(
        "instacart_etl_rnn.jobs.create_reorder_size_training_data_job."
        "transform_reorder_size_data",
        return_value=transformed_data,
    )

    mocked_pad = mocker.patch(
        "instacart_etl_rnn.jobs.create_reorder_size_training_data_job."
        "pad_column_arrays",
        return_value=reorder_size_data,
    )

    mocked_load_contract = mocker.patch(
        "instacart_etl_rnn.jobs.create_reorder_size_training_data_job.load_contract",
        return_value=contract,
    )

    mocked_validate = mocker.patch(
        "instacart_etl_rnn.jobs.create_reorder_size_training_data_job.validate_dataset",
    )

    mocked_write = mocker.patch(
        "instacart_etl_rnn.jobs.create_reorder_size_training_data_job.write_parquet",
    )

    manager = mocker.Mock()
    manager.attach_mock(mocked_join, "join")
    manager.attach_mock(mocked_read, "read")
    manager.attach_mock(mocked_transform, "transform")
    manager.attach_mock(mocked_pad, "pad")
    manager.attach_mock(mocked_load_contract, "load")
    manager.attach_mock(mocked_validate, "validate")
    manager.attach_mock(mocked_write, "write")

    run_reorder_size_training_data(
        spark=spark,
        input_path="silver",
        output_path="gold",
        contract_path="contracts",
        pad_length=5,
    )

    assert manager.mock_calls == [
        call.join("silver", "user_data"),
        call.read("silver/user_data", spark),
        call.transform(user_data),
        call.pad(transformed_data, 5),
        call.join("contracts", "reorder_size_training_data.yaml"),
        call.load("contracts/reorder_size_training_data.yaml"),
        call.validate(reorder_size_data, contract=contract),
        call.join("gold", "reorder_size_training_data"),
        call.write("gold/reorder_size_training_data", reorder_size_data),
    ]

    df = mocked_validate.call_args.args[0]
    written_df = mocked_write.call_args.args[1]
    assert df is written_df

    reorder_size_data.persist.assert_called_once_with(StorageLevel.MEMORY_AND_DISK)
    reorder_size_data.unpersist.assert_called_once_with()


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
        )

    reorder_size_data.persist.assert_called_once_with(StorageLevel.MEMORY_AND_DISK)

    mocked_validate.assert_called_once_with(
        reorder_size_data,
        contract=contract,
    )

    mocked_write.assert_not_called()

    reorder_size_data.unpersist.assert_called_once_with()
