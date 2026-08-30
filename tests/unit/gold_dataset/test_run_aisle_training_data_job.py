from unittest.mock import call

import pytest
from pyspark import StorageLevel
from pyspark.sql import DataFrame

from instacart_etl_rnn.jobs.create_aisle_training_data_job import (
    run_aisle_training_data_job,
)
from instacart_etl_rnn.validation.exceptions import DataValidationError
from instacart_etl_rnn.validation.models import ValidationReport


def test_run_aisle_training_data_job_runs_pipeline(
    spark,
    mocker,
):
    aisle_history_data = mocker.sentinel.aisle_history_data
    contract = mocker.sentinel.contract
    aisle_training_data = mocker.Mock(spec=DataFrame)

    mocked_join_path = mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_training_data_job.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )

    mocked_read = mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_training_data_job.read_parquet",
        return_value=aisle_history_data,
    )

    mocked_parse = mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_training_data_job.parse_aisle_seq_data",
        return_value=aisle_training_data,
    )

    mocked_load_contract = mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_training_data_job.load_contract",
        return_value=contract,
    )

    mocked_validate = mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_training_data_job.validate_dataset",
    )

    mocked_write = mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_training_data_job.write_parquet",
    )

    manager = mocker.Mock()
    manager.attach_mock(mocked_join_path, "join")
    manager.attach_mock(mocked_read, "read")
    manager.attach_mock(mocked_parse, "parse")
    manager.attach_mock(mocked_load_contract, "load")
    manager.attach_mock(mocked_validate, "validate")
    manager.attach_mock(mocked_write, "write")

    run_aisle_training_data_job(
        spark=spark,
        input_path="gold",
        output_path="training",
        contract_path="contracts",
        pad_length=40,
        mode="train",
    )

    assert manager.mock_calls == [
        call.join("gold", "aisle_history_data_train"),
        call.read("gold/aisle_history_data_train", spark),
        call.parse(aisle_history_data, 40),
        call.join("contracts", "aisle_training_data.yaml"),
        call.load("contracts/aisle_training_data.yaml"),
        call.validate(aisle_training_data, contract=contract),
        call.join("training", "aisle_training_data_train"),
        call.write("training/aisle_training_data_train", aisle_training_data),
    ]

    df = mocked_validate.call_args.args[0]
    written_df = mocked_write.call_args.args[1]
    assert df is written_df

    aisle_training_data.persist.assert_called_once_with(StorageLevel.MEMORY_AND_DISK)
    aisle_training_data.unpersist.assert_called_once_with()


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
