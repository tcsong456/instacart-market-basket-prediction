from unittest.mock import call

import pytest
from pyspark.sql import DataFrame

from instacart_etl_rnn.jobs.create_product_training_data_job import (
    run_product_training_data_job,
)
from instacart_etl_rnn.validation.exceptions import DataValidationError
from instacart_etl_rnn.validation.models import ValidationReport


def test_run_product_seq_data_job_orchestrates_pipeline(
    spark,
    mocker,
):
    products = mocker.sentinel.products
    product_history_data = mocker.sentinel.product_history_data
    word_index = mocker.Mock(spec=DataFrame)
    word_index.persist.return_value = word_index
    encoded_product_name = mocker.Mock(spec=DataFrame)
    encoded_product_name.persist.return_value = encoded_product_name
    product_training_data = mocker.Mock(spec=DataFrame)
    contract = mocker.sentinel.contract

    mocked_join_path = mocker.patch(
        "instacart_etl_rnn.jobs.create_product_training_data_job.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )

    mocked_read = mocker.patch(
        "instacart_etl_rnn.jobs.create_product_training_data_job.read_parquet",
        side_effect=[products, product_history_data],
    )

    mocked_word_idx = mocker.patch(
        "instacart_etl_rnn.jobs.create_product_training_data_job.build_word_idx",
        return_value=word_index,
    )

    mocked_encoded_name = mocker.patch(
        "instacart_etl_rnn.jobs.create_product_training_data_job.encode_product_names",
        return_value=encoded_product_name,
    )

    mocked_training_data = mocker.patch(
        "instacart_etl_rnn.jobs.create_product_training_data_job.build_product_training_data",
        return_value=product_training_data,
    )

    mocked_contract_load = mocker.patch(
        "instacart_etl_rnn.jobs.create_product_training_data_job.load_contract",
        return_value=contract,
    )

    mocked_validate = mocker.patch(
        "instacart_etl_rnn.jobs.create_product_training_data_job.validate_dataset",
    )

    mocked_write = mocker.patch(
        "instacart_etl_rnn.jobs.create_product_training_data_job.write_parquet"
    )

    manager = mocker.Mock()
    manager.attach_mock(mocked_join_path, "join")
    manager.attach_mock(mocked_read, "read")
    manager.attach_mock(mocked_word_idx, "word_idx")
    manager.attach_mock(mocked_encoded_name, "encoded_name")
    manager.attach_mock(mocked_training_data, "train")
    manager.attach_mock(mocked_contract_load, "load")
    manager.attach_mock(mocked_validate, "validate")
    manager.attach_mock(mocked_write, "write")

    run_product_training_data_job(
        spark=spark,
        raw_path="data",
        input_path="gold",
        output_path="gold",
        contract_path="contracts",
        min_word_freq=2,
        product_name_length=3,
        encode_length=4,
    )

    assert manager.mock_calls == [
        call.join("data", "products"),
        call.read("data/products", spark),
        call.join("gold", "product_history_data"),
        call.read("gold/product_history_data", spark),
        call.word_idx(products, 2),
        call.encoded_name(products, word_index),
        call.train(
            product_history_data=product_history_data,
            encoded_product_name=encoded_product_name,
            product_name_length=3,
            encode_length=4,
        ),
        call.join("contracts", "product_training_data.yaml"),
        call.load("contracts/product_training_data.yaml"),
        call.validate(product_training_data, contract=contract),
        call.join("gold", "product_training_data"),
        call.write("gold/product_training_data", product_training_data),
    ]

    df = mocked_validate.call_args.args[0]
    written_df = mocked_write.call_args.args[1]
    assert df is written_df


def test_run_product_training_data_job_does_not_write_when_validation_fails(
    spark,
    mocker,
):
    products = mocker.sentinel.products
    product_history_data = mocker.sentinel.product_history_data
    word_index = mocker.Mock(spec=DataFrame)
    encoded_product_name = mocker.Mock(spec=DataFrame)
    training_data = mocker.Mock(spec=DataFrame)
    contract = mocker.sentinel.contract

    mocker.patch(
        "instacart_etl_rnn.jobs.create_product_training_data_job.read_parquet",
        side_effect=[
            products,
            product_history_data,
        ],
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_product_training_data_job.build_word_idx",
        return_value=word_index,
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_product_training_data_job.encode_product_names",
        return_value=encoded_product_name,
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_product_training_data_job.build_product_training_data",
        return_value=training_data,
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_product_training_data_job.load_contract",
        return_value=contract,
    )

    report = ValidationReport(dataset_name="product_training_data", results=[])
    validation_error = DataValidationError(report)

    mocker.patch(
        "instacart_etl_rnn.jobs.create_product_training_data_job.validate_dataset",
        side_effect=validation_error,
    )

    mocked_write = mocker.patch(
        "instacart_etl_rnn.jobs.create_product_training_data_job.write_parquet",
    )

    with pytest.raises(DataValidationError):
        run_product_training_data_job(
            spark=spark,
            raw_path="raw",
            input_path="silver",
            output_path="gold",
            contract_path="contracts",
            min_word_freq=3,
            product_name_length=20,
            encode_length=40,
        )

    mocked_write.assert_not_called()
