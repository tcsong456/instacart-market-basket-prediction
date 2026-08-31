import pytest
from pyspark.sql import DataFrame

from instacart_etl_rnn.jobs.create_product_training_data_job import (
    run_product_training_data_job,
)
from instacart_etl_rnn.validation.exceptions import DataValidationError
from instacart_etl_rnn.validation.models import ValidationReport


def test_run_product_training_data_job_does_not_write_when_validation_fails(
    spark,
    mocker,
):
    products = mocker.sentinel.products
    product_history_data = mocker.sentinel.product_history_data
    word_index = mocker.Mock(spec=DataFrame)
    word_index.persist.return_value = word_index
    encoded_product_name = mocker.Mock(spec=DataFrame)
    encoded_product_name.persist.return_value = encoded_product_name
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
            mode="validation",
        )

    mocked_write.assert_not_called()
    training_data.unpersist.assert_called_once_with()
    encoded_product_name.unpersist.assert_called_once_with()
    word_index.unpersist.assert_called_once_with()
