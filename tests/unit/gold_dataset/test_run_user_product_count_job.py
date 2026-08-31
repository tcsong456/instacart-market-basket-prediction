import pytest

from instacart_etl_rnn.jobs.create_user_product_count_data_job import (
    run_user_product_count_job,
)
from instacart_etl_rnn.validation.exceptions import DataValidationError
from instacart_etl_rnn.validation.models import ValidationReport


def test_run_user_product_count_job_does_not_write_when_validation_fails(
    spark,
    mocker,
):
    order_products = mocker.sentinel.order_products
    user_product_count = mocker.sentinel.user_product_count
    contract = mocker.sentinel.contract

    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_product_count_data_job.read_parquet",
        return_value=order_products,
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_product_count_data_job.build_user_product_count",
        return_value=user_product_count,
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_product_count_data_job.load_contract",
        return_value=contract,
    )

    report = ValidationReport(dataset_name="user_product_count", results=[])
    validate_error = DataValidationError(report)
    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_product_count_data_job.validate_dataset",
        side_effect=validate_error,
    )

    mocked_write = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_product_count_data_job.write_parquet",
    )

    with pytest.raises(DataValidationError):
        run_user_product_count_job(
            spark=spark,
            input_path="silver",
            output_path="gold",
            contract_path="contracts",
            mode="validation",
        )

    mocked_write.assert_not_called()
