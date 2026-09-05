import pytest

from instacart_etl_rnn.jobs.create_user_product_count_data_job import (
    run_user_product_count_job,
)
from instacart_etl_rnn.validation.exceptions import DataValidationError
from instacart_etl_rnn.validation.models import ValidationReport

MODULE_PATH = "instacart_etl_rnn.jobs.create_user_product_count_data_job"


def test_run_user_product_count_job_rejects_unsupported_mode():
    with pytest.raises(ValueError, match="Unsupported mode: test"):
        run_user_product_count_job(
            spark=None,
            input_path="input",
            output_path="output",
            contract_path="contracts",
            mode="test",
        )


def test_run_user_product_count_job_does_not_write_when_validation_fails(mocker):
    spark = mocker.sentinel.spark
    order_products = mocker.sentinel.order_products
    user_product_count = mocker.sentinel.user_product_count
    contract = mocker.sentinel.contract

    mocker.patch(
        f"{MODULE_PATH}.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )
    mocker.patch(
        f"{MODULE_PATH}.read_parquet",
        return_value=order_products,
    )
    mocker.patch(
        f"{MODULE_PATH}.build_user_product_count",
        return_value=user_product_count,
    )
    mocker.patch(
        f"{MODULE_PATH}.load_contract",
        return_value=contract,
    )

    report = ValidationReport(dataset_name="user_product_count", results=[])
    mocker.patch(
        f"{MODULE_PATH}.validate_dataset",
        side_effect=DataValidationError(report),
    )
    mock_write = mocker.patch(f"{MODULE_PATH}.write_parquet")

    with pytest.raises(DataValidationError):
        run_user_product_count_job(
            spark=spark,
            input_path="silver",
            output_path="gold",
            contract_path="contracts",
            mode="validation",
        )

    mock_write.assert_not_called()
