from unittest.mock import call

import pytest

from instacart_etl_rnn.jobs.create_user_product_count_data_job import (
    run_user_product_count_job,
)
from instacart_etl_rnn.validation.exceptions import DataValidationError
from instacart_etl_rnn.validation.models import ValidationReport


def test_run_user_product_count_job_orchestrates_pipeline(
    spark,
    mocker,
):
    order_products = mocker.sentinel.order_products
    user_product_count = mocker.sentinel.user_product_count
    contract = mocker.sentinel.contract

    mocked_join_path = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_product_count_data_job.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )

    mocked_read = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_product_count_data_job.read_parquet",
        return_value=order_products,
    )

    mocked_build = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_product_count_data_job.build_user_product_count",
        return_value=user_product_count,
    )

    mocked_load_contract = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_product_count_data_job.load_contract",
        return_value=contract,
    )

    mocked_validate = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_product_count_data_job.validate_dataset",
    )

    mocked_write = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_product_count_data_job.write_parquet",
    )

    manager = mocker.Mock()
    manager.attach_mock(mocked_join_path, "join")
    manager.attach_mock(mocked_read, "read")
    manager.attach_mock(mocked_build, "build")
    manager.attach_mock(mocked_load_contract, "load")
    manager.attach_mock(mocked_validate, "validate")
    manager.attach_mock(mocked_write, "write")

    run_user_product_count_job(
        spark=spark,
        input_path="silver",
        output_path="gold",
        contract_path="contracts",
    )

    assert manager.mock_calls == [
        call.join("silver", "order_products"),
        call.read("silver/order_products", spark),
        call.build(order_products),
        call.join("contracts", "user_product_count.yaml"),
        call.load("contracts/user_product_count.yaml"),
        call.validate(user_product_count, contract=contract),
        call.join("gold", "user_product_count"),
        call.write("gold/user_product_count", user_product_count),
    ]

    df = mocked_validate.call_args.args[0]
    written_df = mocked_write.call_args.args[1]
    assert df is written_df


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
        )

    mocked_write.assert_not_called()
