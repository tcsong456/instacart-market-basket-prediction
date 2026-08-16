from unittest.mock import call

import pytest
from pyspark.sql import DataFrame

from instacart_etl_rnn.jobs.create_product_history_data_job import (
    run_product_history_job,
)
from instacart_etl_rnn.validation.exceptions import DataValidationError
from instacart_etl_rnn.validation.models import ValidationReport


def test_run_product_history_job_calls_validation_before_write(mocker):
    spark = mocker.sentinel.spark

    user_data = mocker.sentinel.user_data
    orders = mocker.sentinel.orders
    products = mocker.sentinel.products
    reorders = mocker.sentinel.reorders
    product_history = mocker.Mock(name="product_history")
    reorder_history = mocker.sentinel.reorder_history
    combined_history = mocker.Mock(spec=DataFrame)
    contract = mocker.sentinel.contract

    mocked_join = mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )

    mocked_read = mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.read_parquet",
        return_value=user_data,
    )
    mocked_filter = mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.filtered_orders",
        return_value=orders,
    )
    mocked_seq = mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.parse_seq",
        side_effect=[products, reorders],
    )
    mocked_product_history = mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.build_each_product_in_order_history",
        return_value=product_history,
    )
    mocked_reorder_history = mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.build_each_reorder_history",
        return_value=reorder_history,
    )

    product_history.unionByName.return_value = combined_history

    mock_load_contract = mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.load_contract",
        return_value=contract,
    )
    mock_validate = mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.validate_dataset"
    )
    mock_write = mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.write_parquet"
    )

    manager = mocker.Mock()
    manager.attach_mock(mocked_join, "join")
    manager.attach_mock(mocked_read, "read")
    manager.attach_mock(mocked_filter, "filter")
    manager.attach_mock(mocked_seq, "seq")
    manager.attach_mock(mocked_product_history, "product_history")
    manager.attach_mock(mocked_reorder_history, "reorder_history")
    manager.attach_mock(mock_load_contract, "load_contract")
    manager.attach_mock(mock_validate, "validate")
    manager.attach_mock(mock_write, "write")

    run_product_history_job(
        spark,
        "input",
        "data",
        "output",
        "contracts",
    )

    assert manager.mock_calls == [
        call.join("input", "user_data"),
        call.read("input/user_data", spark),
        call.filter("data", spark),
        call.seq(user_data, "product_ids", "products", True),
        call.seq(user_data, "reorders", "reorders"),
        call.product_history(df=products, path="data", orders=orders, spark=spark),
        call.reorder_history(reorders, orders),
        call.join("contracts", "product_history_data.yaml"),
        call.load_contract("contracts/product_history_data.yaml"),
        call.validate(combined_history, contract=contract),
        call.join("output", "product_history_data"),
        call.write("output/product_history_data", combined_history),
    ]


def test_run_product_history_job_does_not_write_when_validation_fails(
    mocker,
):
    spark = mocker.sentinel.spark

    user_data = mocker.sentinel.user_data
    orders = mocker.sentinel.orders
    products = mocker.sentinel.products
    reorders = mocker.sentinel.reorders
    product_history = mocker.Mock(name="product_history")
    reorder_history = mocker.sentinel.reorder_history
    combined_history = mocker.Mock(spec=DataFrame)
    contract = mocker.sentinel.contract

    mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.read_parquet",
        return_value=user_data,
    )
    mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.filtered_orders",
        return_value=orders,
    )
    mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.parse_seq",
        side_effect=[products, reorders],
    )
    mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.build_each_product_in_order_history",
        return_value=product_history,
    )
    mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.build_each_reorder_history",
        return_value=reorder_history,
    )

    product_history.unionByName.return_value = combined_history

    mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.load_contract",
        return_value=contract,
    )

    report = ValidationReport(dataset_name="product_history_data", results=[])
    mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.validate_dataset",
        side_effect=DataValidationError(report),
    )

    mock_write = mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.write_parquet"
    )

    with pytest.raises(DataValidationError):
        run_product_history_job(
            spark,
            "input",
            "data",
            "output",
            "contracts",
        )

    mock_write.assert_not_called()
