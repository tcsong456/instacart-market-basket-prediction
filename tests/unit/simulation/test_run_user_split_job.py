from unittest.mock import call

import pytest
from pyspark import StorageLevel
from pyspark.sql import DataFrame

from instacart_etl_rnn.jobs.create_user_split_data_job import run_user_split_job
from instacart_etl_rnn.validation.exceptions import DataValidationError
from instacart_etl_rnn.validation.models import ValidationReport


def test_run_user_split_job_runs_pipeline_in_order(
    mocker,
):
    spark = mocker.Mock(name="spark")

    orders = mocker.Mock(spec=DataFrame)
    filtered_orders = mocker.Mock(spec=DataFrame)
    orders.filter.return_value = filtered_orders
    user_split = mocker.sentinel.user_split
    order_split = mocker.sentinel.order_split
    available_orders = mocker.Mock(spec=DataFrame)
    contract = mocker.sentinel.contract

    read_parquet = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.read_parquet",
        return_value=orders,
    )
    join_path = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )
    build_user_simulation_split = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.build_user_simulation_split",
        return_value=user_split,
    )
    build_order_simulation_split = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.build_order_simulation_split",
        return_value=order_split,
    )
    add_order_role = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.add_order_role",
        return_value=available_orders,
    )
    load_contract = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.load_contract",
        return_value=contract,
    )
    validate_dataset = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.validate_dataset",
    )
    write_parquet = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.write_parquet",
    )

    manager = mocker.Mock()
    manager.attach_mock(read_parquet, "read")
    manager.attach_mock(join_path, "join")
    manager.attach_mock(build_user_simulation_split, "build_user")
    manager.attach_mock(build_order_simulation_split, "build_order")
    manager.attach_mock(add_order_role, "add_role")
    manager.attach_mock(load_contract, "load")
    manager.attach_mock(validate_dataset, "validate")
    manager.attach_mock(write_parquet, "write")

    run_user_split_job(
        spark=spark,
        input_path="input",
        output_path="output",
        contract_path="contracts",
        period="t1",
    )

    assert manager.mock_calls == [
        call.join("input", "orders"),
        call.read("input/orders", spark),
        call.build_user(
            orders=filtered_orders,
        ),
        call.build_order(orders=filtered_orders, user_split=user_split),
        call.add_role(order_split, "t1"),
        call.join("contracts", "user_split_data.yaml"),
        call.load("contracts/user_split_data.yaml"),
        call.validate(available_orders, contract=contract),
        call.join("output", "available_order"),
        call.write("output/available_order", available_orders),
    ]

    available_orders.persist.assert_called_once_with(
        storageLevel=StorageLevel.MEMORY_AND_DISK,
    )

    available_orders.unpersist.assert_called_once()


def test_run_user_split_job_unpersists_when_validation_fails(
    mocker,
):
    spark = mocker.Mock()

    orders = mocker.MagicMock()
    filtered_orders = mocker.MagicMock()
    orders.filter.return_value = filtered_orders

    user_split = mocker.Mock()
    order_split = mocker.Mock()
    available_orders = mocker.MagicMock()
    contract = mocker.Mock()

    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.read_parquet",
        return_value=orders,
    )
    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )
    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.build_user_simulation_split",
        return_value=user_split,
    )
    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.build_order_simulation_split",
        return_value=order_split,
    )
    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.add_order_role",
        return_value=available_orders,
    )
    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.load_contract",
        return_value=contract,
    )

    report = ValidationReport(dataset_name="available_orders", results=[])
    validation_error = DataValidationError(report)
    validate_dataset = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.validate_dataset",
        side_effect=validation_error,
    )

    write_parquet = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.write_parquet",
    )

    with pytest.raises(DataValidationError):
        run_user_split_job(
            spark=spark,
            input_path="input",
            output_path="output",
            contract_path="contracts",
            period="t1",
        )

    validate_dataset.assert_called_once_with(
        available_orders,
        contract=contract,
    )

    write_parquet.assert_not_called()

    available_orders.unpersist.assert_called_once()
