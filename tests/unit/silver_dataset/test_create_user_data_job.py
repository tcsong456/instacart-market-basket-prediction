from unittest.mock import call

import pytest

from instacart_etl_rnn.jobs.create_user_data_job import run_user_data_job


def test_run_user_data_job(mocker):
    order_products = mocker.sentinel.order_products
    order_group_data = mocker.sentinel.order_group_data
    user_data = mocker.sentinel.user_data
    spark = mocker.sentinel.spark

    mocked_read = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_data_job.read_input_datasets",
        return_value=order_products,
    )

    mockded_build_order = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_data_job.build_order_level_data",
        return_value=order_group_data,
    )

    mocked_build_user = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_data_job.build_user_level_data",
        return_value=user_data,
    )

    mocked_write = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_data_job.write_parquet",
    )

    manager = mocker.Mock()
    manager.attach_mock(mocked_read, "read")
    manager.attach_mock(mockded_build_order, "build_order")
    manager.attach_mock(mocked_build_user, "build_user")
    manager.attach_mock(mocked_write, "write")

    run_user_data_job(spark=spark, path="silver", contract_path="contracts")

    assert manager.mock_calls == [
        call.read(spark=spark, input_path="silver", contract_path="contracts"),
        call.build_order(order_products),
        call.build_user(order_group_data),
        call.write("silver", user_data),
    ]


def test_run_user_data_job_does_not_write_when_user_build_fails(
    mocker,
):
    spark = mocker.sentinel.spark

    order_products_df = mocker.sentinel.order_products_df
    order_group_df = mocker.sentinel.order_group_df

    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_data_job.read_input_datasets",
        return_value=order_products_df,
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_data_job.build_order_level_data",
        return_value=order_group_df,
    )

    mocked_build_user = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_data_job.build_user_level_data",
        side_effect=RuntimeError("user build failed"),
    )

    mocked_write = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_data_job.write_parquet",
    )

    with pytest.raises(
        RuntimeError,
        match="user build failed",
    ):
        run_user_data_job(
            spark=spark,
            path="silver",
            contract_path="contracts",
        )

    mocked_build_user.assert_called_once_with(
        order_group_df,
    )

    mocked_write.assert_not_called()
