from unittest.mock import call

from instacart_etl_rnn.silver.create_user_data import read_input_datasets


def test_read_input_datasets(mocker):
    df = mocker.sentinel.df
    contract = mocker.sentinel.contract
    spark = mocker.sentinel.spark

    mocked_join = mocker.patch(
        "instacart_etl_rnn.silver.create_user_data.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )

    mocked_read = mocker.patch(
        "instacart_etl_rnn.silver.create_user_data.read_parquet", return_value=df
    )

    mocked_load = mocker.patch(
        "instacart_etl_rnn.silver.create_user_data.load_contract", return_value=contract
    )

    mocked_validate = mocker.patch(
        "instacart_etl_rnn.silver.create_user_data.validate_dataset"
    )

    manager = mocker.Mock()
    manager.attach_mock(mocked_join, "join")
    manager.attach_mock(mocked_read, "read")
    manager.attach_mock(mocked_load, "load")
    manager.attach_mock(mocked_validate, "validate")

    result = read_input_datasets(
        spark=spark, input_path="silver", contract_path="contracts"
    )

    assert result is df

    assert manager.mock_calls == [
        call.join("silver", "order_products"),
        call.read("silver/order_products", spark),
        call.join("contracts", "order_products_silver.yaml"),
        call.load("contracts/order_products_silver.yaml"),
        call.validate(df, contract=contract),
    ]
