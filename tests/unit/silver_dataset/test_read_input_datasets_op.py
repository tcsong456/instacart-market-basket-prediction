from unittest.mock import call

from instacart_etl_rnn.silver.create_order_products import (
    read_input_datasets,
)


def test_read_input_datasets(mocker, spark):
    mock_datasets = ["products", "order_products__prior", "order_products__train"]

    mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.DATASETS",
        new=mock_datasets,
    )

    mock_join_path = mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )

    products_df = mocker.sentinel.products_df
    prior_df = mocker.sentinel.prior_df
    train_df = mocker.sentinel.train_df
    orders_df = mocker.sentinel.orders_df

    mock_read_parquet = mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.read_parquet",
        side_effect=[
            products_df,
            prior_df,
            train_df,
            orders_df,
        ],
    )

    result = read_input_datasets(
        spark=spark,
        input_path="bronze",
        order_path="simulation/initial",
    )

    assert result == {
        "products": products_df,
        "order_products__prior": prior_df,
        "order_products__train": train_df,
        "orders": orders_df,
    }

    assert mock_join_path.call_args_list == [
        call("bronze", "products"),
        call("bronze", "order_products__prior"),
        call("bronze", "order_products__train"),
        call("simulation/initial", "available_orders"),
    ]

    assert mock_read_parquet.call_args_list == [
        call("bronze/products", spark),
        call("bronze/order_products__prior", spark),
        call("bronze/order_products__train", spark),
        call("simulation/initial/available_orders", spark),
    ]
