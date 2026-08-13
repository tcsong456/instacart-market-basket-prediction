from unittest.mock import call

from instacart_etl_rnn.silver.create_order_products import (
    read_input_datasets,
)


def test_read_input_datasets_reads_all_datasets(
    mocker,
):
    spark = mocker.sentinel.spark

    orders_df = mocker.sentinel.orders_df
    products_df = mocker.sentinel.products_df
    prior_df = mocker.sentinel.prior_df
    train_df = mocker.sentinel.train_df

    mocked_join_path = mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )

    mocked_read_parquet = mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.read_parquet",
        side_effect=[
            products_df,
            orders_df,
            prior_df,
            train_df,
        ],
    )

    result = read_input_datasets(
        spark=spark,
        input_path="bronze",
    )

    assert result == {
        "orders": orders_df,
        "products": products_df,
        "order_products__prior": prior_df,
        "order_products__train": train_df,
    }

    assert mocked_join_path.call_args_list == [
        call("bronze", "products"),
        call("bronze", "orders"),
        call("bronze", "order_products__prior"),
        call("bronze", "order_products__train"),
    ]

    assert mocked_read_parquet.call_args_list == [
        call("bronze/products", spark),
        call("bronze/orders", spark),
        call("bronze/order_products__prior", spark),
        call("bronze/order_products__train", spark),
    ]
