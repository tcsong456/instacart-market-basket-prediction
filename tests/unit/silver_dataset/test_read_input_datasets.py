from unittest.mock import call

from instacart_etl_rnn.silver.create_order_products import (
    build_order_products,
    read_input_datasets,
)


def test_read_input_datasets(mocker):
    spark = mocker.sentinel.spark

    orders_contract = {}
    products_contract = {
        "relationships": [
            {"parent": {"dataset": "aisles"}},
            {"parent": {"dataset": "departments"}},
        ]
    }
    order_products_contract = {
        "relationships": [
            {"parent": {"dataset": "orders"}},
            {"parent": {"dataset": "products"}},
        ]
    }

    products_df = mocker.sentinel.products_df
    orders_df = mocker.sentinel.orders_df
    order_train = mocker.sentinel.order_train
    order_prior = mocker.sentinel.order_prior
    aisles_df = mocker.sentinel.aisles_df
    departments_df = mocker.sentinel.departments_df

    mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )

    mocked_load_contract = mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.load_contract",
        side_effect=[
            products_contract,
            orders_contract,
            order_products_contract,
            order_products_contract,
        ],
    )

    mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.read_parquet",
        side_effect=[
            products_df,
            aisles_df,
            departments_df,
            orders_df,
            order_prior,
            orders_df,
            products_df,
            order_train,
            orders_df,
            products_df,
        ],
    )

    mocked_validate = mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.validate_dataset",
    )

    result = read_input_datasets(
        spark=spark,
        input_path="gs://bronze-dataset",
        contract_path="gs://contracts",
        validation=True,
    )

    assert result == {
        "products": products_df,
        "orders": orders_df,
        "order_products__prior": order_prior,
        "order_products__train": order_train,
    }

    assert mocked_load_contract.call_args_list == [
        call("gs://contracts/products.yaml"),
        call("gs://contracts/orders.yaml"),
        call("gs://contracts/order_products.yaml"),
        call("gs://contracts/order_products.yaml"),
    ]

    assert mocked_validate.call_args_list == [
        call(
            products_df,
            contract=products_contract,
            reference_datasets={"aisles": aisles_df, "departments": departments_df},
        ),
        call(
            orders_df,
            contract=orders_contract,
            reference_datasets={},
        ),
        call(
            order_prior,
            contract=order_products_contract,
            reference_datasets={
                "orders": orders_df,
                "products": products_df,
            },
        ),
        call(
            order_train,
            contract=order_products_contract,
            reference_datasets={
                "orders": orders_df,
                "products": products_df,
            },
        ),
    ]


def test_read_input_datasets_skips_validation_when_disabled(
    mocker,
):
    spark = mocker.sentinel.spark

    products_df = mocker.sentinel.products_df
    orders_df = mocker.sentinel.orders_df
    order_train = mocker.sentinel.order_train
    order_prior = mocker.sentinel.order_prior

    mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )

    mocked_read_parquet = mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.read_parquet",
        side_effect=[
            products_df,
            orders_df,
            order_prior,
            order_train,
        ],
    )

    mocked_load_contract = mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.load_contract",
    )

    mocked_validate = mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.validate_dataset",
    )

    result = read_input_datasets(
        spark=spark,
        input_path="bronze",
        contract_path="contracts",
        validation=False,
    )

    assert result == {
        "products": products_df,
        "orders": orders_df,
        "order_products__prior": order_prior,
        "order_products__train": order_train,
    }

    mocked_load_contract.assert_not_called()
    mocked_validate.assert_not_called()

    assert mocked_read_parquet.call_count == 4


def test_build_order_products_builds_and_writes_dataset(
    spark,
    mocker,
):
    prior_df = spark.createDataFrame(
        [
            (1, 10, 1, 0),
        ],
        [
            "order_id",
            "product_id",
            "add_to_cart_order",
            "reordered",
        ],
    )

    train_df = spark.createDataFrame(
        [
            (2, 20, 1, 1),
        ],
        [
            "order_id",
            "product_id",
            "add_to_cart_order",
            "reordered",
        ],
    )

    orders_df = spark.createDataFrame(
        [
            (1, 100, "prior", 1, None),
            (2, 200, "train", 2, 5.0),
        ],
        [
            "order_id",
            "user_id",
            "eval_set",
            "order_number",
            "days_since_prior_order",
        ],
    )

    products_df = spark.createDataFrame(
        [
            (10, "Apple"),
            (20, "Banana"),
        ],
        [
            "product_id",
            "product_name",
        ],
    )

    mocked_read = mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.read_input_datasets",
        return_value={
            "order_products__prior": prior_df,
            "order_products__train": train_df,
            "orders": orders_df,
            "products": products_df,
        },
    )

    mocked_write = mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.write_parquet"
    )

    build_order_products(
        spark=spark,
        input_path="bronze",
        contract_path="contracts",
        output_path="silver/order_products",
        validation=True,
    )

    captured_df = mocked_write.call_args.kwargs["df"]

    mocked_read.assert_called_once_with(
        spark=spark,
        input_path="bronze",
        contract_path="contracts",
        validation=True,
    )

    mocked_write.assert_called_once()

    result = captured_df.orderBy("order_id").collect()

    assert result[0]["order_id"] == 1
    assert result[0]["product_id"] == 10
    assert result[0]["user_id"] == 100
    assert result[0]["product_name"] == "Apple"
    assert result[0]["days_since_prior_order"] == -1

    assert result[1]["order_id"] == 2
    assert result[1]["product_id"] == 20
    assert result[1]["user_id"] == 200
    assert result[1]["product_name"] == "Banana"
    assert result[1]["days_since_prior_order"] == 5.0


def test_build_order_products_passes_validation_flag(
    spark,
    mocker,
):
    empty_df = spark.createDataFrame(
        [],
        "order_id INT, product_id INT, add_to_cart_order INT, reordered INT",
    )

    orders_df = spark.createDataFrame(
        [],
        "order_id INT, days_since_prior_order DOUBLE",
    )

    products_df = spark.createDataFrame(
        [],
        "product_id INT",
    )

    mocked_read = mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.read_input_datasets",
        return_value={
            "order_products__prior": empty_df,
            "order_products__train": empty_df,
            "orders": orders_df,
            "products": products_df,
        },
    )

    mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.write_parquet",
    )

    build_order_products(
        spark=spark,
        input_path="bronze",
        contract_path="contracts",
        output_path="silver/order_products",
        validation=False,
    )

    mocked_read.assert_called_once_with(
        spark=spark,
        input_path="bronze",
        contract_path="contracts",
        validation=False,
    )
