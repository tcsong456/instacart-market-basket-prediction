import pytest

from instacart_etl_rnn.silver.create_order_products import build_order_products
from instacart_etl_rnn.validation.exceptions import DataValidationError
from instacart_etl_rnn.validation.models import ValidationReport


def test_build_order_products_builds_validates_and_writes(
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
            "products": products_df,
            "orders": orders_df,
            "order_products__prior": prior_df,
            "order_products__train": train_df,
        },
    )

    mocked_join = mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )

    contract = mocker.sentinel.contract
    mocked_load_contract = mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.load_contract",
        return_value=contract,
    )

    mocked_validate = mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.validate_dataset"
    )

    mocked_write = mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.write_parquet"
    )

    build_order_products(
        spark=mocker.sentinel.spark,
        input_path="bronze",
        output_path="silver/order_products",
        contract_path="contracts",
    )

    mocked_read.assert_called_once_with(
        spark=mocker.sentinel.spark, input_path="bronze"
    )

    mocked_join.assert_called_once_with("contracts", "order_products_silver.yaml")

    mocked_load_contract.assert_called_once_with("contracts/order_products_silver.yaml")

    df = mocked_validate.call_args.args[0]

    assert mocked_validate.call_args.kwargs == {"contract": contract}

    mocked_write.assert_called_once_with(path="silver/order_products", df=df)

    written_df = mocked_write.call_args.kwargs["df"]
    results = written_df.orderBy("order_id").collect()

    assert len(results) == 2

    assert results[0]["order_id"] == 1
    assert results[0]["product_id"] == 10
    assert results[0]["add_to_cart_order"] == 1
    assert results[0]["reordered"] == 0
    assert results[0]["user_id"] == 100
    assert results[0]["eval_set"] == "prior"
    assert results[0]["order_number"] == 1
    assert results[0]["days_since_prior_order"] == -1.0

    assert results[1]["order_id"] == 2
    assert results[1]["product_id"] == 20
    assert results[1]["add_to_cart_order"] == 1
    assert results[1]["reordered"] == 1
    assert results[1]["user_id"] == 200
    assert results[1]["eval_set"] == "train"
    assert results[1]["order_number"] == 2
    assert results[1]["days_since_prior_order"] == 5.0

    assert written_df is df


def test_build_order_products_does_not_write_when_validation_fails(
    spark,
    mocker,
):
    prior_df = spark.createDataFrame(
        [(1, 10, 1, 0)],
        [
            "order_id",
            "product_id",
            "add_to_cart_order",
            "reordered",
        ],
    )

    train_df = spark.createDataFrame(
        [],
        prior_df.schema,
    )

    orders_df = spark.createDataFrame(
        [(1, 100, "prior", 1, None)],
        "order_id INT, "
        "user_id INT, "
        "eval_set STRING, "
        "order_number INT, "
        "days_since_prior_order DOUBLE",
    )

    products_df = spark.createDataFrame(
        [(10, "Apple")],
        ["product_id", "product_name"],
    )

    mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.read_input_datasets",
        return_value={
            "order_products__prior": prior_df,
            "order_products__train": train_df,
            "orders": orders_df,
            "products": products_df,
        },
    )

    mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.join_path",
        return_value="contracts/order_products_silver.yaml",
    )

    mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.load_contract",
        return_value=mocker.sentinel.contract,
    )

    report = ValidationReport(dataset_name="order_products", results=[])
    mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.validate_dataset",
        side_effect=DataValidationError(report),
    )

    mocked_write = mocker.patch(
        "instacart_etl_rnn.silver.create_order_products.write_parquet",
    )

    with pytest.raises(DataValidationError):
        build_order_products(
            spark=spark,
            input_path="bronze",
            output_path="silver/order_products",
            contract_path="contracts",
        )

    mocked_write.assert_not_called()
