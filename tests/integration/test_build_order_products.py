import pytest

from instacart_etl_rnn.common.io import read_parquet
from instacart_etl_rnn.silver.create_order_products import build_order_products
from instacart_etl_rnn.validation.exceptions import DataValidationError


def test_build_order_products_integration(
    spark, tmp_path, order_products_silver_contract
):
    products = spark.createDataFrame(
        [(5, "a"), (10, "b")], ("product_id", "product_name")
    )
    orders = spark.createDataFrame(
        [(1, 10, "prior", 1, None), (2, 10, "train", 2, 20.0)],
        ["order_id", "user_id", "eval_set", "order_number", "days_since_prior_order"],
    )
    order_prior = spark.createDataFrame(
        [(1, 5, 1, 0)], ["order_id", "product_id", "add_to_cart_order", "reordered"]
    )
    order_train = spark.createDataFrame(
        [(2, 10, 2, 1)], ["order_id", "product_id", "add_to_cart_order", "reordered"]
    )

    input_path = tmp_path
    output_path = tmp_path / "order_products"

    products.write.parquet(str(input_path / "products"))
    orders.write.parquet(str(input_path / "orders"))
    order_prior.write.parquet(str(input_path / "order_products__prior"))
    order_train.write.parquet(str(input_path / "order_products__train"))

    build_order_products(
        spark=spark,
        input_path=input_path,
        output_path=output_path,
        contract_path=order_products_silver_contract,
    )

    order_products = read_parquet(output_path, spark)

    rows = order_products.orderBy("order_id").collect()

    assert len(rows) == 2

    assert rows[0]["order_id"] == 1
    assert rows[0]["eval_set"] == "prior"
    assert rows[0]["order_number"] == 1
    assert rows[0]["days_since_prior_order"] == -1.0
    assert rows[0]["reordered"] == 0
    assert rows[0]["product_id"] == 5
    assert rows[0]["product_name"] == "a"

    assert rows[1]["order_id"] == 2
    assert rows[1]["eval_set"] == "train"
    assert rows[1]["order_number"] == 2
    assert rows[1]["days_since_prior_order"] == 20.0
    assert rows[1]["reordered"] == 1
    assert rows[1]["product_id"] == 10
    assert rows[1]["product_name"] == "b"


def test_build_order_products_does_not_write_invalid_silver_dataset(
    spark, tmp_path, order_products_silver_contract
):
    orders_df = spark.createDataFrame(
        [
            (1, 100, "prior", 1, None),
            (2, 100, "train", 2, 5.0),
        ],
        """
        order_id INT,
        user_id INT,
        eval_set STRING,
        order_number INT,
        days_since_prior_order DOUBLE
        """,
    )

    products_df = spark.createDataFrame(
        [
            (10, "Apple"),
            (20, "Banana"),
            (30, "Milk"),
        ],
        """
        product_id INT,
        product_name STRING
        """,
    )

    bad_prior_df = spark.createDataFrame(
        [
            (1, 999, 1, 0),
        ],
        """
        order_id INT,
        product_id INT,
        add_to_cart_order INT,
        reordered INT
        """,
    )

    order_products_train_df = spark.createDataFrame(
        [
            (2, 10, 1, 1),
            (2, 30, 2, 0),
        ],
        """
        order_id INT,
        product_id INT,
        add_to_cart_order INT,
        reordered INT
        """,
    )

    bronze_path = tmp_path / "bronze"

    orders_df.write.parquet(str(bronze_path / "orders"))

    products_df.write.parquet(str(bronze_path / "products"))

    bad_prior_df.write.parquet(str(bronze_path / "order_products__prior"))

    order_products_train_df.write.parquet(str(bronze_path / "order_products__train"))

    output_path = tmp_path / "silver" / "order_products"

    with pytest.raises(DataValidationError):
        build_order_products(
            spark=spark,
            input_path=str(bronze_path),
            output_path=str(output_path),
            contract_path=str(order_products_silver_contract),
        )

    assert not output_path.exists()
