from instacart_etl_rnn.common.io import read_parquet
from instacart_etl_rnn.silver.create_order_products import build_order_products


def test_build_order_products_integration(spark, tmp_path):
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
    contract_path = "unused"

    products.write.parquet(str(input_path / "products"))
    orders.write.parquet(str(input_path / "orders"))
    order_prior.write.parquet(str(input_path / "order_products__prior"))
    order_train.write.parquet(str(input_path / "order_products__train"))

    build_order_products(
        spark=spark,
        input_path=input_path,
        output_path=output_path,
        contract_path=contract_path,
        validation=False,
    )

    order_products = read_parquet(output_path, spark)

    rows = order_products.orderBy("order_id").collect()

    assert len(rows) == 2

    assert rows[0]["order_id"] == 1
    assert rows[0]["eval_set"] == "prior"
    assert rows[0]["order_number"] == 1
    assert rows[0]["days_since_prior_order"] == -1
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
