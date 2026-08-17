from pathlib import Path

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.jobs.create_product_history_data_job import (
    run_product_history_job,
)


def test_run_product_history_job_end_to_end(spark, tmp_path):
    user_data = spark.createDataFrame(
        [
            (
                1,
                "10_20_40 10_30 50_30",
                "0_0 1_0 1_0",
                "1 2 3",
                "10 11 12",
                "-1.0 10.0 5.0",
                "1 2 3",
            ),
            (
                2,
                "30 40",
                "0 0",
                "2 3",
                "15 16",
                "-1.0 7.0",
                "1 2",
            ),
        ],
        """
        user_id INT,
        product_ids STRING,
        reorders STRING,
        order_dows STRING,
        order_hours STRING,
        days_since_prior_orders STRING,
        order_numbers STRING
        """,
    )

    orders = spark.createDataFrame(
        [
            (1, 1, "prior"),
            (3, 2, "prior"),
            (2, 1, "train"),
            (4, 2, "test"),
        ],
        """
        order_id INT,
        user_id INT,
        eval_set STRING
        """,
    )

    products = spark.createDataFrame(
        [
            (10, "Apple", 1, 1),
            (20, "Banana", 2, 1),
            (30, "Milk", 3, 2),
            (40, "Bread", 4, 2),
        ],
        """
        product_id INT,
        product_name STRING,
        aisle_id INT,
        department_id INT
        """,
    )

    write_parquet(str(tmp_path / "user_data"), user_data)
    write_parquet(str(tmp_path / "data" / "orders"), orders)
    write_parquet(str(tmp_path / "data" / "products"), products)

    contract_path = (
        Path(__file__).resolve().parents[2] / "src" / "instacart_etl_rnn" / "contracts"
    )

    run_product_history_job(
        spark=spark,
        input_path=tmp_path,
        data_path=tmp_path / "data",
        output_path=tmp_path / "output",
        contract_path=contract_path,
    )

    result = read_parquet(tmp_path / "output" / "product_history_data", spark)

    rows = result.collect()

    map_results = {}

    for row in rows:
        map_results.setdefault(row.user_id, {})[row.product_id] = row

    row = map_results[1]
    assert row[10]["is_ordered_history"] == "1 1"
    assert row[10]["position_in_order_history"] == "1 1"
    assert row[10]["history_order_size"] == "3 2"
    assert row[10]["history_reorder_size"] == "0 1"
    assert row[10]["order_dows"] == "1 2 3"
    assert row[10]["order_hours"] == "10 11 12"
    assert row[10]["days_since_prior_orders"] == "-1.0 10.0 5.0"
    assert row[10]["order_numbers"] == "1 2 3"
    assert row[10]["aisle_id"] == 1
    assert row[10]["department_id"] == 1
    assert row[10]["product_name"] == "Apple"
    assert row[10]["label"] == 0

    assert row[20]["is_ordered_history"] == "1 0"
    assert row[20]["position_in_order_history"] == "2 0"
    assert row[20]["aisle_id"] == 2
    assert row[20]["department_id"] == 1
    assert row[20]["product_name"] == "Banana"
    assert row[20]["label"] == 0

    assert row[30]["is_ordered_history"] == "0 1"
    assert row[30]["position_in_order_history"] == "0 2"
    assert row[30]["aisle_id"] == 3
    assert row[30]["department_id"] == 2
    assert row[30]["product_name"] == "Milk"
    assert row[30]["label"] == 1

    assert row[40]["is_ordered_history"] == "1 0"
    assert row[40]["position_in_order_history"] == "3 0"
    assert row[40]["aisle_id"] == 4
    assert row[40]["department_id"] == 2
    assert row[40]["product_name"] == "Bread"
    assert row[40]["label"] == 0

    assert row[0]["is_ordered_history"] == "1 0"
    assert row[0]["position_in_order_history"] == "0 0"
    assert row[0]["aisle_id"] == 0
    assert row[0]["department_id"] == 0
    assert row[0]["product_name"] == ""
    assert row[0]["label"] == 0

    row = map_results[2]
    assert row[30]["is_ordered_history"] == "1"
    assert row[30]["position_in_order_history"] == "1"
    assert row[30]["history_order_size"] == "1"
    assert row[30]["history_reorder_size"] == "0"
    assert row[30]["order_dows"] == "2 3"
    assert row[30]["order_hours"] == "15 16"
    assert row[30]["days_since_prior_orders"] == "-1.0 7.0"
    assert row[30]["order_numbers"] == "1 2"
    assert row[30]["aisle_id"] == 3
    assert row[30]["department_id"] == 2
    assert row[30]["product_name"] == "Milk"
    assert row[30]["label"] == -1

    assert row[0]["is_ordered_history"] == "1"
    assert row[0]["position_in_order_history"] == "0"
    assert row[0]["aisle_id"] == 0
    assert row[0]["department_id"] == 0
    assert row[0]["product_name"] == ""
    assert row[0]["label"] == -1
