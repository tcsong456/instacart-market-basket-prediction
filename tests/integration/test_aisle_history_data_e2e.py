from pathlib import Path

from instacart_etl_rnn.jobs.create_aisle_history_data_job import (
    run_aisle_history_job,
)

CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "instacart_etl_rnn" / "contracts"
)


def test_run_aisle_history_job_end_to_end(spark, tmp_path):
    user_data = spark.createDataFrame(
        [
            (
                1,
                "10_10_20 20_30 10_30",
                "1 2 3",
                "10 11 12",
                "-1.0 5.0 7.0",
                "1 2 3",
                "train",
            ),
            (
                2,
                "40_40_50",
                "4",
                "15",
                "-1.0",
                "1",
                "prior",
            ),
        ],
        """
        user_id INT,
        aisle_ids STRING,
        order_dows STRING,
        order_hours STRING,
        days_since_prior_orders STRING,
        order_numbers STRING,
        eval_set STRING
        """,
    )

    products = spark.createDataFrame(
        [
            (100, "A", 10, 1),
            (101, "B", 10, 1),
            (200, "C", 20, 2),
            (300, "D", 30, 3),
            (400, "E", 40, 4),
            (401, "F", 40, 4),
            (500, "G", 50, 5),
        ],
        """
        product_id INT,
        product_name STRING,
        aisle_id INT,
        department_id INT
        """,
    )

    input_path = tmp_path / "silver"
    data_path = tmp_path / "bronze"
    output_path = tmp_path / "gold"

    user_data.write.parquet(str(input_path / "user_data_train"))

    products.write.parquet(str(data_path / "products"))

    run_aisle_history_job(
        spark=spark,
        input_path=input_path,
        data_path=data_path,
        output_path=output_path,
        contract_path=CONTRACT_PATH,
        mode="train",
    )

    result = spark.read.parquet(str(output_path / "aisle_history_data_train"))

    actual = {
        (
            row.user_id,
            row.aisle_id,
        ): row.asDict(recursive=True)
        for row in result.collect()
    }

    assert set(actual) == {
        (1, 10),
        (1, 20),
        (1, 30),
        (2, 40),
        (2, 50),
    }

    assert actual[(1, 10)] == {
        "user_id": 1,
        "aisle_id": 10,
        "department_id": 1,
        "eval_set": "train",
        "is_ordered_history": "1 0",
        "position_in_order": "1 0",
        "num_products_from_aisle": "2 0",
        "aisle_history_size": "2 2",
        "order_dows": "1 2 3",
        "order_hours": "10 11 12",
        "days_since_prior_orders": "-1.0 5.0 7.0",
        "order_numbers": "1 2 3",
    }

    assert actual[(1, 20)] == {
        "user_id": 1,
        "aisle_id": 20,
        "department_id": 2,
        "eval_set": "train",
        "is_ordered_history": "1 1",
        "position_in_order": "2 1",
        "num_products_from_aisle": "1 1",
        "aisle_history_size": "2 2",
        "order_dows": "1 2 3",
        "order_hours": "10 11 12",
        "days_since_prior_orders": "-1.0 5.0 7.0",
        "order_numbers": "1 2 3",
    }

    assert actual[(1, 30)] == {
        "user_id": 1,
        "aisle_id": 30,
        "department_id": 3,
        "eval_set": "train",
        "is_ordered_history": "0 1",
        "position_in_order": "0 2",
        "num_products_from_aisle": "0 1",
        "aisle_history_size": "2 2",
        "order_dows": "1 2 3",
        "order_hours": "10 11 12",
        "days_since_prior_orders": "-1.0 5.0 7.0",
        "order_numbers": "1 2 3",
    }

    assert actual[(2, 40)] == {
        "user_id": 2,
        "aisle_id": 40,
        "department_id": 4,
        "eval_set": "prior",
        "is_ordered_history": "1",
        "position_in_order": "1",
        "num_products_from_aisle": "2",
        "aisle_history_size": "2",
        "order_dows": "4",
        "order_hours": "15",
        "days_since_prior_orders": "-1.0",
        "order_numbers": "1",
    }

    assert actual[(2, 50)] == {
        "user_id": 2,
        "aisle_id": 50,
        "department_id": 5,
        "eval_set": "prior",
        "is_ordered_history": "1",
        "position_in_order": "2",
        "num_products_from_aisle": "1",
        "aisle_history_size": "2",
        "order_dows": "4",
        "order_hours": "15",
        "days_since_prior_orders": "-1.0",
        "order_numbers": "1",
    }
