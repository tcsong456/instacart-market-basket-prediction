from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.jobs.create_user_data_job import run_user_data_job


def test_user_data_complies_with_contract(spark, user_data_contract_path, tmp_path):
    order_products = spark.createDataFrame(
        [
            (10, 1, 1, 0, 1, "prior", 1, 0, 10, -1.0, "Apple", 24, 4),
            (20, 1, 2, 0, 1, "prior", 1, 0, 10, -1.0, "Banana", 28, 5),
            (
                10,
                2,
                1,
                1,
                1,
                "train",
                2,
                1,
                12,
                5.0,
                "Apple",
                24,
                4,
            ),
            (30, 2, 2, 0, 1, "train", 2, 1, 12, 5.0, "Milk", 84, 16),
            (40, 3, 1, 0, 2, "prior", 1, 2, 15, -1.0, "Bread", 112, 3),
            (40, 4, 1, 1, 2, "test", 2, 3, 16, 7.0, "Bread", 112, 3),
            (50, 4, 2, 0, 2, "test", 2, 3, 16, 7.0, "Eggs", 86, 16),
        ],
        """
        product_id INT,
        order_id INT,
        add_to_cart_order INT,
        reordered INT,
        user_id INT,
        eval_set STRING,
        order_number INT,
        order_dow INT,
        order_hour_of_day INT,
        days_since_prior_order DOUBLE,
        product_name STRING,
        aisle_id INT,
        department_id INT
        """,
    )

    write_parquet(tmp_path / "order_products_train", order_products)

    run_user_data_job(
        spark,
        tmp_path,
        user_data_contract_path,
        mode="train",
    )

    result = read_parquet(tmp_path / "user_data_train", spark)

    actual = {row.user_id: row.asDict(recursive=True) for row in result.collect()}

    expected = {
        1: {
            "user_id": 1,
            "product_ids": "10_20 10_30",
            "reorders": "0_0 1_0",
            "order_dows": "0 1",
            "order_hours": "10 12",
            "days_since_prior_orders": "-1.0 5.0",
            "order_numbers": "1 2",
            "eval_set": "train",
            "order_ids": "1 2",
            "aisle_ids": "24_28 24_84",
            "department_ids": "4_5 4_16",
        },
        2: {
            "user_id": 2,
            "product_ids": "40 40_50",
            "reorders": "0 1_0",
            "order_dows": "2 3",
            "order_hours": "15 16",
            "days_since_prior_orders": "-1.0 7.0",
            "order_numbers": "1 2",
            "eval_set": "test",
            "order_ids": "3 4",
            "aisle_ids": "112 112_86",
            "department_ids": "3 3_16",
        },
    }

    assert actual == expected
