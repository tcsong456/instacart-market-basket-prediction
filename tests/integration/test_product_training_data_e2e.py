from pathlib import Path

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.jobs.create_product_training_data_job import (
    run_product_training_data_job,
)

CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "instacart_etl_rnn" / "contracts"
)


def test_run_product_training_data_job_end_to_end(spark, tmp_path):
    product_history_data = spark.createDataFrame(
        [
            (
                1,
                10,
                0,
                1,
                1,
                "Apple Juice",
                "train",
                "1 0",
                "1 0",
                "2 1",
                "0 0",
                "0 1 2",
                "10 11 12",
                "-1.0 5.0 7.0",
                "1 2 3",
            ),
            (
                1,
                20,
                1,
                2,
                1,
                "Banana",
                "train",
                "0 1",
                "0 1",
                "2 1",
                "0 0",
                "0 1 2",
                "10 11 12",
                "-1.0 5.0 7.0",
                "1 2 3",
            ),
            (
                1,
                0,
                0,
                0,
                0,
                "",
                "train",
                "1 0",
                "0 0",
                "2 1",
                "0 0",
                "0 1 2",
                "10 11 12",
                "-1.0 5.0 7.0",
                "1 2 3",
            ),
        ],
        """
        user_id INT,
        product_id INT,
        label INT,
        aisle_id INT,
        department_id INT,
        product_name STRING,
        eval_set STRING,
        is_ordered_history STRING,
        position_in_order_history STRING,
        history_order_size STRING,
        history_reorder_size STRING,
        order_dows STRING,
        order_hours STRING,
        days_since_prior_orders STRING,
        order_numbers STRING
        """,
    )

    products = spark.createDataFrame(
        [
            (10, "Apple Juice", 1, 1),
            (20, "Banana", 2, 1),
            (30, "Milk", 3, 2),
        ],
        """
        product_id INT,
        product_name STRING,
        aisle_id INT,
        department_id INT
        """,
    )

    input_path = tmp_path / "gold"
    raw_path = tmp_path / "bronze"
    output_path = tmp_path / "training"

    write_parquet(
        str(input_path / "product_history_data_train"),
        product_history_data,
    )
    write_parquet(str(raw_path / "products"), products)

    run_product_training_data_job(
        spark=spark,
        raw_path=str(raw_path),
        input_path=str(input_path),
        output_path=str(output_path),
        contract_path=str(CONTRACT_PATH),
        mode="train",
        min_word_freq=1,
        product_name_length=30,
        encode_length=100,
    )

    result = read_parquet(
        output_path / "product_training_data_train",
        spark,
    )

    actual = {
        (row.user_id, row.product_id): row.asDict(recursive=True)
        for row in result.collect()
    }

    assert set(actual) == {(1, 10), (1, 20), (1, 0)}

    apple = actual[(1, 10)]
    assert apple["eval_set"] == "train"
    assert apple["label"] == 0
    assert apple["history_length"] == 2
    assert len(apple["product_name_encoded"]) == 30
    assert len(apple["is_ordered_history"]) == 100
    assert apple["is_ordered_history"][:2] == [1, 0]
    assert apple["product_name_length"] >= 1
    assert apple["order_numbers"][:3] == [1, 2, 3]

    banana = actual[(1, 20)]
    assert banana["label"] == 1
    assert banana["is_ordered_history"][:2] == [0, 1]

    none_product = actual[(1, 0)]
    assert none_product["product_name_length"] == 0
    assert none_product["product_name_encoded"] == [0] * 30
