from pathlib import Path

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.jobs.create_aisle_training_data_job import (
    run_aisle_training_data_job,
)

CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "instacart_etl_rnn" / "contracts"
)


def test_run_aisle_training_data_job_end_to_end(spark, tmp_path):
    aisle_history_data = spark.createDataFrame(
        [
            (
                1,
                10,
                1,
                "train",
                "1 0",
                "1 0",
                "2 0",
                "2 2",
                "0 1 2",
                "10 11 12",
                "-1.0 5.0 7.0",
                "1 2 3",
            ),
            (
                1,
                20,
                2,
                "train",
                "0 1",
                "0 1",
                "0 1",
                "2 2",
                "0 1 2",
                "10 11 12",
                "-1.0 5.0 7.0",
                "1 2 3",
            ),
        ],
        """
        user_id INT,
        aisle_id INT,
        department_id INT,
        eval_set STRING,
        is_ordered_history STRING,
        position_in_order STRING,
        num_products_from_aisle STRING,
        aisle_history_size STRING,
        order_dows STRING,
        order_hours STRING,
        days_since_prior_orders STRING,
        order_numbers STRING
        """,
    )

    input_path = tmp_path / "gold"
    output_path = tmp_path / "training"

    write_parquet(
        str(input_path / "aisle_history_data_train"),
        aisle_history_data,
    )

    run_aisle_training_data_job(
        spark=spark,
        input_path=str(input_path),
        output_path=str(output_path),
        mode="train",
        contract_path=str(CONTRACT_PATH),
        pad_length=100,
    )

    result = read_parquet(
        output_path / "aisle_training_data_train",
        spark,
    )

    actual = {
        (row.user_id, row.aisle_id): row.asDict(recursive=True)
        for row in result.collect()
    }

    assert set(actual) == {(1, 10), (1, 20)}

    aisle_10 = actual[(1, 10)]
    assert aisle_10["department_id"] == 1
    assert aisle_10["eval_set"] == "train"
    assert aisle_10["history_length"] == 2
    assert len(aisle_10["is_ordered_history"]) == 100
    assert aisle_10["is_ordered_history"][:2] == [1, 0]
    assert aisle_10["position_in_order"][:2] == [1, 0]
    assert aisle_10["num_products_from_aisle"][:2] == [2, 0]
    assert aisle_10["aisle_history_size"][:2] == [2, 2]
    assert aisle_10["order_numbers"][:3] == [1, 2, 3]
    assert aisle_10["days_since_prior_orders"][:3] == [-1.0, 5.0, 7.0]

    aisle_20 = actual[(1, 20)]
    assert aisle_20["is_ordered_history"][:2] == [0, 1]
    assert aisle_20["position_in_order"][:2] == [0, 1]
    assert aisle_20["num_products_from_aisle"][:2] == [0, 1]
