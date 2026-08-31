from pathlib import Path

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.jobs.create_reorder_size_training_data_job import (
    run_reorder_size_training_data,
)

CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "instacart_etl_rnn" / "contracts"
)


def test_run_reorder_size_training_data_job_end_to_end(spark, tmp_path):
    user_data = spark.createDataFrame(
        [
            (
                1,
                "1 2 3",
                "0_1_0 1_1_0 1_0_1",
                "0 1 2",
                "10 11 12",
                "-1.0 5.0 7.0",
                "1 2 3",
                "train",
            ),
            (
                2,
                "10 11",
                "0_0 1_0",
                "3 4",
                "8 9",
                "-1.0 4.0",
                "1 2",
                "prior",
            ),
        ],
        """
        user_id INT,
        order_ids STRING,
        reorders STRING,
        order_dows STRING,
        order_hours STRING,
        days_since_prior_orders STRING,
        order_numbers STRING,
        eval_set STRING
        """,
    )

    input_path = tmp_path / "silver"
    output_path = tmp_path / "training"

    write_parquet(str(input_path / "user_data_train"), user_data)

    run_reorder_size_training_data(
        spark=spark,
        input_path=str(input_path),
        output_path=str(output_path),
        contract_path=str(CONTRACT_PATH),
        pad_length=100,
        mode="train",
    )

    result = read_parquet(
        output_path / "reorder_size_training_data_train",
        spark,
    )

    actual = {row.user_id: row.asDict(recursive=True) for row in result.collect()}

    assert set(actual) == {1, 2}

    user_1 = actual[1]
    assert user_1["eval_set"] == "train"
    assert user_1["label"] == 2
    assert user_1["history_length"] == 2
    assert len(user_1["order_sizes"]) == 100
    assert user_1["order_sizes"][:2] == [3, 3]
    assert user_1["reorder_sizes"][:2] == [1, 2]
    assert user_1["order_numbers"][:3] == [1, 2, 3]
    assert user_1["days_since_prior_orders"][:3] == [-1.0, 5.0, 7.0]

    user_2 = actual[2]
    assert user_2["eval_set"] == "prior"
    assert user_2["label"] == 1
    assert user_2["history_length"] == 1
    assert user_2["order_sizes"][:1] == [2]
    assert user_2["reorder_sizes"][:1] == [0]
    assert user_2["order_numbers"][:2] == [1, 2]
