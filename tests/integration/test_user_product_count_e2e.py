from pathlib import Path

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.jobs.create_user_product_count_data_job import (
    run_user_product_count_job,
)

CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "instacart_etl_rnn" / "contracts"
)


def test_run_user_product_count_job_end_to_end(spark, tmp_path):
    order_products = spark.createDataFrame(
        [
            (1, 10, "prior"),
            (1, 10, "prior"),
            (1, 20, "prior"),
            (1, 10, "train"),
            (2, 10, "prior"),
            (2, 30, "prior"),
            (2, 30, "train"),
        ],
        """
        user_id INT,
        product_id INT,
        eval_set STRING
        """,
    )

    input_path = tmp_path / "silver"
    output_path = tmp_path / "gold"

    write_parquet(
        str(input_path / "order_products_train"),
        order_products,
    )

    run_user_product_count_job(
        spark=spark,
        input_path=str(input_path),
        output_path=str(output_path),
        contract_path=str(CONTRACT_PATH),
        mode="train",
    )

    result = read_parquet(
        output_path / "user_product_count_train",
        spark,
    )

    actual = {(row.user_id, row.product_id): row["count"] for row in result.collect()}

    assert actual == {
        (1, 10): 2,
        (1, 20): 1,
        (2, 10): 1,
        (2, 30): 1,
    }

    assert set(result.columns) == {
        "user_id",
        "product_id",
        "count",
    }
