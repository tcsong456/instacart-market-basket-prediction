from pathlib import Path

from pyspark.sql.types import IntegerType

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.jobs.create_user_product_count_data_job import (
    run_user_product_count_job,
)

CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "instacart_etl_rnn" / "contracts"
)


def test_run_user_product_count_job_excludes_validation_target_orders(
    spark,
    tmp_path,
):
    order_products = spark.createDataFrame(
        [
            (1, 101, 1, 10),
            (1, 101, 1, 20),
            (1, 102, 2, 10),
            (1, 103, 3, 10),
            (1, 103, 3, 30),
            (2, 201, 1, 20),
            (2, 202, 2, 20),
        ],
        ["user_id", "order_id", "order_number", "product_id"],
    )

    input_path = tmp_path / "snapshots"
    output_path = tmp_path / "training"
    write_parquet(
        input_path / "order_products_validation",
        order_products,
    )

    run_user_product_count_job(
        spark=spark,
        input_path=str(input_path),
        output_path=str(output_path),
        contract_path=str(CONTRACT_PATH),
        mode="validation",
    )

    result = read_parquet(
        output_path / "user_product_count_validation",
        spark,
    )
    actual = {(row.user_id, row.product_id): row["count"] for row in result.collect()}

    assert actual == {
        (1, 10): 2,
        (1, 20): 1,
        (2, 20): 1,
    }
    assert result.columns == ["user_id", "product_id", "count"]
    assert isinstance(result.schema["count"].dataType, IntegerType)
