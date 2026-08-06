import pytest
from pyspark.errors import AnalysisException

from instacart_etl_rnn.bronze.create_bronze_dataset import convert_csv_to_parquet
from instacart_etl_rnn.validation.exceptions import DataValidationError
from tests.helpers import find_result, write_csv


def test_convert_csv_to_parquet_writes_valid_dataset(
    spark,
    tmp_path,
    validate_dataset_orders_contract,
    validate_dataset_users_df,
):
    input_path = tmp_path / "orders.csv"
    output_path = tmp_path / "bronze" / "orders"

    write_csv(
        input_path,
        content=(
            "order_id,user_id,eval_set,"
            "order_number,days_since_prior_order\n"
            "1,1,prior,1,\n"
            "2,1,train,2,5.0\n"
            "3,2,test,1,\n"
        ),
    )

    result = convert_csv_to_parquet(
        spark,
        input_path=input_path,
        output_path=output_path,
        contract=validate_dataset_orders_contract,
        reference_datasets={
            "users": validate_dataset_users_df,
        },
    )

    assert result is None
    assert output_path.exists()

    loaded_df = spark.read.parquet(str(output_path))

    assert loaded_df.columns == [
        "order_id",
        "user_id",
        "eval_set",
        "order_number",
        "days_since_prior_order",
    ]

    assert dict(loaded_df.dtypes) == {
        "order_id": "int",
        "user_id": "int",
        "eval_set": "string",
        "order_number": "int",
        "days_since_prior_order": "double",
    }

    actual_rows = [tuple(row) for row in loaded_df.orderBy("order_id").collect()]

    assert actual_rows == [
        (1, 1, "prior", 1, None),
        (2, 1, "train", 2, 5.0),
        (3, 2, "test", 1, None),
    ]


def test_convert_csv_to_parquet_rejects_missing_parent_key(
    spark,
    tmp_path,
    validate_dataset_orders_contract,
    validate_dataset_users_df,
):
    input_path = tmp_path / "orders.csv"
    output_path = tmp_path / "bronze" / "orders"

    write_csv(
        input_path,
        content=(
            "order_id,user_id,eval_set,"
            "order_number,days_since_prior_order\n"
            "1,1,prior,1\n"
            "2,1,train,2,15.0\n"
            "3,999,train,2,5.0\n"
        ),
    )

    with pytest.raises(DataValidationError) as exc_info:
        convert_csv_to_parquet(
            spark,
            input_path=input_path,
            output_path=output_path,
            contract=validate_dataset_orders_contract,
            reference_datasets={
                "users": validate_dataset_users_df,
            },
        )

    report = exc_info.value.report

    foreign_key_results = find_result(report, rule_name="orders_user_fk")

    assert foreign_key_results.failed_count == 1

    assert not output_path.exists()


def test_convert_csv_to_parquet_rejects_missing_input_file(
    spark,
    tmp_path,
    validate_dataset_orders_contract,
    validate_dataset_users_df,
):
    input_path = tmp_path / "missing.csv"
    output_path = tmp_path / "bronze" / "orders"

    with pytest.raises(AnalysisException):
        convert_csv_to_parquet(
            spark,
            input_path=input_path,
            output_path=output_path,
            contract=validate_dataset_orders_contract,
            reference_datasets={
                "users": validate_dataset_users_df,
            },
        )

    assert not output_path.exists()
