from unittest.mock import call

import pytest
from pyspark.sql.types import StructType

from instacart_etl_rnn.bronze.create_bronze_dataset import convert_csv_to_parquet
from instacart_etl_rnn.validation.exceptions import DataValidationError


def test_convert_csv_to_parquet_reads_validates_and_writes(spark, mocker):
    expected_schema = StructType([])
    mocked_build_schema = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.build_spark_schema",
        return_value=expected_schema,
    )

    expected_df = mocker.sentinel.dataframe
    mocked_read_csv = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.read_csv",
        return_value=expected_df,
    )

    mocked_validate_dataset = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.validate_dataset"
    )

    mocked_write_parquet = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.write_parquet"
    )

    contract = {
        "dataset": {"name": "orders"},
        "schema": [{"name": "order_id", "type": "integer", "nullable": False}],
    }

    input_path = "data/raw/orders.csv"
    output_path = "data/bronze/orders"
    result = convert_csv_to_parquet(
        spark, input_path=input_path, output_path=output_path, contract=contract
    )

    assert result is None

    mocked_build_schema.assert_called_once_with(contract)

    mocked_validate_dataset.assert_called_once_with(
        expected_df, contract=contract, reference_datasets=None
    )

    mocked_read_csv.assert_called_once_with(
        path=input_path, spark=spark, schema=expected_schema
    )

    mocked_write_parquet.assert_called_once_with(path=output_path, df=expected_df)


def test_convert_csv_to_parquet_validates_before_writing(spark, mocker):
    expected_schema = StructType([])
    mocked_build_schema = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.build_spark_schema",
        return_value=expected_schema,
    )

    expected_df = mocker.sentinel.dataframe
    mocked_read_csv = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.read_csv",
        return_value=expected_df,
    )

    mocked_validate_dataset = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.validate_dataset"
    )

    mocked_write_parquet = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.write_parquet"
    )

    manager = mocker.Mock()

    manager.attach_mock(mocked_build_schema, "build_schema")

    manager.attach_mock(mocked_read_csv, "read_csv")

    manager.attach_mock(mocked_validate_dataset, "validate_dataset")

    manager.attach_mock(mocked_write_parquet, "write_parquet")

    contract = {"dataset": {"name": "orders"}}
    convert_csv_to_parquet(
        spark,
        input_path="data/orders.csv",
        output_path="data/orders",
        contract=contract,
    )

    assert manager.mock_calls == [
        call.build_schema(contract),
        call.read_csv(spark=spark, path="data/orders.csv", schema=expected_schema),
        call.validate_dataset(expected_df, contract=contract, reference_datasets=None),
        call.write_parquet(path="data/orders", df=expected_df),
    ]


def test_convert_csv_to_parquet_does_not_write_when_validation_fails(spark, mocker):
    mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.build_spark_schema",
        return_value=StructType([]),
    )

    expected_df = mocker.sentinel.dataframe
    mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.read_csv",
        return_value=expected_df,
    )

    validation_error = DataValidationError(mocker.sentinel.report)
    mocked_validate = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.validate_dataset",
        side_effect=validation_error,
    )

    mocked_write = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.write_parquet"
    )

    contract = {"dataset": {"name": "orders"}}
    with pytest.raises(DataValidationError) as exc_info:
        convert_csv_to_parquet(
            spark,
            input_path="data/orders.csv",
            output_path="data/orders",
            contract=contract,
        )

    assert exc_info.value is validation_error

    mocked_validate.assert_called_once_with(
        expected_df, contract=contract, reference_datasets=None
    )

    mocked_write.assert_not_called()


def test_convert_csv_to_parquet_stops_when_csv_read_fails(spark, mocker):
    mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.build_spark_schema",
        return_value=StructType([]),
    )

    read_error = OSError("Unable to read csv files")
    mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.read_csv",
        side_effect=read_error,
    )

    mocked_validate = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.validate_dataset"
    )

    mocked_write = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.write_parquet"
    )

    contract = {"dataset": {"name": "orders"}}
    with pytest.raises(OSError, match="Unable to read csv files"):
        convert_csv_to_parquet(
            spark,
            input_path="data/orders.csv",
            output_path="data/orders",
            contract=contract,
        )

    mocked_validate.assert_not_called()

    mocked_write.assert_not_called()
