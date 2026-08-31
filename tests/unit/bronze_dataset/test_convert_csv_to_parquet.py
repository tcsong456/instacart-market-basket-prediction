import pytest
from pyspark.sql.types import StructType

from instacart_etl_rnn.bronze.create_bronze_dataset import convert_csv_to_parquet
from instacart_etl_rnn.validation.exceptions import DataValidationError
from instacart_etl_rnn.validation.models import ValidationReport


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

    report = ValidationReport(dataset_name="orders", results=[])
    validation_error = DataValidationError(report)
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
