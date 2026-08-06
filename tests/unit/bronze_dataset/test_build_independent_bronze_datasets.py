from unittest.mock import call

import pytest

from instacart_etl_rnn.bronze.create_bronze_dataset import (
    build_independent_bronze_datasets,
)
from instacart_etl_rnn.validation.exceptions import (
    DataValidationError,
    InvalidContractError,
)
from instacart_etl_rnn.validation.models import ValidationReport


def test_build_independent_bronze_datasets_processes_all_datasets(mocker):
    spark = mocker.sentinel.spark

    orders_contract = {"dataset": {"name": "orders"}}
    aisles_contract = {"dataset": {"name": "aisles"}}
    departments_contract = {"dataset": {"name": "departments"}}

    mocked_join_path = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.join_path",
        side_effect=[
            "gs://raw-bucket/orders.csv",
            "gs://bronze-bucket/orders",
            "contract/orders.yaml",
            "gs://raw-bucket/aisles.csv",
            "gs://bronze-bucket/aisles",
            "contract/aisles.yaml",
            "gs://raw-bucket/departments.csv",
            "gs://bronze-bucket/departments",
            "contract/departments.yaml",
        ],
    )

    mocked_load_contract = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.load_contract",
        side_effect=[orders_contract, aisles_contract, departments_contract],
    )

    input_path = "gs://raw-bucket"
    output_path = "gs://bronze-bucket"
    contract_path = "contract"

    mocked_convert = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.convert_csv_to_parquet"
    )

    result = build_independent_bronze_datasets(
        spark,
        input_path=input_path,
        output_path=output_path,
        contract_path=contract_path,
    )

    assert result is None

    assert mocked_join_path.call_args_list == [
        call(input_path, "orders.csv"),
        call(output_path, "orders"),
        call(contract_path, "orders.yaml"),
        call(input_path, "aisles.csv"),
        call(output_path, "aisles"),
        call(contract_path, "aisles.yaml"),
        call(input_path, "departments.csv"),
        call(output_path, "departments"),
        call(contract_path, "departments.yaml"),
    ]

    assert mocked_load_contract.call_args_list == [
        call("contract/orders.yaml"),
        call("contract/aisles.yaml"),
        call("contract/departments.yaml"),
    ]

    assert mocked_convert.call_args_list == [
        call(
            spark=spark,
            input_path="gs://raw-bucket/orders.csv",
            output_path="gs://bronze-bucket/orders",
            contract=orders_contract,
        ),
        call(
            spark=spark,
            input_path="gs://raw-bucket/aisles.csv",
            output_path="gs://bronze-bucket/aisles",
            contract=aisles_contract,
        ),
        call(
            spark=spark,
            input_path="gs://raw-bucket/departments.csv",
            output_path="gs://bronze-bucket/departments",
            contract=departments_contract,
        ),
    ]


def test_build_independent_bronze_datasets_stops_on_load_contract_failure(mocker):
    spark = mocker.sentinel.spark
    load_contract_error = InvalidContractError("Invalid contract")

    mocked_join_path = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.join_path",
        side_effect=[
            "gs://bucket-raw/orders.csv",
            "gs://bronze/orders",
            "contract/orders.yaml",
        ],
    )

    mocked_load_contract = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.load_contract",
        side_effect=load_contract_error,
    )

    mocked_convert = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.convert_csv_to_parquet"
    )

    with pytest.raises(InvalidContractError) as exc_info:
        build_independent_bronze_datasets(
            spark,
            input_path="gs://bucket-raw",
            output_path="gs://bronze",
            contract_path="contract",
        )

    assert exc_info.value is load_contract_error

    assert mocked_join_path.call_args_list == [
        call("gs://bucket-raw", "orders.csv"),
        call("gs://bronze", "orders"),
        call("contract", "orders.yaml"),
    ]

    mocked_load_contract.assert_called_once_with("contract/orders.yaml")

    mocked_convert.assert_not_called()


def test_build_independent_bronze_datasets_stops_on_validation_failure(
    mocker,
):
    spark = mocker.sentinel.spark
    report = ValidationReport(dataset_name="orders", results=[])
    validation_error = DataValidationError(report)

    mocked_load_contract = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.load_contract",
        return_value={"dataset": {"name": "orders"}},
    )

    mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.join_path",
        side_effect=["raw/orders.csv", "bronze/orders", "contract/orders.yaml"],
    )

    mocked_convert = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.convert_csv_to_parquet",
        side_effect=validation_error,
    )

    with pytest.raises(DataValidationError) as exc_info:
        build_independent_bronze_datasets(
            spark,
            input_path="raw",
            output_path="bronze",
            contract_path="contract",
        )

    assert exc_info.value is validation_error

    mocked_load_contract.assert_called_once_with("contract/orders.yaml")

    mocked_convert.assert_called_once()

    assert mocked_load_contract.call_count == 1
