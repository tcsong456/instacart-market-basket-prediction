from pathlib import Path
from unittest.mock import call

import pytest

from instacart_etl_rnn.bronze.create_bronze_dataset import (
    build_dependent_bronze_datasets,
)
from instacart_etl_rnn.validation.exceptions import DataValidationError
from instacart_etl_rnn.validation.models import ValidationReport


def test_build_dependent_bronze_datasets_processes_all_datasets(
    mocker,
):
    spark = mocker.sentinel.spark

    csv_path = "gs://raw-bucket"
    parquet_path = "gs://bronze-bucket"
    contract_path = Path("contracts")

    products_contract = {
        "dataset": {
            "name": "products",
        }
    }
    order_products_contract = {
        "dataset": {
            "name": "order_products",
        }
    }

    aisles_df = mocker.sentinel.aisles_df
    departments_df = mocker.sentinel.departments_df
    orders_df = mocker.sentinel.orders_df
    products_df = mocker.sentinel.products_df

    mocked_load_contract = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.load_contract",
        side_effect=[
            products_contract,
            order_products_contract,
        ],
    )

    mocked_join_path = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.join_path",
        side_effect=[
            "gs://bronze-bucket/aisles",
            "gs://bronze-bucket/departments",
            "gs://raw-bucket/products.csv",
            "gs://bronze-bucket/products",
            "gs://bronze-bucket/orders",
            "gs://bronze-bucket/products",
            "gs://raw-bucket/order_products__prior.csv",
            "gs://bronze-bucket/order_products__prior",
            "gs://raw-bucket/order_products__train.csv",
            "gs://bronze-bucket/order_products__train",
        ],
    )

    mocked_read_parquet = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.read_parquet",
        side_effect=[
            aisles_df,
            departments_df,
            orders_df,
            products_df,
        ],
    )

    mocked_convert = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.convert_csv_to_parquet",
    )

    result = build_dependent_bronze_datasets(
        spark,
        parquet_path=parquet_path,
        csv_path=csv_path,
        contract_path=contract_path,
    )

    assert result is None

    assert mocked_load_contract.call_args_list == [
        call(contract_path / "products.yaml"),
        call(contract_path / "order_products.yaml"),
    ]

    assert mocked_join_path.call_args_list == [
        call(parquet_path, "aisles"),
        call(parquet_path, "departments"),
        call(csv_path, "products.csv"),
        call(parquet_path, "products"),
        call(parquet_path, "orders"),
        call(parquet_path, "products"),
        call(csv_path, "order_products__prior.csv"),
        call(parquet_path, "order_products__prior"),
        call(csv_path, "order_products__train.csv"),
        call(parquet_path, "order_products__train"),
    ]

    assert mocked_read_parquet.call_args_list == [
        call(
            "gs://bronze-bucket/aisles",
            spark,
        ),
        call(
            "gs://bronze-bucket/departments",
            spark,
        ),
        call(
            "gs://bronze-bucket/orders",
            spark,
        ),
        call(
            "gs://bronze-bucket/products",
            spark,
        ),
    ]

    assert mocked_convert.call_args_list == [
        call(
            spark,
            input_path="gs://raw-bucket/products.csv",
            output_path="gs://bronze-bucket/products",
            contract=products_contract,
            reference_datasets={
                "aisles": aisles_df,
                "departments": departments_df,
            },
        ),
        call(
            spark,
            input_path=("gs://raw-bucket/order_products__prior.csv"),
            output_path=("gs://bronze-bucket/order_products__prior"),
            contract=order_products_contract,
            reference_datasets={
                "orders": orders_df,
                "products": products_df,
            },
        ),
        call(
            spark,
            input_path=("gs://raw-bucket/order_products__train.csv"),
            output_path=("gs://bronze-bucket/order_products__train"),
            contract=order_products_contract,
            reference_datasets={
                "orders": orders_df,
                "products": products_df,
            },
        ),
    ]


def test_build_dependent_bronze_datasets_stops_when_products_fail(
    mocker,
):
    spark = mocker.sentinel.spark

    products_contract = {
        "dataset": {
            "name": "products",
        }
    }
    order_products_contract = {
        "dataset": {
            "name": "order_products",
        }
    }

    aisles_df = mocker.sentinel.aisles_df
    departments_df = mocker.sentinel.departments_df

    report = ValidationReport(dataset_name="products", results=[])
    validation_error = DataValidationError(report)

    mocked_load_contract = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.load_contract",
        side_effect=[
            products_contract,
            order_products_contract,
        ],
    )

    mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.join_path",
        side_effect=[
            "bronze/aisles",
            "bronze/departments",
            "raw/products.csv",
            "bronze/products",
        ],
    )

    mocked_read_parquet = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.read_parquet",
        side_effect=[
            aisles_df,
            departments_df,
        ],
    )

    mocked_convert = mocker.patch(
        "instacart_etl_rnn.bronze.create_bronze_dataset.convert_csv_to_parquet",
        side_effect=validation_error,
    )

    with pytest.raises(DataValidationError) as exc_info:
        build_dependent_bronze_datasets(
            spark,
            parquet_path="bronze",
            csv_path="raw",
            contract_path=Path("contracts"),
        )

    assert exc_info.value is validation_error

    assert mocked_load_contract.call_count == 2
    assert mocked_read_parquet.call_count == 2
    assert mocked_convert.call_count == 1

    mocked_convert.assert_called_once_with(
        spark,
        input_path="raw/products.csv",
        output_path="bronze/products",
        contract=products_contract,
        reference_datasets={
            "aisles": aisles_df,
            "departments": departments_df,
        },
    )
