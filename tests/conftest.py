import logging
from pathlib import Path

import pandas as pd
import pytest
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from instacart_etl_rnn.common.spark import create_spark_session
from instacart_etl_rnn.validation.loader import load_contract


@pytest.fixture(scope="session")
def spark():
    spark = create_spark_session("unit-test")
    yield spark
    spark.stop()


@pytest.fixture
def raw_dir(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


@pytest.fixture
def sample_dir(tmp_path):
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    return sample_dir


@pytest.fixture
def tiny_fake_testset_csv(raw_dir):
    orders = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5, 6],
            "user_id": [10, 10, 20, 20, 30, 30],
            "eval_set": ["prior", "train", "prior", "train", "prior", "test"],
            "order_number": [1, 2, 1, 2, 1, 2],
            "order_dow": [1, 2, 1, 2, 1, 2],
            "order_hour_of_day": [10, 11, 10, 11, 10, 11],
            "days_since_prior_order": [None, 7, None, 5, None, 3],
        }
    )

    prior = pd.DataFrame(
        {
            "order_id": [1, 3, 5],
            "product_id": [101, 201, 301],
            "add_to_cart_order": [1, 1, 1],
            "reordered": [0, 0, 0],
        }
    )

    train = pd.DataFrame(
        {
            "order_id": [2, 4],
            "product_id": [102, 202],
            "add_to_cart_order": [1, 1],
            "reordered": [1, 1],
        }
    )

    products = pd.DataFrame(
        {
            "product_id": [101, 102, 201, 202, 301],
            "product_name": ["a", "b", "c", "d", "e"],
            "aisle_id": [1, 1, 2, 2, 3],
            "department_id": [10, 10, 20, 20, 30],
        }
    )

    aisles = pd.DataFrame(
        {
            "aisle_id": [1, 2, 3],
            "aisle": ["fresh", "dairy", "snacks"],
        }
    )

    departments = pd.DataFrame(
        {
            "department_id": [10, 20, 30],
            "department": ["produce", "frozen", "pantry"],
        }
    )

    orders.to_csv(raw_dir / "orders.csv", index=False)
    prior.to_csv(raw_dir / "order_products__prior.csv", index=False)
    train.to_csv(raw_dir / "order_products__train.csv", index=False)
    products.to_csv(raw_dir / "products.csv", index=False)
    aisles.to_csv(raw_dir / "aisles.csv", index=False)
    departments.to_csv(raw_dir / "departments.csv", index=False)

    return raw_dir


@pytest.fixture
def order_products_contract():
    return {
        "dataset": {
            "name": "order_products",
        },
        "relationships": [
            {
                "name": "order_products_orders_fk",
                "type": "foreign_key",
                "child_columns": ["order_id"],
                "parent": {
                    "dataset": "orders",
                    "columns": ["order_id"],
                },
            },
            {
                "name": "order_products_products_fk",
                "type": "foreign_key",
                "child_columns": ["product_id"],
                "parent": {
                    "dataset": "products",
                    "columns": ["product_id"],
                },
            },
        ],
    }


@pytest.fixture
def row_logic_contract() -> dict:
    return {
        "derived_fields": [
            {
                "name": "train_or_test_count",
                "aggregation": "conditional_count",
                "condition": "eval_set IN ('train', 'test')",
                "partition_by": ["user_id"],
            },
            {
                "name": "user_min_order_number",
                "aggregation": "min",
                "column": "order_number",
                "partition_by": ["user_id"],
            },
            {
                "name": "user_max_order_number",
                "aggregation": "max",
                "column": "order_number",
                "partition_by": ["user_id"],
            },
            {
                "name": "user_distinct_order_number_count",
                "aggregation": "count_distinct",
                "column": "order_number",
                "partition_by": ["user_id"],
            },
            {
                "name": "is_last_order",
                "expression": ("order_number = user_max_order_number"),
            },
        ],
        "rules": [
            {
                "name": "first_order_has_no_prior_interval",
                "expression": ("order_number <> 1 OR days_since_prior_order IS NULL"),
            },
            {
                "name": "later_orders_must_have_prior_interval",
                "expression": (
                    "order_number = 1 OR days_since_prior_order IS NOT NULL"
                ),
            },
            {
                "name": "last_order_per_user_is_train_or_test",
                "expression": ("NOT is_last_order OR eval_set IN ('train', 'test')"),
            },
            {
                "name": "only_one_train_or_test_per_user",
                "expression": "train_or_test_count = 1",
            },
            {
                "name": "contiguous_order_numbers",
                "expression": (
                    "user_min_order_number = 1 "
                    "AND user_max_order_number = "
                    "user_distinct_order_number_count"
                ),
            },
        ],
    }


@pytest.fixture
def orders_schema():
    return StructType(
        [
            StructField("user_id", IntegerType(), nullable=False),
            StructField("order_number", IntegerType(), nullable=True),
            StructField("eval_set", StringType(), nullable=True),
            StructField("days_since_prior_order", DoubleType(), nullable=True),
        ]
    )


@pytest.fixture
def apply_thresholds_contract():
    return {
        "schema": [
            {
                "name": "order_id",
                "thresholds": {
                    "range": {
                        "max_failed_percent": 1.0,
                        "severity": "error",
                    },
                    "uniqueness": {
                        "max_failed_percent": 0.0,
                        "severity": "critical",
                    },
                },
            },
            {
                "name": "user_id",
                "thresholds": {},
            },
        ]
    }


ORDERS_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "instacart_etl_rnn"
    / "contracts"
    / "orders.yaml"
)


@pytest.fixture
def validate_dataset_orders_contract():
    return load_contract(ORDERS_CONTRACT_PATH)


@pytest.fixture
def validate_dataset_orders_schema():
    return StructType(
        [
            StructField(
                "order_id",
                IntegerType(),
                nullable=False,
            ),
            StructField(
                "user_id",
                IntegerType(),
                nullable=False,
            ),
            StructField(
                "eval_set",
                StringType(),
                nullable=False,
            ),
            StructField(
                "order_number",
                IntegerType(),
                nullable=False,
            ),
            StructField(
                "order_dow",
                IntegerType(),
                nullable=False,
            ),
            StructField(
                "order_hour_of_day",
                IntegerType(),
                nullable=False,
            ),
            StructField(
                "days_since_prior_order",
                DoubleType(),
                nullable=True,
            ),
        ]
    )


@pytest.fixture
def validate_dataset_orders_df(spark, validate_dataset_orders_schema):
    return spark.createDataFrame(
        [
            (1, 1, "prior", 1, 0, 10, None),
            (2, 1, "prior", 2, 1, 11, 5.0),
            (3, 1, "train", 3, 2, 12, 7.0),
            (4, 2, "prior", 1, 3, 13, None),
            (5, 2, "test", 2, 4, 14, 10.0),
        ],
        schema=validate_dataset_orders_schema,
    )


@pytest.fixture
def aisle_history_df(spark):
    return spark.createDataFrame(
        [
            (
                1,
                24,
                "train",
                "1 0",
                "1 0",
                "2 0",
                "3 2",
                "1 2",
                "10 12",
                "-1.0 5.0",
                "1 2",
            ),
        ],
        """
        user_id INT,
        aisle_id INT,
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


@pytest.fixture
def aisle_product_df(spark):
    return spark.createDataFrame(
        [
            (10, 24, 4),
            (20, 24, 4),
            (30, 84, 16),
        ],
        """
        product_id INT,
        aisle_id INT,
        department_id INT
        """,
    )


@pytest.fixture
def preserve_root_logger():
    root_logger = logging.getLogger()

    original_handlers = root_logger.handlers.copy()
    original_level = root_logger.level

    yield root_logger

    # Close handlers created by the test.
    for handler in root_logger.handlers:
        if handler not in original_handlers:
            handler.close()

    root_logger.handlers.clear()
    root_logger.handlers.extend(original_handlers)
    root_logger.setLevel(original_level)
