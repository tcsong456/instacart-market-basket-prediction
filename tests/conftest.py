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
