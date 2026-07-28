import pytest
from pyspark.sql.types import (
    IntegerType,
    StructField,
    StructType,
)

from instacart_etl_rnn.validation.exceptions import InvalidContractError
from instacart_etl_rnn.validation.referential_integrity import validate_foreign_keys


def test_validate_foreign_keys_passes_when_all_parent_keys_exist(
    spark, order_products_contract
):
    child_df = spark.createDataFrame(
        [
            (1, 10),
            (1, 20),
            (2, 10),
        ],
        ["order_id", "product_id"],
    )

    orders_df = spark.createDataFrame(
        [
            (1,),
            (2,),
        ],
        ["order_id"],
    )

    products_df = spark.createDataFrame(
        [
            (10,),
            (20,),
        ],
        ["product_id"],
    )

    results = validate_foreign_keys(
        child_df=child_df,
        contract=order_products_contract,
        datasets={"orders": orders_df, "products": products_df},
    )

    results = {result.rule_name: result for result in results}

    order_result = results["order_products_orders_fk"]
    product_result = results["order_products_products_fk"]

    assert len(results) == 2

    assert order_result.passed is True
    assert order_result.invalid_rows is None
    assert order_result.failed_count == 0
    assert order_result.message == (
        "All non-null child key values 'order_id' exist in parent columns 'order_id'"
    )

    assert product_result.passed is True
    assert product_result.invalid_rows is None
    assert product_result.failed_count == 0
    assert product_result.message == (
        "All non-null child key values 'product_id' exist "
        "in parent columns 'product_id'"
    )


def test_validate_foreign_keys_detects_missing_parent_values(
    spark, order_products_contract
):
    child_df = spark.createDataFrame(
        [
            (1, 10),
            (99, 20),
            (99, 10),
        ],
        ["order_id", "product_id"],
    )

    orders_df = spark.createDataFrame(
        [
            (1,),
            (2,),
        ],
        ["order_id"],
    )

    products_df = spark.createDataFrame(
        [
            (10,),
            (20,),
        ],
        ["product_id"],
    )

    results = validate_foreign_keys(
        child_df=child_df,
        contract=order_products_contract,
        datasets={"orders": orders_df, "products": products_df},
    )
    results_by_rule = {r.rule_name: r for r in results}
    order_result = results_by_rule["order_products_orders_fk"]
    product_result = results_by_rule["order_products_products_fk"]

    assert order_result.passed is False
    assert order_result.failed_count == 2
    invalid_rows = {row["order_id"] for row in order_result.invalid_rows.collect()}
    assert invalid_rows == {99}
    assert order_result.message == (
        "Found 2 child row(s) whose key values "
        "'order_id' do not exist in parent columns 'order_id'"
    )

    assert product_result.passed is True
    assert product_result.failed_count == 0


def test_validate_foreign_keys_reports_each_relationship_separately(
    spark,
    order_products_contract,
):
    child_df = spark.createDataFrame(
        [
            (1, 10),
            (99, 10),
            (1, 999),
            (98, 998),
        ],
        ["order_id", "product_id"],
    )

    orders_df = spark.createDataFrame(
        [(1,)],
        ["order_id"],
    )

    products_df = spark.createDataFrame(
        [(10,)],
        ["product_id"],
    )

    results = validate_foreign_keys(
        child_df=child_df,
        contract=order_products_contract,
        datasets={
            "orders": orders_df,
            "products": products_df,
        },
    )

    results_by_rule = {result.rule_name: result for result in results}

    order_result = results_by_rule["order_products_orders_fk"]
    product_result = results_by_rule["order_products_products_fk"]

    assert order_result.passed is False
    assert order_result.failed_count == 2

    invalid_order_ids = {row["order_id"] for row in order_result.invalid_rows.collect()}
    assert invalid_order_ids == {98, 99}

    assert product_result.passed is False
    assert product_result.failed_count == 2

    invalid_product_ids = {
        row["product_id"] for row in product_result.invalid_rows.collect()
    }
    assert invalid_product_ids == {998, 999}


def test_validate_foreign_keys_ignores_null_child_keys(spark):
    order_contract = {
        "dataset": {"name": "order_products"},
        "relationships": [
            {
                "name": "order_products_orders_fk",
                "type": "foreign_key",
                "child_columns": ["order_id"],
                "parent": {"dataset": "orders", "columns": ["order_id"]},
            }
        ],
    }

    schema = StructType(
        [
            StructField("order_id", IntegerType(), True),
            StructField("product_id", IntegerType(), False),
        ]
    )
    child_df = spark.createDataFrame([(1, 10), (None, 10), (2, 20)], schema=schema)

    order_df = spark.createDataFrame([(1,), (2,)], ["order_id"])

    results = validate_foreign_keys(
        child_df=child_df,
        contract=order_contract,
        datasets={"orders": order_df},
    )

    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].failed_count == 0


def test_validate_foreign_keys_raises_when_parent_dataset_is_unavailable(spark):
    child_df = spark.createDataFrame(
        [(1,)],
        ["order_id"],
    )

    contract = {
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
        ],
    }

    with pytest.raises(
        ValueError,
        match="Parent dataset 'orders' is not available",
    ):
        validate_foreign_keys(
            child_df=child_df,
            contract=contract,
            datasets={},
        )


def test_validate_foreign_keys_supports_composite_keys(spark):
    contract = {
        "relationships": [
            {
                "name": "item_order_version_fk",
                "type": "foreign_key",
                "child_columns": ["order_id", "order_version"],
                "parent": {"dataset": "orders", "columns": ["id", "version"]},
            }
        ]
    }

    child_df = spark.createDataFrame(
        [(10, 1, "a"), (20, 3, "b"), (30, 2, "c")],
        ["order_id", "order_version", "order_value"],
    )

    parent_df = spark.createDataFrame([(30, 2), (10, 1), (20, 3)], ["id", "version"])

    results = validate_foreign_keys(
        child_df=child_df, datasets={"orders": parent_df}, contract=contract
    )

    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].failed_count == 0


def test_validate_foreign_keys_detects_invalid_composite_key(spark):
    contract = {
        "relationships": [
            {
                "name": "item_order_version_fk",
                "type": "foreign_key",
                "child_columns": ["order_id", "order_version"],
                "parent": {"dataset": "orders", "columns": ["id", "version"]},
            }
        ]
    }

    child_df = spark.createDataFrame(
        [(10, 1), (21, 4), (40, 5)], ["order_id", "order_version"]
    )

    parent_df = spark.createDataFrame([(30, 2), (10, 1), (20, 3)], ["id", "version"])

    result = validate_foreign_keys(
        child_df=child_df, datasets={"orders": parent_df}, contract=contract
    )[0]

    assert result.passed is False
    assert result.failed_count == 2
    invalid_rows = {
        (row["order_id"], row["order_version"]) for row in result.invalid_rows.collect()
    }
    assert invalid_rows == {(21, 4), (40, 5)}


def test_validate_foreign_keys_returns_empty_list_when_no_relationships_exist(spark):
    child_df = spark.createDataFrame(
        [(1,)],
        ["order_id"],
    )

    results = validate_foreign_keys(
        child_df=child_df,
        contract={
            "dataset": {
                "name": "orders",
            },
        },
        datasets={},
    )

    assert results == []


def test_validate_foreign_keys_ignores_other_relationship_types(spark):
    child_df = spark.createDataFrame(
        [(1,)],
        ["order_id"],
    )

    contract = {
        "relationships": [
            {
                "name": "some_dependency",
                "type": "dependency",
                "child_columns": ["order_id"],
                "parent": {
                    "dataset": "orders",
                    "columns": ["order_id"],
                },
            },
        ],
    }

    results = validate_foreign_keys(
        child_df=child_df,
        contract=contract,
        datasets={},
    )

    assert results == []


@pytest.mark.parametrize(
    ("relationship", "exception_message"),
    [
        (
            {"type": "foreign_key", "child_columns": []},
            "Foreign-key relationship must contain a non-empty",
        ),
        (
            {
                "type": "foreign_key",
                "child_columns": ["order_id"],
                "parent": ["order_id"],
            },
            "Foreign-key relationship must contain a 'parent' mapping",
        ),
        (
            {
                "type": "foreign_key",
                "child_columns": ["order_id"],
                "parent": {"dataset": "   "},
            },
            "Foreign-key parent must contain a valid dataset name",
        ),
        (
            {
                "type": "foreign_key",
                "child_columns": ["order_id"],
                "parent": {"dataset": "orders", "columns": []},
            },
            "Foreign-key parent must contain a non-empty",
        ),
        (
            {
                "type": "foreign_key",
                "child_columns": ["order_id", "order_version"],
                "parent": {"dataset": "orders", "columns": ["id"]},
            },
            "Foreign-key child_columns and parent columns must have the same length",
        ),
        (
            {
                "type": "foreign_key",
                "child_columns": ["order_id", "order_id"],
                "parent": {"dataset": "orders", "columns": ["id", "version"]},
            },
            "child_columns cannot contain duplicate column names",
        ),
        (
            {
                "type": "foreign_key",
                "child_columns": ["order_id", "order_version"],
                "parent": {"dataset": "orders", "columns": ["id", "id"]},
            },
            "parent_columns cannot contain duplicate column names",
        ),
        (
            {
                "type": "foreign_key",
                "child_columns": ["id"],
                "parent": {"dataset": "orders", "columns": ["id"]},
            },
            "Child DataFrame is missing column(s)",
        ),
        (
            {
                "type": "foreign_key",
                "child_columns": ["order_id"],
                "parent": {"dataset": "orders", "columns": ["order_id"]},
            },
            "Parent DataFrame is missing column(s)",
        ),
    ],
)
def test_validate_foreign_keys_rejects_invalid_contracts(
    spark, relationship, exception_message
):
    contract = {"relationships": [relationship]}

    child_df = spark.createDataFrame(
        [(1, 1)],
        ["order_id", "version"],
    )

    orders_df = spark.createDataFrame([(1,), (2,)], ["id"])

    with pytest.raises(InvalidContractError, match=exception_message):
        validate_foreign_keys(
            child_df=child_df, datasets={"orders": orders_df}, contract=contract
        )


def test_validate_foreign_keys_invalid_rows_limit(spark):
    contract = {
        "relationships": [
            {
                "type": "foreign_key",
                "child_columns": ["order_id"],
                "parent": {"dataset": "orders", "columns": ["order_id"]},
            }
        ]
    }

    child_df = spark.createDataFrame([(value,) for value in range(50)], ["order_id"])

    parent_df = spark.createDataFrame([(50,)], ["order_id"])

    result = validate_foreign_keys(
        child_df=child_df, datasets={"orders": parent_df}, contract=contract
    )[0]

    assert result.passed is False
    assert result.failed_count == 50
    assert result.invalid_rows.count() == 20
    invalid_rows = {row["order_id"] for row in result.invalid_rows.collect()}
    assert set(invalid_rows).issubset(range(50))
