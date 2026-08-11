import pytest

from instacart_etl_rnn.validation.exceptions import InvalidConstraintError
from instacart_etl_rnn.validation.string_length import (
    _build_invalid_condition,
    string_length_validator,
)


def test_string_length_validator_builds_expected_metrics():
    contract = {
        "schema": [
            {
                "name": "product_name",
                "constraints": {
                    "minimum_string_length": 1,
                    "maximum_string_length": 100,
                },
            },
            {
                "name": "eval_set",
                "constraints": {},
            },
        ]
    }

    metrics = string_length_validator(contract)

    assert len(metrics) == 1

    metric = metrics[0]

    assert metric.alias == "product_name_string_length"
    assert metric.rule_name == "product_name.string_length"
    assert metric.validation_type == "string_length"
    assert metric.columns == ("product_name",)


def test_string_length_invalid_condition_identifies_exact_invalid_rows(
    spark,
):
    df = spark.createDataFrame(
        [
            (1, ""),
            (2, "a"),
            (3, "ab"),
            (4, "abcde"),
            (5, "abcdef"),
            (6, None),
        ],
        "id INT, product_name STRING",
    )

    condition = _build_invalid_condition(
        "product_name",
        minimum=2,
        maximum=5,
    )

    invalid_ids = {row["id"] for row in (df.filter(condition).select("id").collect())}

    assert invalid_ids == {1, 2, 5}


def test_string_length_metric_counts_invalid_values(spark):
    df = spark.createDataFrame(
        [
            ("",),
            ("a",),
            ("ab",),
            ("abc",),
            ("abcdef",),
            (None,),
        ],
        "product_name STRING",
    )

    contract = {
        "schema": [
            {
                "name": "product_name",
                "constraints": {
                    "minimum_string_length": 2,
                    "maximum_string_length": 5,
                },
            }
        ]
    }

    metrics = string_length_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["product_name_string_length"] == 3


def test_string_length_metric_returns_zero_when_all_values_are_valid(
    spark,
):
    df = spark.createDataFrame(
        [
            ("ab",),
            ("abc",),
            ("abcd",),
            ("abcde",),
        ],
        "product_name STRING",
    )

    contract = {
        "schema": [
            {
                "name": "product_name",
                "constraints": {
                    "minimum_string_length": 2,
                    "maximum_string_length": 5,
                },
            }
        ]
    }

    metrics = string_length_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["product_name_string_length"] == 0


def test_string_length_metric_ignores_null_values(spark):
    df = spark.createDataFrame(
        [
            (None,),
            (None,),
            (None,),
        ],
        "product_name STRING",
    )

    contract = {
        "schema": [
            {
                "name": "product_name",
                "constraints": {
                    "minimum_string_length": 2,
                    "maximum_string_length": 5,
                },
            }
        ]
    }

    metrics = string_length_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["product_name_string_length"] == 0


def test_string_length_metric_supports_minimum_only(spark):
    df = spark.createDataFrame(
        [
            ("",),
            ("a",),
            ("ab",),
            ("abc",),
        ],
        "product_name STRING",
    )

    contract = {
        "schema": [
            {
                "name": "product_name",
                "constraints": {
                    "minimum_string_length": 2,
                },
            }
        ]
    }

    metrics = string_length_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["product_name_string_length"] == 2


def test_string_length_metric_supports_maximum_only(spark):
    df = spark.createDataFrame(
        [
            ("a",),
            ("abc",),
            ("abcde",),
            ("abcdef",),
        ],
        "product_name STRING",
    )

    contract = {
        "schema": [
            {
                "name": "product_name",
                "constraints": {
                    "maximum_string_length": 5,
                },
            }
        ]
    }

    metrics = string_length_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["product_name_string_length"] == 1


def test_multiple_string_length_metrics_can_be_aggregated_together(
    spark,
):
    df = spark.createDataFrame(
        [
            ("a", "UK"),
            ("valid", "USA"),
            ("toolong", "X"),
        ],
        "product_name STRING, country STRING",
    )

    contract = {
        "schema": [
            {
                "name": "product_name",
                "constraints": {
                    "minimum_string_length": 2,
                    "maximum_string_length": 5,
                },
            },
            {
                "name": "country",
                "constraints": {
                    "minimum_string_length": 2,
                    "maximum_string_length": 2,
                },
            },
        ]
    }

    metrics = string_length_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["product_name_string_length"] == 2
    assert row["country_string_length"] == 2


@pytest.mark.parametrize(
    "constraint_name,value",
    [
        ("minimum_string_length", -1),
        ("minimum_string_length", "2"),
        ("minimum_string_length", True),
        ("maximum_string_length", -1),
        ("maximum_string_length", "10"),
        ("maximum_string_length", False),
        ("minimum_string_length", 0.0),
        ("maximum_string_length", 5.0),
    ],
)
def test_string_length_validator_rejects_invalid_length_constraints(
    constraint_name,
    value,
):
    contract = {
        "schema": [
            {
                "name": "product_name",
                "constraints": {
                    constraint_name: value,
                },
            }
        ]
    }

    with pytest.raises(
        InvalidConstraintError,
        match="must be a non-negative integer",
    ):
        string_length_validator(contract)


def test_string_length_validator_rejects_minimum_greater_than_maximum():
    contract = {
        "schema": [
            {
                "name": "product_name",
                "constraints": {
                    "minimum_string_length": 10,
                    "maximum_string_length": 5,
                },
            }
        ]
    }

    with pytest.raises(
        InvalidConstraintError,
        match="minimum length cannot be greater than",
    ):
        string_length_validator(contract)


def test_string_length_metric_supports_equal_minimum_and_maximum(
    spark,
):
    df = spark.createDataFrame(
        [
            ("a",),
            ("ab",),
            ("xy",),
            ("abc",),
            (None,),
        ],
        "product_name STRING",
    )

    contract = {
        "schema": [
            {
                "name": "product_name",
                "constraints": {
                    "minimum_string_length": 2,
                    "maximum_string_length": 2,
                },
            }
        ]
    }

    metrics = string_length_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["product_name_string_length"] == 2


def test_string_length_validator_accepts_zero_as_minimum(spark):
    df = spark.createDataFrame(
        [
            ("",),
            ("a",),
            ("abc",),
        ],
        "product_name STRING",
    )

    contract = {
        "schema": [
            {
                "name": "product_name",
                "constraints": {
                    "minimum_string_length": 0,
                    "maximum_string_length": 3,
                },
            }
        ]
    }

    metrics = string_length_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["product_name_string_length"] == 0
