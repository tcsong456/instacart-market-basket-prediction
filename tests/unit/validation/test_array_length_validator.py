import pytest

from instacart_etl_rnn.validation.array_length import (
    _build_invalid_condition,
    array_length_validator,
)
from instacart_etl_rnn.validation.exceptions import InvalidConstraintError


def test_array_length_validator_builds_only_for_constrained_columns():
    contract = {
        "schema": [
            {
                "name": "product_ids",
                "constraints": {
                    "minimum_array_length": 1,
                    "maximum_array_length": 5,
                },
            },
            {
                "name": "aisle_ids",
                "constraints": {},
            },
        ]
    }

    metrics = array_length_validator(contract)

    assert len(metrics) == 1

    metric = metrics[0]

    assert metric.alias == "product_ids_array_length"
    assert metric.rule_name == "product_ids.array_length"
    assert metric.validation_type == "array_length"
    assert metric.columns == ("product_ids",)


def test_array_length_invalid_condition_identifies_exact_invalid_rows(
    spark,
):
    df = spark.createDataFrame(
        [
            (2, []),
            (1, [1]),
            (3, [1, 2]),
            (5, [1, 2, 3]),
            (4, [1, 2, 3, 4]),
            (6, None),
        ],
        "id INT, product_ids ARRAY<INT>",
    )

    condition = _build_invalid_condition(
        "product_ids",
        minimum=1,
        maximum=3,
    )

    invalid_ids = {row["id"] for row in (df.filter(condition).select("id").collect())}

    assert invalid_ids == {2, 4}


def test_array_length_metric_counts_mixed_valid_and_invalid_arrays(
    spark,
):
    df = spark.createDataFrame(
        [
            ([],),
            ([1],),
            ([1, 2],),
            ([1, 2, 3],),
            ([1, 2, 3, 4],),
            ([1, 2, 3, 4, 5],),
            (None,),
        ],
        "product_ids ARRAY<INT>",
    )

    contract = {
        "schema": [
            {
                "name": "product_ids",
                "constraints": {
                    "minimum_array_length": 1,
                    "maximum_array_length": 3,
                },
            }
        ]
    }

    metrics = array_length_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["product_ids_array_length"] == 3


def test_array_length_metric_returns_zero_when_all_arrays_are_valid(
    spark,
):
    df = spark.createDataFrame(
        [
            ([1],),
            ([1, 2],),
            ([1, 2, 3],),
        ],
        "product_ids ARRAY<INT>",
    )

    contract = {
        "schema": [
            {
                "name": "product_ids",
                "constraints": {
                    "minimum_array_length": 1,
                    "maximum_array_length": 3,
                },
            }
        ]
    }

    metrics = array_length_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["product_ids_array_length"] == 0


def test_array_length_metric_accepts_exact_minimum_and_maximum(
    spark,
):
    df = spark.createDataFrame(
        [
            ([1],),
            ([1, 2, 3],),
        ],
        "product_ids ARRAY<INT>",
    )

    contract = {
        "schema": [
            {
                "name": "product_ids",
                "constraints": {
                    "minimum_array_length": 1,
                    "maximum_array_length": 3,
                },
            }
        ]
    }

    metrics = array_length_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["product_ids_array_length"] == 0


def test_array_length_metric_ignores_null_arrays(spark):
    df = spark.createDataFrame(
        [
            (None,),
            (None,),
        ],
        "product_ids ARRAY<INT>",
    )

    contract = {
        "schema": [
            {
                "name": "product_ids",
                "constraints": {
                    "minimum_array_length": 1,
                    "maximum_array_length": 3,
                },
            }
        ]
    }

    metrics = array_length_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["product_ids_array_length"] == 0


def test_array_length_metric_supports_minimum_only(spark):
    df = spark.createDataFrame(
        [
            ([],),
            ([1],),
            ([1, 2],),
        ],
        "product_ids ARRAY<INT>",
    )

    contract = {
        "schema": [
            {
                "name": "product_ids",
                "constraints": {
                    "minimum_array_length": 2,
                },
            }
        ]
    }

    metrics = array_length_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["product_ids_array_length"] == 2


def test_array_length_metric_supports_maximum_only(spark):
    df = spark.createDataFrame(
        [
            ([],),
            ([1],),
            ([1, 2],),
            ([1, 2, 3],),
        ],
        "product_ids ARRAY<INT>",
    )

    contract = {
        "schema": [
            {
                "name": "product_ids",
                "constraints": {
                    "maximum_array_length": 2,
                },
            }
        ]
    }

    metrics = array_length_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["product_ids_array_length"] == 1


def test_array_length_metric_supports_equal_minimum_and_maximum(
    spark,
):
    df = spark.createDataFrame(
        [
            ([1],),
            ([1, 2],),
            ([3, 4],),
            ([1, 2, 3],),
        ],
        "product_ids ARRAY<INT>",
    )

    contract = {
        "schema": [
            {
                "name": "product_ids",
                "constraints": {
                    "minimum_array_length": 2,
                    "maximum_array_length": 2,
                },
            }
        ]
    }

    metrics = array_length_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["product_ids_array_length"] == 2


def test_array_length_validator_accepts_zero_as_minimum(spark):
    df = spark.createDataFrame(
        [
            ([],),
            ([1],),
            ([1, 2],),
        ],
        "product_ids ARRAY<INT>",
    )

    contract = {
        "schema": [
            {
                "name": "product_ids",
                "constraints": {
                    "minimum_array_length": 0,
                    "maximum_array_length": 2,
                },
            }
        ]
    }

    metrics = array_length_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["product_ids_array_length"] == 0


@pytest.mark.parametrize(
    "constraint_name,value",
    [
        ("minimum_array_length", -1),
        ("minimum_array_length", 2.0),
        ("minimum_array_length", "2"),
        ("minimum_array_length", True),
        ("maximum_array_length", -1),
        ("maximum_array_length", 3.0),
        ("maximum_array_length", "3"),
        ("maximum_array_length", False),
    ],
)
def test_array_length_validator_rejects_invalid_constraints(
    constraint_name,
    value,
):
    contract = {
        "schema": [
            {
                "name": "product_ids",
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
        array_length_validator(contract)


def test_array_length_validator_rejects_minimum_greater_than_maximum():
    contract = {
        "schema": [
            {
                "name": "product_ids",
                "constraints": {
                    "minimum_array_length": 5,
                    "maximum_array_length": 2,
                },
            }
        ]
    }

    with pytest.raises(
        InvalidConstraintError,
        match="minimum array length cannot be greater",
    ):
        array_length_validator(contract)


def test_multiple_array_length_metrics_can_be_aggregated_together(
    spark,
):
    df = spark.createDataFrame(
        [
            ([1], [10, 20]),
            ([], [10]),
            ([1, 2, 3], [10, 20, 30]),
        ],
        "product_ids ARRAY<INT>, aisle_ids ARRAY<INT>",
    )

    contract = {
        "schema": [
            {
                "name": "product_ids",
                "constraints": {
                    "minimum_array_length": 1,
                    "maximum_array_length": 2,
                },
            },
            {
                "name": "aisle_ids",
                "constraints": {
                    "minimum_array_length": 2,
                    "maximum_array_length": 2,
                },
            },
        ]
    }

    metrics = array_length_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["product_ids_array_length"] == 2
    assert row["aisle_ids_array_length"] == 2
