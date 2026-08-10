import pytest

from instacart_etl_rnn.validation.exceptions import InvalidConstraintError
from instacart_etl_rnn.validation.range import (
    _build_range_invalid_condition,
    range_validator,
)


def test_build_range_metrics_builds_expected_metrics():
    contract = {
        "schema": [
            {
                "name": "order_id",
                "constraints": {
                    "minimum": 1,
                },
            },
            {
                "name": "order_dow",
                "constraints": {
                    "minimum": 0,
                    "maximum": 6,
                },
            },
            {
                "name": "eval_set",
                "constraints": {},
            },
        ]
    }

    metrics = range_validator(contract)

    assert len(metrics) == 2

    assert [metric.alias for metric in metrics] == [
        "order_id_range",
        "order_dow_range",
    ]

    assert [metric.rule_name for metric in metrics] == [
        "order_id.range",
        "order_dow.range",
    ]

    assert [metric.validation_type for metric in metrics] == [
        "range",
        "range",
    ]

    assert [metric.columns for metric in metrics] == [
        ("order_id",),
        ("order_dow",),
    ]


def test_range_metric_counts_values_below_minimum(spark):
    df = spark.createDataFrame(
        [
            (0,),
            (1,),
            (2,),
            (10,),
            (None,),
        ],
        "order_id INT",
    )

    contract = {
        "schema": [
            {
                "name": "order_id",
                "constraints": {
                    "minimum": 1,
                },
            }
        ]
    }

    metrics = range_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["order_id_range"] == 1


def test_range_metric_counts_values_above_maximum(spark):
    df = spark.createDataFrame(
        [
            (0,),
            (6,),
            (7,),
            (10,),
            (None,),
        ],
        "order_dow INT",
    )

    contract = {
        "schema": [
            {
                "name": "order_dow",
                "constraints": {
                    "maximum": 6,
                },
            }
        ]
    }

    metrics = range_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["order_dow_range"] == 2


def test_range_metric_ignores_null_values(spark):
    df = spark.createDataFrame(
        [
            (None,),
            (None,),
            (None,),
        ],
        "order_dow INT",
    )

    contract = {
        "schema": [
            {
                "name": "order_dow",
                "constraints": {
                    "minimum": 0,
                    "maximum": 6,
                },
            }
        ]
    }

    metrics = range_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["order_dow_range"] == 0


def test_multiple_range_metrics_can_be_aggregated_together(spark):
    df = spark.createDataFrame(
        [
            (0, -1, 25),
            (1, 0, 23),
            (2, 6, 12),
            (3, 7, 24),
        ],
        "order_id INT, order_dow INT, order_hour INT",
    )

    contract = {
        "schema": [
            {
                "name": "order_id",
                "constraints": {
                    "minimum": 1,
                },
            },
            {
                "name": "order_dow",
                "constraints": {
                    "minimum": 0,
                    "maximum": 6,
                },
            },
            {
                "name": "order_hour",
                "constraints": {
                    "minimum": 0,
                    "maximum": 23,
                },
            },
        ]
    }

    metrics = range_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["order_id_range"] == 1
    assert row["order_dow_range"] == 2
    assert row["order_hour_range"] == 2


@pytest.mark.parametrize(
    "constraint_name,value",
    [
        ("minimum", "1"),
        ("minimum", True),
        ("minimum", []),
        ("maximum", "10"),
        ("maximum", False),
        ("maximum", {}),
    ],
)
def test_build_range_metrics_rejects_non_numeric_constraints(
    constraint_name,
    value,
):
    contract = {
        "schema": [
            {
                "name": "order_id",
                "constraints": {
                    constraint_name: value,
                },
            }
        ]
    }

    with pytest.raises(
        InvalidConstraintError,
        match="must be a valid number",
    ):
        range_validator(contract)


def test_build_range_metrics_rejects_minimum_greater_than_maximum():
    contract = {
        "schema": [
            {
                "name": "order_dow",
                "constraints": {
                    "minimum": 10,
                    "maximum": 5,
                },
            }
        ]
    }

    with pytest.raises(
        InvalidConstraintError,
        match="minimum cannot be greater than maximum",
    ):
        range_validator(contract)


def test_range_metric_allows_equal_minimum_and_maximum(spark):
    df = spark.createDataFrame(
        [
            (4,),
            (5,),
            (6,),
        ],
        "value INT",
    )

    contract = {
        "schema": [
            {
                "name": "value",
                "constraints": {
                    "minimum": 5,
                    "maximum": 5,
                },
            }
        ]
    }

    metrics = range_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["value_range"] == 2


def test_range_metric_returns_zero_for_values_within_inclusive_range(
    spark,
):
    df = spark.createDataFrame(
        [
            (0,),
            (1,),
            (3,),
            (5,),
            (6,),
        ],
        "order_dow INT",
    )

    contract = {
        "schema": [
            {
                "name": "order_dow",
                "constraints": {
                    "minimum": 0,
                    "maximum": 6,
                },
            }
        ]
    }

    metrics = range_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["order_dow_range"] == 0


def test_range_invalid_condition_identifies_exact_invalid_rows(spark):
    df = spark.createDataFrame(
        [
            (1, -1),
            (2, 0),
            (3, 3),
            (4, 6),
            (5, 7),
            (6, None),
        ],
        "id INT, order_dow INT",
    )

    condition = _build_range_invalid_condition(
        "order_dow",
        minimum=0,
        maximum=6,
    )

    actual = {row.id for row in df.filter(condition).collect()}

    assert actual == {1, 5}
