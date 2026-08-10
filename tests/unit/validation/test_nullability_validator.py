import pytest

from instacart_etl_rnn.validation.exceptions import InvalidContractError
from instacart_etl_rnn.validation.nullability import nullability_validator


def test_build_nullability_metrics_builds_expected_metrics():
    contract = {
        "schema": [
            {
                "name": "order_id",
                "nullable": False,
            },
            {
                "name": "user_id",
                "nullable": False,
            },
            {
                "name": "days_since_prior_order",
                "nullable": True,
            },
        ]
    }

    metrics = nullability_validator(contract)

    assert len(metrics) == 2

    assert [metric.alias for metric in metrics] == [
        "order_id_nullability",
        "user_id_nullability",
    ]

    assert [metric.rule_name for metric in metrics] == [
        "order_id.nullability",
        "user_id.nullability",
    ]

    assert [metric.validation_type for metric in metrics] == [
        "nullability",
        "nullability",
    ]

    assert [metric.columns for metric in metrics] == [
        ("order_id",),
        ("user_id",),
    ]


@pytest.mark.parametrize(
    "nullable",
    [
        "false",
        0,
        1,
        [],
        {},
        None,
    ],
)
def test_build_nullability_metrics_rejects_non_boolean_nullable(
    nullable,
):
    contract = {
        "schema": [
            {
                "name": "order_id",
                "nullable": nullable,
            },
        ]
    }

    with pytest.raises(
        InvalidContractError,
        match="must be a boolean",
    ):
        nullability_validator(contract)


def test_nullability_metrics_count_nulls_correctly(spark):
    df = spark.createDataFrame(
        [
            (1, 10, "prior"),
            (2, None, "train"),
            (None, 20, None),
            (4, None, "test"),
        ],
        schema=("order_id INT, user_id INT, eval_set STRING"),
    )

    contract = {
        "schema": [
            {
                "name": "order_id",
                "nullable": False,
            },
            {
                "name": "user_id",
                "nullable": False,
            },
            {
                "name": "eval_set",
                "nullable": False,
            },
        ]
    }

    metrics = nullability_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["order_id_nullability"] == 1
    assert row["user_id_nullability"] == 2
    assert row["eval_set_nullability"] == 1


def test_nullability_metrics_return_zero_when_no_nulls(spark):
    df = spark.createDataFrame(
        [
            (1, 10),
            (2, 20),
            (3, 30),
        ],
        "order_id INT, user_id INT",
    )

    contract = {
        "schema": [
            {
                "name": "order_id",
                "nullable": False,
            },
            {
                "name": "user_id",
                "nullable": False,
            },
        ]
    }

    metrics = nullability_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["order_id_nullability"] == 0
    assert row["user_id_nullability"] == 0


def test_multiple_nullability_metrics_can_be_aggregated_together(
    spark,
):
    df = spark.createDataFrame(
        [
            (1, None, None),
            (None, 2, None),
        ],
        "a INT, b INT, c INT",
    )

    contract = {
        "schema": [
            {"name": "a", "nullable": False},
            {"name": "b", "nullable": False},
            {"name": "c", "nullable": False},
        ]
    }

    metrics = nullability_validator(contract)

    assert len(metrics) == 3

    result = df.agg(*[metric.expression for metric in metrics]).first()

    assert result["a_nullability"] == 1
    assert result["b_nullability"] == 1
    assert result["c_nullability"] == 2


def test_build_nullability_metrics_skips_nullable_column_without_threshold():
    contract = {
        "schema": [
            {
                "name": "comment",
                "nullable": True,
            }
        ]
    }

    metrics = nullability_validator(contract)

    assert metrics == []


def test_build_nullability_metrics_includes_nullable_column_with_threshold():
    contract = {
        "schema": [
            {
                "name": "comment",
                "nullable": True,
                "thresholds": {
                    "nullability": {
                        "max_failed_percent": 10.0,
                    }
                },
            }
        ]
    }

    metrics = nullability_validator(contract)

    assert len(metrics) == 1

    metric = metrics[0]

    assert metric.alias == "comment_nullability"
    assert metric.rule_name == "comment.nullability"
    assert metric.validation_type == "nullability"
    assert metric.columns == ("comment",)
