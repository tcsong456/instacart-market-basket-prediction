import pytest

from instacart_etl_rnn.validation.allowed_values import (
    _build_invalid_condition,
    allowed_values_validator,
)
from instacart_etl_rnn.validation.exceptions import InvalidConstraintError


def test_invalid_condition_identifies_exact_invalid_rows(spark):
    df = spark.createDataFrame(
        [
            (1, "prior"),
            (2, "train"),
            (3, "test"),
            (4, "wrong"),
            (5, "invalid"),
            (6, None),
        ],
        "id INT, eval_set STRING",
    )

    condition = _build_invalid_condition(
        "eval_set",
        ["prior", "train", "test"],
    )

    invalid_ids = {row["id"] for row in (df.filter(condition).select("id").collect())}

    assert invalid_ids == {4, 5}


def test_allowed_values_metric_counts_invalid_values(spark):
    df = spark.createDataFrame(
        [
            ("prior",),
            ("train",),
            ("wrong",),
            ("test",),
            ("invalid",),
            (None,),
        ],
        "eval_set STRING",
    )

    contract = {
        "schema": [
            {
                "name": "eval_set",
                "constraints": {
                    "allowed_values": [
                        "prior",
                        "train",
                        "test",
                    ]
                },
            }
        ]
    }

    metrics = allowed_values_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["eval_set_allowed_values"] == 2


def test_allowed_values_metric_returns_zero_when_all_values_are_allowed(
    spark,
):
    df = spark.createDataFrame(
        [
            ("prior",),
            ("train",),
            ("test",),
        ],
        "eval_set STRING",
    )

    contract = {
        "schema": [
            {
                "name": "eval_set",
                "constraints": {
                    "allowed_values": [
                        "prior",
                        "train",
                        "test",
                    ]
                },
            }
        ]
    }

    metrics = allowed_values_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["eval_set_allowed_values"] == 0


def test_allowed_values_metric_ignores_null_values(spark):
    df = spark.createDataFrame(
        [
            (None,),
            (None,),
            (None,),
        ],
        "eval_set STRING",
    )

    contract = {
        "schema": [
            {
                "name": "eval_set",
                "constraints": {
                    "allowed_values": [
                        "prior",
                        "train",
                        "test",
                    ]
                },
            }
        ]
    }

    metrics = allowed_values_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["eval_set_allowed_values"] == 0


def test_build_allowed_values_metrics_builds_only_for_constrained_columns():
    contract = {
        "schema": [
            {
                "name": "eval_set",
                "constraints": {
                    "allowed_values": [
                        "prior",
                        "train",
                        "test",
                    ]
                },
            },
            {
                "name": "order_id",
                "constraints": {},
            },
        ]
    }

    metrics = allowed_values_validator(contract)

    assert len(metrics) == 1

    metric = metrics[0]

    assert metric.alias == "eval_set_allowed_values"
    assert metric.rule_name == "eval_set.allowed_values"
    assert metric.validation_type == "allowed_values"
    assert metric.columns == ("eval_set",)


def test_multiple_allowed_values_metrics_can_be_aggregated_together(
    spark,
):
    df = spark.createDataFrame(
        [
            ("prior", "UK"),
            ("wrong", "UK"),
            ("train", "US"),
            ("test", "XX"),
        ],
        "eval_set STRING, country STRING",
    )

    contract = {
        "schema": [
            {
                "name": "eval_set",
                "constraints": {
                    "allowed_values": [
                        "prior",
                        "train",
                        "test",
                    ]
                },
            },
            {
                "name": "country",
                "constraints": {
                    "allowed_values": [
                        "UK",
                        "US",
                    ]
                },
            },
        ]
    }

    metrics = allowed_values_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["eval_set_allowed_values"] == 1
    assert row["country_allowed_values"] == 1


@pytest.mark.parametrize(
    "allowed_values",
    [
        "prior",
        1,
        {},
        True,
    ],
)
def test_build_allowed_values_metrics_rejects_non_list_allowed_values(
    allowed_values,
):
    contract = {
        "schema": [
            {
                "name": "eval_set",
                "constraints": {
                    "allowed_values": allowed_values,
                },
            }
        ]
    }

    with pytest.raises(
        InvalidConstraintError,
        match="must be a list",
    ):
        allowed_values_validator(contract)


def test_build_allowed_values_metrics_rejects_empty_allowed_values():
    contract = {
        "schema": [
            {
                "name": "eval_set",
                "constraints": {
                    "allowed_values": [],
                },
            }
        ]
    }

    with pytest.raises(
        InvalidConstraintError,
        match="must not be empty",
    ):
        allowed_values_validator(contract)


def test_invalid_condition_treats_empty_string_as_normal_value(spark):
    df = spark.createDataFrame(
        [
            (1, ""),
            (2, "prior"),
            (3, None),
        ],
        "id INT, eval_set STRING",
    )

    condition = _build_invalid_condition(
        "eval_set",
        ["prior", "train", "test"],
    )

    invalid_ids = {row["id"] for row in df.filter(condition).collect()}

    assert invalid_ids == {1}


def test_allowed_values_metric_supports_numeric_values(spark):
    df = spark.createDataFrame(
        [
            (1, 0),
            (2, 1),
            (3, 2),
            (4, -1),
            (5, None),
        ],
        "id INT, reordered INT",
    )

    contract = {
        "schema": [
            {
                "name": "reordered",
                "constraints": {
                    "allowed_values": [0, 1],
                },
            }
        ]
    }

    metrics = allowed_values_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["reordered_allowed_values"] == 2


def test_empty_string_is_valid_when_explicitly_allowed(spark):
    df = spark.createDataFrame(
        [
            (1, ""),
            (2, "prior"),
            (3, "train"),
            (4, "wrong"),
            (5, None),
        ],
        "id INT, eval_set STRING",
    )

    condition = _build_invalid_condition(
        "eval_set",
        ["", "prior", "train"],
    )

    invalid_ids = {row["id"] for row in (df.filter(condition).select("id").collect())}

    assert invalid_ids == {4}
