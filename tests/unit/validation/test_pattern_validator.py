import pytest

from instacart_etl_rnn.validation.exceptions import InvalidConstraintError
from instacart_etl_rnn.validation.pattern import (
    _build_invalid_condition,
    pattern_validator,
)


def test_pattern_invalid_condition_identifies_exact_invalid_rows(
    spark,
):
    df = spark.createDataFrame(
        [
            (1, "AB12"),
            (2, "XY99"),
            (3, "ab12"),
            (4, "ABC12"),
            (5, ""),
            (6, None),
        ],
        "id INT, code STRING",
    )

    condition = _build_invalid_condition(
        "code",
        r"^[A-Z]{2}[0-9]{2}$",
    )

    invalid_ids = {row["id"] for row in (df.filter(condition).select("id").collect())}

    assert invalid_ids == {3, 4, 5}


def test_pattern_metric_counts_invalid_values(spark):
    df = spark.createDataFrame(
        [
            ("AB12",),
            ("XY99",),
            ("ab12",),
            ("ABC12",),
            ("",),
            (None,),
        ],
        "code STRING",
    )

    contract = {
        "schema": [
            {
                "name": "code",
                "constraints": {
                    "pattern": r"^[A-Z]{2}[0-9]{2}$",
                },
            }
        ]
    }

    metrics = pattern_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["code_pattern"] == 3


def test_pattern_metric_returns_zero_when_all_values_match(
    spark,
):
    df = spark.createDataFrame(
        [
            ("AB12",),
            ("CD34",),
            ("XY99",),
        ],
        "code STRING",
    )

    contract = {
        "schema": [
            {
                "name": "code",
                "constraints": {
                    "pattern": r"^[A-Z]{2}[0-9]{2}$",
                },
            }
        ]
    }

    metrics = pattern_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["code_pattern"] == 0


def test_pattern_metric_ignores_null_values(spark):
    df = spark.createDataFrame(
        [
            (None,),
            (None,),
        ],
        "code STRING",
    )

    contract = {
        "schema": [
            {
                "name": "code",
                "constraints": {
                    "pattern": r"^[A-Z]{2}[0-9]{2}$",
                },
            }
        ]
    }

    metrics = pattern_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["code_pattern"] == 0


def test_pattern_validator_builds_expected_metric():
    contract = {
        "schema": [
            {
                "name": "postcode",
                "constraints": {
                    "pattern": r"^[A-Z]{2}[0-9]{2}$",
                },
            },
            {
                "name": "order_id",
                "constraints": {},
            },
        ]
    }

    metrics = pattern_validator(contract)

    assert len(metrics) == 1

    metric = metrics[0]

    assert metric.alias == "postcode_pattern"
    assert metric.rule_name == "postcode.pattern"
    assert metric.validation_type == "pattern"
    assert metric.columns == ("postcode",)


def test_multiple_pattern_metrics_can_be_aggregated_together(
    spark,
):
    df = spark.createDataFrame(
        [
            ("AB12", "SW1A 1AA"),
            ("wrong", "SW1A 1AA"),
            ("CD34", "invalid"),
        ],
        "code STRING, postcode STRING",
    )

    contract = {
        "schema": [
            {
                "name": "code",
                "constraints": {
                    "pattern": r"^[A-Z]{2}[0-9]{2}$",
                },
            },
            {
                "name": "postcode",
                "constraints": {
                    "pattern": (
                        r"^[A-Z]{1,2}[0-9][0-9A-Z]? "
                        r"[0-9][A-Z]{2}$"
                    ),
                },
            },
        ]
    }

    metrics = pattern_validator(contract)

    row = df.agg(*[metric.expression for metric in metrics]).first()

    assert row["code_pattern"] == 1
    assert row["postcode_pattern"] == 1


@pytest.mark.parametrize(
    "pattern",
    [
        123,
        True,
        [],
        {},
    ],
)
def test_pattern_validator_rejects_non_string_pattern(
    pattern,
):
    contract = {
        "schema": [
            {
                "name": "code",
                "constraints": {
                    "pattern": pattern,
                },
            }
        ]
    }

    with pytest.raises(
        InvalidConstraintError,
        match="must be a string",
    ):
        pattern_validator(contract)


@pytest.mark.parametrize(
    "pattern",
    [
        "",
        " ",
        "   ",
        "\t",
        "\n",
    ],
)
def test_pattern_validator_rejects_blank_pattern(pattern):
    contract = {
        "schema": [
            {
                "name": "code",
                "constraints": {
                    "pattern": pattern,
                },
            }
        ]
    }

    with pytest.raises(
        InvalidConstraintError,
        match="must not be empty or whitespace-only",
    ):
        pattern_validator(contract)
