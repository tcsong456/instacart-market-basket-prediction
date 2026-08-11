import re

import pytest
from pyspark.sql.types import IntegerType, StructField, StructType

from instacart_etl_rnn.validation.exceptions import InvalidContractError
from instacart_etl_rnn.validation.row_logic import (
    _build_aggregate_expression,
    _group_aggregate_fields,
    validate_row_logic,
)


def test_validate_row_logic_all_rules_pass(
    spark,
    orders_schema,
    row_logic_contract,
):
    df = spark.createDataFrame(
        [
            (1, 1, "prior", None),
            (1, 2, "prior", 7.0),
            (1, 3, "train", 4.0),
            (2, 1, "prior", None),
            (2, 2, "test", 5.0),
        ],
        schema=orders_schema,
    )

    results = validate_row_logic(
        df,
        contract=row_logic_contract,
    )

    assert len(results) == 5

    for result in results:
        assert result.category == "row_logic"
        assert result.passed is True
        assert result.failed_count == 0
        assert result.invalid_rows is None
        assert result.message == f"Rule {result.rule_name!r} passed"


def test_validate_row_logic_returns_expected_failure_counts(
    spark,
    orders_schema,
    row_logic_contract,
):
    df = spark.createDataFrame(
        [
            (1, 1, "prior", 5.0),
            (1, 2, "prior", None),
            (1, 3, "prior", 4.0),
            (2, 1, "prior", None),
            (2, 3, "train", 6.0),
        ],
        schema=orders_schema,
    )

    results = validate_row_logic(
        df,
        contract=row_logic_contract,
    )

    results_by_name = {result.rule_name: result for result in results}

    expected_failed_counts = {
        "first_order_has_no_prior_interval": 1,
        "later_orders_must_have_prior_interval": 1,
        "last_order_per_user_is_train_or_test": 1,
        "only_one_train_or_test_per_user": 3,
        "contiguous_order_numbers": 2,
    }

    assert set(results_by_name) == set(expected_failed_counts)

    for rule_name, expected_count in expected_failed_counts.items():
        result = results_by_name[rule_name]

        assert result.passed is False
        assert result.failed_count == expected_count
        assert result.invalid_rows is not None
        assert result.message == (
            f"Rule {rule_name!r} failed for {expected_count} row(s)"
        )


def test_validate_row_logic_returns_correct_invalid_rows(
    spark,
    orders_schema,
    row_logic_contract,
):
    df = spark.createDataFrame(
        [
            (1, 1, "prior", None),
            (1, 3, "train", 5.0),
            (2, 1, "prior", 10.0),
            (2, 2, "test", 20.0),
        ],
        schema=orders_schema,
    )

    results = validate_row_logic(
        df,
        contract=row_logic_contract,
    )

    results_by_name = {result.rule_name: result for result in results}

    result = results_by_name["first_order_has_no_prior_interval"]

    assert result.invalid_rows is not None

    invalid_rows = result.invalid_rows.collect()

    assert len(invalid_rows) == 1

    row = invalid_rows[0]

    assert row["user_id"] == 2
    assert row["order_number"] == 1
    assert row["days_since_prior_order"] == 10.0

    result = results_by_name["contiguous_order_numbers"]

    assert result.invalid_rows is not None

    actual_rows = {
        (
            row["user_id"],
            row["order_number"],
            row["days_since_prior_order"],
        )
        for row in result.invalid_rows.collect()
    }

    expected_rows = {
        (1, 1, None),
        (1, 3, 5.0),
    }

    assert actual_rows == expected_rows

    passed_rules = [
        "later_orders_must_have_prior_interval",
        "last_order_per_user_is_train_or_test",
        "only_one_train_or_test_per_user",
    ]

    for rule_name in passed_rules:
        assert results_by_name[rule_name].passed


def test_validate_row_logic_limits_invalid_rows(
    spark,
    orders_schema,
    row_logic_contract,
):
    df = spark.createDataFrame(
        [(user_id, 1, "prior", 10.0) for user_id in range(1, 31)],
        schema=orders_schema,
    )

    results = validate_row_logic(
        df,
        contract=row_logic_contract,
    )

    results_by_name = {result.rule_name: result for result in results}

    result = results_by_name["first_order_has_no_prior_interval"]

    assert result.failed_count == 30
    assert result.invalid_rows is not None
    assert result.invalid_rows.count() == 20


def test_validate_row_logic_returns_empty_when_no_rules(
    spark,
    orders_schema,
):
    df = spark.createDataFrame(
        [
            (1, 1, "prior", None),
        ],
        schema=orders_schema,
    )

    contract = {
        "rules": [],
    }

    results = validate_row_logic(
        df,
        contract=contract,
    )

    assert results == []


@pytest.mark.parametrize(
    "name",
    [
        None,
        "",
        "   ",
        123,
        [],
    ],
)
def test_validate_row_logic_rejects_invalid_rule_name(
    spark,
    orders_schema,
    name,
):
    df = spark.createDataFrame(
        [(1, 1, "prior", None)],
        schema=orders_schema,
    )

    contract = {
        "rules": [
            {
                "name": name,
                "expression": "order_number > 0",
            }
        ]
    }

    with pytest.raises(
        InvalidContractError,
        match="Every business rule must have a non-empty name",
    ):
        validate_row_logic(
            df,
            contract=contract,
        )


def test_validate_row_logic_rejects_duplicate_rule_names(
    spark,
    orders_schema,
):
    df = spark.createDataFrame(
        [(1, 1, "prior", None)],
        schema=orders_schema,
    )

    contract = {
        "rules": [
            {
                "name": "same_rule",
                "expression": "order_number > 0",
            },
            {
                "name": "same_rule",
                "expression": "order_number < 10",
            },
        ]
    }

    with pytest.raises(
        InvalidContractError,
        match=re.escape("Duplicate business rule name: 'same_rule'"),
    ):
        validate_row_logic(
            df,
            contract=contract,
        )


@pytest.mark.parametrize(
    "expression",
    [
        None,
        "",
        "   ",
        123,
        [],
    ],
)
def test_validate_row_logic_rejects_invalid_rule_expression(
    spark,
    orders_schema,
    expression,
):
    df = spark.createDataFrame(
        [(1, 1, "prior", None)],
        schema=orders_schema,
    )

    contract = {
        "rules": [
            {
                "name": "test_rule",
                "expression": expression,
            }
        ]
    }

    with pytest.raises(
        InvalidContractError,
        match=re.escape("Rule 'test_rule' must define a non-empty expression"),
    ):
        validate_row_logic(
            df,
            contract=contract,
        )


@pytest.mark.parametrize(
    "name",
    [
        None,
        "",
        "   ",
        123,
    ],
)
def test_validate_row_logic_rejects_invalid_derived_field_name(
    spark,
    orders_schema,
    name,
):
    df = spark.createDataFrame(
        [(1, 1, "prior", None)],
        schema=orders_schema,
    )

    contract = {
        "derived_fields": [
            {
                "name": name,
                "aggregation": "max",
                "column": "order_number",
                "partition_by": ["user_id"],
            }
        ],
        "rules": [
            {
                "name": "dummy",
                "expression": "order_number > 0",
            }
        ],
    }

    with pytest.raises(
        InvalidContractError,
        match="Every derived field must have a non-empty name",
    ):
        validate_row_logic(
            df,
            contract=contract,
        )


def test_validate_row_logic_rejects_duplicate_derived_field_names(
    spark,
    orders_schema,
):
    df = spark.createDataFrame(
        [(1, 1, "prior", None)],
        schema=orders_schema,
    )

    contract = {
        "derived_fields": [
            {
                "name": "user_order_number",
                "aggregation": "min",
                "column": "order_number",
                "partition_by": ["user_id"],
            },
            {
                "name": "user_order_number",
                "aggregation": "max",
                "column": "order_number",
                "partition_by": ["user_id"],
            },
        ],
        "rules": [
            {
                "name": "dummy",
                "expression": "order_number > 0",
            }
        ],
    }

    with pytest.raises(
        InvalidContractError,
        match=re.escape("Duplicate derived field name: 'user_order_number'"),
    ):
        validate_row_logic(
            df,
            contract=contract,
        )


def test_validate_row_logic_rejects_missing_aggregation_and_expression(
    spark,
    orders_schema,
):
    df = spark.createDataFrame(
        [(1, 1, "prior", None)],
        schema=orders_schema,
    )

    contract = {
        "derived_fields": [
            {
                "name": "train_or_test_count",
                "type": "integer",
            }
        ],
        "rules": [
            {
                "name": "dummy",
                "expression": "order_number > 0",
            }
        ],
    }

    with pytest.raises(
        InvalidContractError,
        match=re.escape(
            "Derived field 'train_or_test_count' "
            "must define either 'aggregation' or 'expression'"
        ),
    ):
        validate_row_logic(
            df,
            contract=contract,
        )


def test_validate_row_logic_rejects_both_aggregation_and_expression(
    spark,
    orders_schema,
):
    df = spark.createDataFrame(
        [(1, 1, "prior", None)],
        schema=orders_schema,
    )

    contract = {
        "derived_fields": [
            {
                "name": "train_or_test_count",
                "aggregation": "min",
                "column": "order_number",
                "partition_by": ["user_id"],
                "expression": "order_number > 0",
            }
        ],
        "rules": [
            {
                "name": "dummy",
                "expression": "order_number > 0",
            }
        ],
    }

    with pytest.raises(
        InvalidContractError,
        match=re.escape(
            "Derived field 'train_or_test_count' "
            "cannot define both 'aggregation' and 'expression'"
        ),
    ):
        validate_row_logic(
            df,
            contract=contract,
        )


def test_validate_row_logic_rejects_unsupported_aggregation(
    spark,
    orders_schema,
):
    df = spark.createDataFrame(
        [(1, 1, "prior", None)],
        schema=orders_schema,
    )

    contract = {
        "derived_fields": [
            {
                "name": "train_or_test_count",
                "aggregation": "min_max",
                "column": "order_number",
                "partition_by": ["user_id"],
            }
        ],
        "rules": [
            {
                "name": "dummy",
                "expression": "order_number > 0",
            }
        ],
    }

    with pytest.raises(
        InvalidContractError,
        match=re.escape(
            "Aggregation 'min_max' for derived field "
            "'train_or_test_count' is not supported"
        ),
    ):
        validate_row_logic(
            df,
            contract=contract,
        )


@pytest.mark.parametrize(
    "column",
    [
        None,
        "",
        "   ",
        123,
    ],
)
def test_validate_row_logic_rejects_invalid_aggregate_column(
    spark,
    orders_schema,
    column,
):
    df = spark.createDataFrame(
        [(1, 1, "prior", None)],
        schema=orders_schema,
    )

    contract = {
        "derived_fields": [
            {
                "name": "user_min_order_number",
                "aggregation": "min",
                "column": column,
                "partition_by": ["user_id"],
            }
        ],
        "rules": [
            {
                "name": "dummy",
                "expression": "order_number > 0",
            }
        ],
    }

    with pytest.raises(
        InvalidContractError,
        match=re.escape(
            "Derived field 'user_min_order_number' using 'min' must define a column"
        ),
    ):
        validate_row_logic(
            df,
            contract=contract,
        )


@pytest.mark.parametrize(
    "partition_by",
    [
        None,
        [],
        "",
        "user_id",
        ["user_id", ""],
        ["user_id", "   "],
        ["user_id", 123],
    ],
)
def test_validate_row_logic_rejects_invalid_partition_by(
    spark,
    orders_schema,
    partition_by,
):
    df = spark.createDataFrame(
        [(1, 1, "prior", None)],
        schema=orders_schema,
    )

    contract = {
        "derived_fields": [
            {
                "name": "user_max_order_number",
                "aggregation": "max",
                "column": "order_number",
                "partition_by": partition_by,
            }
        ],
        "rules": [
            {
                "name": "dummy",
                "expression": "order_number > 0",
            }
        ],
    }

    with pytest.raises(
        InvalidContractError,
        match=re.escape(
            "Derived aggregate field 'user_max_order_number' "
            "must define a non-empty 'partition_by' list"
        ),
    ):
        validate_row_logic(
            df,
            contract=contract,
        )


def test_validate_row_logic_rejects_duplicate_partition_columns(
    spark,
    orders_schema,
):
    df = spark.createDataFrame(
        [(1, 1, "prior", None)],
        schema=orders_schema,
    )

    contract = {
        "derived_fields": [
            {
                "name": "user_max_order_number",
                "aggregation": "max",
                "column": "order_number",
                "partition_by": [
                    "user_id",
                    "user_id",
                ],
            }
        ],
        "rules": [
            {
                "name": "dummy",
                "expression": "order_number > 0",
            }
        ],
    }

    with pytest.raises(
        InvalidContractError,
        match=re.escape(
            "'partition_by' for derived field "
            "'user_max_order_number' cannot contain "
            "duplicate columns"
        ),
    ):
        validate_row_logic(
            df,
            contract=contract,
        )


@pytest.mark.parametrize(
    "condition",
    [
        None,
        "",
        "   ",
        123,
    ],
)
def test_validate_row_logic_rejects_invalid_conditional_count_condition(
    spark,
    orders_schema,
    condition,
):
    df = spark.createDataFrame(
        [(1, 1, "prior", None)],
        schema=orders_schema,
    )

    contract = {
        "derived_fields": [
            {
                "name": "train_or_test_count",
                "aggregation": "conditional_count",
                "condition": condition,
                "partition_by": ["user_id"],
            }
        ],
        "rules": [
            {
                "name": "dummy",
                "expression": "order_number > 0",
            }
        ],
    }

    with pytest.raises(
        InvalidContractError,
        match=re.escape(
            "Derived field 'train_or_test_count' using "
            "'conditional_count' must define a non-empty "
            "condition"
        ),
    ):
        validate_row_logic(
            df,
            contract=contract,
        )


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "   ",
        123,
    ],
)
def test_validate_row_logic_rejects_invalid_derived_expression(
    spark,
    orders_schema,
    expression,
):
    df = spark.createDataFrame(
        [(1, 1, "prior", None)],
        schema=orders_schema,
    )

    contract = {
        "derived_fields": [
            {
                "name": "is_valid_order",
                "expression": expression,
            }
        ],
        "rules": [
            {
                "name": "dummy",
                "expression": "order_number > 0",
            }
        ],
    }

    with pytest.raises(
        InvalidContractError,
        match=re.escape(
            "Expression for derived field 'is_valid_order' must be a non-empty string"
        ),
    ):
        validate_row_logic(
            df,
            contract=contract,
        )


def test_group_aggregate_fields_groups_same_partitions():
    fields = [
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
            "name": "product_order_count",
            "aggregation": "count",
            "column": "order_id",
            "partition_by": ["product_id"],
        },
    ]

    grouped = _group_aggregate_fields(fields)

    assert set(grouped) == {
        ("user_id",),
        ("product_id",),
    }

    assert [field["name"] for field in grouped[("user_id",)]] == [
        "user_min_order_number",
        "user_max_order_number",
    ]

    assert [field["name"] for field in grouped[("product_id",)]] == [
        "product_order_count",
    ]


@pytest.mark.parametrize(
    ("field", "data", "expected"),
    [
        (
            {
                "aggregation": "min",
                "column": "value",
            },
            [(3,), (1,), (2,), (None,)],
            1,
        ),
        (
            {
                "aggregation": "max",
                "column": "value",
            },
            [(3,), (1,), (2,), (None,)],
            3,
        ),
        (
            {
                "aggregation": "sum",
                "column": "value",
            },
            [(3,), (1,), (2,), (None,)],
            6,
        ),
        (
            {
                "aggregation": "count",
                "column": "value",
            },
            [(3,), (1,), (2,), (None,)],
            3,
        ),
        (
            {
                "aggregation": "count_distinct",
                "column": "value",
            },
            [(1,), (1,), (2,), (None,)],
            2,
        ),
    ],
)
def test_build_aggregate_expression(
    spark,
    field,
    data,
    expected,
):
    df = spark.createDataFrame(
        data,
        ["value"],
    )

    result = df.agg(_build_aggregate_expression(field).alias("result")).first()[
        "result"
    ]

    assert result == expected


def test_build_aggregate_expression_conditional_count(
    spark,
):
    df = spark.createDataFrame(
        [
            (1,),
            (2,),
            (3,),
            (4,),
            (None,),
        ],
        ["value"],
    )

    field = {
        "aggregation": "conditional_count",
        "condition": "value > 2",
    }

    result = df.agg(_build_aggregate_expression(field).alias("result")).first()[
        "result"
    ]

    assert result == 2


def test_build_aggregate_expression_conditional_count_treats_null_as_false(
    spark,
):
    df = spark.createDataFrame(
        [
            (1,),
            (None,),
        ],
        ["value"],
    )

    field = {
        "aggregation": "conditional_count",
        "condition": "value > 0",
    }

    result = df.agg(_build_aggregate_expression(field).alias("result")).first()[
        "result"
    ]

    assert result == 1


def test_build_aggregate_expression_rejects_unsupported_aggregation():
    field = {
        "aggregation": "average",
        "column": "value",
    }

    with pytest.raises(
        InvalidContractError,
        match=re.escape("Unsupported aggregation: 'average'"),
    ):
        _build_aggregate_expression(field)


@pytest.mark.parametrize(
    (
        "rule",
        "failed_count",
        "passed",
        "failed_rows",
    ),
    [
        (
            {
                "name": "first_order_has_no_prior_interval",
                "expression": ("order_number <> 1 OR days_since_prior_order IS NULL"),
            },
            0,
            True,
            set(),
        ),
        (
            {
                "name": "later_orders_must_have_prior_interval",
                "expression": (
                    "order_number = 1 OR days_since_prior_order IS NOT NULL"
                ),
            },
            1,
            False,
            {
                (None, 2, None, None),
            },
        ),
        (
            {
                "name": "last_order_per_user_is_train_or_test",
                "expression": ("NOT is_last_order OR eval_set IN ('train', 'test')"),
            },
            1,
            False,
            {
                (1, 3, "prior", 20.0),
            },
        ),
        (
            {
                "name": "only_one_train_or_test_per_user",
                "expression": ("train_or_test_count = 1"),
            },
            3,
            False,
            {
                (1, 1, "prior", None),
                (1, 3, "prior", 20.0),
                (1, None, None, 10.0),
            },
        ),
        (
            {
                "name": "contiguous_order_numbers",
                "expression": (
                    "user_min_order_number = 1 "
                    "AND user_max_order_number = "
                    "user_distinct_order_number_count"
                ),
            },
            3,
            False,
            {
                (1, 1, "prior", None),
                (1, 3, "prior", 20.0),
                (1, None, None, 10.0),
            },
        ),
    ],
)
def test_validate_row_logic_handles_null_rule_results(
    spark,
    orders_schema,
    row_logic_contract,
    rule,
    failed_count,
    passed,
    failed_rows,
):
    schema = StructType(
        [
            StructField(
                "user_id",
                IntegerType(),
                True,
            ),
            *orders_schema.fields[1:],
        ]
    )

    df = spark.createDataFrame(
        [
            (1, 1, "prior", None),
            (None, None, "prior", 5.0),
            (None, 2, None, None),
            (1, 3, "prior", 20.0),
            (1, None, None, 10.0),
        ],
        schema=schema,
    )

    contract = {
        "derived_fields": (row_logic_contract["derived_fields"]),
        "rules": [
            rule,
        ],
    }

    result = validate_row_logic(
        df,
        contract=contract,
    )[0]

    assert result.passed is passed
    assert result.failed_count == failed_count

    actual_invalid_rows = (
        {
            (
                row["user_id"],
                row["order_number"],
                row["eval_set"],
                row["days_since_prior_order"],
            )
            for row in result.invalid_rows.collect()
        }
        if result.invalid_rows is not None
        else set()
    )

    assert actual_invalid_rows == failed_rows


@pytest.mark.parametrize(
    "derived_fields",
    [
        {},
        "not-a-list",
        123,
        True,
        None,
    ],
)
def test_validate_row_logic_rejects_non_list_derived_fields(
    spark,
    orders_schema,
    derived_fields,
):
    df = spark.createDataFrame(
        [(1, 1, "prior", None)],
        schema=orders_schema,
    )

    contract = {
        "derived_fields": derived_fields,
        "rules": [
            {
                "name": "dummy",
                "expression": "order_number > 0",
            }
        ],
    }

    with pytest.raises(
        InvalidContractError,
        match="'derived_fields' must be a list",
    ):
        validate_row_logic(
            df,
            contract=contract,
        )


@pytest.mark.parametrize(
    "rules",
    [
        {},
        "not-a-list",
        123,
        True,
        None,
    ],
)
def test_validate_row_logic_rejects_non_list_rules(
    spark,
    orders_schema,
    rules,
):
    df = spark.createDataFrame(
        [(1, 1, "prior", None)],
        schema=orders_schema,
    )

    contract = {
        "rules": rules,
    }

    with pytest.raises(
        InvalidContractError,
        match="'rules' must be a list",
    ):
        validate_row_logic(
            df,
            contract=contract,
        )
