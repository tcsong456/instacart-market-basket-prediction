import pytest

from instacart_etl_rnn.validation.exceptions import InvalidContractError
from instacart_etl_rnn.validation.uniqueness import validate_uniqueness


def test_uniqueness_passes_when_values_are_unique(spark):
    df = spark.createDataFrame(
        [
            (1, "a"),
            (2, "b"),
            (3, "c"),
        ],
        "id INT, value STRING",
    )

    result = validate_uniqueness(
        df,
        columns=["id"],
    )

    assert result.passed
    assert result.failed_count == 0
    assert result.invalid_rows is None
    assert result.rule_name == "id.uniqueness"
    assert result.metadata["duplicate_row_count"] == 0
    assert result.metadata["duplicate_key_count"] == 0


def test_uniqueness_counts_all_rows_participating_in_duplicates(
    spark,
):
    df = spark.createDataFrame(
        [
            (1, "a"),
            (1, "b"),
            (1, "c"),
            (2, "d"),
            (2, "e"),
            (3, "f"),
        ],
        "id INT, value STRING",
    )

    result = validate_uniqueness(
        df,
        columns=["id"],
    )

    assert result.passed is False
    assert result.failed_count == 5
    assert result.metadata["duplicate_row_count"] == 5
    assert result.metadata["duplicate_key_count"] == 2


def test_uniqueness_returns_exact_duplicate_rows(spark):
    df = spark.createDataFrame(
        [
            (1, "a"),
            (1, "b"),
            (2, "c"),
            (3, "d"),
            (3, "e"),
        ],
        "id INT, value STRING",
    )

    result = validate_uniqueness(
        df,
        columns=["id"],
    )

    actual = {(row["id"], row["value"]) for row in result.invalid_rows.collect()}

    assert actual == {
        (1, "a"),
        (1, "b"),
        (3, "d"),
        (3, "e"),
    }


def test_uniqueness_supports_composite_keys(spark):
    df = spark.createDataFrame(
        [
            (1, 10, "a"),
            (1, 10, "b"),
            (1, 20, "c"),
            (2, 10, "d"),
            (2, 10, "e"),
            (3, 30, "f"),
        ],
        "user_id INT, order_id INT, value STRING",
    )

    result = validate_uniqueness(
        df,
        columns=["user_id", "order_id"],
    )

    assert result.passed is False
    assert result.failed_count == 4
    assert result.rule_name == ("user_id, order_id.uniqueness")


def test_uniqueness_ignores_null_single_column_keys(spark):
    df = spark.createDataFrame(
        [
            (1,),
            (2,),
            (None,),
            (None,),
            (None,),
        ],
        "id INT",
    )

    result = validate_uniqueness(
        df,
        columns=["id"],
    )

    assert result.passed
    assert result.failed_count == 0


def test_uniqueness_ignores_partial_null_composite_keys(spark):
    df = spark.createDataFrame(
        [
            (1, 10),
            (2, 20),
            (1, None),
            (1, None),
            (None, 10),
            (None, 10),
            (None, None),
            (None, None),
        ],
        "user_id INT, order_id INT",
    )

    result = validate_uniqueness(
        df,
        columns=["user_id", "order_id"],
    )

    assert result.passed
    assert result.failed_count == 0


def test_uniqueness_ignores_null_keys_but_detects_non_null_duplicates(
    spark,
):
    df = spark.createDataFrame(
        [
            (1, 10),
            (1, 10),
            (2, None),
            (2, None),
            (None, 20),
            (None, 20),
        ],
        "user_id INT, order_id INT",
    )

    result = validate_uniqueness(
        df,
        columns=["user_id", "order_id"],
    )

    assert not result.passed
    assert result.failed_count == 2
    assert result.metadata["duplicate_key_count"] == 1


def test_uniqueness_rejects_duplicate_column_names(spark):
    df = spark.createDataFrame(
        [(1,)],
        "id INT",
    )

    with pytest.raises(
        InvalidContractError,
        match="duplicate column names",
    ):
        validate_uniqueness(
            df,
            columns=["id", "id"],
        )


@pytest.mark.parametrize(
    "columns",
    [["id", ""], ["id", "   "], ["id", 123], ["id", None], [], {}, {"id"}],
)
def test_uniqueness_rejects_invalid_column_names(
    spark,
    columns,
):
    df = spark.createDataFrame(
        [(1,)],
        "id INT",
    )

    with pytest.raises(
        InvalidContractError,
        match="non-empty list of column names",
    ):
        validate_uniqueness(
            df,
            columns=columns,
        )


def test_uniqueness_counts_every_row_in_duplicate_group(spark):
    df = spark.createDataFrame(
        [
            (5,),
            (5,),
            (5,),
            (5,),
        ],
        "id INT",
    )

    result = validate_uniqueness(
        df,
        columns=["id"],
    )

    assert result.failed_count == 4
