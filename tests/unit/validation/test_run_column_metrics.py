from unittest.mock import call

import pytest
from pyspark.sql import functions as F

from instacart_etl_rnn.validation.dataset import _run_column_metrics
from instacart_etl_rnn.validation.exceptions import EmptyDatasetError
from instacart_etl_rnn.validation.models import ValidationMetric


def test_run_column_metrics_aggregates_and_evaluates_metrics(
    spark,
    mocker,
):
    df = spark.createDataFrame(
        [
            (1,),
            (None,),
            (-1,),
        ],
        ["value"],
    )

    null_metric = ValidationMetric(
        rule_name="value.nullability",
        validation_type="nullability",
        columns=["value"],
        alias="value_nullability",
        expression=F.sum(
            F.when(
                F.col("value").isNull(),
                1,
            ).otherwise(0)
        ).alias("value_nullability"),
    )

    range_metric = ValidationMetric(
        rule_name="value.range",
        validation_type="range",
        columns=["value"],
        alias="value_range",
        expression=F.sum(
            F.when(
                F.col("value") < 0,
                1,
            ).otherwise(0)
        ).alias("value_range"),
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset._build_column_metrics",
        return_value=[
            null_metric,
            range_metric,
        ],
    )

    null_result = mocker.sentinel.null_result
    range_result = mocker.sentinel.range_result

    mocked_evaluate = mocker.patch(
        "instacart_etl_rnn.validation.dataset._evaluate_metric",
        side_effect=[
            null_result,
            range_result,
        ],
    )

    contract = mocker.sentinel.contract

    results, total_rows = _run_column_metrics(
        df,
        contract=contract,
    )

    assert total_rows == 3

    assert results == [
        null_result,
        range_result,
    ]

    assert mocked_evaluate.call_args_list == [
        call(
            null_metric,
            failed_count=1,
            total_rows=3,
            contract=contract,
        ),
        call(
            range_metric,
            failed_count=1,
            total_rows=3,
            contract=contract,
        ),
    ]


def test_run_column_metrics_treats_null_metric_result_as_zero(
    spark,
    mocker,
):
    df = spark.createDataFrame(
        [
            (1,),
            (2,),
        ],
        ["value"],
    )

    metric = ValidationMetric(
        rule_name="value.range",
        validation_type="range",
        columns=["value"],
        alias="value_range",
        expression=F.sum(
            F.when(
                F.col("value") < 0,
                1,
            )
        ).alias("value_range"),
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset._build_column_metrics",
        return_value=[metric],
    )

    expected_result = mocker.sentinel.result
    mocked_evaluate = mocker.patch(
        "instacart_etl_rnn.validation.dataset._evaluate_metric",
        return_value=expected_result,
    )

    contract = mocker.sentinel.contract

    results, total_rows = _run_column_metrics(df, contract=contract)

    assert total_rows == 2

    assert results == [expected_result]

    mocked_evaluate.assert_called_once_with(
        metric, failed_count=0, total_rows=2, contract=contract
    )


def test_run_column_metrics_returns_empty_results_when_no_metrics(
    spark,
    mocker,
):
    df = spark.createDataFrame(
        [
            (1,),
            (2,),
        ],
        ["value"],
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset._build_column_metrics",
        return_value=[],
    )

    mocked_evaluate = mocker.patch(
        "instacart_etl_rnn.validation.dataset._evaluate_metric",
    )

    results, total_rows = _run_column_metrics(
        df,
        contract=mocker.sentinel.contract,
    )

    assert total_rows == 2
    assert results == []

    mocked_evaluate.assert_not_called()


def test_run_column_metrics_empty_df_raises_error(spark, mocker):
    df = spark.createDataFrame([], "value INT")

    mocker.patch(
        "instacart_etl_rnn.validation.dataset._build_column_metrics",
        return_value=[],
    )

    with pytest.raises(EmptyDatasetError, match="DataFrame must not be empty"):
        _run_column_metrics(df, contract=mocker.sentinel.contract)
