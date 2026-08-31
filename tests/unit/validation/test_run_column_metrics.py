import pytest

from instacart_etl_rnn.validation.dataset import _run_column_metrics
from instacart_etl_rnn.validation.exceptions import EmptyDatasetError


def test_run_column_metrics_empty_df_raises_error(spark, mocker):
    df = mocker.Mock()
    df.agg.return_value.first.return_value = {"total_rows": 0}

    mocker.patch(
        "instacart_etl_rnn.validation.dataset._build_column_metrics",
        return_value=[],
    )

    with pytest.raises(EmptyDatasetError, match="DataFrame must not be empty"):
        _run_column_metrics(df, contract=mocker.sentinel.contract)
