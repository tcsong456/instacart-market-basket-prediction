from instacart_etl_rnn.validation.dataset import _build_column_metrics


def test_build_column_metrics_combines_all_validator_metrics(
    mocker,
):
    contract = mocker.sentinel.contract

    nullability_metric = mocker.sentinel.nullability_metric
    range_metric = mocker.sentinel.range_metric
    allowed_metric = mocker.sentinel.allowed_metric
    string_length_metric = mocker.sentinel.string_length_metric
    pattern_metric = mocker.sentinel.pattern_metric
    array_length_metric = mocker.sentinel.array_length_metric

    mocked_nullability = mocker.patch(
        "instacart_etl_rnn.validation.dataset.nullability_validator",
        return_value=[nullability_metric],
    )

    mocked_range = mocker.patch(
        "instacart_etl_rnn.validation.dataset.range_validator",
        return_value=[range_metric],
    )

    mocked_allowed = mocker.patch(
        "instacart_etl_rnn.validation.dataset.allowed_values_validator",
        return_value=[allowed_metric],
    )

    mocked_string_length = mocker.patch(
        "instacart_etl_rnn.validation.dataset.string_length_validator",
        return_value=[string_length_metric],
    )

    mocked_pattern = mocker.patch(
        "instacart_etl_rnn.validation.dataset.pattern_validator",
        return_value=[pattern_metric],
    )

    mocked_array_length = mocker.patch(
        "instacart_etl_rnn.validation.dataset.array_length_validator",
        return_value=[array_length_metric],
    )

    result = _build_column_metrics(contract)

    assert result == [
        nullability_metric,
        range_metric,
        allowed_metric,
        string_length_metric,
        pattern_metric,
        array_length_metric,
    ]

    mocked_nullability.assert_called_once_with(contract)
    mocked_range.assert_called_once_with(contract)
    mocked_allowed.assert_called_once_with(contract)
    mocked_string_length.assert_called_once_with(contract)
    mocked_pattern.assert_called_once_with(contract)
    mocked_array_length.assert_called_once_with(contract)


def test_build_column_metrics_handles_validators_returning_no_metrics(
    mocker,
):
    contract = mocker.sentinel.contract

    mocker.patch(
        "instacart_etl_rnn.validation.dataset.nullability_validator",
        return_value=[],
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.range_validator",
        return_value=[],
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.allowed_values_validator",
        return_value=[],
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.string_length_validator",
        return_value=[],
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.pattern_validator",
        return_value=[],
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.array_length_validator",
        return_value=[],
    )

    result = _build_column_metrics(contract)

    assert result == []
