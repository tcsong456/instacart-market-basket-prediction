from instacart_etl_rnn.validation.dataset import _evaluate_metric
from instacart_etl_rnn.validation.models import (
    ThresholdConfig,
    ValidationMetric,
    ValidationSeverity,
    ValidationStatus,
)


def test_evaluate_metric_returns_passed_when_no_rows_fail(
    mocker,
):
    metric = ValidationMetric(
        rule_name="order_id.range",
        validation_type="range",
        columns=["order_id"],
        expression=mocker.sentinel.expression,
        alias="order_id_range",
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset._get_metric_threshold",
        return_value=ThresholdConfig(
            maximum_failed_percent=5.0,
            severity=ValidationSeverity.ERROR,
        ),
    )

    result = _evaluate_metric(
        metric,
        failed_count=0,
        total_rows=100,
        contract=mocker.sentinel.contract,
    )

    assert result.rule_name == "order_id.range"
    assert result.category == "range"
    assert result.passed is True
    assert result.status == ValidationStatus.PASSED
    assert result.severity == ValidationSeverity.INFO
    assert result.failed_count == 0
    assert result.invalid_rows is None
    assert result.message == "order_id.range passed"

    assert result.metadata == {
        "columns": ["order_id"],
        "failed_percent": 0.0,
        "maximum_failed_percent": 5.0,
    }


def test_evaluate_metric_returns_warning_when_failure_is_within_threshold(
    mocker,
):
    metric = ValidationMetric(
        rule_name="order_id.range",
        validation_type="range",
        columns=["order_id"],
        expression=mocker.sentinel.expression,
        alias="order_id_range",
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset._get_metric_threshold",
        return_value=ThresholdConfig(
            maximum_failed_percent=5.0,
            severity=ValidationSeverity.CRITICAL,
        ),
    )

    result = _evaluate_metric(
        metric,
        failed_count=3,
        total_rows=100,
        contract=mocker.sentinel.contract,
    )

    assert result.passed is True
    assert result.status == ValidationStatus.WARNING
    assert result.severity == ValidationSeverity.WARNING
    assert result.failed_count == 3

    assert result.metadata == {
        "columns": ["order_id"],
        "failed_percent": 3.0,
        "maximum_failed_percent": 5.0,
    }

    assert "produced a warning" in result.message
    assert "3 row(s)" in result.message
    assert "3.00%" in result.message


def test_evaluate_metric_returns_failed_when_threshold_is_exceeded(
    mocker,
):
    metric = ValidationMetric(
        rule_name="order_id.range",
        validation_type="range",
        columns=["order_id"],
        expression=mocker.sentinel.expression,
        alias="order_id_range",
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset._get_metric_threshold",
        return_value=ThresholdConfig(
            maximum_failed_percent=2.0,
            severity=ValidationSeverity.CRITICAL,
        ),
    )

    result = _evaluate_metric(
        metric,
        failed_count=5,
        total_rows=100,
        contract=mocker.sentinel.contract,
    )

    assert result.passed is False
    assert result.status == ValidationStatus.FAILED
    assert result.severity == ValidationSeverity.CRITICAL
    assert result.failed_count == 5

    assert result.metadata == {
        "columns": ["order_id"],
        "failed_percent": 5.0,
        "maximum_failed_percent": 2.0,
    }

    assert "failed" in result.message
    assert "5 row(s)" in result.message
    assert "5.00%" in result.message


def test_evaluate_metric_returns_warning_when_failure_equals_threshold(
    mocker,
):
    metric = ValidationMetric(
        rule_name="order_id.range",
        validation_type="range",
        columns=["order_id"],
        expression=mocker.sentinel.expression,
        alias="order_id_range",
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset._get_metric_threshold",
        return_value=ThresholdConfig(
            maximum_failed_percent=5.0,
            severity=ValidationSeverity.ERROR,
        ),
    )

    result = _evaluate_metric(
        metric,
        failed_count=5,
        total_rows=100,
        contract=mocker.sentinel.contract,
    )

    assert result.status == ValidationStatus.WARNING
    assert result.severity == ValidationSeverity.WARNING
    assert result.passed is True


def test_get_metric_threshold_receives_right_arguments_inside_evaluate_metric(mocker):
    mocked_threshold = mocker.patch(
        "instacart_etl_rnn.validation.dataset._get_metric_threshold",
        return_value=ThresholdConfig(
            maximum_failed_percent=5.0,
            severity=ValidationSeverity.ERROR,
        ),
    )

    metric = mocker.Mock()
    metric.rule_name = "order_id.unique"
    metric.validation_type = "uniqueness"
    metric.columns = ("order_id",)

    contract = mocker.sentinel.contract

    _evaluate_metric(
        metric,
        failed_count=1,
        total_rows=100,
        contract=contract,
    )

    mocked_threshold.assert_called_once_with(
        metric,
        contract=contract,
    )
