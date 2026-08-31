import pytest

from instacart_etl_rnn.validation.dataset import _get_metric_threshold
from instacart_etl_rnn.validation.exceptions import InvalidConstraintError
from instacart_etl_rnn.validation.models import (
    ThresholdConfig,
    ValidationMetric,
    ValidationSeverity,
)


def test_get_metric_threshold_uses_defaults_when_threshold_missing(
    mocker,
):
    metric = ValidationMetric(
        rule_name="order_id.range",
        validation_type="range",
        columns=["order_id"],
        expression=mocker.sentinel.expression,
        alias="order_id_range",
    )

    contract = {
        "schema": [
            {
                "name": "order_id",
            }
        ]
    }

    result = _get_metric_threshold(
        metric,
        contract=contract,
    )

    assert result == ThresholdConfig(
        maximum_failed_percent=0.0,
        severity=ValidationSeverity.ERROR,
    )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        -0.1,
        "0.5",
        None,
        True,
    ],
)
def test_get_metric_threshold_rejects_invalid_max_failed_percent(
    mocker,
    invalid_threshold,
):
    metric = ValidationMetric(
        rule_name="order_id.range",
        validation_type="range",
        columns=["order_id"],
        expression=mocker.sentinel.expression,
        alias="order_id_range",
    )

    contract = {
        "schema": [
            {
                "name": "order_id",
                "thresholds": {
                    "range": {
                        "max_failed_percent": invalid_threshold,
                    }
                },
            }
        ]
    }

    with pytest.raises(
        InvalidConstraintError,
        match="max_failed_percent must be a non-negative number",
    ):
        _get_metric_threshold(
            metric,
            contract=contract,
        )


@pytest.mark.parametrize(
    "invalid_severity",
    [
        "warning",
        "info",
        "",
        None,
    ],
)
def test_get_metric_threshold_rejects_invalid_severity(
    mocker,
    invalid_severity,
):
    metric = ValidationMetric(
        rule_name="order_id.range",
        validation_type="range",
        columns=["order_id"],
        expression=mocker.sentinel.expression,
        alias="order_id_range",
    )

    contract = {
        "schema": [
            {
                "name": "order_id",
                "thresholds": {
                    "range": {
                        "max_failed_percent": 0.0,
                        "severity": invalid_severity,
                    }
                },
            }
        ]
    }

    with pytest.raises(
        InvalidConstraintError,
        match="severity must be either 'error' or 'critical'",
    ):
        _get_metric_threshold(
            metric,
            contract=contract,
        )
