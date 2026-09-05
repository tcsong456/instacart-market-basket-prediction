import logging

import pytest

from instacart_etl_rnn.validation.dataset import _log_validation_results
from instacart_etl_rnn.validation.models import (
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)


def test_log_validation_results_skips_passed_results(caplog):
    result = ValidationResult(
        status=ValidationStatus.PASSED,
        severity=ValidationSeverity.INFO,
        message="order_id.range passed",
    )

    with caplog.at_level(logging.DEBUG):
        _log_validation_results([result])

    assert caplog.records == []


@pytest.mark.parametrize(
    ("severity", "expected_level"),
    [
        (
            ValidationSeverity.WARNING,
            logging.WARNING,
        ),
        (
            ValidationSeverity.ERROR,
            logging.ERROR,
        ),
        (
            ValidationSeverity.CRITICAL,
            logging.CRITICAL,
        ),
    ],
)
def test_log_validation_results_uses_correct_log_level(
    caplog,
    severity,
    expected_level,
):
    result = ValidationResult(
        status=ValidationStatus.FAILED,
        severity=severity,
        message="validation failed",
    )

    with caplog.at_level(logging.WARNING):
        _log_validation_results([result])

    assert len(caplog.records) == 1

    record = caplog.records[0]

    assert record.levelno == expected_level
    assert record.message == "validation failed"


def test_log_validation_results_logs_only_non_passed_results(caplog):
    results = [
        ValidationResult(
            status=ValidationStatus.PASSED,
            severity=ValidationSeverity.INFO,
            message="passed",
        ),
        ValidationResult(
            status=ValidationStatus.WARNING,
            severity=ValidationSeverity.WARNING,
            message="warning",
        ),
        ValidationResult(
            status=ValidationStatus.FAILED,
            severity=ValidationSeverity.ERROR,
            message="error",
        ),
    ]

    with caplog.at_level(logging.WARNING):
        _log_validation_results(results)

    assert [(record.levelno, record.message) for record in caplog.records] == [
        (logging.WARNING, "warning"),
        (logging.ERROR, "error"),
    ]
