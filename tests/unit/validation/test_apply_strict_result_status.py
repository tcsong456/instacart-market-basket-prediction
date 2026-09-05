from instacart_etl_rnn.validation.dataset import (
    _apply_strict_result_status,
)
from instacart_etl_rnn.validation.models import (
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)


def test_apply_strict_result_status_marks_zero_failures_as_passed():
    result = ValidationResult(
        failed_count=0,
        passed=False,
        status=ValidationStatus.FAILED,
        severity=ValidationSeverity.CRITICAL,
        metadata={"columns": ["order_id"]},
    )

    updated = _apply_strict_result_status(
        result,
        total_rows=100,
    )

    assert updated.passed is True
    assert updated.status == ValidationStatus.PASSED
    assert updated.severity == ValidationSeverity.INFO
    assert updated.failed_count == 0

    assert updated.metadata == {
        "columns": ["order_id"],
        "failed_percent": 0.0,
    }


def test_apply_strict_result_status_marks_failures_as_failed():
    result = ValidationResult(
        failed_count=5,
        passed=True,
        status=ValidationStatus.PASSED,
        severity=ValidationSeverity.INFO,
        metadata={"columns": ["order_id"]},
    )

    updated = _apply_strict_result_status(
        result,
        total_rows=100,
    )

    assert updated.passed is False
    assert updated.status == ValidationStatus.FAILED
    assert updated.severity == ValidationSeverity.CRITICAL
    assert updated.failed_count == 5

    assert updated.metadata == {
        "columns": ["order_id"],
        "failed_percent": 5.0,
    }


def test_apply_strict_result_status_uses_given_failure_severity():
    result = ValidationResult(
        failed_count=2,
        metadata={},
    )

    updated = _apply_strict_result_status(
        result,
        total_rows=100,
        severity=ValidationSeverity.ERROR,
    )

    assert updated.status == ValidationStatus.FAILED
    assert updated.severity == ValidationSeverity.ERROR


def test_apply_strict_result_status_does_not_modify_original_result():
    result = ValidationResult(
        failed_count=5,
        passed=True,
        status=ValidationStatus.WARNING,
        severity=ValidationSeverity.WARNING,
        metadata={"source": "uniqueness"},
    )

    updated = _apply_strict_result_status(
        result,
        total_rows=100,
    )

    assert updated is not result

    assert result.passed is True
    assert result.status == ValidationStatus.WARNING
    assert result.severity == ValidationSeverity.WARNING
    assert result.metadata == {
        "source": "uniqueness",
    }
