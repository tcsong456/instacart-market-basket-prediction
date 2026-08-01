import pytest

from instacart_etl_rnn.validation.dataset import _apply_thresholds
from instacart_etl_rnn.validation.exceptions import InvalidConstraintError
from instacart_etl_rnn.validation.models import (
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)


def test_apply_thresholds_leaves_passed_result_unchanged(apply_thresholds_contract):
    result = ValidationResult(
        rule_name="order_id.range",
        failed_count=0,
        status=ValidationStatus.PASSED,
        severity=ValidationSeverity.INFO,
    )

    returned = _apply_thresholds(
        [result],
        validation_type="range",
        total_rows=100,
        contract=apply_thresholds_contract,
    )

    assert returned == [result]
    assert result.status is ValidationStatus.PASSED
    assert result.severity is ValidationSeverity.INFO


def test_apply_thresholds_downgrades_failure_within_threshold(
    apply_thresholds_contract,
):
    result = ValidationResult(
        rule_name="order_id.range",
        failed_count=1,
    )

    result = _apply_thresholds(
        [result],
        validation_type="range",
        total_rows=200,
        contract=apply_thresholds_contract,
    )[0]

    assert result.status is ValidationStatus.WARNING
    assert result.severity is ValidationSeverity.WARNING
    assert "0.50% of total rows" in result.message
    assert "did not exceed" in result.message


def test_apply_thresholds_marks_failure_when_threshold_exceeded(
    apply_thresholds_contract,
):
    result = ValidationResult(
        rule_name="order_id.range",
        failed_count=2,
    )

    result = _apply_thresholds(
        [result],
        validation_type="range",
        total_rows=100,
        contract=apply_thresholds_contract,
    )[0]

    assert result.status is ValidationStatus.FAILED
    assert result.severity is ValidationSeverity.ERROR
    assert "2.00% of total rows" in result.message


def test_apply_thresholds_allows_failure_equal_to_threshold(apply_thresholds_contract):
    result = ValidationResult(
        rule_name="order_id.range",
        failed_count=1,
    )

    result = _apply_thresholds(
        [result],
        validation_type="range",
        total_rows=100,
        contract=apply_thresholds_contract,
    )

    assert result[0].status is ValidationStatus.WARNING
    assert result[0].severity is ValidationSeverity.WARNING


def test_apply_thresholds_applies_critical_severity(apply_thresholds_contract):
    result = ValidationResult(
        rule_name="order_id.uniqueness",
        failed_count=1,
    )

    result = _apply_thresholds(
        [result],
        validation_type="uniqueness",
        total_rows=100,
        contract=apply_thresholds_contract,
    )

    assert result[0].status is ValidationStatus.FAILED
    assert result[0].severity is ValidationSeverity.CRITICAL


def test_apply_thresholds_marks_composite_uniqueness_as_critical(
    apply_thresholds_contract,
):
    result = ValidationResult(
        rule_name="user_id, order_number.uniqueness",
        failed_count=2,
    )

    result = _apply_thresholds(
        [result],
        validation_type="uniqueness",
        total_rows=100,
        contract=apply_thresholds_contract,
    )

    assert result[0].status is ValidationStatus.FAILED
    assert result[0].severity is ValidationSeverity.CRITICAL


def test_apply_thresholds_failed_result_without_matching_column(
    apply_thresholds_contract,
):
    result = ValidationResult(
        rule_name="unknown_column.range",
        failed_count=2,
    )

    result = _apply_thresholds(
        [result],
        validation_type="range",
        total_rows=100,
        contract=apply_thresholds_contract,
    )[0]

    assert result.status is ValidationStatus.FAILED
    assert result.severity is ValidationSeverity.ERROR
    assert "no matching column threshold was found" in result.message


def test_apply_thresholds_fails_unsupported_validation_type(apply_thresholds_contract):
    with pytest.raises(
        ValueError,
        match="validation type: .* is not supported",
    ):
        _apply_thresholds(
            [],
            validation_type="unknown",
            total_rows=100,
            contract=apply_thresholds_contract,
        )


@pytest.mark.parametrize(
    "value",
    [
        -1,
        "1",
        True,
        [],
    ],
)
def test_apply_thresholds_rejects_invalid_max_failed_percent(value):
    contract = {
        "schema": [
            {
                "name": "order_id",
                "thresholds": {
                    "range": {
                        "max_failed_percent": value,
                    },
                },
            },
        ]
    }

    result = ValidationResult(
        rule_name="order_id.range",
        failed_count=1,
    )

    with pytest.raises(
        InvalidConstraintError,
        match="max_failed_percent must be a non-negative number",
    ):
        _apply_thresholds(
            [result],
            validation_type="range",
            total_rows=100,
            contract=contract,
        )


def test_apply_thresholds_rejects_invalid_severity():
    contract = {
        "schema": [
            {
                "name": "order_id",
                "thresholds": {
                    "range": {
                        "severity": "warning",
                    },
                },
            },
        ]
    }

    result = ValidationResult(
        rule_name="order_id.range",
        failed_count=1,
    )

    with pytest.raises(
        InvalidConstraintError,
        match="either 'error' or 'critical'",
    ):
        _apply_thresholds(
            [result],
            validation_type="range",
            total_rows=100,
            contract=contract,
        )


def test_apply_thresholds_hard_fail_with_no_threshold_block(apply_thresholds_contract):
    result = ValidationResult(
        rule_name="user_id.range",
        failed_count=2,
    )

    result = _apply_thresholds(
        [result],
        validation_type="range",
        total_rows=100,
        contract=apply_thresholds_contract,
    )[0]

    assert result.status == ValidationStatus.FAILED
    assert result.severity == ValidationSeverity.ERROR


def test_apply_thresholds_with_empty_results(apply_thresholds_contract):
    result = _apply_thresholds(
        [],
        validation_type="range",
        total_rows=100,
        contract=apply_thresholds_contract,
    )

    assert result == []


def test_apply_thresholds_critical_downgrades_to_warning_within_max_percent_fail():
    contract = {
        "schema": [
            {
                "name": "order_id",
                "thresholds": {
                    "nullability": {
                        "max_failed_percent": 2.0,
                        "severity": "critical",
                    },
                },
            }
        ]
    }

    result = ValidationResult(rule_name="order_id.nullability", failed_count=1)

    result = _apply_thresholds(
        [result],
        validation_type="nullability",
        total_rows=100,
        contract=contract,
    )[0]

    assert result.status == ValidationStatus.WARNING
    assert result.severity == ValidationSeverity.WARNING
