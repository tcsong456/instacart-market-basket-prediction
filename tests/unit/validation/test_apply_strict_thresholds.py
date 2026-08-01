from instacart_etl_rnn.validation.dataset import _apply_strict_thresholds
from instacart_etl_rnn.validation.models import (
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)


def test_apply_strict_thresholds_marks_failed_rule_as_critical():
    result = ValidationResult(
        rule_name="first_order_has_no_prior_interval",
        failed_count=3,
    )

    result = _apply_strict_thresholds([result], total_rows=100, prefix="business")[0]

    assert result.status == ValidationStatus.FAILED
    assert result.severity == ValidationSeverity.CRITICAL
    assert "business rule: first_order_has_no_prior_interval " in result.message
    assert "failed 3 rows" in result.message
    assert "3.00% of total rows" in result.message


def test_apply_strict_thresholds_leaves_passing_rule_unchanged():
    result = ValidationResult(
        rule_name="first_order_has_no_prior_interval",
        failed_count=0,
        message="This rule has passed",
    )

    result = _apply_strict_thresholds([result], total_rows=100)[0]

    assert result.status == ValidationStatus.PASSED
    assert result.severity == ValidationSeverity.INFO
    assert result.message == "This rule has passed"


def test_apply_strict_thresholds_pass_and_fail_cases():
    pass_result = ValidationResult(
        rule_name="order_id.referential_integrity",
        failed_count=0,
        message="This rule has passed",
    )

    fail_result = ValidationResult(
        rule_name="order_id.referential_integrity", failed_count=10
    )

    results = _apply_strict_thresholds(
        [pass_result, fail_result], total_rows=100, prefix="RI"
    )

    assert results[0].status == ValidationStatus.PASSED
    assert results[0].severity == ValidationSeverity.INFO
    assert results[0].message == "This rule has passed"

    assert results[1].status == ValidationStatus.FAILED
    assert results[1].severity == ValidationSeverity.CRITICAL
    assert "RI rule: order_id.referential_integrity " in results[1].message
    assert "failed 10 rows" in results[1].message
    assert "10.00% of total rows" in results[1].message


def test_apply_strict_thresholds_empty_result():
    result = _apply_strict_thresholds([], total_rows=100)

    assert result == []
