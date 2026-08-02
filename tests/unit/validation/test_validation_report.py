import pytest

from instacart_etl_rnn.validation.models import (
    ValidationReport,
    ValidationResult,
    ValidationStatus,
)


@pytest.mark.parametrize(
    ("statuses", "has_errors", "has_warning", "can_continue"),
    [
        (
            [ValidationStatus.PASSED],
            False,
            False,
            True,
        ),
        (
            [ValidationStatus.WARNING],
            False,
            True,
            True,
        ),
        (
            [ValidationStatus.FAILED],
            True,
            False,
            False,
        ),
        (
            [ValidationStatus.WARNING, ValidationStatus.FAILED],
            True,
            True,
            False,
        ),
    ],
)
def test_validation_report_properties(
    statuses,
    has_errors,
    has_warning,
    can_continue,
):
    report = ValidationReport(
        dataset_name="orders",
        results=[
            ValidationResult(
                rule_name=f"rule_{i}",
                category="test",
                status=status,
                failed_count=1,
                message="",
            )
            for i, status in enumerate(statuses)
        ],
    )

    assert report.has_errors is has_errors
    assert report.has_warning is has_warning
    assert report.can_continue is can_continue
