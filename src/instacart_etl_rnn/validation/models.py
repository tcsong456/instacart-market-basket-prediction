from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pyspark.sql import DataFrame


class ValidationStatus(str, Enum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


class ValidationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    rule_name: str
    category: str
    passed: bool
    message: str
    failed_count: int = 0
    invalid_rows: DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str | None = ValidationStatus.PASSED
    severity: str | None = ValidationSeverity.INFO


@dataclass
class ValidationReport:
    dataset_name: str
    results: list[ValidationResult]

    @property
    def has_errors(self) -> bool:
        return any(result.status == ValidationStatus.FAILED for result in self.results)

    @property
    def has_warning(self) -> bool:
        return any(result.status == ValidationStatus.WARNING for result in self.results)

    @property
    def can_continue(self) -> bool:
        return not self.has_errors
