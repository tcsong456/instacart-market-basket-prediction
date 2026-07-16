from dataclasses import dataclass, field
from typing import Any

from pyspark.sql import DataFrame


@dataclass
class ValidationResult:
    rule_name: str
    category: str
    passed: bool
    message: str
    failed_count: int = 0
    invalid_rows: DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
