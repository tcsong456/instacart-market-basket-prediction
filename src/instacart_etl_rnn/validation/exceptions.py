from instacart_etl_rnn.validation.models import ValidationReport


class ContractError(Exception):
    """Base class for contract-related errors."""


class InvalidContractError(ContractError):
    """The contract YAML is malformed."""


class ValidationError(Exception):
    """Base exception for validation-system errors."""


class InvalidConstraintError(ValidationError, ValueError):
    """Raised when a validation rule is configured incorrectly."""


class DataValidationError(RuntimeError):
    """Raised when data fails one or more blocking validation rules."""

    def __init__(self, report: ValidationReport):
        self.report = report

        super().__init__(f"Dataset: {report.dataset_name} failed the data validation!")


class EmptyDatasetError(ValueError):
    """Raised when a dataset expected to contain rows is empty."""
