class ContractError(Exception):
    """Base class for contract-related errors."""


class InvalidContractError(ContractError):
    """The contract YAML is malformed."""


class ValidationError(Exception):
    """Base exception for validation-system errors."""


class InvalidConstraintError(ValidationError, ValueError):
    """Raised when a validation rule is configured incorrectly."""
