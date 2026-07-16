class ContractError(Exception):
    """Base class for contract-related errors."""


class InvalidContractError(ContractError):
    """The contract YAML is malformed."""
