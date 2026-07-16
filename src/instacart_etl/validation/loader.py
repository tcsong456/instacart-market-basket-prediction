from pathlib import Path
from typing import Any

import yaml
from yaml import YAMLError

from instacart_etl.validation.exceptions import InvalidContractError


def load_contract(contract_path: str | Path) -> dict[str, Any]:
    """
    Load a YAML data contract.

    Args:
        contract_path: Path to the YAML contract.

    Returns:
        Parsed contract dictionary.

    Raises:
        InvalidContractError: If the contract cannot be interpreted.
    """

    path = Path(contract_path)
    if not path.exists():
        raise FileNotFoundError(f"contract not found on path: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            contract = yaml.safe_load(f)
    except YAMLError as exc:
        raise InvalidContractError(f"Invalid syntax in yaml contract: {exc}") from exc

    if contract is None:
        raise InvalidContractError("Contract is empty!")

    if not isinstance(contract, dict):
        raise InvalidContractError("Contract must contain a top-level YAML mapping")

    schema = contract.get("schema")

    if schema is None:
        raise InvalidContractError("Contract must contain a 'schema' section")

    if not isinstance(schema, list):
        raise InvalidContractError("The contract schema section must be a list")

    if not schema:
        raise InvalidContractError("The contract 'schema' section must not be empty.")

    for index, column in enumerate(schema):
        if not isinstance(column, dict):
            raise InvalidContractError(
                f"Schema entry at index {index} must be a mapping."
            )

        if "name" not in column:
            raise InvalidContractError(
                f"Schema entry at index {index} is missing 'name'."
            )

        if "type" not in column:
            raise InvalidContractError(
                f"Schema entry '{column['name']}' is missing 'type'."
            )

        if not isinstance(column["name"], str) or not column["name"].strip():
            raise InvalidContractError(
                f"Schema entry at index {index} has an invalid column name."
            )

        if not isinstance(column["type"], str):
            raise InvalidContractError(
                f"Column '{column['name']}' must have a string type."
            )

        if "nullable" not in column or not isinstance(column["nullable"], bool):
            raise InvalidContractError(
                f"Column '{column['name']}' has a non-boolean 'nullable' value."
            )

        constraints = column.get("constraints", {})

        if not isinstance(constraints, dict):
            raise InvalidContractError(
                f"Constraints for column '{column['name']}' must be a mapping."
            )

    column_names = [column["name"] for column in schema]
    if len(column_names) != len(set(column_names)):
        raise InvalidContractError("Contract schema contains duplicate column names.")

    return contract
