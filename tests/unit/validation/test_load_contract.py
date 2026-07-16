import pytest
import yaml

from instacart_etl.validation.exceptions import InvalidContractError
from instacart_etl.validation.loader import load_contract


def test_load_contract_returns_parsed_contract(tmp_path):
    contract_path = tmp_path / "orders.yaml"
    contract_path.write_text(
        """
        dataset:
          name: orders
        schema:
          - name: order_id
            type: integer
            nullable: false
            constraints:
              unique: true
              minimum: 1
        """
    )

    contract = load_contract(contract_path)

    assert contract["schema"] == [
        {
            "name": "order_id",
            "type": "integer",
            "nullable": False,
            "constraints": {"unique": True, "minimum": 1},
        }
    ]
    assert contract["dataset"]["name"] == "orders"


def test_load_contract_raises_when_file_does_not_exist(tmp_path):
    contract_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="contract not found on path"):
        load_contract(contract_path)


def test_load_contract_raises_for_invalid_yaml(tmp_path):
    contract_path = tmp_path / "invalid.yaml"
    contract_path.write_text(
        """
        schema:
          - name: order_id
              type: integer
        nullable: false
          
        """
    )

    with pytest.raises(
        InvalidContractError, match="Invalid syntax in yaml contract"
    ) as error:
        load_contract(contract_path)

    assert isinstance(error.value.__cause__, yaml.YAMLError)


def test_load_contract_rejects_non_mapping_root(tmp_path):
    contract_path = tmp_path / "list.yaml"
    contract_path.write_text(
        """
        - order_id
        - product_id
        """
    )

    with pytest.raises(InvalidContractError, match="contain a top-level YAML mapping"):
        load_contract(contract_path)


def test_load_contract_requires_schema_section(tmp_path):
    contract_path = tmp_path / "no_schema.yaml"
    contract_path.write_text(
        """
        dataset:
          name: orders
        """
    )

    with pytest.raises(
        InvalidContractError, match="Contract must contain a 'schema' section"
    ):
        load_contract(contract_path)


def test_load_contract_requires_schema_to_be_list(tmp_path):
    contract_path = tmp_path / "schema_dict.yaml"
    contract_path.write_text(
        """
        schema:
          name: order_id
          type: integer
          nullable: false
        """
    )

    with pytest.raises(
        InvalidContractError, match="The contract schema section must be a list"
    ):
        load_contract(contract_path)


def test_load_contract_raises_for_empty_schema(tmp_path):
    contract_path = tmp_path / "orders.yaml"
    contract_path.write_text(
        """
        dataset:
          name: orders

        schema: []
        """
    )

    with pytest.raises(
        InvalidContractError,
        match="The contract 'schema' section must not be empty.",
    ):
        load_contract(contract_path)


def test_load_contract_raises_for_non_mapping_schema_entry(tmp_path):
    contract_path = tmp_path / "empty_schema.yaml"
    contract_path.write_text(
        """
        dataset:
          name: orders

        schema:
          - invalid
        """
    )

    with pytest.raises(
        InvalidContractError,
        match="Schema entry at index 0 must be a mapping.",
    ):
        load_contract(contract_path)


def test_load_contract_raises_for_schema_missing_name(tmp_path):
    contract_path = tmp_path / "schema_no_name.yaml"
    contract_path.write_text(
        """
        schema:
          - type: integer
        """
    )

    with pytest.raises(
        InvalidContractError, match="Schema entry at index 0 is missing 'name'."
    ):
        load_contract(contract_path)


def test_load_contract_raises_for_schema_missing_type(tmp_path):
    contract_path = tmp_path / "schema_no_type.yaml"
    contract_path.write_text(
        """
        schema:
          - name: order_id
        """
    )

    with pytest.raises(
        InvalidContractError, match="Schema entry 'order_id' is missing 'type'."
    ):
        load_contract(contract_path)


def test_load_contract_raises_for_non_string_column_name(tmp_path):
    contract_path = tmp_path / "non_string_name.yaml"

    contract_path.write_text(
        """
        dataset:
          name: orders

        schema:
          - name: 123
            type: integer
        """
    )

    with pytest.raises(
        InvalidContractError,
        match="Schema entry at index 0 has an invalid column name.",
    ):
        load_contract(contract_path)


def test_load_contract_raises_for_empty_string_column_name(tmp_path):
    contract_path = tmp_path / "empty_string_name.yaml"
    contract_path.write_text(
        """
        schema:
          - name: ""
            type: integer
        """
    )

    with pytest.raises(
        InvalidContractError,
        match="Schema entry at index 0 has an invalid column name.",
    ):
        load_contract(contract_path)


def test_load_contract_raises_for_space_only_column_name(tmp_path):
    contract_path = tmp_path / "space_string_name.yaml"
    contract_path.write_text(
        """
        schema:
          - name: "    "
            type: short
        """
    )

    with pytest.raises(
        InvalidContractError,
        match="Schema entry at index 0 has an invalid column name.",
    ):
        load_contract(contract_path)


def test_load_contract_raises_for_non_string_type(tmp_path):
    contract_path = tmp_path / "non_string_type.yaml"

    contract_path.write_text(
        """
        dataset:
          name: orders

        schema:
          - name: order_id
            type: 123
            nullable: false
        """
    )

    with pytest.raises(
        InvalidContractError,
        match="Column 'order_id' must have a string type.",
    ):
        load_contract(contract_path)


def test_load_contract_raises_when_nullable_missing(tmp_path):
    contract_path = tmp_path / "nullable_missing_schema.yaml"

    contract_path.write_text(
        """
        dataset:
          name: orders

        schema:
          - name: order_id
            type: integer
        """
    )

    with pytest.raises(
        InvalidContractError,
        match="Column 'order_id' has a non-boolean 'nullable' value.",
    ):
        load_contract(contract_path)


def test_load_contract_raises_for_non_boolean_nullable(tmp_path):
    contract_path = tmp_path / "nullable_non_boolean.yaml"

    contract_path.write_text(
        """
        dataset:
          name: orders

        schema:
          - name: order_id
            type: integer
            nullable: "yes"
        """
    )

    with pytest.raises(
        InvalidContractError,
        match="Column 'order_id' has a non-boolean 'nullable' value.",
    ):
        load_contract(contract_path)


def test_load_contract_raises_for_invalid_constraints(tmp_path):
    contract_path = tmp_path / "invalid_constraints.yaml"

    contract_path.write_text(
        """
        dataset:
          name: orders

        schema:
          - name: order_id
            type: integer
            nullable: false
            constraints: invalid
        """
    )

    with pytest.raises(
        InvalidContractError,
        match="Constraints for column 'order_id' must be a mapping.",
    ):
        load_contract(contract_path)


def test_load_contract_raises_for_duplicate_column_names(tmp_path):
    contract_path = tmp_path / "orders.yaml"

    contract_path.write_text(
        """
        dataset:
          name: orders

        schema:
          - name: order_id
            type: integer
            nullable: false

          - name: order_id
            type: string
            nullable: true
        """
    )

    with pytest.raises(
        InvalidContractError,
        match="Contract schema contains duplicate column names.",
    ):
        load_contract(contract_path)
