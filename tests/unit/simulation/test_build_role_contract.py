import pytest

from instacart_etl_rnn.jobs.create_period_split_data_job import build_role_contract


def test_build_role_contract_builds_expected_contract():
    base_contract = {
        "dataset": {
            "name": "order_products",
        },
        "schema": [
            {
                "name": "user_id",
                "type": "integer",
            },
            {
                "name": "order_role",
                "type": "string",
                "constraints": {
                    "nullable": False,
                },
            },
        ],
    }

    result = build_role_contract(
        base_contract,
        "train_label",
    )

    assert result["dataset"]["name"] == "order_products_train_label"

    order_role_field = next(
        field for field in result["schema"] if field["name"] == "order_role"
    )

    assert order_role_field["constraints"] == {
        "nullable": False,
        "allowed_values": ["train_label"],
    }

    user_id_field = next(
        field for field in result["schema"] if field["name"] == "user_id"
    )

    assert user_id_field == {
        "name": "user_id",
        "type": "integer",
    }

    assert result["rules"] == [
        {
            "name": "contains_only_train_label_rows",
            "expression": "order_role = 'train_label'",
        }
    ]


def test_build_role_contract_does_not_mutate_base_contract():
    base_contract = {
        "dataset": {
            "name": "order_products",
        },
        "schema": [
            {
                "name": "order_role",
                "type": "string",
                "constraints": {
                    "allowed_values": [
                        "history",
                        "train_label",
                        "validation_label",
                    ],
                },
            }
        ],
        "rules": [],
    }

    build_role_contract(
        base_contract,
        "history",
    )

    assert base_contract["dataset"]["name"] == "order_products"

    assert base_contract["schema"][0]["constraints"]["allowed_values"] == [
        "history",
        "train_label",
        "validation_label",
    ]

    assert base_contract["rules"] == []


@pytest.mark.parametrize(
    "role",
    [
        "future",
        "invalid",
        "",
    ],
)
def test_build_role_contract_rejects_unsupported_role(role):
    with pytest.raises(
        ValueError,
        match=f"Unsupported order role: {role}",
    ):
        build_role_contract(
            {
                "dataset": {"name": "order_products"},
                "schema": [],
                "rules": [],
            },
            role,
        )
