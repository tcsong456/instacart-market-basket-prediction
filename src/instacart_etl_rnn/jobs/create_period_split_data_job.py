import copy

from pyspark.sql import SparkSession

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.common.paths import join_path
from instacart_etl_rnn.simulation.create_order_product_split import (
    select_base_model_users,
    select_stacking_model_users,
    split_order_products_by_role,
)
from instacart_etl_rnn.validation.dataset import validate_dataset
from instacart_etl_rnn.validation.loader import load_contract

COLUMNS = [
    "product_id",
    "order_id",
    "add_to_cart_order",
    "reordered",
    "user_id",
    "eval_set",
    "order_number",
    "order_dow",
    "order_hour_of_day",
    "product_name",
    "aisle_id",
    "department_id",
]


def build_role_contract(
    base_contract: dict,
    role: str,
) -> dict:
    if role not in {
        "history",
        "train_label",
        "validation_label",
    }:
        raise ValueError(f"Unsupported order role: {role}")

    contract = copy.deepcopy(base_contract)

    contract["dataset"]["name"] = f"order_products_{role}"

    contract["schema"] = [
        {
            **field,
            **(
                {
                    "constraints": {
                        **field.get("constraints", {}),
                        "allowed_values": [role],
                    }
                }
                if field["name"] == "order_role"
                else {}
            ),
        }
        for field in contract["schema"]
    ]

    contract["rules"] = [
        {
            "name": f"contains_only_{role}_rows",
            "expression": f"order_role = '{role}'",
        }
    ]

    return contract


def run_order_products_split_job(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    mode: str,
    contract_path: str,
) -> None:
    if mode not in ["base_train", "stacking_train"]:
        raise ValueError(
            f"mode must be either base_train or stacking_train, but received {mode}"
        )

    order_products = read_parquet(join_path(input_path, "order_products"), spark)

    if mode == "base_train":
        model = select_base_model_users(order_products)
    else:
        model = select_stacking_model_users(order_products)

    history, train_label, validation_label = split_order_products_by_role(model)

    base_contract = load_contract(
        join_path(
            contract_path,
            "order_products_split_base.yaml",
        )
    )

    role_datasets = {
        "history": history,
        "train_label": train_label,
        "validation_label": validation_label,
    }

    for role, df in role_datasets.items():
        contract = build_role_contract(
            base_contract,
            role,
        )

        validate_dataset(
            df,
            contract=contract,
        )

        write_parquet(
            join_path(output_path, f"{mode}_{role}"),
            df.select(COLUMNS),
        )
