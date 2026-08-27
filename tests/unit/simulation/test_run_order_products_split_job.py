import pytest

from instacart_etl_rnn.jobs.create_period_split_data_job import (
    run_order_products_split_job,
)


@pytest.mark.parametrize(
    ("mode"),
    [
        ("base_train"),
        ("stacking_train"),
    ],
)
def test_run_order_products_split_job_runs_expected_pipeline(
    mocker,
    mode,
):
    module_path = "instacart_etl_rnn.jobs.create_period_split_data_job"

    spark = mocker.sentinel.spark
    order_products = mocker.sentinel.order_products
    selected_model = mocker.sentinel.selected_model

    history = mocker.Mock(name="history")
    train_label = mocker.Mock(name="train_label")
    validation_label = mocker.Mock(name="validation_label")

    history_selected = mocker.sentinel.history_selected
    train_selected = mocker.sentinel.trained_selected
    validation_selected = mocker.sentinel.validation_selected

    history.select.return_value = history_selected
    train_label.select.return_value = train_selected
    validation_label.select.return_value = validation_selected

    base_contract = mocker.sentinel.base_contract

    history_contract = mocker.sentinel.history_contract
    train_contract = mocker.sentinel.train_contract
    validation_contract = mocker.sentinel.validation_contract

    mocked_join = mocker.patch(
        f"{module_path}.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )

    read_parquet = mocker.patch(
        f"{module_path}.read_parquet",
        return_value=order_products,
    )

    base_selector = mocker.patch(
        f"{module_path}.select_base_model_users",
        return_value=selected_model,
    )

    stacking_selector = mocker.patch(
        f"{module_path}.select_stacking_model_users",
        return_value=selected_model,
    )

    split_order_products_by_role = mocker.patch(
        f"{module_path}.split_order_products_by_role",
        return_value=(
            history,
            train_label,
            validation_label,
        ),
    )

    load_contract = mocker.patch(
        f"{module_path}.load_contract",
        return_value=base_contract,
    )

    build_role_contract = mocker.patch(
        f"{module_path}.build_role_contract",
        side_effect=[
            history_contract,
            train_contract,
            validation_contract,
        ],
    )

    validate_dataset = mocker.patch(
        f"{module_path}.validate_dataset",
    )

    write_parquet = mocker.patch(
        f"{module_path}.write_parquet",
    )

    mocker.patch(
        f"{module_path}.COLUMNS",
        ["user_id", "order_id"],
    )

    run_order_products_split_job(
        spark=spark,
        input_path="input",
        output_path="output",
        mode=mode,
        contract_path="contracts",
    )

    join_args_list = [
        mocker.call("input", "order_products"),
        mocker.call("contracts", "order_products_split_base.yaml"),
    ]

    read_parquet.assert_called_once_with(
        "input/order_products",
        spark,
    )

    if mode == "base_train":
        base_selector.assert_called_once_with(order_products)
        stacking_selector.assert_not_called()
        assert mocked_join.call_args_list == [
            *join_args_list,
            mocker.call("output", "base_train_history"),
            mocker.call("output", "base_train_train_label"),
            mocker.call("output", "base_train_validation_label"),
        ]
    else:
        stacking_selector.assert_called_once_with(order_products)
        base_selector.assert_not_called()
        assert mocked_join.call_args_list == [
            *join_args_list,
            mocker.call("output", "stacking_train_history"),
            mocker.call("output", "stacking_train_train_label"),
            mocker.call("output", "stacking_train_validation_label"),
        ]

    split_order_products_by_role.assert_called_once_with(
        selected_model,
    )

    load_contract.assert_called_once_with(
        "contracts/order_products_split_base.yaml",
    )

    assert build_role_contract.call_args_list == [
        mocker.call(base_contract, "history"),
        mocker.call(base_contract, "train_label"),
        mocker.call(base_contract, "validation_label"),
    ]

    assert validate_dataset.call_args_list == [
        mocker.call(
            history,
            contract=history_contract,
        ),
        mocker.call(
            train_label,
            contract=train_contract,
        ),
        mocker.call(
            validation_label,
            contract=validation_contract,
        ),
    ]

    history.select.assert_called_once_with(["user_id", "order_id"])
    train_label.select.assert_called_once_with(["user_id", "order_id"])
    validation_label.select.assert_called_once_with(["user_id", "order_id"])

    assert write_parquet.call_args_list == [
        mocker.call(
            f"output/{mode}_history",
            history_selected,
        ),
        mocker.call(
            f"output/{mode}_train_label",
            train_selected,
        ),
        mocker.call(
            f"output/{mode}_validation_label",
            validation_selected,
        ),
    ]


@pytest.mark.parametrize(
    "mode",
    [
        "base_model",
        "stack_model",
        "invalid",
        "",
    ],
)
def test_run_order_products_split_job_rejects_invalid_mode(
    mocker,
    mode,
):
    spark = mocker.Mock()

    with pytest.raises(
        ValueError,
        match="mode must be either base_train or stacking_train",
    ):
        run_order_products_split_job(
            spark=spark,
            input_path="input",
            output_path="output",
            mode=mode,
            contract_path="contracts",
        )
