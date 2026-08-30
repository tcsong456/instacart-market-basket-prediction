import pytest

from instacart_etl_rnn.jobs.create_period_split_data_job import (
    COLUMNS,
    run_order_products_split_job,
)

MODULE_PATH = "instacart_etl_rnn.jobs.create_period_split_data_job"


def test_run_order_products_split_job_base_train(
    mocker,
    spark,
):
    order_products = mocker.Mock(name="order_products")

    model = mocker.Mock(name="base_model_users")

    train_history = mocker.Mock(name="train_history")
    evaluation_history = mocker.Mock(name="evaluation_history")
    validation_history = mocker.Mock(name="validation_history")

    train_output = mocker.sentinel.train_output
    evaluation_output = mocker.sentinel.evaluation_output
    validation_output = mocker.sentinel.validation_output

    train_history.select.return_value = train_output
    evaluation_history.select.return_value = evaluation_output
    validation_history.select.return_value = validation_output

    base_contract = {
        "dataset": {
            "name": "order_products_split",
        }
    }

    mock_join_path = mocker.patch(
        f"{MODULE_PATH}.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )

    mock_read_parquet = mocker.patch(
        f"{MODULE_PATH}.read_parquet",
        return_value=order_products,
    )

    mock_select_base = mocker.patch(
        f"{MODULE_PATH}.select_base_model_users",
        return_value=model,
    )

    mock_select_stacking = mocker.patch(
        f"{MODULE_PATH}.select_stacking_model_users",
    )

    mock_split = mocker.patch(
        f"{MODULE_PATH}.split_order_products_by_role",
        return_value=(
            train_history,
            evaluation_history,
            validation_history,
        ),
    )

    mock_load_contract = mocker.patch(
        f"{MODULE_PATH}.load_contract",
        return_value=base_contract,
    )

    mock_validate = mocker.patch(
        f"{MODULE_PATH}.validate_dataset",
    )

    mock_write = mocker.patch(
        f"{MODULE_PATH}.write_parquet",
    )

    run_order_products_split_job(
        spark=spark,
        input_path="gs://bucket/input",
        output_path="gs://bucket/output",
        mode="base_train",
        period="t1",
        contract_path="gs://bucket/contracts",
    )

    assert mock_join_path.call_args_list == [
        mocker.call("gs://bucket/input", "order_products"),
        mocker.call("gs://bucket/contracts", "order_products_split_base.yaml"),
        mocker.call("gs://bucket/output/t1", "order_products_train"),
        mocker.call("gs://bucket/output/t1", "order_products_validation"),
        mocker.call("gs://bucket/output/t1", "order_products_evaluation"),
    ]

    mock_read_parquet.assert_called_once_with(
        "gs://bucket/input/order_products",
        spark,
    )

    mock_select_base.assert_called_once_with(
        order_products,
    )

    mock_select_stacking.assert_not_called()

    mock_split.assert_called_once_with(
        model,
    )

    mock_load_contract.assert_called_once_with(
        "gs://bucket/contracts/order_products_split_base.yaml",
    )

    train_history.select.assert_called_once_with(COLUMNS)
    evaluation_history.select.assert_called_once_with(COLUMNS)
    validation_history.select.assert_called_once_with(COLUMNS)

    assert mock_validate.call_count == 3

    assert mock_validate.call_args_list == [
        mocker.call(
            train_output,
            contract={
                "dataset": {
                    "name": "order_products_train",
                }
            },
        ),
        mocker.call(
            validation_output,
            contract={
                "dataset": {
                    "name": "order_products_validation",
                }
            },
        ),
        mocker.call(
            evaluation_output,
            contract={
                "dataset": {
                    "name": "order_products_evaluation",
                }
            },
        ),
    ]

    assert mock_write.call_args_list == [
        mocker.call(
            "gs://bucket/output/t1/order_products_train",
            train_output,
        ),
        mocker.call(
            "gs://bucket/output/t1/order_products_validation",
            validation_output,
        ),
        mocker.call(
            "gs://bucket/output/t1/order_products_evaluation",
            evaluation_output,
        ),
    ]


def test_run_order_products_split_job_stacking_train(
    mocker,
    spark,
):
    order_products = mocker.Mock(name="order_products")

    model = mocker.Mock(name="stacking_model_users")

    train_history = mocker.Mock(name="train_history")
    evaluation_history = mocker.Mock(name="evaluation_history")
    validation_history = mocker.Mock(name="validation_history")

    train_output = mocker.sentinel.train_output
    validation_output = mocker.sentinel.validation_output

    train_history.select.return_value = train_output
    validation_history.select.return_value = validation_output

    base_contract = {
        "dataset": {
            "name": "order_products_split",
        }
    }

    mocker.patch(
        f"{MODULE_PATH}.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )

    mock_read_parquet = mocker.patch(
        f"{MODULE_PATH}.read_parquet",
        return_value=order_products,
    )

    mock_select_base = mocker.patch(
        f"{MODULE_PATH}.select_base_model_users",
    )

    mock_select_stacking = mocker.patch(
        f"{MODULE_PATH}.select_stacking_model_users",
        return_value=model,
    )

    mock_split = mocker.patch(
        f"{MODULE_PATH}.split_order_products_by_role",
        return_value=(
            train_history,
            evaluation_history,
            validation_history,
        ),
    )

    mocker.patch(
        f"{MODULE_PATH}.load_contract",
        return_value=base_contract,
    )

    mock_validate = mocker.patch(
        f"{MODULE_PATH}.validate_dataset",
    )

    mock_write = mocker.patch(
        f"{MODULE_PATH}.write_parquet",
    )

    run_order_products_split_job(
        spark=spark,
        input_path="gs://bucket/input",
        output_path="gs://bucket/output",
        mode="stacking_train",
        period="t1",
        contract_path="gs://bucket/contracts",
    )

    mock_read_parquet.assert_called_once_with(
        "gs://bucket/input/order_products",
        spark,
    )

    mock_select_base.assert_not_called()

    mock_select_stacking.assert_called_once_with(
        order_products,
    )

    mock_split.assert_called_once_with(
        model,
    )

    train_history.select.assert_called_once_with(COLUMNS)
    validation_history.select.assert_called_once_with(COLUMNS)

    evaluation_history.select.assert_not_called()

    assert mock_validate.call_count == 2

    assert mock_validate.call_args_list == [
        mocker.call(
            train_output,
            contract={
                "dataset": {
                    "name": "order_products_train",
                }
            },
        ),
        mocker.call(
            validation_output,
            contract={
                "dataset": {
                    "name": "order_products_validation",
                }
            },
        ),
    ]

    assert mock_write.call_args_list == [
        mocker.call(
            "gs://bucket/output/stacking_train/order_products_train",
            train_output,
        ),
        mocker.call(
            "gs://bucket/output/stacking_train/order_products_validation",
            validation_output,
        ),
    ]


@pytest.mark.parametrize(
    "mode",
    [
        "invalid",
        "train",
        "evaluation",
        "",
    ],
)
def test_run_order_products_split_job_rejects_invalid_mode(
    spark,
    mode,
):
    with pytest.raises(
        ValueError,
        match="mode must be either base_train or stacking_train",
    ):
        run_order_products_split_job(
            spark=spark,
            input_path="input",
            output_path="output",
            mode=mode,
            period="initial",
            contract_path="contracts",
        )


@pytest.mark.parametrize(
    "period",
    [
        "t3",
        "train",
        "final",
        "",
    ],
)
def test_run_order_products_split_job_rejects_invalid_period(
    spark,
    period,
):
    with pytest.raises(
        ValueError,
        match=r"period can only be one of \[initial, t1, t2\]",
    ):
        run_order_products_split_job(
            spark=spark,
            input_path="input",
            output_path="output",
            mode="base_train",
            period=period,
            contract_path="contracts",
        )
