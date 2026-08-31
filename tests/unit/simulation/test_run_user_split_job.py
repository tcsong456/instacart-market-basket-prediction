import pytest
from pyspark.sql import DataFrame

from instacart_etl_rnn.jobs.create_user_split_data_job import run_simulation_split_job
from instacart_etl_rnn.validation.exceptions import DataValidationError
from instacart_etl_rnn.validation.models import ValidationReport


def test_run_user_split_job_unpersists_when_validation_fails(
    mocker,
):
    spark = mocker.Mock()

    orders = mocker.MagicMock()
    filtered_orders = mocker.MagicMock()
    orders.filter.return_value = filtered_orders

    user_split = mocker.Mock()
    order_split = mocker.Mock()
    available_orders = mocker.MagicMock(spec=DataFrame)
    contract = mocker.Mock()

    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.read_parquet",
        return_value=orders,
    )
    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )
    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.build_user_simulation_split",
        return_value=user_split,
    )
    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.build_order_simulation_split",
        return_value=order_split,
    )
    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.add_order_role",
        return_value=available_orders,
    )
    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.load_contract",
        return_value=contract,
    )

    report = ValidationReport(dataset_name="available_orders", results=[])
    validation_error = DataValidationError(report)
    validate_dataset = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.validate_dataset",
        side_effect=validation_error,
    )

    write_parquet = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_split_data_job.write_parquet",
    )

    with pytest.raises(DataValidationError):
        run_simulation_split_job(
            spark=spark,
            input_path="input",
            output_path="output",
            contract_path="contracts",
            period="t1",
        )

    validate_dataset.assert_called_once_with(
        available_orders,
        contract=contract,
    )

    write_parquet.assert_not_called()

    available_orders.unpersist.assert_called_once()
