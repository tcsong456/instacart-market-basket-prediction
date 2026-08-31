import pytest
from pyspark.sql import DataFrame

from instacart_etl_rnn.jobs.create_user_data_job import run_user_data_job
from instacart_etl_rnn.validation.exceptions import DataValidationError
from instacart_etl_rnn.validation.models import ValidationReport


def test_run_user_data_job_does_not_write_when_user_build_fails(
    mocker,
):
    spark = mocker.sentinel.spark

    order_products_df = mocker.sentinel.order_products_df
    order_group_df = mocker.sentinel.order_group_df

    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_data_job.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_data_job.read_parquet",
        return_value=order_products_df,
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_data_job.build_order_level_data",
        return_value=order_group_df,
    )

    mocked_build_user = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_data_job.build_user_level_data",
        side_effect=RuntimeError("user build failed"),
    )

    mocked_validate = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_data_job.validate_dataset"
    )

    mocked_write = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_data_job.write_parquet",
    )

    with pytest.raises(
        RuntimeError,
        match="user build failed",
    ):
        run_user_data_job(
            spark=spark, path="silver", contract_path="contracts", mode="train"
        )

    mocked_build_user.assert_called_once_with(
        order_group_df,
    )

    mocked_validate.assert_not_called()

    mocked_write.assert_not_called()


def test_run_user_data_job_does_not_write_when_validation_fails(
    mocker,
):
    spark = mocker.sentinel.spark

    order_products_df = mocker.sentinel.order_products_df
    order_group_df = mocker.sentinel.order_group_df
    user_group_df = mocker.Mock(spec=DataFrame)
    contract = mocker.sentinel.contract

    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_data_job.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_data_job.read_parquet",
        return_value=order_products_df,
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_data_job.build_order_level_data",
        return_value=order_group_df,
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_data_job.build_user_level_data",
        return_value=user_group_df,
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_user_data_job.load_contract",
        return_value=contract,
    )

    report = ValidationReport(dataset_name="user_data", results=[])
    mocked_validate = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_data_job.validate_dataset",
        side_effect=DataValidationError(report),
    )

    mocked_write = mocker.patch(
        "instacart_etl_rnn.jobs.create_user_data_job.write_parquet",
    )

    with pytest.raises(DataValidationError):
        run_user_data_job(
            spark=spark, path="silver", contract_path="contracts", mode="train"
        )

    mocked_validate.assert_called_once_with(user_group_df, contract=contract)

    mocked_write.assert_not_called()
    user_group_df.unpersist.assert_called_once_with()


@pytest.mark.parametrize(
    "mode",
    [
        "invalid",
        "training",
        "test",
        "",
    ],
)
def test_run_user_data_job_rejects_invalid_mode(
    spark,
    mode,
):
    with pytest.raises(
        ValueError,
        match="mode must be either train or validation or evaluation",
    ):
        run_user_data_job(
            spark=spark,
            path="gs://bucket/data",
            contract_path="gs://bucket/contracts",
            mode=mode,
        )
