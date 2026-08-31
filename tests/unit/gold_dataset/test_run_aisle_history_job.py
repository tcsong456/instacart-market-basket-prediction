import pytest

from instacart_etl_rnn.jobs.create_aisle_history_data_job import (
    run_aisle_history_job,
)
from instacart_etl_rnn.validation.exceptions import (
    DataValidationError,
)
from instacart_etl_rnn.validation.models import ValidationReport


def test_run_aisle_history_job_does_not_write_when_validation_fails(
    spark, mocker, aisle_history_df, aisle_product_df
):
    user_data = mocker.sentinel.user_data
    parsed_df = mocker.sentinel.parsed_df

    mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_history_data_job.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_history_data_job.read_parquet",
        side_effect=[
            user_data,
            aisle_product_df,
        ],
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_history_data_job.parse_seq",
        return_value=parsed_df,
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_history_data_job.build_aisle_history_data",
        return_value=aisle_history_df,
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_history_data_job.load_contract",
        return_value=mocker.sentinel.contract,
    )

    report = ValidationReport(dataset_name="aisle_history_data", results=[])
    mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_history_data_job.validate_dataset",
        side_effect=DataValidationError(report),
    )

    mocked_write = mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_history_data_job.write_parquet",
    )

    with pytest.raises(DataValidationError):
        run_aisle_history_job(
            spark=spark,
            input_path="silver",
            data_path="bronze",
            output_path="gold",
            contract_path="contracts",
            mode="validation",
        )

    mocked_write.assert_not_called()
