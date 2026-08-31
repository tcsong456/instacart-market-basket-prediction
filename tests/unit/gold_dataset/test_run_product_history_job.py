import pytest
from pyspark.sql import DataFrame

from instacart_etl_rnn.jobs.create_product_history_data_job import (
    run_product_history_job,
)
from instacart_etl_rnn.validation.exceptions import DataValidationError
from instacart_etl_rnn.validation.models import ValidationReport


def test_run_product_history_job_does_not_write_when_validation_fails(
    mocker,
):
    spark = mocker.sentinel.spark

    user_data = mocker.sentinel.user_data
    products = mocker.sentinel.products
    reorders = mocker.sentinel.reorders
    product_history = mocker.Mock(name="product_history")
    reorder_history = mocker.sentinel.reorder_history
    combined_history = mocker.Mock(spec=DataFrame)
    contract = mocker.sentinel.contract

    mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.read_parquet",
        return_value=user_data,
    )
    mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.parse_seq",
        side_effect=[products, reorders],
    )
    mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.build_each_product_in_order_history",
        return_value=product_history,
    )
    mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.build_each_reorder_history",
        return_value=reorder_history,
    )

    product_history.unionByName.return_value = combined_history

    mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.load_contract",
        return_value=contract,
    )

    report = ValidationReport(dataset_name="product_history_data", results=[])
    mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.validate_dataset",
        side_effect=DataValidationError(report),
    )

    mock_write = mocker.patch(
        "instacart_etl_rnn.jobs.create_product_history_data_job.write_parquet"
    )

    with pytest.raises(DataValidationError):
        run_product_history_job(spark, "input", "data", "output", "contracts", "train")

    mock_write.assert_not_called()
    combined_history.unpersist.assert_called_once_with()
