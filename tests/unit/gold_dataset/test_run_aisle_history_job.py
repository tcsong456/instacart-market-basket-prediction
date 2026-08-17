from unittest.mock import call

import pytest

from instacart_etl_rnn.jobs.create_aisle_history_data_job import (
    run_aisle_history_job,
)
from instacart_etl_rnn.validation.exceptions import (
    DataValidationError,
)
from instacart_etl_rnn.validation.models import ValidationReport


def test_run_aisle_history_job_builds_validates_and_writes(spark, mocker, tmp_path):
    user_data = mocker.sentinel.user_data

    parsed_df = mocker.sentinel.parsed_df

    aisle_history_df = spark.createDataFrame(
        [
            (
                1,
                24,
                "train",
                "1 0",
                "1 0",
                "2 0",
                "3 2",
                "1 2",
                "10 12",
                "-1.0 5.0",
                "1 2",
            ),
        ],
        """
        user_id INT,
        aisle_id INT,
        eval_set STRING,
        is_ordered_history STRING,
        position_in_order STRING,
        num_products_from_aisle STRING,
        aisle_history_size STRING,
        order_dows STRING,
        order_hours STRING,
        days_since_prior_orders STRING,
        order_numbers STRING
        """,
    )

    products_df = spark.createDataFrame(
        [
            (10, 24, 4),
            (20, 24, 4),
            (30, 84, 16),
        ],
        """
        product_id INT,
        aisle_id INT,
        department_id INT
        """,
    )

    contract = mocker.sentinel.contract

    mocked_join_path = mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_history_data_job.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )

    mocked_read = mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_history_data_job.read_parquet",
        side_effect=[
            user_data,
            products_df,
        ],
    )

    mocked_parse = mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_history_data_job.parse_seq",
        return_value=parsed_df,
    )

    mocked_build = mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_history_data_job.build_aisle_history_data",
        return_value=aisle_history_df,
    )

    mocked_load_contract = mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_history_data_job.load_contract",
        return_value=contract,
    )

    mocked_validate = mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_history_data_job.validate_dataset",
    )

    mocked_write = mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_history_data_job.write_parquet",
    )

    manager = mocker.Mock()
    manager.attach_mock(mocked_join_path, "join")
    manager.attach_mock(mocked_read, "read")
    manager.attach_mock(mocked_parse, "parse")
    manager.attach_mock(mocked_build, "build")
    manager.attach_mock(mocked_load_contract, "load")
    manager.attach_mock(mocked_validate, "validate")
    manager.attach_mock(mocked_write, "write")

    run_aisle_history_job(
        spark=spark,
        input_path="silver",
        data_path="bronze",
        output_path=tmp_path / "gold",
        contract_path="contracts",
    )

    validated_df = mocked_validate.call_args.args[0]
    written_path, written_df = mocked_write.call_args.args

    assert manager.mock_calls == [
        call.join("silver", "user_data"),
        call.read("silver/user_data", spark),
        call.parse(user_data),
        call.build(parsed_df),
        call.join("bronze", "products"),
        call.read("bronze/products", spark),
        call.join("contracts", "aisle_history_data.yaml"),
        call.load("contracts/aisle_history_data.yaml"),
        call.validate(validated_df, contract=contract),
        call.join(tmp_path / "gold", "aisle_history_data"),
        call.write(written_path, written_df),
    ]

    actual = {
        (
            row.user_id,
            row.aisle_id,
        ): row.asDict(recursive=True)
        for row in validated_df.collect()
    }

    assert actual[(1, 24)]["department_id"] == 4
    assert actual[(1, 24)]["eval_set"] == "train"
    assert "department_id" not in aisle_history_df.columns

    assert written_path == f"{tmp_path}/gold/aisle_history_data"
    assert validated_df.collect() == written_df.collect()


def test_run_aisle_history_job_does_not_write_when_validation_fails(
    spark,
    mocker,
):
    user_data = mocker.sentinel.user_data
    parsed_df = mocker.sentinel.parsed_df

    aisle_history_df = spark.createDataFrame(
        [
            (
                1,
                24,
                "train",
                "1 0",
                "1 0",
                "2 0",
                "3 2",
                "1 2",
                "10 12",
                "-1.0 5.0",
                "1 2",
            ),
        ],
        """
            user_id INT,
            aisle_id INT,
            eval_set STRING,
            is_ordered_history STRING,
            position_in_order STRING,
            num_products_from_aisle STRING,
            aisle_history_size STRING,
            order_dows STRING,
            order_hours STRING,
            days_since_prior_orders STRING,
            order_numbers STRING
            """,
    )

    products_df = spark.createDataFrame(
        [
            (10, 24, 4),
            (20, 24, 4),
            (30, 84, 16),
        ],
        """
            product_id INT,
            aisle_id INT,
            department_id INT
            """,
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_history_data_job.join_path",
        side_effect=lambda base, name: f"{base}/{name}",
    )

    mocker.patch(
        "instacart_etl_rnn.jobs.create_aisle_history_data_job.read_parquet",
        side_effect=[
            user_data,
            products_df,
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
        )

    mocked_write.assert_not_called()
