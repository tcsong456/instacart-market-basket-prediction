import pytest

from instacart_etl_rnn.cli.build_order_products_dataset import main


def test_main_runs_job_and_stops_spark(mocker):
    args = mocker.Mock(
        input_path="bronze",
        contract_path="contracts",
        output_path="silver/order_products",
        validation=False,
    )
    spark = mocker.Mock()

    mocker.patch(
        "instacart_etl_rnn.cli.build_order_products_dataset.parse_args",
        return_value=args,
    )
    mocker.patch(
        "instacart_etl_rnn.cli.build_order_products_dataset.configure_logging",
    )
    mocked_create_spark = mocker.patch(
        "instacart_etl_rnn.cli.build_order_products_dataset.create_spark_session",
        return_value=spark,
    )
    mocked_build = mocker.patch(
        "instacart_etl_rnn.cli.build_order_products_dataset.build_order_products",
    )

    result = main()

    assert result is None

    mocked_create_spark.assert_called_once_with("build_order_products")

    mocked_build.assert_called_once_with(
        spark=spark,
        input_path="bronze",
        contract_path="contracts",
        output_path="silver/order_products",
        validation=False,
    )

    spark.stop.assert_called_once()


def test_main_stops_spark_when_job_fails(mocker):
    args = mocker.Mock(
        input_path="bronze",
        contract_path="contracts",
        output_path="silver/order_products",
        validation=True,
    )
    spark = mocker.Mock()
    job_error = RuntimeError("job failed")

    mocker.patch(
        "instacart_etl_rnn.cli.build_order_products_dataset.parse_args",
        return_value=args,
    )
    mocker.patch(
        "instacart_etl_rnn.cli.build_order_products_dataset.configure_logging",
    )
    mocker.patch(
        "instacart_etl_rnn.cli.build_order_products_dataset.create_spark_session",
        return_value=spark,
    )
    mocker.patch(
        "instacart_etl_rnn.cli.build_order_products_dataset.build_order_products",
        side_effect=job_error,
    )

    mocked_logger = mocker.Mock()
    mocker.patch(
        "instacart_etl_rnn.cli.build_order_products_dataset.logging.getLogger",
        return_value=mocked_logger,
    )

    with pytest.raises(RuntimeError) as exc_info:
        main()

    assert exc_info.value is job_error

    mocked_logger.exception.assert_called_once_with(
        "Building order_proudcts dataset failed"
    )

    spark.stop.assert_called_once()
