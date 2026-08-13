import pytest

from instacart_etl_rnn.cli.build_order_products_dataset import main


def test_main_runs_build_order_products_and_stops_spark(mocker):
    args = mocker.Mock(
        input_path="bronze",
        contract_path="contracts",
        output_path="silver/order_products",
    )
    spark = mocker.Mock()

    mocked_parse_args = mocker.patch(
        "instacart_etl_rnn.cli.build_order_products_dataset.parse_args",
        return_value=args,
    )

    mocked_configure_logging = mocker.patch(
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

    mocked_parse_args.assert_called_once_with()
    mocked_configure_logging.assert_called_once_with()

    mocked_create_spark.assert_called_once_with("build_order_products")

    mocked_build.assert_called_once_with(
        spark=spark,
        input_path="bronze",
        contract_path="contracts",
        output_path="silver/order_products",
    )

    spark.stop.assert_called_once_with()


def test_main_logs_reraises_and_stops_spark_when_build_fails(
    mocker,
):
    args = mocker.Mock(
        input_path="bronze",
        contract_path="contracts",
        output_path="silver/order_products",
    )
    spark = mocker.Mock()
    error = RuntimeError("build failed")

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
        side_effect=error,
    )

    mocked_logger = mocker.Mock()

    mocker.patch(
        "instacart_etl_rnn.cli.build_order_products_dataset.logging.getLogger",
        return_value=mocked_logger,
    )

    with pytest.raises(RuntimeError) as exc_info:
        main()

    assert exc_info.value is error

    mocked_logger.exception.assert_called_once_with(
        "Building order_products dataset failed"
    )

    spark.stop.assert_called_once_with()
