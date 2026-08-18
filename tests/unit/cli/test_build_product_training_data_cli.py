import pytest

from instacart_etl_rnn.cli.build_product_training_dataset import main


def test_main_runs_product_training_job_and_stops_spark(
    mocker,
):
    args = mocker.Mock()
    args.input_path = "silver"
    args.raw_path = "bronze"
    args.output_path = "gold"
    args.contract_path = "contracts"
    args.min_word_freq = 5
    args.product_name_length = 50
    args.encode_length = 40

    spark = mocker.Mock()

    mocked_configure_logging = mocker.patch(
        "instacart_etl_rnn.cli.build_product_training_dataset.configure_logging"
    )

    mocked_parse_args = mocker.patch(
        "instacart_etl_rnn.cli.build_product_training_dataset.parse_args",
        return_value=args,
    )

    mocked_create_spark = mocker.patch(
        "instacart_etl_rnn.cli.build_product_training_dataset.create_spark_session",
        return_value=spark,
    )

    mocked_run_job = mocker.patch(
        "instacart_etl_rnn.cli.build_product_training_dataset."
        "run_product_training_data_job"
    )

    main()

    mocked_configure_logging.assert_called_once_with()
    mocked_parse_args.assert_called_once_with()

    mocked_create_spark.assert_called_once_with("build_product_training_data")

    mocked_run_job.assert_called_once_with(
        spark=spark,
        input_path="silver",
        raw_path="bronze",
        output_path="gold",
        contract_path="contracts",
        min_word_freq=5,
        product_name_length=50,
        encode_length=40,
    )

    spark.stop.assert_called_once_with()


def test_main_logs_reraises_and_stops_spark_when_job_fails(
    mocker,
):
    args = mocker.Mock()
    args.input_path = "silver"
    args.raw_path = "bronze"
    args.output_path = "gold"
    args.contract_path = "contracts"
    args.min_word_freq = 5
    args.product_name_length = 50
    args.encode_length = 40

    spark = mocker.Mock()

    mocker.patch(
        "instacart_etl_rnn.cli.build_product_training_dataset.configure_logging"
    )

    mocker.patch(
        "instacart_etl_rnn.cli.build_product_training_dataset.parse_args",
        return_value=args,
    )

    mocker.patch(
        "instacart_etl_rnn.cli.build_product_training_dataset.create_spark_session",
        return_value=spark,
    )

    error = RuntimeError("job failed")

    mocker.patch(
        "instacart_etl_rnn.cli.build_product_training_dataset."
        "run_product_training_data_job",
        side_effect=error,
    )

    mocked_logger = mocker.Mock()

    mocker.patch(
        "instacart_etl_rnn.cli.build_product_training_dataset.logging.getLogger",
        return_value=mocked_logger,
    )

    with pytest.raises(RuntimeError, match="job failed"):
        main()

    mocked_logger.exception.assert_called_once_with(
        "Build product training data failed!"
    )

    spark.stop.assert_called_once_with()
