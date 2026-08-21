from instacart_etl_rnn.cli.build_reorder_size_training_dataset import main


def test_main_runs_reorder_size_training_job_and_stops_spark(
    mocker,
):
    args = mocker.Mock()
    args.input_path = "silver"
    args.output_path = "gold"
    args.contract_path = "contracts"
    args.pad_length = 5

    spark = mocker.Mock()

    mocked_configure_logging = mocker.patch(
        "instacart_etl_rnn.cli.build_reorder_size_training_dataset.configure_logging"
    )

    mocked_parse_args = mocker.patch(
        "instacart_etl_rnn.cli.build_reorder_size_training_dataset.parse_args",
        return_value=args,
    )

    mocked_create_spark = mocker.patch(
        "instacart_etl_rnn.cli.build_reorder_size_training_dataset."
        "create_spark_session",
        return_value=spark,
    )

    mocked_run_job = mocker.patch(
        "instacart_etl_rnn.cli.build_reorder_size_training_dataset."
        "run_reorder_size_training_data"
    )

    main()

    mocked_configure_logging.assert_called_once_with()
    mocked_parse_args.assert_called_once_with()

    mocked_create_spark.assert_called_once_with("build_reorder_size_training_data")

    mocked_run_job.assert_called_once_with(
        spark=spark,
        input_path="silver",
        output_path="gold",
        contract_path="contracts",
        pad_length=5,
    )

    spark.stop.assert_called_once_with()
