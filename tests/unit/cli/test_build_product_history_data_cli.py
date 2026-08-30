from instacart_etl_rnn.cli.build_product_history_dataset import main


def test_main_runs_product_history_job_and_stops_spark(
    mocker,
):
    args = mocker.Mock(
        input_path="silver",
        data_path="bronze",
        output_path="gold",
        contract_path="contracts",
        mode="train",
    )

    spark = mocker.Mock(name="spark")

    mocked_configure_logging = mocker.patch(
        "instacart_etl_rnn.cli.build_product_history_dataset.configure_logging"
    )

    mocked_parse_args = mocker.patch(
        "instacart_etl_rnn.cli.build_product_history_dataset.parse_args",
        return_value=args,
    )

    mocked_create_spark = mocker.patch(
        "instacart_etl_rnn.cli.build_product_history_dataset.create_spark_session",
        return_value=spark,
    )

    mocked_run_job = mocker.patch(
        "instacart_etl_rnn.cli.build_product_history_dataset.run_product_history_job"
    )

    main()

    mocked_configure_logging.assert_called_once_with()
    mocked_parse_args.assert_called_once_with()

    mocked_create_spark.assert_called_once_with("build_product_history_data")

    mocked_run_job.assert_called_once_with(
        spark=spark,
        input_path="silver",
        data_path="bronze",
        output_path="gold",
        contract_path="contracts",
        mode="train",
    )

    spark.stop.assert_called_once_with()
