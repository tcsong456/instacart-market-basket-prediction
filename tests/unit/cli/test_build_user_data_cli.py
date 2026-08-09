import pytest

from instacart_etl_rnn.cli.build_user_dataset import main


def test_main_runs_user_data_job_and_stops_spark(
    mocker,
):
    args = mocker.Mock(
        path="gs://silver/user_data",
        contract_path="gs://contracts",
    )

    spark = mocker.Mock()

    mocked_logging = mocker.patch(
        "instacart_etl_rnn.cli.build_user_dataset.configure_logging"
    )

    mocked_parse_args = mocker.patch(
        "instacart_etl_rnn.cli.build_user_dataset.parse_args",
        return_value=args,
    )

    mocked_create_spark = mocker.patch(
        "instacart_etl_rnn.cli.build_user_dataset.create_spark_session",
        return_value=spark,
    )

    mocked_run_job = mocker.patch(
        "instacart_etl_rnn.cli.build_user_dataset.run_user_data_job"
    )

    main()

    mocked_logging.assert_called_once_with()

    mocked_parse_args.assert_called_once_with()

    mocked_create_spark.assert_called_once_with("build_user_data")

    mocked_run_job.assert_called_once_with(
        spark=spark,
        path=args.path,
        contract_path=args.contract_path,
    )

    spark.stop.assert_called_once()


def test_main_logs_and_reraises_when_job_fails(
    mocker,
):
    args = mocker.Mock(
        path="silver",
        contract_path="contracts",
    )

    spark = mocker.Mock()
    error = RuntimeError("job failed")

    mocker.patch("instacart_etl_rnn.cli.build_user_dataset.configure_logging")

    mocker.patch(
        "instacart_etl_rnn.cli.build_user_dataset.parse_args",
        return_value=args,
    )

    mocker.patch(
        "instacart_etl_rnn.cli.build_user_dataset.create_spark_session",
        return_value=spark,
    )

    mocker.patch(
        "instacart_etl_rnn.cli.build_user_dataset.run_user_data_job",
        side_effect=error,
    )

    mocked_logger = mocker.Mock()

    mocker.patch(
        "instacart_etl_rnn.cli.build_user_dataset.logging.getLogger",
        return_value=mocked_logger,
    )

    with pytest.raises(RuntimeError) as exc_info:
        main()

    assert exc_info.value is error

    mocked_logger.exception.assert_called_once_with("Build user data failed!")

    spark.stop.assert_called_once()
