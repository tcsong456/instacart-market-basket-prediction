import logging
import sys

import pytest

from instacart_etl_rnn.common.setup_logging import configure_logging


def test_configure_logging_configures_console_handler(
    preserve_root_logger,
):
    root_logger = preserve_root_logger

    configure_logging(
        level="DEBUG",
    )

    assert root_logger.level == logging.DEBUG
    assert len(root_logger.handlers) == 1

    handler = root_logger.handlers[0]

    assert type(handler) is logging.StreamHandler
    assert handler.stream is sys.stdout

    assert handler.formatter is not None
    assert handler.formatter._fmt == (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    assert handler.formatter.datefmt == ("%Y-%m-%d %H:%M:%S")


def test_configure_logging_replaces_existing_handlers(
    preserve_root_logger,
):
    root_logger = preserve_root_logger

    existing_handler = logging.NullHandler()
    root_logger.addHandler(existing_handler)

    configure_logging()

    assert existing_handler not in root_logger.handlers

    assert len(root_logger.handlers) == 1
    assert type(root_logger.handlers[0]) is logging.StreamHandler


def test_configure_logging_adds_file_handler(
    preserve_root_logger,
    tmp_path,
):
    root_logger = preserve_root_logger

    log_file = tmp_path / "logs" / "application.log"

    configure_logging(
        level="WARNING",
        log_file=log_file,
    )

    assert root_logger.level == logging.WARNING

    assert len(root_logger.handlers) == 2

    console_handler = root_logger.handlers[0]
    file_handler = root_logger.handlers[1]

    assert type(console_handler) is logging.StreamHandler
    assert isinstance(file_handler, logging.FileHandler)

    assert log_file.parent.exists()
    assert log_file.exists()

    assert file_handler.baseFilename == str(log_file.resolve())

    logger = logging.getLogger("test_logger")
    logger.warning("test message")

    file_handler.flush()

    assert "test message" in log_file.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("debug", logging.DEBUG),
        ("info", logging.INFO),
        ("warning", logging.WARNING),
        ("error", logging.ERROR),
        ("critical", logging.CRITICAL),
    ],
)
def test_configure_logging_accepts_lowercase_levels(
    preserve_root_logger,
    level,
    expected,
):
    root_logger = preserve_root_logger

    configure_logging(
        level=level,
    )

    assert root_logger.level == expected
