from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(
    *,
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """
    Configure application logging.

    Parameters
    ----------
    level
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file
        Optional log file path.
    """

    root_logger = logging.getLogger()

    if root_logger.handlers:
        root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt=("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_handler = logging.FileHandler(
            log_file,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)

        root_logger.addHandler(file_handler)

    root_logger.setLevel(level.upper())
