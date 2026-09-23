import json
import os
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import gcsfs

from instacart_etl_rnn.common.paths import is_gcs_url, join_path


@dataclass(frozen=True)
class RunPaths:
    root: str

    @property
    def run_json(self) -> str:
        return join_path(self.root, "run.json")

    @property
    def config_json(self) -> str:
        return join_path(self.root, "config.json")

    @property
    def metrics_json(self) -> str:
        return join_path(self.root, "metrics.json")

    @property
    def checkpoints(self) -> str:
        return join_path(self.root, "checkpoints")

    @property
    def best_checkpoint(self) -> str:
        return join_path(self.checkpoints, "best.pt")

    @property
    def last_checkpoint(self) -> str:
        return join_path(self.checkpoints, "last.pt")

    @property
    def artifacts(self) -> str:
        return join_path(self.root, "artifacts")

    @property
    def success_marker(self) -> str:
        return join_path(self.root, "_SUCCESS")


def build_run_paths(
    *,
    runs_root: str,
    model_name: str,
    run_id: str,
) -> RunPaths:
    return RunPaths(
        root=join_path(
            f"{runs_root}/{model_name}",
            run_id,
        )
    )


def _write_text(path: str, text: str) -> None:
    if is_gcs_url(path):
        fs = gcsfs.GCSFileSystem()

        with fs.open(path, "w") as file:
            file.write(text)

        return

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def write_json(
    path: str,
    payload: dict[str, Any],
) -> None:
    _write_text(
        path,
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        ),
    )


def write_dataclass_json(
    path: str,
    value: Any,
) -> None:
    if not is_dataclass(value):
        raise TypeError("value must be a dataclass instance")

    write_json(path, asdict(value))


def write_success_marker(path: str) -> None:
    _write_text(path, "")
