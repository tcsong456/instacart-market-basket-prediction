import json
from pathlib import Path

from instacart_platform.runs import (
    build_run_paths,
    write_json,
    write_success_marker,
)


def test_build_run_paths_uses_model_and_run_id_under_local_runs_root(tmp_path):
    paths = build_run_paths(
        runs_root=str(tmp_path),
        model_name="product",
        run_id="run-1",
    )

    run_root = tmp_path / "product" / "run-1"

    assert Path(paths.root) == run_root
    assert Path(paths.run_json) == run_root / "run.json"
    assert Path(paths.best_checkpoint) == run_root / "checkpoints" / "best.pt"
    assert Path(paths.last_checkpoint) == run_root / "checkpoints" / "last.pt"
    assert Path(paths.success_marker) == run_root / "_SUCCESS"


def test_build_run_paths_keeps_gcs_urls_as_strings():
    paths = build_run_paths(
        runs_root="gs://bucket/runs",
        model_name="product",
        run_id="run-1",
    )

    assert paths.root == "gs://bucket/runs/product/run-1"
    assert paths.run_json == "gs://bucket/runs/product/run-1/run.json"
    assert paths.best_checkpoint == (
        "gs://bucket/runs/product/run-1/checkpoints/best.pt"
    )
    assert paths.last_checkpoint == (
        "gs://bucket/runs/product/run-1/checkpoints/last.pt"
    )
    assert paths.success_marker == "gs://bucket/runs/product/run-1/_SUCCESS"


def test_write_json_creates_parent_directories_and_writes_payload(tmp_path):
    path = tmp_path / "product" / "run-1" / "metrics.json"

    write_json(str(path), {"best_epoch": 2, "best_validation_loss": 0.25})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "best_epoch": 2,
        "best_validation_loss": 0.25,
    }


def test_write_success_marker_creates_empty_file(tmp_path):
    path = tmp_path / "product" / "run-1" / "_SUCCESS"

    write_success_marker(str(path))

    assert path.exists()
    assert path.read_text(encoding="utf-8") == ""
