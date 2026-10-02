import json
import logging
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from instacart_rnn.stacking.folds import (
    STACKING_MODELS,
    assign_user_folds,
    generate_stack_id,
    main,
    parse_args,
    select_stacking_artifacts,
    stack_output_dir,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_users(path: Path, user_ids: list[int]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"user_id": pa.array(user_ids, type=pa.int64())}),
        path / "part.parquet",
    )


def _write_run(
    runs_root: Path,
    *,
    model_name: str,
    run_id: str,
    status: str,
    completed_at: str | None,
    timeline: str,
    mode: str = "sample",
    success: bool = True,
    artifact: bool = True,
    user_ids: list[int] | None = None,
) -> None:
    run_root = runs_root / model_name / run_id
    _write_json(
        run_root / "run.json",
        {
            "status": status,
            "completed_at": completed_at,
            "run_id": run_id,
            "model_name": model_name,
        },
    )
    _write_json(
        run_root / "config.json",
        {
            "train_path": (
                f"gs://gold/training/{mode}/{timeline}/{model_name}_training_data_train"
            )
        },
    )
    if success:
        (run_root / "_SUCCESS").write_text("", encoding="utf-8")
    if artifact:
        _write_users(run_root / "artifacts" / "stacking_train", user_ids or [1])


def test_select_stacking_artifacts_keeps_the_latest_completed_timeline_run(tmp_path):
    for model_name in STACKING_MODELS:
        _write_run(
            tmp_path,
            model_name=model_name,
            run_id="20260101T000000Z_old",
            status="completed",
            completed_at="2026-01-01T00:00:00+00:00",
            timeline="t1",
        )
        _write_run(
            tmp_path,
            model_name=model_name,
            run_id="20260201T000000Z_new",
            status="completed",
            completed_at="2026-02-01T00:00:00+00:00",
            timeline="t1",
            user_ids=[10, 11],
        )
        _write_run(
            tmp_path,
            model_name=model_name,
            run_id="20260301T000000Z_other",
            status="completed",
            completed_at="2026-03-01T00:00:00+00:00",
            timeline="t2",
        )
        _write_run(
            tmp_path,
            model_name=model_name,
            run_id="20260401T000000Z_failed",
            status="failed",
            completed_at="2026-04-01T00:00:00+00:00",
            timeline="t1",
        )
        _write_run(
            tmp_path,
            model_name=model_name,
            run_id="20260501T000000Z_incomplete",
            status="completed",
            completed_at="2026-05-01T00:00:00+00:00",
            timeline="t1",
            success=False,
        )

    selected = select_stacking_artifacts(
        runs_root=str(tmp_path),
        mode="sample",
        timeline="t1",
    )

    assert set(selected) == set(STACKING_MODELS)
    for model_name, artifact in selected.items():
        assert artifact.run_id == "20260201T000000Z_new"
        assert Path(artifact.artifact_path) == (
            tmp_path
            / model_name
            / "20260201T000000Z_new"
            / "artifacts"
            / "stacking_train"
        )
        assert "/training/sample/t1/" in artifact.train_path


def test_select_model_artifact_breaks_equal_timestamps_by_run_id(tmp_path):
    _write_run(
        tmp_path,
        model_name="product",
        run_id="20260201T000000Z_aaa",
        status="completed",
        completed_at="2026-02-01T00:00:00+00:00",
        timeline="t1",
    )
    _write_run(
        tmp_path,
        model_name="product",
        run_id="20260201T000000Z_zzz",
        status="completed",
        completed_at="2026-02-01T00:00:00+00:00",
        timeline="t1",
    )
    for model_name in STACKING_MODELS:
        if model_name == "product":
            continue
        _write_run(
            tmp_path,
            model_name=model_name,
            run_id="20260201T000000Z_only",
            status="completed",
            completed_at="2026-02-01T00:00:00+00:00",
            timeline="t1",
        )

    selected = select_stacking_artifacts(
        runs_root=str(tmp_path),
        mode="sample",
        timeline="t1",
    )

    assert selected["product"].run_id == "20260201T000000Z_zzz"


def test_select_stacking_artifacts_raises_when_a_model_has_no_match(tmp_path):
    for model_name in STACKING_MODELS:
        if model_name == "aisle":
            continue
        _write_run(
            tmp_path,
            model_name=model_name,
            run_id="20260201T000000Z_ok",
            status="completed",
            completed_at="2026-02-01T00:00:00+00:00",
            timeline="t1",
        )

    with pytest.raises(FileNotFoundError, match="aisle"):
        select_stacking_artifacts(
            runs_root=str(tmp_path),
            mode="sample",
            timeline="t1",
        )


def test_assign_user_folds_is_stable_and_independent_of_order():
    user_ids = [40, 7, 18, 3, 99]

    first = assign_user_folds(user_ids, n_folds=4, seed=42)
    second = assign_user_folds(list(reversed(user_ids)), n_folds=4, seed=42)
    five_folds = assign_user_folds(user_ids, n_folds=5, seed=42)

    assert first == second
    assert set(first) == set(user_ids)
    assert set(first.values()) <= set(range(4))
    assert set(five_folds.values()) <= set(range(5))
    assert first[7] == assign_user_folds([7], n_folds=4, seed=42)[7]


def test_assign_user_folds_rejects_an_unsupported_fold_count():
    with pytest.raises(ValueError, match="n_folds"):
        assign_user_folds([1], n_folds=3, seed=42)


def test_parse_args_defaults_to_four_folds(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "stacking_train.py",
            "--runs-root",
            "gs://runs",
            "--mode",
            "sample",
            "--timeline",
            "t1",
        ],
    )

    args = parse_args()

    assert args.n_folds == 4
    assert args.seed == 42
    assert args.timeline == "t1"
    assert not hasattr(args, "output_dir")


def test_generate_stack_id_uses_utc_timestamp_and_hex_suffix():
    stack_id = generate_stack_id()

    assert re.fullmatch(r"\d{8}T\d{6}Z_[0-9a-f]{6}", stack_id)


def test_stack_output_dir_nests_the_stack_id_under_runs_root():
    assert stack_output_dir("gs://runs/", "20260201T000000Z_abc123") == (
        "gs://runs/stack/20260201T000000Z_abc123"
    )


def test_stack_output_dir_rejects_a_blank_stack_id():
    with pytest.raises(ValueError, match="stack_id"):
        stack_output_dir("gs://runs", "  ")


def test_main_writes_the_stack_under_runs_root(tmp_path, monkeypatch, capsys):
    for model_name in STACKING_MODELS:
        _write_run(
            tmp_path,
            model_name=model_name,
            run_id="20260201T000000Z_ok",
            status="completed",
            completed_at="2026-02-01T00:00:00+00:00",
            timeline="t1",
            user_ids=[4, 9],
        )
    monkeypatch.setattr(
        "sys.argv",
        [
            "folds.py",
            "--runs-root",
            str(tmp_path),
            "--mode",
            "sample",
            "--timeline",
            "t1",
        ],
    )
    monkeypatch.setattr(
        "instacart_rnn.stacking.folds.generate_stack_id",
        lambda: "20260201T000000Z_abc123",
    )

    root_logger = logging.getLogger()
    handlers_before = list(root_logger.handlers)
    level_before = root_logger.level

    main()

    assert list(root_logger.handlers) == handlers_before
    assert root_logger.level == level_before

    stack_dir = tmp_path / "stack" / "20260201T000000Z_abc123"
    payload = json.loads((stack_dir / "selected_artifacts.json").read_text())
    folds = pq.read_table(stack_dir / "user_folds.parquet")
    captured = capsys.readouterr()

    assert payload["stack_id"] == "20260201T000000Z_abc123"
    assert set(payload["models"]) == set(STACKING_MODELS)
    assert folds.column("user_id").to_pylist() == [4, 9]
    assert Path(captured.out.strip()) == stack_dir
    assert "stack_id=20260201T000000Z_abc123" in captured.err
