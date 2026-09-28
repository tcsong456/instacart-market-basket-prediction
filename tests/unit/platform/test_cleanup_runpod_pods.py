import json
from datetime import datetime, timedelta, timezone
from io import StringIO

import pytest

from instacart_platform.cleanup_runpod_pods import (
    cleanup_reason,
    cleanup_runpod_pods,
    parse_args,
    parse_created_at,
    parse_managed_pod,
    read_run_status,
)
from instacart_platform.models import CleanupReason, ManagedPod

NOW = datetime(2026, 9, 27, 19, 0, tzinfo=timezone.utc)
RUNS_ROOT = "gs://runs"
MAX_POD_AGE = timedelta(hours=6.5)


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


class _RecordingBackend:
    def __init__(self, pods):
        self._pods = pods
        self.deleted: list[str] = []

    def list_pods(self):
        return self._pods

    def delete_pod(self, pod_id: str) -> None:
        self.deleted.append(pod_id)


def _pod(pod_id, name, created_at):
    return {"id": pod_id, "name": name, "createdAt": created_at}


def _patch_run_files(mocker, files):
    filesystem = mocker.Mock()

    def open_path(path, mode="r"):
        payload = files.get(path)
        if payload is None:
            raise FileNotFoundError(path)
        return StringIO(json.dumps(payload))

    filesystem.open.side_effect = open_path
    mocker.patch(
        "instacart_platform.cleanup_runpod_pods.gcsfs.GCSFileSystem",
        return_value=filesystem,
    )
    return filesystem


def _freeze_now(mocker):
    mocker.patch(
        "instacart_platform.cleanup_runpod_pods.datetime",
        _FrozenDateTime,
    )


@pytest.mark.parametrize(
    ("run_status", "age", "expected"),
    [
        ("completed", timedelta(minutes=1), CleanupReason.RUN_FINISHED),
        ("failed", timedelta(hours=9), CleanupReason.RUN_FINISHED),
        ("running", timedelta(hours=9), CleanupReason.POD_EXPIRED),
        (None, timedelta(hours=9), CleanupReason.POD_EXPIRED),
        ("running", timedelta(hours=6.5), None),
        ("running", timedelta(hours=1), None),
        (None, timedelta(minutes=1), None),
    ],
)
def test_cleanup_reason_selects_finished_runs_before_age(
    run_status,
    age,
    expected,
):
    reason = cleanup_reason(
        run_status=run_status,
        pod_created_at=NOW - age,
        now=NOW,
        max_pod_age=MAX_POD_AGE,
    )

    assert reason is expected


def test_parse_managed_pod_reads_the_model_and_run_id():
    managed = parse_managed_pod("instacart-product-20260927T191405Z_d76a15")

    assert managed == ManagedPod(
        model_name="product",
        run_id="20260927T191405Z_d76a15",
    )


def test_parse_managed_pod_keeps_hyphens_in_the_run_id():
    managed = parse_managed_pod("instacart-product-run-1-extra")

    assert managed == ManagedPod(model_name="product", run_id="run-1-extra")


@pytest.mark.parametrize(
    "pod_name",
    [
        "other-pod",
        "instacart-",
        "instacart-product",
        "instacart-product-",
        "instacart--run-1",
    ],
)
def test_parse_managed_pod_rejects_names_outside_the_managed_pattern(pod_name):
    assert parse_managed_pod(pod_name) is None


def test_parse_created_at_returns_utc_for_a_zulu_timestamp():
    created_at = parse_created_at(
        {"id": "pod-1", "createdAt": "2026-09-27T18:30:00Z"},
    )

    assert created_at == datetime(2026, 9, 27, 18, 30, tzinfo=timezone.utc)


def test_parse_created_at_converts_an_offset_to_utc():
    created_at = parse_created_at(
        {"id": "pod-1", "createdAt": "2026-09-27T15:00:00-04:00"},
    )

    assert created_at == datetime(2026, 9, 27, 19, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("pod", "message"),
    [
        ({"id": "pod-1"}, "invalid createdAt"),
        ({"id": "pod-1", "createdAt": ""}, "invalid createdAt"),
        ({"id": "pod-1", "createdAt": 1}, "invalid createdAt"),
        ({"id": "pod-1", "createdAt": "yesterday"}, "invalid createdAt"),
        (
            {"id": "pod-1", "createdAt": "2026-09-27T19:00:00"},
            "createdAt has no timezone",
        ),
    ],
)
def test_parse_created_at_rejects_an_unusable_timestamp(pod, message):
    with pytest.raises(RuntimeError, match=message):
        parse_created_at(pod)


@pytest.mark.parametrize("runs_root", ["gs://runs", "gs://runs/"])
@pytest.mark.parametrize("status", ["running", "completed", "failed"])
def test_read_run_status_returns_the_stored_status(mocker, runs_root, status):
    filesystem = _patch_run_files(
        mocker,
        {"gs://runs/product/run-1/run.json": {"status": status}},
    )

    result = read_run_status(
        runs_root=runs_root,
        model_name="product",
        run_id="run-1",
    )

    assert result == status
    filesystem.open.assert_called_once_with(
        "gs://runs/product/run-1/run.json",
        "r",
    )


def test_read_run_status_returns_none_when_the_run_file_is_missing(mocker):
    _patch_run_files(mocker, {})

    status = read_run_status(
        runs_root=RUNS_ROOT,
        model_name="product",
        run_id="run-1",
    )

    assert status is None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "missing or invalid 'status'"),
        ({"status": None}, "missing or invalid 'status'"),
        ({"status": "succeeded"}, "Received status: succeeded is invalid"),
    ],
)
def test_read_run_status_rejects_an_invalid_run_file(mocker, payload, message):
    _patch_run_files(
        mocker,
        {"gs://runs/product/run-1/run.json": payload},
    )

    with pytest.raises(RuntimeError, match=message):
        read_run_status(
            runs_root=RUNS_ROOT,
            model_name="product",
            run_id="run-1",
        )


def test_cleanup_deletes_finished_and_expired_pods_and_keeps_a_live_run(mocker):
    _freeze_now(mocker)
    _patch_run_files(
        mocker,
        {
            "gs://runs/product/run-finished/run.json": {"status": "completed"},
            "gs://runs/product/run-failed/run.json": {"status": "failed"},
            "gs://runs/product/run-live/run.json": {"status": "running"},
            "gs://runs/product/run-old/run.json": {"status": "running"},
        },
    )
    backend = _RecordingBackend(
        [
            _pod("pod-other", "scratch", "2026-09-27T10:00:00Z"),
            _pod(
                "pod-finished",
                "instacart-product-run-finished",
                "2026-09-27T18:00:00Z",
            ),
            _pod(
                "pod-failed",
                "instacart-product-run-failed",
                "2026-09-27T18:00:00Z",
            ),
            _pod(
                "pod-live",
                "instacart-product-run-live",
                "2026-09-27T18:00:00Z",
            ),
            _pod(
                "pod-old",
                "instacart-product-run-old",
                "2026-09-27T10:00:00Z",
            ),
            _pod(
                "pod-missing",
                "instacart-product-run-missing",
                "2026-09-27T18:30:00Z",
            ),
        ]
    )

    cleanup_runpod_pods(
        backend=backend,
        runs_root=RUNS_ROOT,
        max_pod_age=6.5,
    )

    assert backend.deleted == ["pod-finished", "pod-failed", "pod-old"]


def test_cleanup_keeps_a_pod_that_is_exactly_the_age_limit(mocker):
    _freeze_now(mocker)
    _patch_run_files(
        mocker,
        {"gs://runs/product/run-1/run.json": {"status": "running"}},
    )
    backend = _RecordingBackend(
        [
            _pod(
                "pod-1",
                "instacart-product-run-1",
                "2026-09-27T12:30:00Z",
            )
        ]
    )

    cleanup_runpod_pods(
        backend=backend,
        runs_root=RUNS_ROOT,
        max_pod_age=6.5,
    )

    assert backend.deleted == []


def test_cleanup_deletes_an_old_pod_that_never_wrote_a_run_file(mocker):
    _freeze_now(mocker)
    _patch_run_files(mocker, {})
    backend = _RecordingBackend(
        [
            _pod(
                "pod-1",
                "instacart-product-run-1",
                "2026-09-27T10:00:00Z",
            )
        ]
    )

    cleanup_runpod_pods(
        backend=backend,
        runs_root=RUNS_ROOT,
        max_pod_age=6.5,
    )

    assert backend.deleted == ["pod-1"]


@pytest.mark.parametrize("max_pod_age", [0, -1])
def test_cleanup_rejects_a_non_positive_age_before_listing_pods(max_pod_age):
    backend = _RecordingBackend([])

    with pytest.raises(RuntimeError, match="max_pod_age must be a positive number"):
        cleanup_runpod_pods(
            backend=backend,
            runs_root=RUNS_ROOT,
            max_pod_age=max_pod_age,
        )

    assert backend.deleted == []


def test_cleanup_stops_when_a_managed_pod_has_no_created_at(mocker):
    _freeze_now(mocker)
    _patch_run_files(
        mocker,
        {"gs://runs/product/run-1/run.json": {"status": "completed"}},
    )
    backend = _RecordingBackend(
        [
            {
                "id": "pod-1",
                "name": "instacart-product-run-1",
            },
            _pod(
                "pod-2",
                "instacart-product-run-2",
                "2026-09-27T10:00:00Z",
            ),
        ]
    )

    with pytest.raises(RuntimeError, match="invalid createdAt"):
        cleanup_runpod_pods(
            backend=backend,
            runs_root=RUNS_ROOT,
            max_pod_age=6.5,
        )

    assert backend.deleted == []


def test_parse_args_reads_the_workflow_flags(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "cleanup_runpod_pods.py",
            "--runs-root",
            "gs://runs",
            "--template-id",
            "tpl-1",
        ],
    )

    args = parse_args()

    assert args.runs_root == "gs://runs"
    assert args.template_id == "tpl-1"
    assert args.max_pod_age == 6.5


def test_parse_args_reads_an_explicit_age(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "cleanup_runpod_pods.py",
            "--runs-root",
            "gs://runs",
            "--template-id",
            "tpl-1",
            "--max-pod-age",
            "8",
        ],
    )

    args = parse_args()

    assert args.max_pod_age == 8


def test_parse_args_requires_the_runs_root_and_template(monkeypatch):
    monkeypatch.setattr("sys.argv", ["cleanup_runpod_pods.py"])

    with pytest.raises(SystemExit):
        parse_args()
