import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

ENTRYPOINT = (
    Path(__file__).resolve().parents[3] / "docker" / "training" / "entrypoint.sh"
)
UPLOAD_HOOK = Path(__file__).resolve().parent / "upload_hook"
VALID_SA_JSON = json.dumps(
    {
        "type": "service_account",
        "project_id": "demo",
    }
)

pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="training-entrypoint is a POSIX bash container script",
)


def _entrypoint_env(
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    command_env = os.environ.copy()
    command_env.pop("GCP_TRAINING_SA_JSON", None)
    command_env.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
    command_env.pop("ENTRYPOINT_UPLOAD_SINK", None)
    command_env.pop("ENTRYPOINT_UPLOAD_FAIL", None)

    if extra_env is not None:
        command_env.update(extra_env)

    if command_env.get("ENTRYPOINT_UPLOAD_SINK") or command_env.get(
        "ENTRYPOINT_UPLOAD_FAIL"
    ):
        hook = str(UPLOAD_HOOK)
        previous = command_env.get("PYTHONPATH", "")
        command_env["PYTHONPATH"] = (
            hook if not previous else hook + os.pathsep + previous
        )

    return command_env


def _run_entrypoint(
    *args: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(ENTRYPOINT), *args],
        capture_output=True,
        text=True,
        env=_entrypoint_env(extra_env),
        check=False,
        timeout=60,
    )


def test_training_entrypoint_requires_a_python_command():
    result = _run_entrypoint()

    assert result.returncode == 64
    assert "No job entrypoint was provided." in result.stderr


def test_training_entrypoint_rejects_invalid_service_account_json():
    result = _run_entrypoint(
        "-c",
        "raise SystemExit('should not run')",
        extra_env={"GCP_TRAINING_SA_JSON": "not-json"},
    )

    assert result.returncode == 78
    assert "GCP_TRAINING_SA_JSON contains invalid JSON." in result.stderr
    assert "should not run" not in result.stderr
    assert "should not run" not in result.stdout


def test_training_entrypoint_exposes_credentials_file_to_python():
    result = _run_entrypoint(
        "-c",
        (
            "import json, os, pathlib, stat;"
            "path = os.environ['GOOGLE_APPLICATION_CREDENTIALS'];"
            "payload = pathlib.Path(path).read_text();"
            "mode = stat.S_IMODE(pathlib.Path(path).stat().st_mode);"
            "print('CREDENTIALS_PATH=' + path);"
            "print('JSON_STILL_SET=' + str('GCP_TRAINING_SA_JSON' in os.environ));"
            "print('MODE=' + oct(mode));"
            "print('PAYLOAD=' + payload)"
        ),
        extra_env={"GCP_TRAINING_SA_JSON": VALID_SA_JSON},
    )

    assert result.returncode == 0, result.stderr

    lines = dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if line.startswith(
            ("CREDENTIALS_PATH=", "JSON_STILL_SET=", "MODE=", "PAYLOAD=")
        )
    )

    assert lines["JSON_STILL_SET"] == "False"
    assert lines["MODE"] == oct(0o600)
    assert lines["PAYLOAD"] == VALID_SA_JSON
    assert json.loads(lines["PAYLOAD"])["project_id"] == "demo"
    assert lines["CREDENTIALS_PATH"]


def test_training_entrypoint_leaves_adc_unset_without_service_account_json():
    result = _run_entrypoint(
        "-c",
        (
            "import os;"
            "print('HAS_ADC=' + str('GOOGLE_APPLICATION_CREDENTIALS' in os.environ))"
        ),
    )

    assert result.returncode == 0, result.stderr
    assert "HAS_ADC=False" in result.stdout


def test_training_entrypoint_returns_the_python_exit_code():
    result = _run_entrypoint("-c", "raise SystemExit(3)")

    assert result.returncode == 3


def test_training_entrypoint_removes_the_local_log_after_exit(tmp_path):
    marker = tmp_path / "log-path"
    result = _run_entrypoint(
        "-c",
        (
            "import os;"
            "from pathlib import Path;"
            f"Path({str(marker)!r}).write_text(os.environ['LOG_FILE'])"
        ),
    )

    assert result.returncode == 0, result.stderr
    log_path = Path(marker.read_text(encoding="utf-8"))
    assert not log_path.exists()


@pytest.mark.parametrize(
    "run_args",
    [
        (
            "--runs-root",
            "gs://bucket/runs/",
            "--model",
            "product",
            "--run-id",
            "run-1",
        ),
        (
            "--runs-root=gs://bucket/runs",
            "--model=product",
            "--run-id=run-1",
        ),
    ],
)
def test_training_entrypoint_uploads_the_run_log(tmp_path, run_args):
    sink = tmp_path / "upload.json"
    result = _run_entrypoint(
        "-c",
        "import sys; print('HELLO'); print('ERR', file=sys.stderr)",
        *run_args,
        extra_env={"ENTRYPOINT_UPLOAD_SINK": str(sink)},
    )

    assert result.returncode == 0, result.stderr
    uploaded = json.loads(sink.read_text(encoding="utf-8"))
    assert uploaded["uri"] == "gs://bucket/runs/product/run-1/train.log"
    assert uploaded["content_type"] == "text/plain; charset=utf-8"
    assert "HELLO" in uploaded["body"]
    assert "ERR" in uploaded["body"]
    assert "PyTorch:" in uploaded["body"]
    assert "CUDA available:" in uploaded["body"]


def test_training_entrypoint_uploads_the_log_when_python_fails(tmp_path):
    sink = tmp_path / "upload.json"
    result = _run_entrypoint(
        "-c",
        "print('partial'); raise RuntimeError('boom')",
        "--runs-root",
        "gs://bucket/runs",
        "--model",
        "product",
        "--run-id",
        "run-1",
        extra_env={"ENTRYPOINT_UPLOAD_SINK": str(sink)},
    )

    assert result.returncode == 1, result.stderr
    uploaded = json.loads(sink.read_text(encoding="utf-8"))
    assert "partial" in uploaded["body"]
    assert "RuntimeError: boom" in uploaded["body"]


@pytest.mark.parametrize(
    "run_args",
    [
        ("--model", "product", "--run-id", "run-1"),
        ("--runs-root", "gs://bucket/runs", "--run-id", "run-1"),
        ("--runs-root", "gs://bucket/runs", "--model", "product"),
        ("--runs-root", "gs://bucket/runs", "--model", "--run-id", "run-1"),
    ],
)
def test_training_entrypoint_does_not_upload_without_a_full_run_identity(
    tmp_path,
    run_args,
):
    sink = tmp_path / "upload.json"
    result = _run_entrypoint(
        "-c",
        "raise SystemExit(0)",
        *run_args,
        extra_env={"ENTRYPOINT_UPLOAD_SINK": str(sink)},
    )

    assert result.returncode == 0, result.stderr
    assert not sink.exists()


def test_training_entrypoint_warns_when_the_log_uri_is_not_gcs(tmp_path):
    sink = tmp_path / "upload.json"
    result = _run_entrypoint(
        "-c",
        "raise SystemExit(0)",
        "--runs-root",
        "http://runs",
        "--model",
        "product",
        "--run-id",
        "run-1",
        extra_env={"ENTRYPOINT_UPLOAD_SINK": str(sink)},
    )

    assert result.returncode == 0, result.stderr
    warning = (
        "WARNING: failed to upload training log to http://runs/product/run-1/train.log"
    )
    assert warning in result.stderr
    assert not sink.exists()


def test_training_entrypoint_keeps_the_python_status_when_upload_fails(tmp_path):
    sink = tmp_path / "upload.json"
    result = _run_entrypoint(
        "-c",
        "raise SystemExit(4)",
        "--runs-root",
        "gs://bucket/runs",
        "--model",
        "product",
        "--run-id",
        "run-1",
        extra_env={
            "ENTRYPOINT_UPLOAD_SINK": str(sink),
            "ENTRYPOINT_UPLOAD_FAIL": "1",
        },
    )

    assert result.returncode == 4
    assert (
        "WARNING: failed to upload training log to "
        "gs://bucket/runs/product/run-1/train.log"
    ) in result.stderr


def test_training_entrypoint_forwards_sigterm_to_the_process_group(tmp_path):
    ready = tmp_path / "ready"
    worker_marker = tmp_path / "worker"
    worker_pid = tmp_path / "worker.pid"
    script = """
import os, signal, time
from pathlib import Path
ready = Path(os.environ["READY_MARKER"])
worker_marker = Path(os.environ["WORKER_MARKER"])
worker_pid = Path(os.environ["WORKER_PID"])
if os.fork() == 0:
    worker_pid.write_text(str(os.getpid()))

    def _mark(value):
        worker_marker.write_text(value)
        os._exit(0)

    signal.signal(signal.SIGTERM, lambda *_: _mark("15"))
    ready.write_text("ready")
    time.sleep(30)
    worker_marker.write_text("missed")
    os._exit(0)
time.sleep(30)
"""
    process = subprocess.Popen(
        ["bash", str(ENTRYPOINT), "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_entrypoint_env(
            {
                "READY_MARKER": str(ready),
                "WORKER_MARKER": str(worker_marker),
                "WORKER_PID": str(worker_pid),
            }
        ),
    )

    try:
        deadline = time.monotonic() + 20
        while not ready.exists():
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise AssertionError(
                    f"entrypoint exited early: code={process.returncode}\n"
                    f"stdout={stdout}\nstderr={stderr}"
                )
            if time.monotonic() > deadline:
                raise AssertionError("python did not become ready")
            time.sleep(0.05)

        process.send_signal(signal.SIGTERM)
        _stdout, stderr = process.communicate(timeout=20)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
        if worker_pid.exists():
            try:
                os.kill(int(worker_pid.read_text(encoding="utf-8")), signal.SIGKILL)
            except OSError:
                pass

    assert process.returncode == 143, stderr
    assert worker_marker.read_text(encoding="utf-8") == "15"
