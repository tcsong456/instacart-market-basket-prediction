import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[3] / "docker" / "vertex" / "entrypoint.sh"
VALID_SA_JSON = json.dumps(
    {
        "type": "service_account",
        "project_id": "demo",
    }
)

pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="vertex-entrypoint is a POSIX bash container script",
)


def _run_entrypoint(
    *args: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command_env = os.environ.copy()
    command_env.pop("GCP_TRAINING_SA_JSON", None)
    command_env.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

    if extra_env is not None:
        command_env.update(extra_env)

    return subprocess.run(
        ["bash", str(ENTRYPOINT), *args],
        capture_output=True,
        text=True,
        env=command_env,
        check=False,
        timeout=60,
    )


def test_vertex_entrypoint_requires_a_python_command():
    result = _run_entrypoint()

    assert result.returncode == 64
    assert "No job entrypoint was provided." in result.stderr


def test_vertex_entrypoint_rejects_invalid_service_account_json():
    result = _run_entrypoint(
        "-c",
        "raise SystemExit('should not run')",
        extra_env={"GCP_TRAINING_SA_JSON": "not-json"},
    )

    assert result.returncode == 78
    assert "GCP_TRAINING_SA_JSON contains invalid JSON." in result.stderr
    assert "should not run" not in result.stderr
    assert "should not run" not in result.stdout


def test_vertex_entrypoint_exposes_credentials_file_to_python():
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


def test_vertex_entrypoint_leaves_adc_unset_without_service_account_json():
    result = _run_entrypoint(
        "-c",
        (
            "import os;"
            "print('HAS_ADC=' + str('GOOGLE_APPLICATION_CREDENTIALS' in os.environ))"
        ),
    )

    assert result.returncode == 0, result.stderr
    assert "HAS_ADC=False" in result.stdout
