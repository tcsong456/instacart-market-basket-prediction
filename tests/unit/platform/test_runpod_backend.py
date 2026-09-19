import pytest
import requests

from instacart_platform.models import TrainingJob, TrainingJobHandle
from instacart_platform.runpod_backend import (
    RUNPOD_API_URL,
    RunpodTrainingBackend,
)

TEMPLATE_ID = "tpl-1"
API_KEY = "rp-test-key"


def _training_job(**overrides) -> TrainingJob:
    payload = {
        "run_id": "run-42",
        "image": "ghcr.io/example/trainer:latest",
        "command": ["-m", "instacart_rnn.train", "--model", "product"],
        "gpu_type": "NVIDIA L4",
        "gpu_count": 1,
    }
    payload.update(overrides)
    return TrainingJob(**payload)


def _backend(**overrides) -> RunpodTrainingBackend:
    payload = {
        "api_key": API_KEY,
        "template_id": TEMPLATE_ID,
    }
    payload.update(overrides)
    return RunpodTrainingBackend(**payload)


def _mock_create_pod_response(mocker, payload):
    response = mocker.Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return mocker.patch(
        "instacart_platform.runpod_backend.requests.post",
        return_value=response,
    )


def test_init_prefers_explicit_api_key_over_environment(monkeypatch, mocker):
    monkeypatch.setenv("RUNPOD_API_KEY", "from-env")
    post = _mock_create_pod_response(mocker, {"id": "pod-123"})

    _backend(api_key=API_KEY).submit(_training_job())

    assert post.call_args.kwargs["headers"]["Authorization"] == f"Bearer {API_KEY}"
    assert post.call_args.kwargs["timeout"] == pytest.approx(30.0)


def test_init_reads_api_key_from_environment(monkeypatch, mocker):
    monkeypatch.setenv("RUNPOD_API_KEY", "from-env")
    post = _mock_create_pod_response(mocker, {"id": "pod-123"})

    _backend(api_key=None).submit(_training_job())

    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer from-env"


def test_init_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    with pytest.raises(
        ValueError,
        match="Runpod API key must be provided either through",
    ):
        _backend(api_key=None)


def test_init_rejects_blank_template_id():
    with pytest.raises(ValueError, match="template_id must not be empty"):
        _backend(template_id="   ")


def test_init_rejects_non_positive_timeout():
    with pytest.raises(
        ValueError,
        match="timeout_seconds must be greater than 0",
    ):
        _backend(timeout_seconds=0)


def test_submit_posts_official_create_pod_schema(mocker):
    post = _mock_create_pod_response(mocker, {"id": "pod-123"})
    job = _training_job()
    backend = _backend(timeout_seconds=15.0)

    handle = backend.submit(job)

    post.assert_called_once_with(
        f"{RUNPOD_API_URL}/pods",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "name": "instacart-run-42",
            "imageName": "ghcr.io/example/trainer:latest",
            "gpuTypeIds": ["NVIDIA L4"],
            "gpuCount": 1,
            "templateId": TEMPLATE_ID,
            "dockerStartCmd": [
                "-m",
                "instacart_rnn.train",
                "--model",
                "product",
            ],
        },
        timeout=15.0,
    )
    assert "dockerArgs" not in post.call_args.kwargs["json"]
    assert handle == TrainingJobHandle(job_id="pod-123", run_id="run-42")


def test_submit_sends_requested_gpu_count(mocker):
    post = _mock_create_pod_response(mocker, {"id": "pod-123"})
    job = _training_job(gpu_count=2)

    _backend().submit(job)

    assert post.call_args.kwargs["json"]["gpuCount"] == 2
    assert post.call_args.kwargs["json"]["gpuTypeIds"] == ["NVIDIA L4"]


def test_submit_raises_when_create_pod_response_omits_id(mocker):
    _mock_create_pod_response(mocker, {})

    with pytest.raises(
        RuntimeError,
        match="did not contain a valid Pod ID",
    ):
        _backend().submit(_training_job())


@pytest.mark.parametrize("pod_id", ["", 123, None])
def test_submit_raises_when_pod_id_is_invalid(mocker, pod_id):
    _mock_create_pod_response(mocker, {"id": pod_id})

    with pytest.raises(
        RuntimeError,
        match="did not contain a valid Pod ID",
    ):
        _backend().submit(_training_job())


def test_submit_raises_for_http_error(mocker):
    response = mocker.Mock()
    response.raise_for_status.side_effect = requests.HTTPError("boom")
    mocker.patch(
        "instacart_platform.runpod_backend.requests.post",
        return_value=response,
    )

    with pytest.raises(requests.HTTPError, match="boom"):
        _backend().submit(_training_job())
