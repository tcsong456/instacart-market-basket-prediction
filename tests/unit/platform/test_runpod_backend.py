import json
from io import StringIO

import pytest
import requests

from instacart_platform.models import (
    JobStatus,
    TrainingJob,
    TrainingJobHandle,
    TrainingResult,
)
from instacart_platform.runpod_backend import (
    RUNPOD_API_URL,
    RUNPOD_V2_API_URL,
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
        "runs_root": "gs://runs",
        "model_name": "product",
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


def _handle(**overrides) -> TrainingJobHandle:
    payload = {
        "job_id": "pod-123",
        "run_id": "run-42",
        "runs_root": "gs://runs",
        "model_name": "product",
    }
    payload.update(overrides)
    return TrainingJobHandle(**payload)


@pytest.fixture(autouse=True)
def gcs(mocker):
    filesystem = mocker.Mock()
    mocker.patch(
        "instacart_platform.runpod_backend.gcsfs.GCSFileSystem",
        return_value=filesystem,
    )
    return filesystem


def _configure_run_folder(gcs, *, status=None, success=False):
    if status is None:
        gcs.open.side_effect = FileNotFoundError
    else:
        payload = json.dumps({"status": status})
        gcs.open.side_effect = lambda *args, **kwargs: StringIO(payload)

    gcs.exists.return_value = success


def _mock_get_pod(mocker, payload):
    response = mocker.Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return mocker.patch(
        "instacart_platform.runpod_backend.requests.get",
        return_value=response,
    )


def _mock_create_pod_response(mocker, payload):
    response = mocker.Mock()
    response.ok = True
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return mocker.patch(
        "instacart_platform.runpod_backend.requests.post",
        return_value=response,
    )


def _capacity_response(mocker, error):
    response = mocker.Mock()
    response.ok = False
    response.status_code = 500
    response.text = error
    response.json.return_value = {"error": error, "status": 500}
    return response


def _http_error_response(mocker, *, status_code, payload=None, text="failure"):
    response = mocker.Mock()
    response.ok = False
    response.status_code = status_code
    response.text = text
    if payload is None:
        response.json.side_effect = requests.JSONDecodeError("bad json", text, 0)
    else:
        response.json.return_value = payload
    response.raise_for_status.side_effect = requests.HTTPError("boom")
    return response


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


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("poll_interval_seconds", "poll_interval_seconds must be greater than 0"),
        ("max_wait_seconds", "max_wait_seconds must be greater than 0"),
        (
            "completion_grace_seconds",
            "completion_grace_seconds must be greater than 0",
        ),
    ],
)
def test_init_rejects_non_positive_wait_timeouts(field, message):
    with pytest.raises(ValueError, match=message):
        _backend(**{field: 0})


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
    assert handle == TrainingJobHandle(
        job_id="pod-123",
        run_id="run-42",
        runs_root="gs://runs",
        model_name="product",
    )


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
    response = _http_error_response(
        mocker,
        status_code=500,
        payload={"error": "internal failure"},
        text="internal failure",
    )
    post = mocker.patch(
        "instacart_platform.runpod_backend.requests.post",
        return_value=response,
    )

    with pytest.raises(requests.HTTPError, match="boom"):
        _backend().submit(_training_job())

    assert post.call_count == 1


def test_submit_rejects_unsupported_gpu_type(mocker):
    post = mocker.patch("instacart_platform.runpod_backend.requests.post")

    with pytest.raises(ValueError, match="Unsupported GPU type: 'Tesla T4'"):
        _backend().submit(_training_job(gpu_type="Tesla T4"))

    post.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        "create pod: There are no instances currently available",
        "There are no longer any instances available",
        "THERE ARE NO INSTANCES CURRENTLY AVAILABLE",
    ],
)
def test_submit_falls_back_to_the_next_gpu_when_preferred_has_no_capacity(
    mocker,
    error,
):
    created = mocker.Mock()
    created.ok = True
    created.json.return_value = {"id": "pod-999"}
    post = mocker.patch(
        "instacart_platform.runpod_backend.requests.post",
        side_effect=[_capacity_response(mocker, error), created],
    )

    handle = _backend().submit(
        _training_job(gpu_type="NVIDIA A40", gpu_count=2),
    )

    assert handle.job_id == "pod-999"
    assert [call.kwargs["json"]["gpuTypeIds"] for call in post.call_args_list] == [
        ["NVIDIA A40"],
        ["NVIDIA L4"],
    ]
    assert post.call_args_list[1].kwargs["json"]["gpuCount"] == 2


def test_submit_raises_when_no_gpu_type_has_capacity(mocker):
    error = "create pod: There are no instances currently available"
    responses = [_capacity_response(mocker, error) for _ in range(5)]
    post = mocker.patch(
        "instacart_platform.runpod_backend.requests.post",
        side_effect=responses,
    )

    with pytest.raises(
        RuntimeError,
        match="No configured RunPod GPU type currently has capacity",
    ) as exc_info:
        _backend().submit(_training_job())

    assert post.call_count == 5
    assert "NVIDIA L4" in str(exc_info.value)
    assert "NVIDIA L40S" in str(exc_info.value)
    for response in responses:
        response.raise_for_status.assert_not_called()


@pytest.mark.parametrize(
    ("status_code", "payload"),
    [
        (500, {"error": "template not found"}),
        (500, {"error": {"message": "There are no instances currently available"}}),
        (400, {"error": "There are no instances currently available"}),
        (500, None),
    ],
)
def test_submit_does_not_try_another_gpu_after_a_non_capacity_error(
    mocker,
    status_code,
    payload,
):
    post = mocker.patch(
        "instacart_platform.runpod_backend.requests.post",
        return_value=_http_error_response(
            mocker,
            status_code=status_code,
            payload=payload,
        ),
    )

    with pytest.raises(requests.HTTPError, match="boom"):
        _backend().submit(_training_job())

    assert post.call_count == 1


def test_submit_does_not_try_another_gpu_when_create_succeeds_without_a_pod_id(
    mocker,
):
    accepted = mocker.Mock()
    accepted.ok = True
    accepted.json.return_value = {}
    post = mocker.patch(
        "instacart_platform.runpod_backend.requests.post",
        side_effect=[
            _capacity_response(
                mocker,
                "create pod: There are no instances currently available",
            ),
            accepted,
        ],
    )

    with pytest.raises(RuntimeError, match="did not contain a valid Pod ID"):
        _backend().submit(_training_job())

    assert post.call_count == 2


def test_terminate_deletes_pod(
    mocker,
):
    response = mocker.Mock()

    delete = mocker.patch(
        "instacart_platform.runpod_backend.requests.delete",
        return_value=response,
    )

    backend = RunpodTrainingBackend(
        api_key="secret-api-key",
        template_id="template-123",
        timeout_seconds=20,
    )

    handle = TrainingJobHandle(
        job_id="pod-123",
        run_id="run-123",
        runs_root="gs://runs",
        model_name="product",
    )

    backend.terminate(handle)

    delete.assert_called_once_with(
        f"{RUNPOD_API_URL}/pods/pod-123",
        headers={
            "Authorization": "Bearer secret-api-key",
            "Content-Type": "application/json",
        },
        timeout=20,
    )

    response.raise_for_status.assert_called_once_with()


def test_terminate_raises_for_http_error(
    mocker,
):
    response = mocker.Mock()
    response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")

    delete = mocker.patch(
        "instacart_platform.runpod_backend.requests.delete",
        return_value=response,
    )
    mocker.patch("instacart_platform.runpod_backend.time.sleep")

    backend = RunpodTrainingBackend(
        api_key="api-key",
        template_id="template-123",
    )

    handle = TrainingJobHandle(
        job_id="pod-123",
        run_id="run-123",
        runs_root="gs://runs",
        model_name="product",
    )

    with pytest.raises(
        RuntimeError,
        match="unable to terminate RunPod job",
    ):
        backend.terminate(handle)

    assert delete.call_count == 5


def test_terminate_ignores_missing_pod(mocker):
    response = mocker.Mock()
    response.status_code = 404
    mocker.patch(
        "instacart_platform.runpod_backend.requests.delete",
        return_value=response,
    )

    _backend().terminate(_handle())

    response.raise_for_status.assert_not_called()


def test_get_status_succeeds_when_run_json_is_completed_and_success_exists(
    gcs,
    mocker,
):
    _configure_run_folder(gcs, status="completed", success=True)
    get = mocker.patch("instacart_platform.runpod_backend.requests.get")

    status = _backend().get_status(_handle())

    assert status is JobStatus.SUCCEEDED
    get.assert_not_called()


def test_get_status_keeps_running_while_completed_run_awaits_success_marker(
    gcs,
    mocker,
):
    _configure_run_folder(gcs, status="completed", success=False)
    mocker.patch(
        "instacart_platform.runpod_backend.time.monotonic",
        return_value=100.0,
    )

    status = _backend(completion_grace_seconds=30).get_status(_handle())

    assert status is JobStatus.RUNNING


def test_get_status_fails_when_completed_run_never_writes_success_marker(
    gcs,
    mocker,
):
    _configure_run_folder(gcs, status="completed", success=False)
    backend = _backend(completion_grace_seconds=10)
    handle = _handle()
    mocker.patch(
        "instacart_platform.runpod_backend.time.monotonic",
        side_effect=[100.0, 111.0],
    )

    assert backend.get_status(handle) is JobStatus.RUNNING
    assert backend.get_status(handle) is JobStatus.FAILED


def test_get_status_fails_when_run_json_is_failed(gcs, mocker):
    _configure_run_folder(gcs, status="failed")
    get = mocker.patch("instacart_platform.runpod_backend.requests.get")

    status = _backend().get_status(_handle())

    assert status is JobStatus.FAILED
    get.assert_not_called()


def test_get_status_gets_pod_from_v2_endpoint(gcs, mocker):
    _configure_run_folder(gcs, status="running")
    get = _mock_get_pod(
        mocker,
        {"status": "RUNNING", "runtime": {"uptime": 12}},
    )
    backend = _backend(timeout_seconds=15.0)

    status = backend.get_status(_handle())

    assert status is JobStatus.RUNNING
    get.assert_called_once_with(
        f"{RUNPOD_V2_API_URL}/pods/pod-123",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        timeout=15.0,
    )


def test_get_status_is_running_when_run_json_is_running_and_runtime_is_present(
    gcs,
    mocker,
):
    _configure_run_folder(gcs, status="running")
    _mock_get_pod(mocker, {"status": "RUNNING", "runtime": {"uptime": 12}})

    status = _backend().get_status(_handle())

    assert status is JobStatus.RUNNING


def test_get_status_keeps_running_when_runtime_is_missing_on_first_sighting(
    gcs,
    mocker,
):
    _configure_run_folder(gcs, status="running")
    _mock_get_pod(mocker, {"status": "RUNNING", "runtime": None})

    status = _backend().get_status(_handle())

    assert status is JobStatus.RUNNING


def test_get_status_fails_when_runtime_disappears_after_run_json_is_running(
    gcs,
    mocker,
):
    _configure_run_folder(gcs, status="running")
    backend = _backend()
    handle = _handle()
    _mock_get_pod(mocker, {"status": "RUNNING", "runtime": {"uptime": 12}})

    assert backend.get_status(handle) is JobStatus.RUNNING

    _mock_get_pod(mocker, {"status": "RUNNING", "runtime": None})

    assert backend.get_status(handle) is JobStatus.FAILED


@pytest.mark.parametrize("pod_status", ["EXITED", "ERROR", "TERMINATED"])
def test_get_status_fails_when_pod_lifecycle_is_dead(gcs, mocker, pod_status):
    _configure_run_folder(gcs)
    _mock_get_pod(mocker, {"status": pod_status, "runtime": None})

    status = _backend().get_status(_handle())

    assert status is JobStatus.FAILED


@pytest.mark.parametrize("pod_status", ["EXITED", "ERROR", "TERMINATED"])
def test_get_status_fails_when_run_json_is_running_and_pod_lifecycle_is_dead(
    gcs,
    mocker,
    pod_status,
):
    _configure_run_folder(gcs, status="running")
    _mock_get_pod(mocker, {"status": pod_status, "runtime": None})

    status = _backend().get_status(_handle())

    assert status is JobStatus.FAILED


def test_get_status_fails_when_pod_is_not_found(gcs, mocker):
    _configure_run_folder(gcs)
    response = mocker.Mock()
    response.status_code = 404
    mocker.patch(
        "instacart_platform.runpod_backend.requests.get",
        return_value=response,
    )

    status = _backend().get_status(_handle())

    assert status is JobStatus.FAILED
    response.raise_for_status.assert_not_called()


def test_get_status_is_pending_when_container_is_up_before_run_json(
    gcs,
    mocker,
):
    _configure_run_folder(gcs)
    _mock_get_pod(mocker, {"status": "RUNNING", "runtime": {"uptime": 3}})

    status = _backend().get_status(_handle())

    assert status is JobStatus.PENDING


def test_get_status_fails_when_runtime_disappears_before_run_json(
    gcs,
    mocker,
):
    _configure_run_folder(gcs)
    backend = _backend()
    handle = _handle()
    _mock_get_pod(mocker, {"status": "RUNNING", "runtime": {"uptime": 3}})

    assert backend.get_status(handle) is JobStatus.PENDING

    _mock_get_pod(mocker, {"status": "RUNNING", "runtime": None})

    assert backend.get_status(handle) is JobStatus.FAILED


@pytest.mark.parametrize("pod_status", ["PROVISIONING", "STARTING"])
def test_get_status_is_pending_during_startup_grace_without_runtime(
    gcs,
    mocker,
    pod_status,
):
    _configure_run_folder(gcs)
    _mock_get_pod(mocker, {"status": pod_status, "runtime": None})
    mocker.patch(
        "instacart_platform.runpod_backend.time.monotonic",
        return_value=50.0,
    )

    status = _backend(startup_grace_seconds=60).get_status(_handle())

    assert status is JobStatus.PENDING


def test_get_status_fails_when_startup_grace_expires_without_runtime(
    gcs,
    mocker,
):
    _configure_run_folder(gcs)
    _mock_get_pod(mocker, {"status": "PROVISIONING", "runtime": None})
    backend = _backend(startup_grace_seconds=10)
    handle = _handle()
    mocker.patch(
        "instacart_platform.runpod_backend.time.monotonic",
        side_effect=[20.0, 31.0],
    )

    assert backend.get_status(handle) is JobStatus.PENDING
    assert backend.get_status(handle) is JobStatus.FAILED


def test_wait_returns_succeeded_when_run_folder_is_complete(gcs):
    _configure_run_folder(gcs, status="completed", success=True)

    result = _backend().wait(_handle())

    assert result == TrainingResult(
        run_id="run-42",
        status=JobStatus.SUCCEEDED,
    )


def test_wait_returns_failed_when_run_folder_is_failed(gcs):
    _configure_run_folder(gcs, status="failed")

    result = _backend().wait(_handle())

    assert result == TrainingResult(
        run_id="run-42",
        status=JobStatus.FAILED,
    )


def test_wait_polls_until_run_succeeds(gcs, mocker):
    _configure_run_folder(gcs)
    backend = _backend(poll_interval_seconds=2)
    mocker.patch.object(
        backend,
        "get_status",
        side_effect=[JobStatus.PENDING, JobStatus.RUNNING, JobStatus.SUCCEEDED],
    )
    sleep = mocker.patch("instacart_platform.runpod_backend.time.sleep")
    mocker.patch(
        "instacart_platform.runpod_backend.time.monotonic",
        side_effect=[0.0, 1.0, 2.0],
    )

    result = backend.wait(_handle())

    assert result.status is JobStatus.SUCCEEDED
    assert sleep.call_count == 2
    sleep.assert_called_with(2)


def test_wait_times_out_when_run_never_finishes(mocker):
    backend = _backend(max_wait_seconds=10, poll_interval_seconds=1)
    mocker.patch.object(backend, "get_status", return_value=JobStatus.PENDING)
    mocker.patch("instacart_platform.runpod_backend.time.sleep")
    mocker.patch(
        "instacart_platform.runpod_backend.time.monotonic",
        side_effect=[0.0, 11.0],
    )

    with pytest.raises(TimeoutError, match="did not complete within 10 seconds"):
        backend.wait(_handle())
