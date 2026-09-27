#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -eq 0 ]]; then
    cat >&2 <<'MSG'
No job entrypoint was provided.
Pass a Python script or "-m <module>" as container arguments, for example:
  -m instacart_rnn.run --model product ...
  -m instacart_rnn.inference --model product ...
MSG
    exit 64
fi

GCP_CREDENTIALS_FILE=""
LOG_FILE=""
TRAIN_LOG_URI=""
child=""
tail_pid=""

finish() {
    if [[ -n "${TRAIN_LOG_URI}" && -n "${LOG_FILE}" && -f "${LOG_FILE}" ]]; then
        python - <<'PY' || echo "WARNING: failed to upload training log to ${TRAIN_LOG_URI}" >&2
import os
from urllib.parse import urlparse

from google.cloud import storage

uri = os.environ["TRAIN_LOG_URI"]
parsed = urlparse(uri)
if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
    raise SystemExit(f"Not a gs:// object path: {uri}")

blob = storage.Client().bucket(parsed.netloc).blob(parsed.path.lstrip("/"))
blob.upload_from_filename(
    os.environ["LOG_FILE"],
    content_type="text/plain; charset=utf-8",
)
PY
    fi

    if [[ -n "${GCP_CREDENTIALS_FILE}" ]]; then
        rm -f "${GCP_CREDENTIALS_FILE}"
    fi
    if [[ -n "${LOG_FILE}" ]]; then
        rm -f "${LOG_FILE}"
    fi
}

# Bash returns from wait when this trap runs, before the child has exited.
# The wait loop below keeps going until the pid is gone.
forward_signal() {
    local signal="$1"

    if [[ -n "${child}" ]]; then
        kill "-${signal}" -- "-${child}" 2>/dev/null || true
        return
    fi

    exit $((128 + signal))
}

trap finish EXIT
trap 'forward_signal 15' TERM
trap 'forward_signal 2' INT

runs_root=""
model_name=""
run_id=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    token="${args[$i]}"
    next="${args[$((i + 1))]:-}"
    case "${token}" in
        --runs-root)
            if [[ -n "${next}" && "${next}" != --* ]]; then
                runs_root="${next}"
            fi
            ;;
        --runs-root=*)
            runs_root="${token#--runs-root=}"
            ;;
        --model)
            if [[ -n "${next}" && "${next}" != --* ]]; then
                model_name="${next}"
            fi
            ;;
        --model=*)
            model_name="${token#--model=}"
            ;;
        --run-id)
            if [[ -n "${next}" && "${next}" != --* ]]; then
                run_id="${next}"
            fi
            ;;
        --run-id=*)
            run_id="${token#--run-id=}"
            ;;
    esac
done

if [[ -n "${runs_root}" && -n "${model_name}" && -n "${run_id}" ]]; then
    TRAIN_LOG_URI="${runs_root%/}/${model_name}/${run_id}/train.log"
fi

if [[ -n "${GCP_TRAINING_SA_JSON:-}" ]]; then
    umask 077

    GCP_CREDENTIALS_FILE="$(mktemp)"

    printf '%s' "${GCP_TRAINING_SA_JSON}" > "${GCP_CREDENTIALS_FILE}"
    chmod 600 "${GCP_CREDENTIALS_FILE}"

    if ! python -m json.tool "${GCP_CREDENTIALS_FILE}" >/dev/null 2>&1; then
        echo "GCP_TRAINING_SA_JSON contains invalid JSON." >&2
        exit 78
    fi

    export GOOGLE_APPLICATION_CREDENTIALS="${GCP_CREDENTIALS_FILE}"
    unset GCP_TRAINING_SA_JSON
fi

LOG_FILE="$(mktemp)"
export LOG_FILE TRAIN_LOG_URI

{
    echo "Python: $(python --version 2>&1)"
    python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
PY
} 2>&1 | tee -a "${LOG_FILE}"

# setsid makes Python the leader of a new process group, so one signal
# reaches DataLoader workers forked into that group.
setsid python "$@" >>"${LOG_FILE}" 2>&1 &
child=$!
tail -n 0 -F "${LOG_FILE}" --pid="${child}" &
tail_pid=$!

set +e
while true; do
    wait "${child}"
    exit_code=$?
    if ! kill -0 "${child}" 2>/dev/null; then
        break
    fi
done
set -e

wait "${tail_pid}" || true
exit "${exit_code}"
