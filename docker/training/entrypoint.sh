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

cleanup() {
    if [[ -n "${GCP_CREDENTIALS_FILE}" ]]; then
        rm -f "${GCP_CREDENTIALS_FILE}"
    fi
}

trap cleanup EXIT

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

exec python "$@"
