#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -eq 0 ]]; then
    cat >&2 <<'MSG'
No job entrypoint was provided.
Pass a Python script or "-m <module>" as container arguments, for example:
  -m instacart_rnn.train --model product ...
  -m instacart_rnn.inference --model product ...
MSG
    exit 64
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
