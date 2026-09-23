import argparse
from datetime import datetime, timezone
from uuid import uuid4

from instacart_platform.models import TrainingJob
from instacart_platform.runpod_backend import RunpodTrainingBackend


def generate_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:6]

    return f"{timestamp}_{suffix}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--template-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--gold-root", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--gpu-type", default="NVIDIA GeForce RTX 4090", type=str)
    parser.add_argument("--mode", default="curated", type=str)

    return parser.parse_args()


def main() -> None:
    run_id = generate_run_id()
    args = parse_args()

    base_path = f"{args.gold_root.rstrip('/')}/training/{args.mode}/{args.timeline}"
    job = TrainingJob(
        run_id=run_id,
        image=args.image,
        gpu_type=args.gpu_type,
        gpu_count=1,
        command=(
            "-m",
            "instacart_rnn.run",
            "--run-id",
            run_id,
            "--runs-root",
            args.runs_root,
            "--data-root",
            base_path,
            "--model",
            args.model_name,
            "--git-commit",
            args.git_commit,
            "--image",
            args.image,
            "--amp",
            "--pin-memory",
        ),
    )

    backend = RunpodTrainingBackend(
        template_id=args.template_id,
    )

    handle = backend.submit(job)

    print(f"Submitted run: {run_id}")
    print(f"RunPod Pod ID: {handle.job_id}")
    print(f"Git commit: {args.git_commit}")
    print(f"Image: {args.image}")


if __name__ == "__main__":
    main()
