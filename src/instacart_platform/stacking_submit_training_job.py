import argparse
import logging

from instacart_etl_rnn.common.setup_logging import configure_logging
from instacart_platform.models import JobStatus, TrainingJob
from instacart_platform.rnn_submit_training_job import generate_run_id
from instacart_platform.runpod_backend import RunpodTrainingBackend

logger = logging.getLogger(__name__)

STACKING_MODEL_NAME = "stacking_gbm"
STACKING_CPU_FLAVORS = ("cpu3m", "cpu5m", "cpu3g", "cpu5g")
STACKING_VCPU_COUNT = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--template-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--mode")
    parser.add_argument("--timeline")
    parser.add_argument("--stack-id")
    parser.add_argument("--n-folds", type=int, default=5)
    return parser.parse_args()


def training_selection_args(args: argparse.Namespace) -> tuple[str, ...]:
    """Return the stack selection arguments for the training container.

    Args:
        args: Submitter arguments. ``--stack-id`` reuses one directory.
            Otherwise ``--mode`` and ``--timeline`` select the latest runs.

    Returns:
        Arguments appended to the stacking training command.
    """

    if args.stack_id and (args.mode or args.timeline):
        raise ValueError("Pass either --stack-id or --mode and --timeline")
    if args.stack_id:
        return ("--stack-dir", stack_directory(args.runs_root, args.stack_id))
    if not args.mode or not args.timeline:
        raise ValueError("mode and timeline are required when --stack-id is omitted")
    return (
        "--mode",
        args.mode,
        "--timeline",
        args.timeline,
        "--n-folds",
        str(args.n_folds),
    )


def stack_directory(runs_root: str, stack_id: str) -> str:
    if not stack_id or "/" in stack_id or stack_id.startswith("gs:"):
        raise ValueError(
            "stack_id must be the directory name under runs/stack, "
            f"received {stack_id!r}"
        )
    return f"{runs_root.rstrip('/')}/stack/{stack_id}"


def main() -> None:
    configure_logging()

    run_id = generate_run_id()
    args = parse_args()
    selection_args = training_selection_args(args)
    job = TrainingJob(
        run_id=run_id,
        image=args.image,
        gpu_type="",
        runs_root=args.runs_root,
        model_name=STACKING_MODEL_NAME,
        gpu_count=0,
        compute_type="CPU",
        cpu_flavor_ids=STACKING_CPU_FLAVORS,
        vcpu_count=STACKING_VCPU_COUNT,
        command=(
            "-m",
            "instacart_rnn.stacking.stacking_train",
            *selection_args,
            "--run-id",
            run_id,
            "--runs-root",
            args.runs_root,
            "--model",
            STACKING_MODEL_NAME,
            "--git-commit",
            args.git_commit,
            "--image",
            args.image,
        ),
    )

    backend = RunpodTrainingBackend(template_id=args.template_id)
    handle = backend.submit(job)

    print(f"Submitted run: {run_id}")
    print(f"RunPod Pod ID: {handle.job_id}")
    print(f"Stack selection: {' '.join(selection_args)}")
    print(f"Git commit: {args.git_commit}")
    print(f"Image: {args.image}")

    training_error = None

    try:
        result = backend.wait(handle)

        if result.status != JobStatus.SUCCEEDED:
            training_error = RuntimeError(
                f"Training run {result.run_id} failed with status {result.status.value}"
            )
        else:
            print(f"Training completed successfully: {result.run_id}")
    except Exception as exc:
        training_error = exc
    finally:
        print(f"Terminating RunPod Pod {handle.job_id}...")

        try:
            backend.terminate(handle)
        except Exception:
            logger.exception(
                "Failed to terminate RunPod job %s",
                handle.job_id,
            )

            if training_error is None:
                raise

    if training_error is not None:
        raise training_error

    print(f"RunPod Pod {handle.job_id} terminated.")


if __name__ == "__main__":
    main()
