from dataclasses import asdict
from datetime import datetime, timezone

from instacart_platform.runs import (
    build_run_paths,
    write_json,
    write_success_marker,
)
from instacart_rnn.training.runner import (
    InferenceRunConfig,
    TrainingRunConfig,
    run_inference,
    run_training,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def execute_run(
    *,
    run_id: str,
    runs_root: str,
    training_config: TrainingRunConfig,
    inference_config: InferenceRunConfig,
    git_commit: str | None = None,
    image: str | None = None,
) -> str:
    paths = build_run_paths(
        runs_root=runs_root,
        model_name=training_config.model_name,
        run_id=run_id,
    )

    started_at = utc_now()

    json_attributes = {
        "run_id": run_id,
        "model_name": training_config.model_name,
        "started_at": started_at,
        "git_commit": git_commit,
        "image": image,
    }

    write_json(
        paths.run_json,
        {"status": "running", "completed_at": None, **json_attributes},
    )

    config_payload = asdict(training_config)
    config_payload.pop("on_validation_end", None)

    write_json(
        paths.config_json,
        config_payload,
    )

    try:
        result = run_training(
            training_config,
            paths=paths,
        )

        resolved_inference_config = InferenceRunConfig(
            **{
                **asdict(inference_config),
                "checkpoint_path": paths.best_checkpoint,
                "output_path": paths.artifacts,
            }
        )

        run_inference(resolved_inference_config)

        completed_at = utc_now()

        write_json(
            paths.metrics_json,
            {
                "best_validation_loss": result.best_validation_loss,
                "best_epoch": result.best_epoch,
            },
        )

        write_json(
            paths.run_json,
            {"status": "completed", "completed_at": completed_at, **json_attributes},
        )

        write_success_marker(paths.success_marker)

        return run_id
    except Exception:
        write_json(
            paths.run_json,
            {"status": "failed", "completed_at": utc_now(), **json_attributes},
        )

        raise
