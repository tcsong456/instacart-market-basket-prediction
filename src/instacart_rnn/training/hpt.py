import argparse
import json
import logging
from pathlib import Path

import optuna

from instacart_etl_rnn.common.setup_logging import configure_logging
from instacart_platform.runs import RunPaths
from instacart_rnn.training.runner import TrainingRunConfig, run_training

logger = logging.getLogger(__name__)


def sqlite_storage_url(path: Path) -> str:
    """Build a SQLite URL that is valid on Windows and POSIX.

    Args:
        path: Filesystem path to the SQLite database file.

    Returns:
        A ``sqlite:///`` URL using a resolved POSIX path.
    """

    return f"sqlite:///{path.resolve().as_posix()}"


def objective(
    trial: optuna.Trial,
    *,
    model_name: str,
    train_path: str,
    validation_path: str,
    output_path: Path,
    epochs: int,
    seed: int,
    amp: bool,
    pin_memory: bool,
    early_stopping: int | None,
) -> float:
    """Run one HPT trial and return its best validation loss."""

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-4,
        3e-3,
        log=True,
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-6,
        1e-2,
        log=True,
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [256, 512, 1024],
    )

    grad_clip_norm = trial.suggest_categorical(
        "grad_clip_norm",
        [0.0, 2.0, 5.0],
    )

    lstm_size = trial.suggest_categorical(
        "lstm_size",
        [64, 128, 256],
    )

    trial_path = output_path / f"trial_{trial.number:04d}"
    trial_path.mkdir(parents=True, exist_ok=True)

    def on_validation_end(
        epoch: int,
        validation_loss: float,
    ) -> None:
        trial.report(
            validation_loss,
            step=epoch,
        )

        logger.info(
            "Trial %d epoch %d: validation_loss=%.6f",
            trial.number,
            epoch,
            validation_loss,
        )

        if trial.should_prune():
            logger.info(
                "Pruning trial %d at epoch %d with validation_loss=%.6f",
                trial.number,
                epoch,
                validation_loss,
            )

            raise optuna.TrialPruned(f"Trial {trial.number} pruned at epoch {epoch}")

    config = TrainingRunConfig(
        model_name=model_name,
        train_path=train_path,
        validation_path=validation_path,
        epochs=epochs,
        batch_size=batch_size,
        read_batch_size=4096,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip_norm=None if grad_clip_norm == 0 else grad_clip_norm,
        lstm_size=lstm_size,
        num_workers=0,
        seed=seed,
        amp=amp,
        pin_memory=pin_memory,
        warm_start=False,
        early_stopping=early_stopping,
        on_validation_end=on_validation_end,
    )
    paths = RunPaths(root=str(trial_path))

    logger.info(
        "Starting Optuna trial %d with parameters: %s",
        trial.number,
        trial.params,
    )

    result = run_training(config, paths=paths)

    validation_loss = result.best_validation_loss

    logger.info(
        "Trial %d finished: validation_loss=%.6f",
        trial.number,
        validation_loss,
    )

    return validation_loss


def run_hpt(
    *,
    model_name: str,
    train_path: str,
    validation_path: str,
    output_path: str,
    n_trials: int,
    epochs: int,
    seed: int,
    amp: bool,
    pin_memory: bool,
    early_stopping: int | None,
) -> optuna.Study:
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    storage_path = output_dir / "optuna.db"

    study = optuna.create_study(
        study_name="product_rnn_hpt",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,
        ),
        storage=sqlite_storage_url(storage_path),
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(
            trial,
            model_name=model_name,
            train_path=train_path,
            validation_path=validation_path,
            output_path=output_dir,
            epochs=epochs,
            seed=seed,
            amp=amp,
            pin_memory=pin_memory,
            early_stopping=early_stopping,
        ),
        n_trials=n_trials,
        n_jobs=1,
    )

    best_result = {
        "trial_number": study.best_trial.number,
        "validation_loss": study.best_value,
        "parameters": study.best_params,
    }

    with (output_dir / "best_params.json").open("w") as file:
        json.dump(best_result, file, indent=2)

    return study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-path", required=True)
    parser.add_argument("--validation-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model", required=True)

    parser.add_argument(
        "--trials",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument("--early-stopping", type=int, default=2)
    parser.add_argument(
        "--amp",
        action="store_true",
    )
    parser.add_argument("--pin-memory", action="store_true")

    return parser.parse_args()


def main() -> None:
    configure_logging()

    args = parse_args()

    study = run_hpt(
        model_name=args.model,
        train_path=args.train_path,
        validation_path=args.validation_path,
        output_path=args.output_path,
        n_trials=args.trials,
        epochs=args.epochs,
        seed=args.seed,
        amp=args.amp,
        pin_memory=args.pin_memory,
        early_stopping=args.early_stopping,
    )

    logger.info(
        "Best trial: %d",
        study.best_trial.number,
    )

    logger.info(
        "Best validation loss: %.6f",
        study.best_value,
    )

    logger.info(
        "Best parameters: %s",
        study.best_params,
    )


if __name__ == "__main__":
    main()
