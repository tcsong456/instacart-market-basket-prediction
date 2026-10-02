import argparse
import logging
from pathlib import Path

import optuna

from instacart_etl_rnn.common.paths import join_path
from instacart_etl_rnn.common.setup_logging import configure_logging
from instacart_platform.runs import write_json
from instacart_rnn.stacking.feature_selection import (
    COMPACT_COLUMNS,
    STATE_SOURCES,
    binary_log_loss,
    columns_with_prefix,
    load_selected_artifacts,
    out_of_fold_probabilities,
    stacking_label_path,
)
from instacart_rnn.stacking.feature_selection import (
    _column_matrices as column_matrices,
)
from instacart_rnn.stacking.feature_selection import (
    _load_frame as load_frame,
)
from instacart_rnn.training.hpt import sqlite_storage_url

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stack-dir",
        required=True,
    )
    parser.add_argument(
        "--label-path",
    )
    parser.add_argument(
        "--output-path",
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def full_feature_columns(frame) -> list[str]:
    missing = [column for column in COMPACT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Compact features are missing {missing}")

    states: list[str] = []
    for _, prefix, _ in STATE_SOURCES:
        source_columns = columns_with_prefix(frame, prefix)
        if not source_columns:
            raise ValueError(f"No columns found for prefix {prefix}")
        states.extend(source_columns)
    return [*COMPACT_COLUMNS, *states]


def suggest_lightgbm_params(trial: optuna.Trial) -> tuple[dict, int]:
    """Sample one LightGBM configuration.

    Learning rate is paired with 500-1000 rounds so the total step stays
    near the feature-selection fit of 0.05 for 100 rounds.

    Args:
        trial: Optuna trial that records the sampled values.

    Returns:
        Booster parameters and the number of boosting rounds.
    """

    params = {
        "objective": "binary",
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.02, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": 1,
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "verbose": -1,
    }
    num_boost_round = trial.suggest_int("num_boost_round", 500, 1000)
    return params, num_boost_round


def fit_predict_lightgbm(
    train_x,
    train_y,
    test_x,
    *,
    params: dict,
    num_boost_round: int,
    seed: int,
):
    """Fit one LightGBM configuration and return test probabilities."""

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "lightgbm is required for stacking GBM search. Install the stacking extra."
        ) from exc

    dataset = lgb.Dataset(train_x, label=train_y)
    booster = lgb.train(
        {**params, "seed": seed},
        dataset,
        num_boost_round=num_boost_round,
    )
    return booster.predict(test_x)


def run_hpt(
    *,
    frame,
    columns: list[str],
    output_dir: Path,
    n_trials: int,
    seed: int,
) -> optuna.Study:
    """Minimize out-of-fold log loss over LightGBM configurations.

    Args:
        frame: Joined stacking rows with a ``fold`` column.
        columns: Feature columns shared by every trial.
        output_dir: Local directory for the Optuna database.
        n_trials: Number of configurations to try.
        seed: Sampler seed.

    Returns:
        The completed Optuna study.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    matrices = column_matrices(frame, columns)
    study = optuna.create_study(
        study_name="stacking_gbm_hpt",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=sqlite_storage_url(output_dir / "optuna.db"),
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params, num_boost_round = suggest_lightgbm_params(trial)
        probabilities, _ = out_of_fold_probabilities(
            frame,
            matrices,
            lambda train_x, train_y, test_x: fit_predict_lightgbm(
                train_x,
                train_y,
                test_x,
                params=params,
                num_boost_round=num_boost_round,
                seed=seed,
            ),
        )
        loss = binary_log_loss(frame["label"].to_numpy(dtype=float), probabilities)
        logger.info(
            "Trial %d out-of-fold log loss %.6f parameters %s",
            trial.number,
            loss,
            trial.params,
        )
        return loss

    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return study


def main() -> None:
    """Search LightGBM settings and write ``gbm_hpt.json``."""

    configure_logging()
    args = parse_args()
    stack_dir = args.stack_dir.rstrip("/")
    manifest = load_selected_artifacts(
        str(join_path(stack_dir, "selected_artifacts.json"))
    )
    label_path = args.label_path or stacking_label_path(
        manifest["models"]["product"]["train_path"]
    )
    frame = load_frame(manifest, stack_dir, label_path)
    columns = full_feature_columns(frame)
    output_dir = (
        Path(args.output_path) if args.output_path else _default_study_dir(stack_dir)
    )
    study = run_hpt(
        frame=frame,
        columns=columns,
        output_dir=output_dir,
        n_trials=args.trials,
        seed=args.seed,
    )
    result = {
        "trial_number": study.best_trial.number,
        "oof_log_loss": study.best_value,
        "parameters": study.best_params,
        "n_features": len(columns),
        "n_rows": int(len(frame)),
        "label_path": label_path,
    }
    write_json(str(output_dir / "best_params.json"), result)
    write_json(str(join_path(stack_dir, "gbm_hpt.json")), result)
    logger.info(
        "Best trial %d out-of-fold log loss %.6f parameters %s",
        study.best_trial.number,
        study.best_value,
        study.best_params,
    )


def _default_study_dir(stack_dir: str) -> Path:
    stack_id = stack_dir.rstrip("/").split("/")[-1]
    return Path("stacking_gbm_hpt") / stack_id


if __name__ == "__main__":
    main()
