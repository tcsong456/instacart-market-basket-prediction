"""Choose a stacking GBM feature version with user-level out-of-fold scores.

Version 1 uses compact model outputs: product and aisle logits, the
reorder-size RNN prediction, and the reorder-size GMM summaries plus
candidate NLLs. Version 2 adds a PCA of each model's hidden state, fit on
the training fold only and capped at 15 components or 90% of variance.
Version 3 adds the raw hidden states. The smallest version is kept unless
it is worse than the best out-of-fold log loss by more than
``LOSS_TOLERANCE``.
"""

import argparse
import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import gcsfs
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads

from instacart_etl_rnn.common.paths import is_gcs_url, join_path
from instacart_etl_rnn.common.setup_logging import configure_logging
from instacart_platform.runs import write_json

logger = logging.getLogger(__name__)

LABEL_COLUMNS = ("user_id", "product_id", "label")
REQUIRED_MODELS = (
    "product",
    "aisle",
    "reorder_size_rnn",
    "reorder_size_gmm",
)
COMPACT_COLUMNS = (
    "product_logit",
    "aisle_logit",
    "reorder_prediction",
    "gmm_nll_0",
    "gmm_mode_size",
    "gmm_candidate_expected_size",
    "gmm_candidate_entropy",
    "gmm_expected_size",
)
STATE_SOURCES = (
    ("product", "product_state_", ("user_id", "product_id")),
    ("aisle", "aisle_state_", ("user_id", "aisle_id")),
    ("reorder_rnn", "reorder_state_", ("user_id",)),
    ("reorder_gmm", "gmm_state_", ("user_id",)),
)
PCA_MAX_COMPONENTS = 15
PCA_VARIANCE = 0.95
LOSS_TOLERANCE = 0.005
PROBABILITY_CLIP = 1e-6

FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
FoldMatrices = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def parse_args() -> argparse.Namespace:
    """Parse feature-selection arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stack-dir",
        required=True,
        help="Directory containing selected_artifacts.json and user_folds.parquet.",
    )
    parser.add_argument(
        "--label-path",
        help=(
            "Product stacking-train gold parquet. Defaults to the timeline "
            "stacking_train file derived from the selected product run."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stacking_label_path(base_train_path: str) -> str:
    """Return the stacking-train gold path for a base-model train path.

    Args:
        base_train_path: ``.../{timeline}/{model}_training_data_train``.

    Returns:
        ``.../{timeline}/stacking_train/{model}_training_data_train``.
    """

    parent, filename = base_train_path.rstrip("/").rsplit("/", 1)
    if not filename.endswith("_training_data_train"):
        raise ValueError(
            f"train_path must end with _training_data_train, received {base_train_path}"
        )
    return f"{parent}/stacking_train/{filename}"


def load_selected_artifacts(path: str) -> dict:
    """Read the artifact manifest written by the fold script.

    Args:
        path: ``selected_artifacts.json``.

    Returns:
        The manifest object.

    Raises:
        ValueError: A required model is missing from ``models``.
    """

    payload = _read_json(path)
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"Artifact manifest {path} has no models")
    missing = [name for name in REQUIRED_MODELS if name not in models]
    if missing:
        raise ValueError(f"Artifact manifest {path} is missing {', '.join(missing)}")
    return payload


def build_feature_frame(
    *,
    product: pd.DataFrame,
    aisle: pd.DataFrame,
    reorder: pd.DataFrame,
    gmm: pd.DataFrame,
    labels: pd.DataFrame,
    folds: pd.DataFrame,
) -> pd.DataFrame:
    """Join product rows to labels, folds, and the other model outputs.

    Args:
        product: One row per user and product, including ``aisle_id``.
        aisle: One row per user and aisle.
        reorder: Reorder-size RNN output, one row per user.
        gmm: Reorder-size GMM output, one row per user.
        labels: Gold ``user_id``, ``product_id``, and ``label``.
        folds: ``user_id`` and ``fold``.

    Returns:
        One row per labeled product row, with a fold on every row.
    """

    _require_unique(product, ["user_id", "product_id"], "product artifact")
    _require_unique(aisle, ["user_id", "aisle_id"], "aisle artifact")
    _require_unique(reorder, ["user_id"], "reorder-size RNN artifact")
    _require_unique(gmm, ["user_id"], "reorder-size GMM artifact")
    _require_unique(labels, ["user_id", "product_id"], "stacking labels")
    _require_unique(folds, ["user_id"], "user folds")

    frame = product.merge(labels, on=["user_id", "product_id"], how="inner")
    frame = frame.merge(folds, on="user_id", how="left")
    if frame["fold"].isna().any():
        raise ValueError("Product rows are missing a user fold")

    frame = frame.merge(aisle, on=["user_id", "aisle_id"], how="left")
    frame = frame.merge(reorder, on="user_id", how="left")
    frame = frame.merge(gmm, on="user_id", how="left")
    if frame.empty:
        raise ValueError("No product rows matched the stacking labels")
    frame["fold"] = frame["fold"].astype(int)
    return frame.reset_index(drop=True)


def compact_columns(frame: pd.DataFrame) -> list[str]:
    """Return version-1 columns present on ``frame``, in model order.

    Args:
        frame: Joined stacking rows.

    Returns:
        Logit, prediction, GMM summary, and candidate-NLL column names.
    """

    missing = [column for column in COMPACT_COLUMNS if column not in frame.columns]
    candidate_nlls = columns_with_prefix(frame, "gmm_candidate_nll_")
    if missing or not candidate_nlls:
        raise ValueError(
            f"Compact features are missing {missing or ['gmm_candidate_nll_']}"
        )
    return [*COMPACT_COLUMNS, *candidate_nlls]


def columns_with_prefix(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns = [column for column in frame.columns if column.startswith(prefix)]
    return sorted(columns, key=_suffix_index)


def out_of_fold_probabilities(
    frame: pd.DataFrame,
    matrices_for_fold: FoldMatrices,
    fit_predict: FitPredict,
) -> tuple[np.ndarray, list[int]]:
    """Predict every row from a model trained on the other folds.

    Args:
        frame: Labeled rows containing a ``fold`` column.
        matrices_for_fold: Builds train and test matrices for one held-out fold.
        fit_predict: Trains on ``(train_x, train_y)`` and returns probabilities
            for ``test_x``.

    Returns:
        One probability per row, in ``frame`` order, and the feature width
        used for each held-out fold.
    """

    folds = frame["fold"].to_numpy()
    labels = frame["label"].to_numpy(dtype=np.float64)
    predictions = np.empty(len(frame), dtype=np.float64)
    widths: list[int] = []

    for fold in sorted(set(folds.tolist())):
        test_mask = folds == fold
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            raise ValueError(f"Fold {fold} cannot be held out")
        train_x, test_x = matrices_for_fold(train_mask, test_mask)
        if train_x.shape[1] != test_x.shape[1]:
            raise ValueError("Train and test feature widths differ")
        widths.append(int(train_x.shape[1]))
        predictions[test_mask] = fit_predict(
            train_x,
            labels[train_mask],
            test_x,
        )

    return predictions, widths


def binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Return the mean binary log loss of ``probabilities`` against ``labels``."""

    clipped = np.clip(probabilities, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    loss = -(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    return np.round(float(np.mean(loss)), 5)


def project_state_block(
    values: np.ndarray,
    keys: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Project hidden states with a PCA fit on unique training-fold rows.

    Args:
        values: Hidden-state matrix with one row per product.
        keys: Grain of ``values``. Repeated keys are fit once.
        train_mask: Rows whose keys may enter the PCA fit.
        test_mask: Rows to project for the held-out fold.

    Returns:
        Training projections, held-out projections, and the component count.
    """

    training = _pca_training_matrix(values, keys, train_mask)
    mean, components = _fit_pca(training)
    return (
        _transform_pca(values[train_mask], mean, components),
        _transform_pca(values[test_mask], mean, components),
        int(components.shape[0]),
    )


def select_feature_set(
    frame: pd.DataFrame,
    fit_predict: FitPredict,
    *,
    tolerance: float = LOSS_TOLERANCE,
) -> tuple[str, list[dict[str, object]]]:
    """Score versions 1-3 and keep the smallest competitive version.

    Args:
        frame: Joined stacking rows.
        fit_predict: Fold trainer used for every version.
        tolerance: A larger version is eligible only when no smaller version
            is within this log-loss gap of the best score.

    Returns:
        The winning version name and one result record per version.
    """

    results: list[dict[str, object]] = []
    for name, complexity, matrices in build_versions(frame):
        probabilities, widths = out_of_fold_probabilities(
            frame,
            matrices,
            fit_predict,
        )
        loss = binary_log_loss(
            frame["label"].to_numpy(dtype=np.float64),
            probabilities,
        )
        results.append(
            {
                "feature_set": name,
                "complexity": complexity,
                "oof_log_loss": loss,
                "n_rows": int(len(frame)),
                "n_features": max(widths),
            }
        )
        logger.info("Feature set %s out-of-fold log loss %.6f", name, loss)

    winner = choose_version(results, tolerance)
    return winner, results


def build_versions(
    frame: pd.DataFrame,
) -> list[tuple[str, int, FoldMatrices]]:
    compact = compact_columns(frame)
    state_columns = _state_columns(frame)
    return [
        ("version_1", 1, _column_matrices(frame, compact)),
        ("version_2", 2, _pca_matrices(frame, compact)),
        ("version_3", 3, _column_matrices(frame, [*compact, *state_columns])),
    ]


def choose_version(
    results: Sequence[Mapping[str, object]],
    tolerance: float = LOSS_TOLERANCE,
) -> str:
    """Return the smallest version within ``tolerance`` of the best loss.

    Args:
        results: Records containing ``feature_set``, ``complexity``, and
            ``oof_log_loss``.
        tolerance: Maximum log-loss gap that still counts as a tie.

    Returns:
        The selected ``feature_set`` name.
    """

    if not results:
        raise ValueError("No feature-set results to choose from")
    best_loss = min(float(item["oof_log_loss"]) for item in results)
    eligible = [
        item
        for item in results
        if _loss_within_tolerance(float(item["oof_log_loss"]), best_loss, tolerance)
    ]
    winner = min(
        eligible,
        key=lambda item: (int(item["complexity"]), str(item["feature_set"])),
    )
    return str(winner["feature_set"])


def _loss_within_tolerance(loss: float, best_loss: float, tolerance: float) -> bool:
    gap = loss - best_loss
    return gap <= tolerance or math.isclose(gap, tolerance)


def fit_predict_lightgbm(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Fit one fixed LightGBM configuration and return test probabilities."""

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "lightgbm is required for feature selection. Install the stacking extra."
        ) from exc

    dataset = lgb.Dataset(train_x, label=train_y)
    booster = lgb.train(
        {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 5,
            "verbose": -1,
            "seed": seed,
        },
        dataset,
        num_boost_round=100,
    )
    return np.asarray(booster.predict(test_x), dtype=np.float64)


def read_model_frame(
    path: str,
    keys: Sequence[str],
    scalars: Mapping[str, str],
    lists: Mapping[str, str],
) -> pd.DataFrame:
    """Read keys, renamed scalars, and flattened list columns.

    Args:
        path: Inference parquet directory or file.
        keys: Columns that identify a row.
        scalars: Source column name to output column name.
        lists: Source fixed-size list column to output prefix.

    Returns:
        One row per inference record.
    """

    column_names = [*keys, *scalars.keys(), *lists.keys()]
    table = pads.dataset(path, format="parquet").to_table(columns=column_names)
    frame = table.select([*keys, *scalars.keys()]).to_pandas()
    frame = frame.rename(columns=dict(scalars))
    for source_name, prefix in lists.items():
        matrix = _fixed_list_matrix(table.column(source_name), source_name)
        for index in range(matrix.shape[1]):
            frame[f"{prefix}{index}"] = matrix[:, index]
    return frame


def main() -> None:
    """Score the three feature versions and write ``feature_selection.json``."""

    configure_logging()
    args = parse_args()
    stack_dir = args.stack_dir.rstrip("/")
    manifest = load_selected_artifacts(
        str(join_path(stack_dir, "selected_artifacts.json"))
    )
    product_train_path = manifest["models"]["product"]["train_path"]
    label_path = args.label_path or stacking_label_path(product_train_path)
    frame = _load_frame(manifest, stack_dir, label_path)
    winner, results = select_feature_set(
        frame,
        lambda train_x, train_y, test_x: fit_predict_lightgbm(
            train_x,
            train_y,
            test_x,
            seed=args.seed,
        ),
    )
    output_path = str(join_path(stack_dir, "feature_selection.json"))
    write_json(
        output_path,
        {
            "label_path": label_path,
            "metric": "binary_log_loss",
            "loss_tolerance": LOSS_TOLERANCE,
            "selected_feature_set": winner,
            "results": results,
        },
    )
    logger.info("Selected feature set %s", winner)


def read_stacking_outputs(
    artifact_paths: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read product, aisle, reorder-RNN, and reorder-GMM inference frames.

    Args:
        artifact_paths: Inference directory for each required model.

    Returns:
        Product, aisle, reorder-RNN, and reorder-GMM frames.
    """

    missing = [name for name in REQUIRED_MODELS if name not in artifact_paths]
    if missing:
        raise ValueError(f"Artifact paths are missing {', '.join(missing)}")
    product = read_model_frame(
        artifact_paths["product"],
        ("user_id", "product_id", "aisle_id"),
        {"final_logits": "product_logit"},
        {"final_states": "product_state_"},
    )
    aisle = read_model_frame(
        artifact_paths["aisle"],
        ("user_id", "aisle_id"),
        {"final_logits": "aisle_logit"},
        {"final_states": "aisle_state_"},
    )
    reorder = read_model_frame(
        artifact_paths["reorder_size_rnn"],
        ("user_id",),
        {"final_predictions": "reorder_prediction"},
        {"final_states": "reorder_state_"},
    )
    gmm = read_model_frame(
        artifact_paths["reorder_size_gmm"],
        ("user_id",),
        {
            "nll_0": "gmm_nll_0",
            "mode_size": "gmm_mode_size",
            "candidate_expected_size": "gmm_candidate_expected_size",
            "candidate_entropy": "gmm_candidate_entropy",
            "gmm_expected_size": "gmm_expected_size",
        },
        {
            "candidate_nlls": "gmm_candidate_nll_",
            "final_states": "gmm_state_",
        },
    )
    return product, aisle, reorder, gmm


def _load_frame(manifest: dict, stack_dir: str, label_path: str) -> pd.DataFrame:
    models = manifest["models"]
    product, aisle, reorder, gmm = read_stacking_outputs(
        {name: models[name]["artifact_path"] for name in REQUIRED_MODELS}
    )
    labels = _read_frame(label_path, list(LABEL_COLUMNS))
    folds = _read_frame(
        str(join_path(stack_dir, "user_folds.parquet")),
        ["user_id", "fold"],
    )
    return build_feature_frame(
        product=product,
        aisle=aisle,
        reorder=reorder,
        gmm=gmm,
        labels=labels,
        folds=folds,
    )


def _column_matrices(frame: pd.DataFrame, columns: Sequence[str]) -> FoldMatrices:
    values = frame.loc[:, list(columns)].to_numpy(dtype=np.float64)

    def matrices(
        train_mask: np.ndarray,
        test_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return values[train_mask], values[test_mask]

    return matrices


def _pca_matrices(frame: pd.DataFrame, compact: Sequence[str]) -> FoldMatrices:
    base = frame.loc[:, list(compact)].to_numpy(dtype=np.float64)
    blocks = []
    for name, prefix, keys in STATE_SOURCES:
        columns = columns_with_prefix(frame, prefix)
        if not columns:
            raise ValueError(f"No columns found for prefix {prefix}")
        blocks.append(
            (
                name,
                frame.loc[:, columns].to_numpy(dtype=np.float64),
                frame.loc[:, list(keys)].to_numpy(),
            )
        )

    def matrices(
        train_mask: np.ndarray,
        test_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        train_parts = [base[train_mask]]
        test_parts = [base[test_mask]]
        for _, values, keys in blocks:
            train_projected, test_projected, _ = project_state_block(
                values,
                keys,
                train_mask,
                test_mask,
            )
            train_parts.append(train_projected)
            test_parts.append(test_projected)
        return np.hstack(train_parts), np.hstack(test_parts)

    return matrices


def _state_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for _, prefix, _ in STATE_SOURCES:
        source_columns = columns_with_prefix(frame, prefix)
        if not source_columns:
            raise ValueError(f"No columns found for prefix {prefix}")
        columns.extend(source_columns)
    return columns


def _pca_training_matrix(
    values: np.ndarray,
    keys: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    chosen: list[np.ndarray] = []
    seen: set[tuple] = set()
    for index in np.flatnonzero(train_mask):
        row = values[index]
        if not np.isfinite(row).all():
            continue
        key = tuple(np.asarray(keys[index]).tolist())
        if key in seen:
            continue
        seen.add(key)
        chosen.append(np.asarray(row, dtype=np.float64))
    if len(chosen) < 2:
        raise ValueError("PCA needs at least two finite training rows")
    return np.vstack(chosen)


def _fit_pca(training: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = training.mean(axis=0)
    centered = training - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    explained = (singular**2) / (training.shape[0] - 1)
    total = float(explained.sum())
    if total <= 0 or not np.isfinite(total):
        return mean, np.zeros((1, training.shape[1]), dtype=np.float64)

    limit = min(
        PCA_MAX_COMPONENTS,
        training.shape[0] - 1,
        training.shape[1],
        len(singular),
    )
    ratios = np.cumsum(explained / total)
    component_count = int(np.searchsorted(ratios[:limit], PCA_VARIANCE, side="left"))
    component_count = min(component_count + 1, limit)
    return mean, vt[:component_count]


def _transform_pca(
    values: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    projected = (values - mean) @ components.T
    invalid = ~np.isfinite(values).all(axis=1)
    projected[invalid] = np.nan
    return projected


def _suffix_index(name: str) -> int:
    suffix = name.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        raise ValueError(f"Expected a numeric suffix on {name}")
    return int(suffix)


def _read_frame(path: str, columns: list[str]) -> pd.DataFrame:
    table = pads.dataset(path, format="parquet").to_table(columns=columns)
    return table.to_pandas()


def _fixed_list_matrix(column: pa.ChunkedArray, name: str) -> np.ndarray:
    combined = column.combine_chunks()
    if not pa.types.is_fixed_size_list(combined.type):
        raise TypeError(f"{name} must be a fixed-size list column")
    width = combined.type.list_size
    values = combined.flatten().to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float64).reshape(-1, width)


def _require_unique(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    if frame.duplicated(keys).any():
        raise ValueError(f"{name} has duplicate rows for {keys}")


def _read_json(path: str) -> dict:
    if is_gcs_url(path):
        filesystem = gcsfs.GCSFileSystem()
        filesystem.invalidate_cache(path)
        with filesystem.open(path, "r") as file:
            payload = json.load(file)
    else:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return payload


if __name__ == "__main__":
    main()
