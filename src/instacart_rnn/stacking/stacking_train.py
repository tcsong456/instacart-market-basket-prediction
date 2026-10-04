import argparse
import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from instacart_etl_rnn.common.paths import is_gcs_url, join_path
from instacart_etl_rnn.common.setup_logging import configure_logging
from instacart_platform.runs import write_json
from instacart_rnn.stacking.feature_selection import (
    COMPACT_COLUMNS,
    LABEL_COLUMNS,
    REQUIRED_MODELS,
    STATE_SOURCES,
    binary_log_loss,
    columns_with_prefix,
    load_selected_artifacts,
    read_stacking_outputs,
    stacking_label_path,
)
from instacart_rnn.stacking.feature_selection import (
    _load_frame as load_training_frame,
)
from instacart_rnn.stacking.feature_selection import (
    _read_frame as read_frame,
)
from instacart_rnn.stacking.feature_selection import (
    _require_unique as require_unique,
)

logger = logging.getLogger(__name__)

NONE_TOKEN = "None"
GBM_PARAMETERS = {
    "objective": "binary",
    "learning_rate": 0.0114,
    "num_leaves": 17,
    "min_child_samples": 129,
    "feature_fraction": 0.585,
    "bagging_fraction": 0.626,
    "bagging_freq": 1,
    "lambda_l2": 6.25,
    "verbose": -1,
}
NUM_BOOST_ROUND = 983


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-dir", required=True)
    parser.add_argument("--label-path")
    parser.add_argument("--evaluation-label-path")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluation_label_path(base_train_path: str) -> str:
    """Return the evaluation gold path for a base-model train path.

    Args:
        base_train_path: ``.../{timeline}/{model}_training_data_train``.

    Returns:
        ``.../{timeline}/{model}_training_data_evaluation``.
    """

    parent, filename = base_train_path.rstrip("/").rsplit("/", 1)
    suffix = "_training_data_train"
    if not filename.endswith(suffix):
        raise ValueError(
            f"train_path must end with {suffix}, received {base_train_path}"
        )
    stem = filename[: -len("_train")]
    return f"{parent}/{stem}_evaluation"


def evaluation_artifact_path(artifact_path: str) -> str:
    """Return the evaluation export beside a stacking-train export.

    Args:
        artifact_path: Directory ending in ``stacking_train``.

    Returns:
        The sibling ``evaluation`` directory.
    """

    if is_gcs_url(artifact_path):
        normalized = artifact_path.rstrip("/")
        suffix = "/stacking_train"
        if not normalized.endswith(suffix):
            raise ValueError(
                f"artifact_path must end with /stacking_train, received {artifact_path}"
            )
        return normalized[: -len(suffix)] + "/evaluation"

    path = Path(artifact_path)
    if path.name != "stacking_train":
        raise ValueError(
            f"artifact_path must end with stacking_train, received {artifact_path}"
        )
    return str(path.parent / "evaluation")


def searched_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return the compact columns and raw hidden states used by HPT.

    Args:
        frame: Joined stacking rows.

    Returns:
        Feature names in model order. Candidate NLLs are excluded.
    """

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


def join_labeled_features(
    *,
    product: pd.DataFrame,
    aisle: pd.DataFrame,
    reorder: pd.DataFrame,
    gmm: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Join product rows to labels and the other model outputs.

    Args:
        product: One row per user and product, including ``aisle_id``.
        aisle: One row per user and aisle.
        reorder: Reorder-size RNN output, one row per user.
        gmm: Reorder-size GMM output, one row per user.
        labels: Gold ``user_id``, ``product_id``, and ``label``.

    Returns:
        One row per labeled product row.
    """

    require_unique(product, ["user_id", "product_id"], "product artifact")
    require_unique(aisle, ["user_id", "aisle_id"], "aisle artifact")
    require_unique(reorder, ["user_id"], "reorder-size RNN artifact")
    require_unique(gmm, ["user_id"], "reorder-size GMM artifact")
    require_unique(labels, ["user_id", "product_id"], "labels")

    frame = product.merge(labels, on=["user_id", "product_id"], how="inner")
    frame = frame.merge(aisle, on=["user_id", "aisle_id"], how="left")
    frame = frame.merge(reorder, on="user_id", how="left")
    frame = frame.merge(gmm, on="user_id", how="left")
    if frame.empty:
        raise ValueError("No product rows matched the labels")
    return frame.reset_index(drop=True)


def train_booster(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    parameters: Mapping[str, object],
    num_boost_round: int,
    seed: int,
):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "lightgbm is required to train the stacking GBM. "
            "Install the stacking extra."
        ) from exc

    dataset = lgb.Dataset(train_x, label=train_y)
    return lgb.train(
        {**parameters, "seed": seed},
        dataset,
        num_boost_round=num_boost_round,
    )


def prefix_expectations(probabilities: np.ndarray) -> np.ndarray:
    """Return the expected set-F1 of every probability prefix.

    Products are treated as independent Bernoulli draws. Column ``k`` is
    the prefix of the ``k`` largest probabilities. Row 0 also includes the
    ``None`` token, and row 1 is that product prefix alone. This is the
    O(n) expectation used by the Instacart F1 prefix search.

    Args:
        probabilities: One reorder probability per candidate product.

    Returns:
        Array of shape ``(2, n + 1)``.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("probabilities must be a non-empty vector")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")

    ranked = np.sort(probabilities)[::-1]
    width = ranked.shape[0]
    none_probability = float(np.prod(1.0 - ranked))
    count = np.zeros((width + 2, width + 1))
    count[0, 0] = 1.0
    for column in range(1, width):
        count[0, column] = (1.0 - ranked[column - 1]) * count[0, column - 1]

    for positives in range(1, width + 1):
        count[positives, positives] = (
            count[positives - 1, positives - 1] * ranked[positives - 1]
        )
        for column in range(positives + 1, width + 1):
            count[positives, column] = (
                ranked[column - 1] * count[positives - 1, column - 1]
                + (1.0 - ranked[column - 1]) * count[positives, column - 1]
            )

    reciprocal = np.zeros((2 * width + 1,))
    none_reciprocal = np.zeros((2 * width + 1,))
    for index in range(1, 2 * width + 1):
        reciprocal[index] = 1.0 / index
        none_reciprocal[index] = 1.0 / (index + 1.0)

    expectations = []
    for prefix in range(width + 1)[::-1]:
        product_f1 = 0.0
        none_f1 = 0.0
        for positives in range(width + 1):
            weight = 2.0 * positives * count[positives, prefix]
            product_f1 += weight * reciprocal[prefix + positives]
            none_f1 += weight * none_reciprocal[prefix + positives]
        for index in range(1, 2 * prefix - 1):
            keep = 1.0 - ranked[prefix - 1]
            take = ranked[prefix - 1]
            reciprocal[index] = keep * reciprocal[index] + take * reciprocal[index + 1]
            none_reciprocal[index] = (
                keep * none_reciprocal[index] + take * none_reciprocal[index + 1]
            )
        expectations.append(
            [none_f1 + 2.0 * none_probability / (2.0 + prefix), product_f1]
        )

    return np.asarray(expectations[::-1], dtype=np.float64).T


def choose_prefix(probabilities: np.ndarray) -> tuple[int, bool]:
    expectations = prefix_expectations(probabilities)
    row, prefix = np.unravel_index(int(np.argmax(expectations)), expectations.shape)
    return int(prefix), bool(row == 0)


def predicted_basket(
    product_ids: np.ndarray,
    probabilities: np.ndarray,
) -> set[object]:
    """Return the product prefix, and ``None`` when that cell wins.

    Args:
        product_ids: Candidate products in the same order as ``probabilities``.
        probabilities: Reorder probability of each candidate.

    Returns:
        Predicted basket. ``None`` is the empty-order token.
    """

    if len(product_ids) != len(probabilities):
        raise ValueError("product_ids and probabilities must have the same length")
    order = np.argsort(-np.asarray(probabilities, dtype=np.float64), kind="mergesort")
    ranked_ids = np.asarray(product_ids)[order]
    prefix, include_none = choose_prefix(np.asarray(probabilities)[order])
    basket: set[object] = {int(product_id) for product_id in ranked_ids[:prefix]}
    if include_none:
        basket.add(NONE_TOKEN)
    return basket


def true_basket(product_ids: np.ndarray, labels: np.ndarray) -> set[object]:
    if len(product_ids) != len(labels):
        raise ValueError("product_ids and labels must have the same length")
    ordered = {
        int(product_id)
        for product_id, label in zip(product_ids, labels, strict=True)
        if float(label) > 0.5
    }
    return ordered or {NONE_TOKEN}


def basket_f1(predicted: set[object], actual: set[object]) -> float:
    if not predicted and not actual:
        return 1.0
    overlap = len(predicted & actual)
    return 2.0 * overlap / (len(predicted) + len(actual))


def mean_order_f1(frame: pd.DataFrame, probabilities: np.ndarray) -> float:
    """Return the mean basket F1 over users.

    Args:
        frame: Labeled product rows containing ``user_id``, ``product_id``,
            and ``label``.
        probabilities: One reorder probability per row of ``frame``.

    Returns:
        Mean set-F1, rounded to 5 decimal places.
    """

    if len(probabilities) != len(frame):
        raise ValueError("probabilities must have one value per frame row")
    scored = frame.loc[:, ["user_id", "product_id", "label"]].copy()
    scored["probability"] = np.asarray(probabilities, dtype=np.float64)
    scores = [
        basket_f1(
            predicted_basket(
                group["product_id"].to_numpy(),
                group["probability"].to_numpy(),
            ),
            true_basket(
                group["product_id"].to_numpy(),
                group["label"].to_numpy(),
            ),
        )
        for _, group in scored.groupby("user_id", sort=False)
    ]
    if not scores:
        raise ValueError("No users to score")
    return float(np.round(float(np.mean(scores)), 5))


def main() -> None:
    configure_logging()
    args = parse_args()
    stack_dir = args.stack_dir.rstrip("/")
    manifest = load_selected_artifacts(
        str(join_path(stack_dir, "selected_artifacts.json"))
    )
    product_train_path = manifest["models"]["product"]["train_path"]
    train_label_path = args.label_path or stacking_label_path(product_train_path)
    evaluation_labels = args.evaluation_label_path or evaluation_label_path(
        product_train_path
    )

    train_frame = load_training_frame(manifest, stack_dir, train_label_path)
    columns = searched_feature_columns(train_frame)
    booster = train_booster(
        train_frame.loc[:, columns].to_numpy(dtype=np.float64),
        train_frame["label"].to_numpy(dtype=np.float64),
        parameters=GBM_PARAMETERS,
        num_boost_round=NUM_BOOST_ROUND,
        seed=args.seed,
    )

    evaluation_frame = _load_evaluation_frame(manifest, evaluation_labels)
    _require_disjoint_users(train_frame, evaluation_frame)
    missing = [column for column in columns if column not in evaluation_frame.columns]
    if missing:
        raise ValueError(f"Evaluation features are missing {missing}")
    probabilities = np.asarray(
        booster.predict(evaluation_frame.loc[:, columns].to_numpy(dtype=np.float64)),
        dtype=np.float64,
    )
    log_loss = binary_log_loss(
        evaluation_frame["label"].to_numpy(dtype=np.float64),
        probabilities,
    )
    order_f1 = mean_order_f1(evaluation_frame, probabilities)
    result = {
        "train_label_path": train_label_path,
        "evaluation_label_path": evaluation_labels,
        "parameters": GBM_PARAMETERS,
        "num_boost_round": NUM_BOOST_ROUND,
        "seed": args.seed,
        "n_features": len(columns),
        "n_train_rows": int(len(train_frame)),
        "n_train_users": int(train_frame["user_id"].nunique()),
        "n_evaluation_rows": int(len(evaluation_frame)),
        "n_evaluation_users": int(evaluation_frame["user_id"].nunique()),
        "evaluation_log_loss": log_loss,
        "evaluation_f1": order_f1,
    }
    write_json(str(join_path(stack_dir, "stacking_train.json")), result)
    logger.info(
        "Evaluation users %d log loss %.5f basket F1 %.5f",
        result["n_evaluation_users"],
        log_loss,
        order_f1,
    )


def _load_evaluation_frame(manifest: dict, label_path: str) -> pd.DataFrame:
    artifact_paths = {
        name: evaluation_artifact_path(manifest["models"][name]["artifact_path"])
        for name in REQUIRED_MODELS
    }
    product, aisle, reorder, gmm = read_stacking_outputs(artifact_paths)
    labels = read_frame(label_path, list(LABEL_COLUMNS))
    return join_labeled_features(
        product=product,
        aisle=aisle,
        reorder=reorder,
        gmm=gmm,
        labels=labels,
    )


def _require_disjoint_users(train_frame: pd.DataFrame, evaluation_frame: pd.DataFrame):
    shared = np.intersect1d(
        train_frame["user_id"].to_numpy(),
        evaluation_frame["user_id"].to_numpy(),
    )
    if len(shared):
        raise ValueError(
            f"Evaluation overlap contains {len(shared)} stacking-train users"
        )


if __name__ == "__main__":
    main()
