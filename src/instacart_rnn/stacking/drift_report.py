"""Compare one frozen champion on two evaluation timelines.

The report records how the same stack behaves on the snapshot where it was
promoted and on a later snapshot. It does not change the promotion rule and
it does not roll the champion back. Comparison figures are written beside the
JSON when matplotlib is installed.

Run it with::

    python -m instacart_rnn.stacking.drift_report \\
        --stack-dir gs://runs/stack/<stack-id> \\
        --runs-root gs://runs \\
        --baseline-timeline t1 \\
        --current-timeline t2 \\
        --baseline-label-path gs://gold/.../t1/product_training_data_evaluation \\
        --current-label-path gs://gold/.../t2/product_training_data_evaluation \\
        --output gs://runs/monitoring/curated/<stack-id>/t2/drift.json
"""

import argparse
import io
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from instacart_etl_rnn.common.paths import is_gcs_url, join_path
from instacart_etl_rnn.common.setup_logging import configure_logging
from instacart_platform.runs import build_run_paths, write_json
from instacart_rnn.stacking.feature_selection import (
    LABEL_COLUMNS,
    REQUIRED_MODELS,
    _read_frame,
    _read_json,
    load_selected_artifacts,
)
from instacart_rnn.stacking.stacking_train import (
    NONE_TOKEN,
    basket_f1,
    predicted_basket,
    true_basket,
)

logger = logging.getLogger(__name__)

HISTORY_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("3-5", 3, 5),
    ("6-10", 6, 10),
    ("11-15", 11, 15),
    ("16-20", 16, 20),
    ("21-25", 21, 25),
    ("26+", 26, None),
)
BIN_NAMES: tuple[str, ...] = tuple(name for name, _, _ in HISTORY_BINS) + ("other",)
_BASELINE_COLOR = "#4C78A8"
_CURRENT_COLOR = "#F58518"
_DELTA_COLORS = ("#4C78A8", "#54A24B", "#E45756")
_F1_METRICS = (
    ("basket_f1", "Basket F1"),
    ("true_size_f1", "True-size F1"),
    ("ranking_f1", "Ranking F1"),
)
_HISTORY_METRICS = (
    *_F1_METRICS,
    ("predicted_none_rate", "Predicted None rate"),
    ("true_none_rate", "True None rate"),
    ("mean_predicted_basket_size", "Mean predicted size"),
    ("mean_true_basket_size", "Mean true size"),
    ("mean_signed_size_error", "Signed size error"),
    ("mean_absolute_size_error", "Absolute size error"),
)
_PRODUCT_EVALUATION = "product_training_data_evaluation"
_SCORE_KEYS = (
    "basket_f1",
    "true_size_f1",
    "ranking_f1",
    "mean_signed_size_error",
    "mean_absolute_size_error",
    "predicted_none_rate",
    "true_none_rate",
    "mean_predicted_basket_size",
    "mean_true_basket_size",
)


def history_bin(length: int) -> str:
    for name, low, high in HISTORY_BINS:
        if length >= low and (high is None or length <= high):
            return name
    return "other"


def f1_at_true_size(
    product_ids: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, bool, float]:
    """Score the probability ranking cut at the true product count.

    A true basket of None scores 1 only when the predicted basket is None.
    That case is a basket decision, so it is excluded from the ranking score.

    Args:
        product_ids: Candidate product ids for one user.
        probabilities: Reorder probabilities aligned with ``product_ids``.
        labels: Binary labels aligned with ``product_ids``.

    Returns:
        Composite true-size F1, whether the true basket has products, and the
        ranking F1. Ranking F1 is NaN when the true basket is None.
    """

    actual = true_basket(product_ids, labels)
    true_products = actual - {NONE_TOKEN}
    if not true_products:
        predicted = predicted_basket(product_ids, probabilities)
        score = 1.0 if predicted == {NONE_TOKEN} else 0.0
        return score, False, float("nan")
    order = np.argsort(-np.asarray(probabilities, dtype=np.float64), kind="mergesort")
    chosen_ids = np.asarray(product_ids)[order][: len(true_products)]
    chosen = {int(product_id) for product_id in chosen_ids}
    ranking = basket_f1(chosen, true_products)
    return ranking, True, ranking


def user_scores(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    history_lengths: Mapping[int, int],
) -> pd.DataFrame:
    """Score every user in one labeled snapshot.

    Args:
        frame: Rows with ``user_id``, ``product_id``, and ``label``.
        probabilities: One probability per row, in ``frame`` order.
        history_lengths: History length keyed by user id.

    Returns:
        One row per user with basket scores and the history bin.
    """

    if len(frame) != len(probabilities):
        raise ValueError("Probabilities must align with the labeled frame")
    scored = frame.loc[:, ["user_id", "product_id", "label"]].copy()
    scored["probability"] = np.asarray(probabilities, dtype=np.float64)
    rows = [
        _user_row(int(user_id), group, history_lengths)
        for user_id, group in scored.groupby("user_id", sort=True)
    ]
    return pd.DataFrame(rows)


def summarize_users(users: pd.DataFrame) -> dict:
    return {
        "overall": _average(users),
        "bins": {name: _average(_bin_users(users, name)) for name in BIN_NAMES},
    }


def compare_snapshots(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    *,
    champion_stack_id: str,
    baseline_timeline: str,
    current_timeline: str,
) -> dict:
    """Build the champion drift report for two scored snapshots.

    Paired deltas use users who appear in both snapshots and stay in the same
    history bin. A user who changes bins is counted separately and is not
    included in the mean. The same user still has one more history order and
    a different next basket. Deltas are current minus baseline.

    Args:
        baseline: User scores on the promotion snapshot.
        current: User scores for the same champion on the later snapshot.
        champion_stack_id: Stack being compared with itself.
        baseline_timeline: Timeline of the promotion snapshot.
        current_timeline: Timeline of the later snapshot.

    Returns:
        JSON-ready report.
    """

    _require_one_row(baseline, "baseline")
    _require_one_row(current, "current")
    baseline_summary = summarize_users(baseline)
    current_summary = summarize_users(current)
    return {
        "champion_stack_id": champion_stack_id,
        "baseline_timeline": baseline_timeline,
        "current_timeline": current_timeline,
        "baseline": baseline_summary,
        "current": current_summary,
        "deltas": _summary_deltas(baseline_summary, current_summary),
        "paired": _paired(baseline, current),
    }


def one_history_length(frame: pd.DataFrame) -> dict[int, int]:
    """Return one history length per user.

    Args:
        frame: Rows with ``user_id`` and ``history_length``.

    Returns:
        History length keyed by user id.

    Raises:
        ValueError: A user has more than one history length.
    """

    counts = frame.groupby("user_id")["history_length"].nunique()
    if bool((counts > 1).any()):
        raise ValueError("history length is not constant per user")
    lengths = frame.groupby("user_id")["history_length"].first()
    return {int(user_id): int(length) for user_id, length in lengths.items()}


def evaluation_inputs(product_label_path: str) -> dict[str, str]:
    """Return the four evaluation gold paths beside a product evaluation path.

    Args:
        product_label_path: ``.../product_training_data_evaluation``.

    Returns:
        Evaluation gold path for each required base model.
    """

    parent, filename = product_label_path.rstrip("/").rsplit("/", 1)
    if filename != _PRODUCT_EVALUATION:
        raise ValueError(
            "label path must end with "
            f"{_PRODUCT_EVALUATION}, received {product_label_path}"
        )
    reorder = str(join_path(parent, "reorder_size_training_data_evaluation"))
    return {
        "product": product_label_path.rstrip("/"),
        "aisle": str(join_path(parent, "aisle_training_data_evaluation")),
        "reorder_size_rnn": reorder,
        "reorder_size_gmm": reorder,
    }


def score_champion_snapshot(
    stack_dir: str,
    runs_root: str,
    *,
    timeline: str,
    label_path: str,
) -> pd.DataFrame:
    """Score one frozen champion on one evaluation timeline.

    A timeline that matches the stack uses its saved evaluation exports.
    Another timeline re-runs the frozen checkpoints on that timeline's inputs.

    Args:
        stack_dir: Champion stack directory containing ``stacking_train.json``.
        runs_root: Runs root used to find checkpoints and write rescore exports.
        timeline: Evaluation timeline to score.
        label_path: Product evaluation gold for ``timeline``.

    Returns:
        One scored row per evaluation user.
    """

    from instacart_rnn.stacking.feature_selection import read_stacking_outputs
    from instacart_rnn.stacking.stacking_train import (
        champion_evaluation_artifacts,
        join_labeled_features,
        load_booster,
    )

    record = _read_json(str(join_path(stack_dir, "stacking_train.json")))
    columns = _feature_columns(record, stack_dir)
    model_path = record.get("model_path")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError(f"{stack_dir} stacking_train.json has no model_path")
    manifest = load_selected_artifacts(
        str(join_path(stack_dir, "selected_artifacts.json"))
    )
    artifacts = champion_evaluation_artifacts(
        _stack_pointer(stack_dir, runs_root, manifest),
        runs_root=runs_root,
        evaluation_timeline=timeline,
        evaluation_inputs=evaluation_inputs(label_path),
    )
    product, aisle, reorder, gmm = read_stacking_outputs(artifacts)
    labels = _read_frame(label_path, list(LABEL_COLUMNS))
    frame = join_labeled_features(
        product=product,
        aisle=aisle,
        reorder=reorder,
        gmm=gmm,
        labels=labels,
    )
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Champion evaluation features are missing {missing}")
    booster = load_booster(model_path)
    probabilities = np.asarray(
        booster.predict(frame.loc[:, list(columns)].to_numpy(dtype=np.float64)),
        dtype=np.float64,
    )
    return user_scores(frame, probabilities, _history_lengths(label_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare one frozen champion on two evaluation timelines."
    )
    parser.add_argument("--stack-dir", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--baseline-timeline", required=True)
    parser.add_argument("--current-timeline", required=True)
    parser.add_argument("--baseline-label-path", required=True)
    parser.add_argument("--current-label-path", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    """Score both snapshots and write the drift report."""

    configure_logging()
    args = build_parser().parse_args()
    baseline = score_champion_snapshot(
        args.stack_dir,
        args.runs_root,
        timeline=args.baseline_timeline,
        label_path=args.baseline_label_path,
    )
    current = score_champion_snapshot(
        args.stack_dir,
        args.runs_root,
        timeline=args.current_timeline,
        label_path=args.current_label_path,
    )
    report = compare_snapshots(
        baseline,
        current,
        champion_stack_id=_stack_id(args.stack_dir),
        baseline_timeline=args.baseline_timeline,
        current_timeline=args.current_timeline,
    )
    write_drift_report(args.output, report)


def _feature_columns(record: dict, stack_dir: str) -> list[str]:
    columns = record.get("feature_columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError(f"{stack_dir} stacking_train.json has no feature_columns")
    if not all(isinstance(column, str) for column in columns):
        raise ValueError(f"{stack_dir} feature_columns must be strings")
    return columns


def _user_row(
    user_id: int,
    group: pd.DataFrame,
    history_lengths: Mapping[int, int],
) -> dict:
    length = _history_length(user_id, history_lengths)
    product_ids = group["product_id"].to_numpy()
    labels = group["label"].to_numpy(dtype=np.float64)
    probabilities = group["probability"].to_numpy(dtype=np.float64)
    predicted = predicted_basket(product_ids, probabilities)
    actual = true_basket(product_ids, labels)
    true_size_f1, nonempty, ranking_f1 = f1_at_true_size(
        product_ids, probabilities, labels
    )
    predicted_size = _product_count(predicted)
    true_size = _product_count(actual)
    return {
        "user_id": user_id,
        "history_length": length,
        "history_bin": history_bin(length),
        "basket_f1": basket_f1(predicted, actual),
        "true_size_f1": true_size_f1,
        "ranking_f1": ranking_f1,
        "nonempty": nonempty,
        "signed_size_error": float(predicted_size - true_size),
        "absolute_size_error": float(abs(predicted_size - true_size)),
        "predicted_none": bool(predicted == {NONE_TOKEN}),
        "true_none": bool(actual == {NONE_TOKEN}),
        "predicted_size": predicted_size,
        "true_size": true_size,
    }


def _history_length(user_id: int, history_lengths: Mapping[int, int]) -> int:
    if user_id not in history_lengths:
        raise ValueError(f"User {user_id} has no history length")
    return int(history_lengths[user_id])


def _product_count(basket: set[int | str]) -> int:
    return len(basket - {NONE_TOKEN})


def _bin_users(users: pd.DataFrame, name: str) -> pd.DataFrame:
    if users.empty:
        return users
    return users.loc[users["history_bin"] == name]


def _average(users: pd.DataFrame) -> dict:
    if users.empty:
        ranking = pd.Series(dtype=float)
        nonempty_count = 0
    else:
        ranking = users.loc[users["nonempty"], "ranking_f1"]
        nonempty_count = int(users["nonempty"].sum())
    return {
        "n_users": int(len(users)),
        "n_nonempty_users": nonempty_count,
        "basket_f1": _mean(users, "basket_f1"),
        "true_size_f1": _mean(users, "true_size_f1"),
        "ranking_f1": _mean_series(ranking),
        "mean_signed_size_error": _mean(users, "signed_size_error"),
        "mean_absolute_size_error": _mean(users, "absolute_size_error"),
        "predicted_none_rate": _mean(users, "predicted_none"),
        "true_none_rate": _mean(users, "true_none"),
        "mean_predicted_basket_size": _mean(users, "predicted_size"),
        "mean_true_basket_size": _mean(users, "true_size"),
    }


def _mean(users: pd.DataFrame, column: str) -> float | None:
    if users.empty:
        return None
    return _mean_series(users[column])


def _mean_series(values: pd.Series) -> float | None:
    if len(values) == 0:
        return None
    return round(float(values.astype(float).mean()), 5)


def _summary_deltas(baseline: dict, current: dict) -> dict:
    return {
        "overall": _metric_deltas(baseline["overall"], current["overall"]),
        "bins": {
            name: _metric_deltas(baseline["bins"][name], current["bins"][name])
            for name in BIN_NAMES
        },
    }


def _metric_deltas(
    baseline: Mapping[str, float | int | None],
    current: Mapping[str, float | int | None],
) -> dict:
    deltas: dict[str, float | int | None] = {
        "n_users": int(current["n_users"]) - int(baseline["n_users"])
    }
    for key in _SCORE_KEYS:
        deltas[key] = _subtract(baseline[key], current[key])
    return deltas


def _subtract(
    baseline: float | int | None, current: float | int | None
) -> float | None:
    if baseline is None or current is None:
        return None
    return round(float(current) - float(baseline), 5)


def _paired(baseline: pd.DataFrame, current: pd.DataFrame) -> dict:
    left = baseline.set_index("user_id")
    right = current.set_index("user_id")
    shared = left.index.intersection(right.index)
    same_bin = _same_bin_users(left, right, shared)
    crossed = shared.difference(same_bin)
    both_nonempty = _nonempty_users(left, right, same_bin)
    return {
        "paired_user_count": int(len(same_bin)),
        "crossed_bin_user_count": int(len(crossed)),
        "baseline_only_user_count": int(len(left.index.difference(right.index))),
        "current_only_user_count": int(len(right.index.difference(left.index))),
        "basket_f1_delta": _paired_delta(left, right, same_bin, "basket_f1"),
        "true_size_f1_delta": _paired_delta(left, right, same_bin, "true_size_f1"),
        "ranking_f1_delta": _paired_delta(left, right, both_nonempty, "ranking_f1"),
        "ranking_paired_user_count": int(len(both_nonempty)),
        "bins": {name: _paired_bin(left, right, same_bin, name) for name in BIN_NAMES},
    }


def _same_bin_users(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    shared: pd.Index,
) -> pd.Index:
    if len(shared) == 0:
        return shared
    baseline_bins = baseline.loc[shared, "history_bin"].to_numpy()
    current_bins = current.loc[shared, "history_bin"].to_numpy()
    return shared[baseline_bins == current_bins]


def _nonempty_users(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    users: pd.Index,
) -> pd.Index:
    if len(users) == 0:
        return users
    baseline_nonempty = baseline.loc[users, "nonempty"].to_numpy()
    current_nonempty = current.loc[users, "nonempty"].to_numpy()
    return users[baseline_nonempty & current_nonempty]


def _paired_bin(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    same_bin: pd.Index,
    name: str,
) -> dict:
    if len(same_bin) == 0:
        users = same_bin
    else:
        users = same_bin[baseline.loc[same_bin, "history_bin"].to_numpy() == name]
    ranking = _nonempty_users(baseline, current, users)
    return {
        "paired_user_count": int(len(users)),
        "basket_f1_delta": _paired_delta(baseline, current, users, "basket_f1"),
        "true_size_f1_delta": _paired_delta(baseline, current, users, "true_size_f1"),
        "ranking_f1_delta": _paired_delta(baseline, current, ranking, "ranking_f1"),
        "ranking_paired_user_count": int(len(ranking)),
    }


def _paired_delta(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    users: pd.Index,
    column: str,
) -> float | None:
    if len(users) == 0:
        return None
    current_values = current.loc[users, column].astype(float)
    baseline_values = baseline.loc[users, column].astype(float)
    return round(float((current_values - baseline_values).mean()), 5)


def _history_lengths(label_path: str) -> dict[int, int]:
    return one_history_length(_read_frame(label_path, ["user_id", "history_length"]))


def _require_one_row(users: pd.DataFrame, name: str) -> None:
    if users["user_id"].duplicated().any():
        raise ValueError(f"{name} has more than one row per user")


def _stack_pointer(stack_dir: str, runs_root: str, manifest: dict):
    from instacart_platform.pointers import ResolvedPointer

    mode = manifest["mode"]
    models = {
        name: _resolved_model(name, manifest["models"][name], runs_root)
        for name in REQUIRED_MODELS
    }
    return ResolvedPointer(
        role="champion",
        mode=mode,
        timeline=manifest["timeline"],
        stack_id=manifest["stack_id"],
        stack_dir=stack_dir,
        model_path=str(join_path(stack_dir, "model.txt")),
        model_sha256="",
        previous_stack_id=None,
        models=models,
        promoted_at="",
    )


def _resolved_model(name: str, entry: Mapping[str, str], runs_root: str):
    from instacart_platform.pointers import ResolvedModel

    paths = build_run_paths(
        runs_root=runs_root,
        model_name=name,
        run_id=entry["run_id"],
    )
    return ResolvedModel(
        model_name=name,
        run_id=entry["run_id"],
        artifact_path=entry["artifact_path"],
        best_checkpoint=str(paths.best_checkpoint),
        checkpoint_sha256="",
    )


def write_drift_report(output: str, report: dict) -> list[str]:
    write_json(output, report)
    logger.info("Wrote champion drift report to %s", output)
    return write_drift_figures(output, report)


def write_drift_figures(output: str, report: dict) -> list[str]:
    """Write the four comparison figures beside a JSON report.

    Args:
        output: JSON path used to name the figures.
        report: Report from ``compare_snapshots``.

    Returns:
        Figure paths, or an empty list when matplotlib is not installed.
    """

    try:
        _pyplot()
    except ImportError:
        logger.warning(
            "matplotlib is not installed; wrote %s without figures",
            output,
        )
        return []
    plots = (
        ("overall_f1", plot_overall_f1),
        ("none_and_size", plot_none_and_size),
        ("history_bins", plot_history_bins),
        ("paired_deltas", plot_paired_deltas),
    )
    paths = [figure_path(output, name) for name, _plot in plots]
    for path, (_name, plot) in zip(paths, plots, strict=True):
        _write_png(path, plot(report))
        logger.info("Wrote champion drift figure to %s", path)
    return paths


def figure_path(output: str, name: str) -> str:
    """Return the PNG path beside a JSON report.

    Args:
        output: JSON report path.
        name: Figure name, such as ``overall_f1``.

    Returns:
        Path ending in ``{stem}_{name}.png``.
    """

    if is_gcs_url(output):
        parent, filename = output.rstrip("/").rsplit("/", 1)
        stem = _report_stem(filename)
        return f"{parent}/{stem}_{name}.png"
    path = Path(output)
    return str(path.with_name(f"{_report_stem(path.name)}_{name}.png"))


def plot_overall_f1(report: dict):
    """Plot baseline and current basket, true-size, and ranking F1.

    Args:
        report: Report from ``compare_snapshots``.

    Returns:
        A matplotlib figure.
    """

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7, 4))
    labels = [label for _key, label in _F1_METRICS]
    _grouped_bars(axis, labels, _overall_series(report, _F1_METRICS))
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("F1")
    axis.set_title(_comparison_title(report, "Overall F1"))
    figure.tight_layout()
    return figure


def plot_none_and_size(report: dict):
    """Plot None rates, basket sizes, and basket-size error.

    Args:
        report: Report from ``compare_snapshots``.

    Returns:
        A matplotlib figure with one panel for each comparison.
    """

    plt = _pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    panels = (
        (
            "None rate",
            ("Predicted", "True"),
            ("predicted_none_rate", "true_none_rate"),
            (0, 1.05),
        ),
        (
            "Basket size",
            ("Predicted", "True"),
            ("mean_predicted_basket_size", "mean_true_basket_size"),
            None,
        ),
        (
            "Size error",
            ("Signed", "Absolute"),
            ("mean_signed_size_error", "mean_absolute_size_error"),
            None,
        ),
    )
    for axis, (title, labels, keys, ylim) in zip(axes, panels, strict=True):
        metrics = tuple(zip(keys, labels, strict=True))
        _grouped_bars(axis, labels, _overall_series(report, metrics))
        axis.set_title(title)
        if ylim is not None:
            axis.set_ylim(*ylim)
    figure.suptitle(_comparison_title(report, "None rate and basket size"))
    figure.tight_layout()
    return figure


def plot_history_bins(report: dict):
    """Plot baseline and current scores inside each history-length bin.

    Args:
        report: Report from ``compare_snapshots``.

    Returns:
        A matplotlib figure with one panel per score.
    """

    plt = _pyplot()
    figure, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
    for axis, (key, title) in zip(axes.ravel(), _HISTORY_METRICS, strict=True):
        _grouped_bars(
            axis,
            BIN_NAMES,
            _bin_series(report, key),
        )
        axis.set_title(title, fontsize=9)
        axis.tick_params(axis="x", labelrotation=45, labelsize=7)
    figure.suptitle(_comparison_title(report, "History bins"))
    figure.tight_layout()
    return figure


def plot_paired_deltas(report: dict):
    """Plot same-bin paired changes, current minus baseline.

    The title records how many users stayed in the same bin and how many
    crossed bins. A crossed user is excluded from the bars.

    Args:
        report: Report from ``compare_snapshots``.

    Returns:
        A matplotlib figure.
    """

    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(10, 4))
    labels = [_paired_tick(report, name) for name in BIN_NAMES]
    series = []
    for (key, label), color in zip(_F1_METRICS, _DELTA_COLORS, strict=True):
        values = [report["paired"]["bins"][name][f"{key}_delta"] for name in BIN_NAMES]
        series.append((label, values, color))
    _grouped_bars(axis, labels, series)
    axis.set_ylabel("Current − baseline")
    paired = report["paired"]
    axis.set_title(
        "Paired same-bin deltas\n"
        f"{paired['paired_user_count']} paired users, "
        f"{paired['crossed_bin_user_count']} crossed bins"
    )
    figure.tight_layout()
    return figure


def _comparison_title(report: dict, heading: str) -> str:
    return (
        f"{heading}\n{report['champion_stack_id']}: "
        f"{report['baseline_timeline']} vs {report['current_timeline']}"
    )


def _paired_tick(report: dict, name: str) -> str:
    count = report["paired"]["bins"][name]["paired_user_count"]
    return f"{name}\n(n={count})"


def _overall_series(
    report: dict,
    metrics: Sequence[tuple[str, str]],
) -> list[tuple[str, list[float | None], str]]:
    return [
        (
            "Baseline",
            [report["baseline"]["overall"][key] for key, _label in metrics],
            _BASELINE_COLOR,
        ),
        (
            "Current",
            [report["current"]["overall"][key] for key, _label in metrics],
            _CURRENT_COLOR,
        ),
    ]


def _bin_series(
    report: dict,
    key: str,
) -> list[tuple[str, list[float | None], str]]:
    return [
        (
            "Baseline",
            [report["baseline"]["bins"][name][key] for name in BIN_NAMES],
            _BASELINE_COLOR,
        ),
        (
            "Current",
            [report["current"]["bins"][name][key] for name in BIN_NAMES],
            _CURRENT_COLOR,
        ),
    ]


def _grouped_bars(ax, labels: Sequence[str], series: Sequence[tuple]) -> None:
    count = len(series)
    positions = np.arange(len(labels))
    width = 0.8 / count
    for index, (name, values, color) in enumerate(series):
        offset = (index - (count - 1) / 2) * width
        _bars_at(ax, positions + offset, values, width, name, color)
    ax.set_xticks(positions, list(labels))
    ax.axhline(0, color="0.6", linewidth=0.6)
    _add_legend(ax, series)


def _bars_at(ax, positions, values, width: float, name: str, color: str) -> None:
    drawn_x = []
    drawn_y = []
    for position, value in zip(positions, values, strict=True):
        if value is None:
            continue
        drawn_x.append(position)
        drawn_y.append(float(value))
    if not drawn_x:
        return
    ax.bar(drawn_x, drawn_y, width, label=name, color=color)


def _add_legend(ax, series: Sequence[tuple]) -> None:
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(facecolor=color, label=name) for name, _values, color in series],
        fontsize=8,
    )


def _write_png(path: str, figure) -> None:
    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    finally:
        _pyplot().close(figure)
    payload = buffer.getvalue()
    if is_gcs_url(path):
        import gcsfs

        filesystem = gcsfs.GCSFileSystem()
        with filesystem.open(path, "wb") as file:
            file.write(payload)
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as file:
        file.write(payload)


def _pyplot():
    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _report_stem(filename: str) -> str:
    if filename.endswith(".json"):
        return filename[: -len(".json")]
    return filename


def _stack_id(stack_dir: str) -> str:
    manifest = load_selected_artifacts(
        str(join_path(stack_dir, "selected_artifacts.json"))
    )
    return str(manifest["stack_id"])


if __name__ == "__main__":
    main()
