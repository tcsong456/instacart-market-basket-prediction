import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

from instacart_rnn.stacking.drift_report import (
    compare_snapshots,
    evaluation_inputs,
    f1_at_true_size,
    figure_path,
    history_bin,
    one_history_length,
    plot_history_bins,
    plot_none_and_size,
    plot_overall_f1,
    plot_paired_deltas,
    summarize_users,
    user_scores,
    write_drift_figures,
    write_drift_report,
)


def test_history_bin_keeps_lengths_below_three_separate() -> None:
    assert history_bin(2) == "other"
    assert history_bin(3) == "3-5"
    assert history_bin(5) == "3-5"
    assert history_bin(6) == "6-10"
    assert history_bin(20) == "16-20"
    assert history_bin(21) == "21-25"
    assert history_bin(25) == "21-25"
    assert history_bin(26) == "26+"


def test_true_size_f1_scores_none_only_when_the_prediction_is_none() -> None:
    matched, matched_nonempty, matched_ranking = f1_at_true_size(
        [1],
        [0.01],
        [0],
    )
    missed, missed_nonempty, missed_ranking = f1_at_true_size(
        [7],
        [0.99],
        [0],
    )

    assert matched == 1.0
    assert matched_nonempty is False
    assert math.isnan(matched_ranking)
    assert missed == 0.0
    assert missed_nonempty is False
    assert math.isnan(missed_ranking)


def test_ranking_f1_cuts_at_the_true_product_count() -> None:
    score, nonempty, ranking = f1_at_true_size(
        [1, 2],
        [0.95, 0.5],
        [1, 0],
    )

    assert nonempty is True
    assert ranking == 1.0
    assert score == 1.0


def test_user_scores_keeps_basket_f1_apart_from_ranking_f1() -> None:
    frame = pd.DataFrame(
        {
            "user_id": [7, 7],
            "product_id": [1, 2],
            "label": [1, 0],
        }
    )

    row = user_scores(frame, [0.95, 0.5], {7: 4}).iloc[0]

    assert row["history_bin"] == "3-5"
    assert row["ranking_f1"] == 1.0
    assert row["basket_f1"] == pytest.approx(2.0 / 3.0)
    assert row["signed_size_error"] == 1.0
    assert row["absolute_size_error"] == 1.0
    assert bool(row["true_none"]) is False
    assert bool(row["predicted_none"]) is False
    assert row["true_size"] == 1
    assert row["predicted_size"] == 2


def test_user_scores_rejects_a_missing_history_length() -> None:
    frame = pd.DataFrame({"user_id": [1], "product_id": [1], "label": [0]})

    with pytest.raises(ValueError, match="no history length"):
        user_scores(frame, [0.01], {})


def test_user_scores_rejects_misaligned_probabilities() -> None:
    frame = pd.DataFrame({"user_id": [1], "product_id": [1], "label": [0]})

    with pytest.raises(ValueError, match="align"):
        user_scores(frame, [0.01, 0.2], {1: 3})


def test_absolute_size_error_keeps_cancelled_mistakes() -> None:
    users = _users(
        [
            _row(1, "3-5", signed=5.0, basket_f1=1.0, ranking_f1=1.0),
            _row(2, "3-5", signed=-5.0, basket_f1=1.0, ranking_f1=1.0),
        ]
    )

    summary = summarize_users(users)

    assert summary["overall"]["mean_signed_size_error"] == 0.0
    assert summary["overall"]["mean_absolute_size_error"] == 5.0
    assert summary["bins"]["3-5"]["n_users"] == 2
    assert summary["bins"]["26+"]["n_users"] == 0
    assert summary["bins"]["26+"]["basket_f1"] is None


def test_ranking_average_excludes_none_baskets() -> None:
    users = _users(
        [
            _row(1, "6-10", ranking_f1=1.0, true_size_f1=1.0, nonempty=True),
            _row(
                2,
                "6-10",
                ranking_f1=float("nan"),
                true_size_f1=0.0,
                nonempty=False,
                true_none=True,
            ),
        ]
    )

    summary = summarize_users(users)["overall"]

    assert summary["n_users"] == 2
    assert summary["n_nonempty_users"] == 1
    assert summary["ranking_f1"] == 1.0
    assert summary["true_size_f1"] == 0.5
    assert summary["true_none_rate"] == 0.5


def test_paired_deltas_exclude_users_who_change_bins() -> None:
    baseline = _users(
        [
            _row(1, "3-5", basket_f1=0.2, true_size_f1=0.4, ranking_f1=0.4),
            _row(2, "3-5", basket_f1=1.0, true_size_f1=1.0, ranking_f1=1.0),
            _row(3, "6-10", basket_f1=1.0, true_size_f1=1.0, ranking_f1=1.0),
        ]
    )
    current = _users(
        [
            _row(1, "3-5", basket_f1=0.5, true_size_f1=0.8, ranking_f1=0.8),
            _row(2, "6-10", basket_f1=0.0, true_size_f1=0.0, ranking_f1=0.0),
            _row(4, "3-5", basket_f1=1.0, true_size_f1=1.0, ranking_f1=1.0),
        ]
    )

    report = compare_snapshots(
        baseline,
        current,
        champion_stack_id="stack-a",
        baseline_timeline="t1",
        current_timeline="t2",
    )

    paired = report["paired"]
    assert report["champion_stack_id"] == "stack-a"
    assert paired["paired_user_count"] == 1
    assert paired["crossed_bin_user_count"] == 1
    assert paired["baseline_only_user_count"] == 1
    assert paired["current_only_user_count"] == 1
    assert paired["basket_f1_delta"] == pytest.approx(0.3)
    assert paired["true_size_f1_delta"] == pytest.approx(0.4)
    assert paired["ranking_f1_delta"] == pytest.approx(0.4)
    assert paired["ranking_paired_user_count"] == 1
    assert paired["bins"]["3-5"]["paired_user_count"] == 1
    assert paired["bins"]["6-10"]["paired_user_count"] == 0
    assert paired["bins"]["6-10"]["basket_f1_delta"] is None
    assert report["deltas"]["overall"]["n_users"] == 0
    assert report["deltas"]["bins"]["26+"]["basket_f1"] is None


def test_ranking_pair_requires_a_product_on_both_snapshots() -> None:
    baseline = _users(
        [
            _row(1, "11-15", ranking_f1=1.0, nonempty=True),
            _row(2, "11-15", ranking_f1=float("nan"), nonempty=False),
        ]
    )
    current = _users(
        [
            _row(1, "11-15", ranking_f1=float("nan"), nonempty=False),
            _row(2, "11-15", ranking_f1=1.0, nonempty=True),
        ]
    )

    paired = compare_snapshots(
        baseline,
        current,
        champion_stack_id="stack-a",
        baseline_timeline="t1",
        current_timeline="t2",
    )["paired"]

    assert paired["paired_user_count"] == 2
    assert paired["ranking_paired_user_count"] == 0
    assert paired["ranking_f1_delta"] is None


def test_one_history_length_rejects_a_user_with_two_lengths() -> None:
    frame = pd.DataFrame({"user_id": [1, 1], "history_length": [4, 5]})

    with pytest.raises(ValueError, match="not constant"):
        one_history_length(frame)


def test_evaluation_inputs_use_sibling_gold_in_the_same_timeline() -> None:
    paths = evaluation_inputs("gs://gold/curated/t2/product_training_data_evaluation")

    assert paths["product"].endswith("/t2/product_training_data_evaluation")
    assert paths["aisle"].endswith("/t2/aisle_training_data_evaluation")
    assert paths["reorder_size_rnn"] == paths["reorder_size_gmm"]
    assert paths["reorder_size_rnn"].endswith(
        "/t2/reorder_size_training_data_evaluation"
    )


def test_evaluation_inputs_reject_a_path_that_is_not_product_evaluation() -> None:
    with pytest.raises(ValueError, match="product_training_data_evaluation"):
        evaluation_inputs("gs://gold/curated/t2/aisle_training_data_evaluation")


def test_figure_path_keeps_the_report_directory() -> None:
    output = "gs://runs/monitoring/curated/stack-a/drift.json"

    assert figure_path(output, "overall_f1") == (
        "gs://runs/monitoring/curated/stack-a/drift_overall_f1.png"
    )


def test_report_is_written_when_matplotlib_is_missing(mocker, tmp_path) -> None:
    mocker.patch.dict(sys.modules, {"matplotlib": None})
    output = tmp_path / "drift.json"

    paths = write_drift_report(str(output), _chart_report())

    assert paths == []
    assert json.loads(output.read_text(encoding="utf-8"))["champion_stack_id"] == (
        "stack-a"
    )
    assert list(tmp_path.glob("*.png")) == []


def test_overall_figure_uses_the_report_scores() -> None:
    pytest.importorskip("matplotlib")
    report = _chart_report()
    figure = plot_overall_f1(report)
    try:
        baseline = report["baseline"]["overall"]
        current = report["current"]["overall"]
        assert _heights(figure.axes[0], "Baseline") == pytest.approx(
            [baseline["basket_f1"], baseline["true_size_f1"], baseline["ranking_f1"]]
        )
        assert _heights(figure.axes[0], "Current") == pytest.approx(
            [current["basket_f1"], current["true_size_f1"], current["ranking_f1"]]
        )
        assert "stack-a" in figure.axes[0].get_title()
        assert "t1 vs t2" in figure.axes[0].get_title()
    finally:
        _close(figure)


def test_none_and_size_figure_keeps_separate_scales() -> None:
    pytest.importorskip("matplotlib")
    report = _chart_report()
    figure = plot_none_and_size(report)
    try:
        baseline = report["baseline"]["overall"]
        assert _heights(figure.axes[0], "Baseline") == pytest.approx(
            [baseline["predicted_none_rate"], baseline["true_none_rate"]]
        )
        assert _heights(figure.axes[1], "Baseline") == pytest.approx(
            [
                baseline["mean_predicted_basket_size"],
                baseline["mean_true_basket_size"],
            ]
        )
        assert _heights(figure.axes[2], "Current") == pytest.approx(
            [
                report["current"]["overall"]["mean_signed_size_error"],
                report["current"]["overall"]["mean_absolute_size_error"],
            ]
        )
    finally:
        _close(figure)


def test_history_figure_leaves_empty_bins_blank() -> None:
    pytest.importorskip("matplotlib")
    report = _chart_report()
    figure = plot_history_bins(report)
    try:
        axis = figure.axes[0]
        assert _heights(axis, "Baseline") == pytest.approx(
            [
                report["baseline"]["bins"]["3-5"]["basket_f1"],
                report["baseline"]["bins"]["6-10"]["basket_f1"],
            ]
        )
        centers = [patch.get_x() + patch.get_width() / 2 for patch in axis.patches]
        assert centers
        assert all(center < 2 for center in centers)
    finally:
        _close(figure)


def test_paired_figure_shows_same_bin_deltas_and_counts() -> None:
    pytest.importorskip("matplotlib")
    report = _chart_report()
    figure = plot_paired_deltas(report)
    try:
        assert _heights(figure.axes[0], "Basket F1") == pytest.approx([0.3])
        assert _heights(figure.axes[0], "True-size F1") == pytest.approx([0.4])
        title = figure.axes[0].get_title()
        assert "1 paired users" in title
        assert "1 crossed bins" in title
    finally:
        _close(figure)


def test_write_drift_report_saves_json_and_four_figures(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    output = tmp_path / "drift.json"

    paths = write_drift_report(str(output), _chart_report())

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["baseline_timeline"] == "t1"
    assert [Path(path).name for path in paths] == [
        "drift_overall_f1.png",
        "drift_none_and_size.png",
        "drift_history_bins.png",
        "drift_paired_deltas.png",
    ]
    for path in paths:
        assert Path(path).read_bytes().startswith(b"\x89PNG")


def test_figures_are_uploaded_beside_a_gcs_report(mocker) -> None:
    pytest.importorskip("matplotlib")
    filesystem = mocker.MagicMock()
    mocker.patch("gcsfs.GCSFileSystem", return_value=filesystem)

    paths = write_drift_figures("gs://runs/monitoring/drift.json", _chart_report())

    assert paths[0] == "gs://runs/monitoring/drift_overall_f1.png"
    assert len(paths) == 4
    assert filesystem.open.call_args_list[0].args == (
        "gs://runs/monitoring/drift_overall_f1.png",
        "wb",
    )
    written = filesystem.open.return_value.__enter__.return_value.write
    assert written.call_args_list[0].args[0].startswith(b"\x89PNG")


def _chart_report() -> dict:
    baseline = _users(
        [
            _row(1, "3-5", basket_f1=0.2, true_size_f1=0.4, ranking_f1=0.4),
            _row(2, "3-5", basket_f1=1.0, true_size_f1=1.0, ranking_f1=1.0),
            _row(3, "6-10", basket_f1=1.0, true_size_f1=1.0, ranking_f1=1.0),
        ]
    )
    current = _users(
        [
            _row(1, "3-5", basket_f1=0.5, true_size_f1=0.8, ranking_f1=0.8),
            _row(2, "6-10", basket_f1=0.0, true_size_f1=0.0, ranking_f1=0.0),
            _row(4, "3-5", basket_f1=1.0, true_size_f1=1.0, ranking_f1=1.0),
        ]
    )
    return compare_snapshots(
        baseline,
        current,
        champion_stack_id="stack-a",
        baseline_timeline="t1",
        current_timeline="t2",
    )


def _heights(axis, label: str) -> list[float]:
    for container in axis.containers:
        if container.get_label() == label:
            return [patch.get_height() for patch in container.patches]
    raise AssertionError(label)


def _close(figure) -> None:
    from matplotlib import pyplot as plt

    plt.close(figure)


def _users(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _row(
    user_id: int,
    history_bin_name: str,
    *,
    basket_f1: float = 1.0,
    true_size_f1: float = 1.0,
    ranking_f1: float = 1.0,
    nonempty: bool = True,
    signed: float = 0.0,
    predicted_none: bool = False,
    true_none: bool = False,
    predicted_size: int = 1,
    true_size: int = 1,
) -> dict:
    return {
        "user_id": user_id,
        "history_length": 4,
        "history_bin": history_bin_name,
        "basket_f1": basket_f1,
        "true_size_f1": true_size_f1,
        "ranking_f1": ranking_f1,
        "nonempty": nonempty,
        "signed_size_error": signed,
        "absolute_size_error": abs(signed),
        "predicted_none": predicted_none,
        "true_none": true_none,
        "predicted_size": predicted_size,
        "true_size": true_size,
    }
