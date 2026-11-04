import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from dsrv.metrics import compute_metrics, predict_at_threshold, threshold_sweep


def test_known_matrix():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1]
    metrics = compute_metrics(y_true, y_pred)

    assert metrics["confusion_matrix"] == [[3, 1], [2, 4]]
    assert metrics["recall_attack"] == pytest.approx(4 / 6)
    assert metrics["precision_attack"] == pytest.approx(4 / 5)
    assert metrics["fpr"] == pytest.approx(1 / 4)
    assert metrics["accuracy"] == pytest.approx(7 / 10)
    assert not any(metrics["zero_division"].values())


def test_fpr_without_negatives_is_zero_and_flagged():
    metrics = compute_metrics([1, 1, 1], [1, 0, 1])

    assert metrics["fpr"] == 0.0
    assert metrics["zero_division"]["fpr"] is True
    assert metrics["zero_division"]["recall_attack"] is False


def test_recall_without_positives_is_zero_and_flagged():
    metrics = compute_metrics([0, 0], [0, 1])

    assert metrics["recall_attack"] == 0.0
    assert metrics["zero_division"]["recall_attack"] is True


def test_precision_without_predicted_positives_is_zero_and_flagged():
    metrics = compute_metrics([0, 1], [0, 0])

    assert metrics["precision_attack"] == 0.0
    assert metrics["zero_division"]["precision_attack"] is True


def test_macro_f1_flags_a_class_with_undefined_f1():
    metrics = compute_metrics([1, 1], [1, 1])

    assert metrics["macro_f1"] == pytest.approx(0.5)
    assert metrics["zero_division"]["macro_f1"] is True


@pytest.mark.parametrize("seed", range(5))
def test_matches_scikit_learn(seed):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, 500)
    y_pred = rng.integers(0, 2, 500)
    metrics = compute_metrics(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    assert metrics["confusion_matrix"] == [[tn, fp], [fn, tp]]
    assert metrics["recall_attack"] == pytest.approx(recall_score(y_true, y_pred))
    assert metrics["precision_attack"] == pytest.approx(precision_score(y_true, y_pred))
    assert metrics["macro_f1"] == pytest.approx(f1_score(y_true, y_pred, average="macro"))
    assert metrics["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))


def test_rejects_invalid_input():
    with pytest.raises(ValueError):
        compute_metrics([0, 1], [0])
    with pytest.raises(ValueError):
        compute_metrics([0, 2], [0, 1])


def test_threshold_is_inclusive():
    assert predict_at_threshold([0.49, 0.5, 0.51], 0.5).tolist() == [0, 1, 1]


def test_sweep_covers_every_threshold():
    y = np.array([0, 0, 1, 1])
    sweep = threshold_sweep(y, np.array([0.1, 0.4, 0.35, 0.8]))

    assert len(sweep) == 91
    assert sweep["threshold"].iloc[0] == 0.05
    assert sweep["threshold"].iloc[-1] == 0.95
    assert sweep["threshold"].tolist() == [i / 100 for i in range(5, 96)]


def test_sweep_matches_compute_metrics_and_is_monotonic():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    score = rng.random(300)
    sweep = threshold_sweep(y, score)

    row = sweep[sweep["threshold"] == 0.5].iloc[0]
    expected = compute_metrics(y, predict_at_threshold(score, 0.5))
    assert row["recall_attack"] == expected["recall_attack"]
    assert [[row["tn"], row["fp"]], [row["fn"], row["tp"]]] == expected["confusion_matrix"]
    assert sweep["recall_attack"].is_monotonic_decreasing
    assert sweep["fpr"].is_monotonic_decreasing


def test_sweep_flags_zero_division():
    sweep = threshold_sweep(np.array([1, 1]), np.array([0.2, 0.9]))

    assert sweep["zero_division"].all()
