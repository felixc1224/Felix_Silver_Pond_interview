"""Tests for the metric functions in src/evaluate.py, on tiny hand-built inputs."""

import numpy as np
import pytest

from src.evaluate import (accuracy, calibration_bins, confusion_matrix, per_class_metrics,
                          top_k_accuracy)

# 3 classes, 6 samples. true:  0 0 1 1 2 2
#                      pred:  0 1 1 1 2 0
Y_TRUE = np.array([0, 0, 1, 1, 2, 2])
Y_PRED = np.array([0, 1, 1, 1, 2, 0])


def test_confusion_matrix_values():
    cm = confusion_matrix(Y_TRUE, Y_PRED, 3)
    expected = np.array([[1, 1, 0],
                         [0, 2, 0],
                         [1, 0, 1]])
    assert np.array_equal(cm, expected)


def test_confusion_matrix_rows_sum_to_true_counts():
    cm = confusion_matrix(Y_TRUE, Y_PRED, 3)
    assert cm.sum() == len(Y_TRUE)
    assert np.array_equal(cm.sum(axis=1), np.bincount(Y_TRUE, minlength=3))
    assert np.array_equal(cm.sum(axis=0), np.bincount(Y_PRED, minlength=3))


def test_accuracy_matches_diagonal():
    cm = confusion_matrix(Y_TRUE, Y_PRED, 3)
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(4 / 6)
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(np.trace(cm) / cm.sum())


def test_per_class_metrics_by_hand():
    pc = per_class_metrics(confusion_matrix(Y_TRUE, Y_PRED, 3))
    # class 0: tp=1, true=2, predicted=2 -> recall .5, precision .5, f1 .5
    # class 1: tp=2, true=2, predicted=3 -> recall 1,  precision 2/3, f1 .8
    # class 2: tp=1, true=2, predicted=1 -> recall .5, precision 1,   f1 2/3
    assert pc["recall"] == pytest.approx([0.5, 1.0, 0.5])
    assert pc["precision"] == pytest.approx([0.5, 2 / 3, 1.0])
    assert pc["f1"] == pytest.approx([0.5, 0.8, 2 / 3])
    assert np.array_equal(pc["support"], [2, 2, 2])


def test_per_class_metrics_zero_safe():
    """A class that is never predicted must give precision 0, not a divide-by-zero."""
    cm = np.array([[2, 0], [2, 0]])          # class 1 never predicted
    pc = per_class_metrics(cm)
    assert pc["precision"][1] == 0 and pc["recall"][1] == 0 and pc["f1"][1] == 0
    assert np.isfinite(pc["f1"]).all()


def test_perfect_predictions():
    y = np.arange(10).repeat(3)
    cm = confusion_matrix(y, y, 10)
    assert np.array_equal(cm, np.eye(10, dtype=int) * 3)
    pc = per_class_metrics(cm)
    assert np.all(pc["f1"] == 1.0)
    assert accuracy(y, y) == 1.0


def test_top_k_accuracy():
    probs = np.array([[0.6, 0.3, 0.1],     # true 1 -> top1 wrong, top2 right
                      [0.1, 0.2, 0.7],     # true 2 -> top1 right
                      [0.5, 0.1, 0.4]])    # true 1 -> not in top2
    y = np.array([1, 2, 1])
    assert top_k_accuracy(probs, y, k=1) == pytest.approx(1 / 3)
    assert top_k_accuracy(probs, y, k=2) == pytest.approx(2 / 3)
    assert top_k_accuracy(probs, y, k=3) == 1.0


def test_calibration_perfectly_calibrated():
    """Confidence 0.75 and exactly 75% correct -> ECE ~ 0."""
    probs = np.tile([0.75, 0.25], (100, 1))
    y = np.array([0] * 75 + [1] * 25)
    cal = calibration_bins(probs, y, n_bins=10)
    assert cal["count"].sum() == 100
    assert cal["ece"] == pytest.approx(0.0, abs=1e-9)


def test_calibration_overconfident():
    """Confidence 0.9 but only 50% correct -> ECE = 0.4."""
    probs = np.tile([0.9, 0.1], (10, 1))
    y = np.array([0] * 5 + [1] * 5)
    cal = calibration_bins(probs, y, n_bins=10)
    assert cal["ece"] == pytest.approx(0.4)
