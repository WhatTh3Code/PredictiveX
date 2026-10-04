"""Metric helpers. Every number reported anywhere in PredictiveX is produced by these functions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "f2": round(float(fbeta_score(y, pred, beta=2, zero_division=0)), 4),
        "accuracy": round(float((tp + tn) / len(y)), 4),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) else 0.0, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "positives": int(np.sum(y)),
        "n": int(len(y)),
        "prevalence": round(float(np.mean(y)), 4),
    }


def full_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    return {**ranking_metrics(y, p), **threshold_metrics(y, p, threshold)}


def best_fbeta_threshold(y: np.ndarray, p: np.ndarray, beta: float) -> float:
    """Threshold that maximises F-beta. Returned as the midpoint between adjacent scores so it
    is not glued to a single training prediction."""
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    denom = beta**2 * prec + rec
    f = np.where(denom > 0, (1 + beta**2) * prec * rec / np.where(denom > 0, denom, 1), 0)
    i = int(np.argmax(f))
    if i > 0:
        return float((thr[i] + thr[i - 1]) / 2)
    return float(thr[i])


def threshold_for_recall(y: np.ndarray, p: np.ndarray, target_recall: float) -> float:
    """Highest threshold whose recall is still >= target_recall."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return float(thr[0])
    return float(thr[ok[-1]])


def curves(y: np.ndarray, p: np.ndarray, max_points: int = 200) -> dict:
    prec, rec, pr_thr = precision_recall_curve(y, p)
    fpr, tpr, _ = roc_curve(y, p)

    def thin(*arrays: np.ndarray) -> list[list[float]]:
        n = len(arrays[0])
        idx = np.unique(np.linspace(0, n - 1, min(n, max_points)).astype(int))
        return [[round(float(a[i]), 4) for a in arrays] for i in idx]

    return {
        "pr": thin(rec, prec),
        "roc": thin(fpr, tpr),
        "pr_thresholds": thin(rec[:-1], prec[:-1], pr_thr),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, edges: tuple[float, ...] = (0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0001)) -> list[dict]:
    """Reliability table on fixed probability bins (empty bins are skipped).

    Fixed bins are used because almost all scores are near zero: equal-count bins would put
    every point in the bottom corner and hide how the high-probability range behaves."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        out.append(
            {
                "bin": [round(float(lo), 2), round(float(min(hi, 1.0)), 2)],
                "mean_predicted": round(float(p[mask].mean()), 4),
                "observed_rate": round(float(y[mask].mean()), 4),
                "n": int(mask.sum()),
            }
        )
    return out


def bootstrap_ci(
    y: np.ndarray, p: np.ndarray, threshold: float, n_boot: int = 1000, seed: int = 0
) -> dict:
    """Percentile bootstrap 95% CIs for test metrics: how much would they move on another sample?"""
    rng = np.random.default_rng(seed)
    stats: dict[str, list[float]] = {k: [] for k in ("pr_auc", "roc_auc", "precision", "recall", "f1")}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        pred = (pb >= threshold).astype(int)
        stats["pr_auc"].append(average_precision_score(yb, pb))
        stats["roc_auc"].append(roc_auc_score(yb, pb))
        stats["precision"].append(precision_score(yb, pred, zero_division=0))
        stats["recall"].append(recall_score(yb, pred, zero_division=0))
        stats["f1"].append(f1_score(yb, pred, zero_division=0))
    return {
        k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
        for k, v in stats.items()
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return [round(float(max(0.0, centre - half)), 4), round(float(min(1.0, centre + half)), 4)]
