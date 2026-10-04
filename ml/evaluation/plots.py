"""Evaluation figures (reports/figures/07-12). Drawn only from computed results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from ml.evaluation import style
from ml.evaluation.metrics import threshold_metrics


def model_comparison(rows: list[dict], path: Path) -> None:
    style.apply()
    metrics = [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("precision", "Precision"), ("recall", "Recall"),
               ("f1", "F1")]
    names = [r["display_name"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    width = 0.8 / len(rows)
    x = np.arange(len(metrics))
    for i, r in enumerate(rows):
        vals = [r["validation"][k] for k, _ in metrics]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width * 0.92, color=style.SERIES[i], label=names[i])
    ax.set_xticks(x, [m for _, m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Validation score")
    ax.set_title("Model comparison on the validation split (thresholds from out-of-fold training predictions)")
    ax.legend(ncol=len(rows), loc="lower center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def pr_roc(y: np.ndarray, p: np.ndarray, thr: float, path: Path) -> None:
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    prec, rec, _ = precision_recall_curve(y, p)
    m = threshold_metrics(y, p, thr)
    ax = axes[0]
    ax.plot(rec, prec, color=style.NORMAL, lw=2)
    ax.axhline(y.mean(), color=style.MUTED, lw=0.8)
    ax.scatter([m["recall"]], [m["precision"]], s=60, color=style.FAILED, zorder=3, edgecolors=style.SURFACE)
    ax.annotate(f"operating point\nP={m['precision']:.2f} R={m['recall']:.2f}", (m["recall"], m["precision"]),
                xytext=(-110, -40), textcoords="offset points", fontsize=8, color=style.INK_2)
    ax.text(0.02, y.mean() + 0.02, f"no-skill = {y.mean():.3f}", fontsize=7.5, color=style.MUTED)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall (test)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax = axes[1]
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, color=style.NORMAL, lw=2)
    ax.plot([0, 1], [0, 1], color=style.MUTED, lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC (test)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def confusion(cm: dict, thr: float, path: Path) -> None:
    style.apply()
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    shade = np.array([[0.15, 0.55], [0.55, 0.9]])
    ax.imshow(shade, cmap="Blues", vmin=0, vmax=1.4)
    labels = [["True negative", "False alarm"], ["Missed failure", "Caught failure"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:,}\n{labels[i][j]}", ha="center", va="center", fontsize=9,
                    color="white" if shade[i, j] > 0.7 else style.INK)
    ax.set_xticks([0, 1], ["Predicted normal", "Predicted failure"])
    ax.set_yticks([0, 1], ["Actual normal", "Actual failure"])
    ax.grid(False)
    ax.set_title(f"Confusion matrix (test, threshold {thr:.3f})")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def importance(shap_imp: list[dict], perm_imp: list[dict], path: Path) -> None:
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, rows, key, title, color in (
        (axes[0], shap_imp, "mean_abs_shap", "Mean |SHAP| (test sample, probability points)", style.NORMAL),
        (axes[1], perm_imp, "mean", "Permutation importance on validation (drop in PR-AUC)", style.NORMAL),
    ):
        rows = rows[::-1]
        ax.barh([r["label"] for r in rows], [r[key] for r in rows], color=color, height=0.6,
                xerr=[r.get("std", 0) for r in rows] if key == "mean" else None, ecolor=style.INK_2)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def calibration(table: list[dict], path: Path) -> None:
    style.apply()
    fig, ax = plt.subplots(figsize=(4.6, 4))
    x = [r["mean_predicted"] for r in table]
    y = [r["observed_rate"] for r in table]
    ax.plot([0, 1], [0, 1], color=style.MUTED, lw=0.8)
    ax.plot(x, y, color=style.NORMAL, lw=2, marker="o", ms=5)
    for r in table:
        ax.annotate(f"n={r['n']}", (r["mean_predicted"], r["observed_rate"]), xytext=(5, -10),
                    textcoords="offset points", fontsize=7, color=style.INK_2)
    ax.set_xlabel("Mean predicted probability (fixed bins)")
    ax.set_ylabel("Observed failure rate")
    ax.set_title("Calibration (test)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def threshold_tradeoff(y: np.ndarray, p: np.ndarray, policy, path: Path) -> None:
    style.apply()
    ts = np.linspace(0.01, 0.95, 95)
    rows = [threshold_metrics(y, p, t) for t in ts]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for key, color, name in (("precision", style.SERIES[0], "Precision"), ("recall", style.SERIES[1], "Recall"),
                             ("f1", style.SERIES[2], "F1")):
        ax.plot(ts, [r[key] for r in rows], color=color, lw=2, label=name)
    for t, name in ((policy.medium_threshold, "MEDIUM"), (policy.high_threshold, "HIGH")):
        ax.axvline(t, color=style.INK_2, lw=0.8)
        ax.text(t + 0.01, 0.04, name, fontsize=8, color=style.INK_2)
    ax.set_xlabel("Decision threshold")
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision / recall trade-off on validation (thresholds fixed from training OOF)")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
