"""Phase 3 - exploratory data analysis.

    python scripts/run_eda.py

All analysis is computed on the TRAINING split only, so nothing learned here (including
feature ideas) is informed by validation or test rows. Each figure answers one question:

  01 Are failures imbalanced, and how are failure modes distributed?
  02 Which operating conditions differ between normal and failed machines?
  03 How does the failure rate change across the range of each variable?
  04 Where in operating space does each failure mode occur?
  05 Does product quality type change the failure rate?
  06 Which variables move together (redundancy)?

Writes reports/figures/*.png and reports/eda_summary.json (consumed by the dashboard).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from ml import config  # noqa: E402
from ml.evaluation import style  # noqa: E402
from ml.features.engineering import add_engineered  # noqa: E402
from ml.preprocessing.data import load_split  # noqa: E402

FEATS = [*config.NUMERIC_INPUTS, "temp_diff_k", "power_w", "strain_nm_min"]


def label(c: str) -> str:
    unit = config.FEATURE_UNITS.get(c)
    return f"{config.FEATURE_LABELS[c]} [{unit}]" if unit else config.FEATURE_LABELS[c]


def fig_imbalance(df: pd.DataFrame, out: Path) -> dict:
    counts = df[config.TARGET].value_counts().sort_index()
    modes = {config.FAILURE_MODE_NAMES[m]: int(df[m].sum()) for m in config.FAILURE_MODES}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    bars = ax.bar(["Normal", "Failed"], counts.values, color=[style.NORMAL, style.FAILED], width=0.55)
    for b, v in zip(bars, counts.values, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / len(df):.1%})", ha="center",
                va="bottom", fontsize=8, color=style.INK_2)
    ax.set_title("Target: 1 failure per ~29 normal records")
    ax.set_ylabel("Records (training split)")
    ax.set_ylim(0, counts.max() * 1.18)
    ax = axes[1]
    names = list(modes)[::-1]
    vals = [modes[n] for n in names]
    ax.barh(names, vals, color=style.FAILED, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 1, i, str(v), va="center", fontsize=8, color=style.INK_2)
    ax.set_title("Failure-mode labels (a record can carry several)")
    ax.set_xlabel("Records")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "01_class_imbalance.png")
    plt.close(fig)
    return {"target_counts": {"normal": int(counts.get(0, 0)), "failed": int(counts.get(1, 0))}, "mode_counts": modes}


def fig_distributions(df: pd.DataFrame, out: Path) -> dict:
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6))
    result = {}
    for ax, c in zip(axes.flat, FEATS, strict=True):
        normal = df.loc[df[config.TARGET] == 0, c]
        failed = df.loc[df[config.TARGET] == 1, c]
        bins = np.histogram_bin_edges(df[c], bins=30)
        ax.hist(normal, bins=bins, density=True, color=style.NORMAL, alpha=0.55, label="Normal")
        ax.hist(failed, bins=bins, density=True, color=style.FAILED, alpha=0.65, label="Failed")
        ax.set_title(label(c), fontsize=9)
        ax.set_yticks([])
        # Standardised mean difference: a scale-free measure of separation.
        pooled = np.sqrt((normal.var() + failed.var()) / 2)
        smd = (failed.mean() - normal.mean()) / pooled if pooled > 0 else 0.0
        ax.text(0.02, 0.95, f"SMD {smd:+.2f}", transform=ax.transAxes, fontsize=7.5, va="top",
                color=style.INK_2)
        n_counts, _ = np.histogram(normal, bins=bins)
        f_counts, _ = np.histogram(failed, bins=bins)
        result[c] = {
            "bins": [round(float(b), 3) for b in bins],
            "normal": n_counts.tolist(),
            "failed": f_counts.tolist(),
            "normal_mean": round(float(normal.mean()), 3),
            "failed_mean": round(float(failed.mean()), 3),
            "normal_median": round(float(normal.median()), 3),
            "failed_median": round(float(failed.median()), 3),
            "standardized_mean_difference": round(float(smd), 3),
        }
    axes.flat[0].legend(loc="upper right")
    fig.suptitle("Which operating conditions differ between normal and failed records? (density)",
                 fontsize=11, color=style.INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "02_distributions_by_failure.png")
    plt.close(fig)
    return result


def fig_rate_by_bin(df: pd.DataFrame, out: Path) -> dict:
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharey=True)
    base = df[config.TARGET].mean()
    result = {}
    for ax, c in zip(axes.flat, FEATS, strict=True):
        q = pd.qcut(df[c], q=10, duplicates="drop")
        g = df.groupby(q, observed=True)[config.TARGET].agg(["mean", "size", "sum"])
        mids = [iv.mid for iv in g.index]
        ax.plot(mids, g["mean"] * 100, color=style.FAILED, lw=2, marker="o", ms=4)
        ax.axhline(base * 100, color=style.MUTED, lw=0.8)
        ax.set_title(label(c), fontsize=9)
        result[c] = [
            {"low": round(float(iv.left), 3), "high": round(float(iv.right), 3), "rate": round(float(r["mean"]), 4),
             "n": int(r["size"]), "failures": int(r["sum"])}
            for iv, (_, r) in zip(g.index, g.iterrows(), strict=True)
        ]
    for ax in axes[:, 0]:
        ax.set_ylabel("Failure rate in decile (%)")
    axes.flat[0].text(0.02, 0.95, f"grey line = overall {base:.1%}", transform=axes.flat[0].transAxes,
                      fontsize=7.5, va="top", color=style.INK_2)
    fig.suptitle("How does the failure rate change across each variable's range? (training-split deciles)",
                 fontsize=11, color=style.INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "03_failure_rate_by_decile.png")
    plt.close(fig)
    return result


MODE_VIEWS = {
    "twf": ("tool_wear_min", "torque_nm"),
    "hdf": ("rotational_speed_rpm", "temp_diff_k"),
    "pwf": ("rotational_speed_rpm", "torque_nm"),
    "osf": ("tool_wear_min", "torque_nm"),
}


def fig_mode_maps(df: pd.DataFrame, out: Path) -> dict:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    result = {}
    for ax, (mode, (x, y)) in zip(axes, MODE_VIEWS.items(), strict=True):
        rest = df[df[mode] == 0]
        hit = df[df[mode] == 1]
        ax.scatter(rest[x], rest[y], s=3, color=style.AXIS, alpha=0.5, linewidths=0, label="Other records")
        ax.scatter(hit[x], hit[y], s=16, color=style.FAILED, edgecolors=style.SURFACE, linewidths=0.6,
                   label=config.FAILURE_MODE_NAMES[mode])
        ax.set_xlabel(label(x))
        ax.set_ylabel(label(y))
        ax.set_title(f"{config.FAILURE_MODE_NAMES[mode]} (n={len(hit)})", fontsize=9.5)
        result[mode] = {
            "x": x,
            "y": y,
            "n": int(len(hit)),
            "x_range": [round(float(hit[x].min()), 2), round(float(hit[x].max()), 2)] if len(hit) else None,
            "y_range": [round(float(hit[y].min()), 2), round(float(hit[y].max()), 2)] if len(hit) else None,
        }
    fig.suptitle("Where in operating space does each failure mode occur? (training split)",
                 fontsize=11, color=style.INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "04_failure_mode_regions.png")
    plt.close(fig)
    return result


def fig_type(df: pd.DataFrame, out: Path) -> dict:
    g = df.groupby("type")[config.TARGET].agg(["mean", "size", "sum"]).reindex(config.PRODUCT_TYPES)
    # Wilson 95% interval: an honest error bar for small failure counts.
    z = 1.96
    n, p = g["size"], g["mean"]
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(g.index, p * 100, color=style.NORMAL, width=0.5)
    ax.errorbar(g.index, centre * 100, yerr=half * 100, fmt="none", ecolor=style.INK_2, capsize=4, lw=1)
    for i, t in enumerate(g.index):
        ax.text(i + 0.28, p[t] * 100, f"{p[t]:.1%}\n{int(g.loc[t, 'sum'])}/{int(n[t])}", fontsize=7.5,
                color=style.INK_2, va="center")
    ax.set_ylabel("Failure rate (%)")
    ax.set_title("Failure rate by product type (95% Wilson interval)")
    fig.tight_layout()
    fig.savefig(out / "05_failure_rate_by_type.png")
    plt.close(fig)
    return {
        t: {"rate": round(float(p[t]), 4), "n": int(n[t]), "failures": int(g.loc[t, "sum"]),
            "ci_low": round(float(centre[t] - half[t]), 4), "ci_high": round(float(centre[t] + half[t]), 4)}
        for t in g.index
    }


def fig_corr(df: pd.DataFrame, out: Path) -> dict:
    cols = [*FEATS, config.TARGET]
    corr = df[cols].corr(method="spearman")
    cmap = LinearSegmentedColormap.from_list(
        "div", [style.DIVERGING_NEG, style.DIVERGING_MID, style.DIVERGING_POS]
    )
    fig, ax = plt.subplots(figsize=(7.4, 6))
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    names = [config.FEATURE_LABELS.get(c, "Machine failure") for c in cols]
    ax.set_xticks(range(len(cols)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(cols)), names)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else style.INK_2)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Spearman ρ")
    ax.set_title("Which variables move together? (Spearman, training split)")
    fig.tight_layout()
    fig.savefig(out / "06_correlation.png")
    plt.close(fig)
    return {"columns": cols, "matrix": [[round(float(v), 3) for v in row] for row in corr.values]}


def outlier_check(df: pd.DataFrame) -> dict:
    """Are IQR 'outliers' errors or a real operating regime? Compare their failure rate."""
    out = {}
    for c in ("rotational_speed_rpm", "torque_nm"):
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)
        out[c] = {
            "n_outliers": int(mask.sum()),
            "failure_rate_outliers": round(float(df.loc[mask, config.TARGET].mean()), 4) if mask.any() else None,
            "failure_rate_rest": round(float(df.loc[~mask, config.TARGET].mean()), 4),
            "power_w_range_outliers": [round(float(df.loc[mask, "power_w"].min()), 1),
                                       round(float(df.loc[mask, "power_w"].max()), 1)] if mask.any() else None,
        }
    out["rpm_torque_spearman"] = round(float(df["rotational_speed_rpm"].corr(df["torque_nm"], method="spearman")), 3)
    return out


def scatter_sample(df: pd.DataFrame, seed: int) -> list[dict]:
    """Every failure plus a fixed random sample of normal rows, for the dashboard scatter."""
    failed = df[df[config.TARGET] == 1]
    normal = df[df[config.TARGET] == 0].sample(n=1200, random_state=seed)
    s = pd.concat([normal, failed])
    return [
        {"rpm": int(r.rotational_speed_rpm), "torque": round(float(r.torque_nm), 1),
         "wear": int(r.tool_wear_min), "temp_diff": round(float(r.temp_diff_k), 2),
         "failed": int(r.machine_failure)}
        for r in s.itertuples()
    ]


def main() -> None:
    style.apply()
    out = config.FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    train = add_engineered(load_split("train"))
    summary = {
        "computed_on": "training split only",
        "n_rows": int(len(train)),
        "imbalance": fig_imbalance(train, out),
        "distributions": fig_distributions(train, out),
        "rate_by_decile": fig_rate_by_bin(train, out),
        "failure_mode_regions": fig_mode_maps(train, out),
        "failure_rate_by_type": fig_type(train, out),
        "correlation": fig_corr(train, out),
        "outlier_check": outlier_check(train),
        "scatter_sample": scatter_sample(train, config.RANDOM_SEED),
    }
    (config.REPORTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=1))
    print("figures:", sorted(p.name for p in out.glob("*.png")))
    print("SMD:", {c: v["standardized_mean_difference"] for c, v in summary["distributions"].items()})
    print("type:", summary["failure_rate_by_type"])
    print("outliers:", summary["outlier_check"])
    print("mode regions:", {m: (v["x_range"], v["y_range"]) for m, v in summary["failure_mode_regions"].items()})


if __name__ == "__main__":
    main()
