"""Draws docs/architecture.png (offline training pipeline + online serving path)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from ml import config  # noqa: E402
from ml.evaluation import style  # noqa: E402

INK, INK2, MUTED, RULE = "#1d2329", "#4a545e", "#7d8791", "#d3d9de"
OFFLINE, ONLINE, ACCENT = "#eef1f3", "#e8f0fb", "#2a78d6"

W, H = 14, 9.2


def box(ax, x, y, w, h, title, sub="", fill="white", edge=RULE, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fill, edgecolor=edge, linewidth=1.6 if bold else 1.0))
    ax.text(x + 0.15, y + h - 0.2, title, fontsize=10.5, fontweight="bold", color=INK, va="top")
    if sub:
        ax.text(x + 0.15, y + h - 0.5, sub, fontsize=8.2, color=INK2, va="top", linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, a, b, color=INK2, style_="-|>", rad=0.0, lw=1.3):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style_, mutation_scale=12, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def main() -> None:
    style.apply()
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.3, H - 0.35, "PredictiveX architecture", fontsize=16, fontweight="bold", color=INK, va="top")
    ax.text(0.3, H - 0.8, "Training happens offline; the API only loads the saved bundle and runs inference.",
            fontsize=9.5, color=INK2, va="top")

    # Lanes
    ax.add_patch(FancyBboxPatch((0.25, 0.3), 5.5, H - 1.5, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=OFFLINE, edgecolor="none"))
    ax.text(0.45, H - 1.45, "Offline: reproducible pipeline (scripts/)", fontsize=9.5, color=MUTED, va="top")
    ax.add_patch(FancyBboxPatch((6.15, 0.3), 7.6, H - 1.5, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=ONLINE, edgecolor="none"))
    ax.text(6.35, H - 1.45, "Online: serving (backend/ + frontend/)", fontsize=9.5, color=MUTED, va="top")

    bw, bh, x0 = 5.0, 0.98, 0.5
    steps = [
        ("Dataset", "AI4I 2020 (UCI 601), 10,000 synthetic records\nchecksum-verified, never modified"),
        ("Data pipeline", "schema, missing, duplicates, outliers, leakage control\nstratified 60/20/20 split, seed 42"),
        ("Feature engineering", "temperature gap, mechanical power, tool strain\nkept only if CV ablation improves PR-AUC"),
        ("Model training", "LogReg, Random Forest, HistGB, XGBoost\n5-fold CV tuning on train (PR-AUC)"),
        ("Model evaluation", "select on validation; thresholds from OOF train\ntest evaluated once + bootstrap + 5 re-splits"),
        ("Saved model bundle", "features + model + risk policy + SHAP explainer\n+ failure-mode model (models/*.joblib)"),
    ]
    ys = []
    top = H - 2.0
    for i, (t, s) in enumerate(steps):
        y = top - i * 1.16 - bh
        box(ax, x0, y, bw, bh, t, s, bold=(t == "Saved model bundle"), edge=ACCENT if t == "Saved model bundle" else RULE)
        ys.append(y)
    for i in range(len(ys) - 1):
        arrow(ax, (x0 + bw / 2, ys[i]), (x0 + bw / 2, ys[i + 1] + bh))

    # Serving side
    api = box(ax, 6.5, 5.55, 3.1, 1.25, "FastAPI", "Pydantic validation, CORS,\n/predict  /simulate  /metrics ...", bold=True)
    engine = box(ax, 6.5, 3.55, 3.1, 1.25, "Prediction engine", "validate → features →\nmodel → probability → risk")
    shap = box(ax, 10.2, 3.55, 3.2, 1.25, "SHAP (exact TreeSHAP)", "explains each prediction:\nper-feature contributions")
    expl = box(ax, 10.2, 1.55, 3.2, 1.25, "Explanation", "plain-language summary,\nnon-causal wording")
    ui = box(ax, 10.2, 5.55, 3.2, 1.25, "React dashboard", "TypeScript, esbuild;\nserved by the API")
    user = box(ax, 10.2, 7.2, 3.2, 0.62, "User", fill="white")
    whatif = box(ax, 6.5, 1.55, 3.1, 1.25, "What-if simulator", "base machine + changed inputs\n→ before / after")

    # bundle -> API
    arrow(ax, (x0 + bw, ys[-1] + bh / 2), (api[0], api[1] + 0.35), color=ACCENT, rad=-0.25, lw=1.8)
    ax.text(5.6, 2.6, "loaded once\nat start-up", fontsize=8, color=ACCENT, ha="left")
    arrow(ax, (api[0] + api[2] / 2, api[1]), (engine[0] + engine[2] / 2, engine[1] + engine[3]))
    arrow(ax, (api[0] + api[2], api[1] + 0.62), (ui[0], ui[1] + 0.62), style_="<|-|>")
    arrow(ax, (ui[0] + ui[2] / 2, ui[1] + ui[3]), (user[0] + user[2] / 2, user[1]), style_="<|-|>")
    arrow(ax, (engine[0] + engine[2], engine[1] + 0.62), (shap[0], shap[1] + 0.62), color=INK)
    arrow(ax, (shap[0] + shap[2] / 2, shap[1]), (expl[0] + expl[2] / 2, expl[1] + expl[3]), color=INK)
    # User -> What-if -> engine
    arrow(ax, (user[0] + user[2], user[1] + 0.31), (13.75, user[1] + 0.31), style_="-")
    arrow(ax, (13.75, user[1] + 0.31), (13.75, 1.2), style_="-")
    arrow(ax, (13.75, 1.2), (whatif[0] + whatif[2] / 2, 1.2), style_="-")
    arrow(ax, (whatif[0] + whatif[2] / 2, 1.2), (whatif[0] + whatif[2] / 2, whatif[1]))
    arrow(ax, (whatif[0] + whatif[2] / 2, whatif[1] + whatif[3]), (engine[0] + engine[2] / 2, engine[1]), color=INK)
    ax.text(11.0, 0.95, "User → What-If → Prediction engine", fontsize=8, color=INK2)

    out = config.ROOT / "docs" / "architecture.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
