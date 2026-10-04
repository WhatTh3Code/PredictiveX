"""Builds and executes notebooks/01-04 (run after the pipeline).

    python scripts/build_notebooks.py

Each code cell is executed in a shared namespace and its stdout / last-expression value /
matplotlib figures are stored as real notebook outputs, so the notebooks open with results
in Jupyter, VS Code or GitHub. The notebooks read the pipeline's modules and reports; they do
not re-implement or re-tune anything.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SETUP = """import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
pd.set_option("display.width", 140); pd.set_option("display.max_columns", 20)
from ml import config
report = lambda name: json.loads((config.REPORTS_DIR / name).read_text())"""

NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "01_eda.ipynb": [
        ("md", "# 01 · Exploratory data analysis\nAI4I 2020 predictive-maintenance data (synthetic). Quality checks use the full raw file; "
               "distribution analysis uses the **training split only** so that nothing is learned from validation/test rows."),
        ("code", SETUP),
        ("code", "from ml.preprocessing.data import load_raw, quality_report\nraw = load_raw()\nraw.head()"),
        ("code", "q = quality_report(raw)\nkeys = ['n_rows', 'missing_values_total', 'duplicate_rows', 'failure_rate',\n"
                 "        'imbalance_ratio', 'failure_mode_counts', 'type_counts', 'label_consistency']\n{k: q[k] for k in keys}"),
        ("md", "**Label consistency.** 9 failures carry no mode flag and 18 RNF flags have no failure. "
               "The official `Machine failure` column is kept as the target; nothing is relabelled."),
        ("code", "pd.DataFrame(q['outliers_iqr']).T"),
        ("md", "## Are the outliers errors?"),
        ("code", "eda = report('eda_summary.json')\neda['outlier_check']"),
        ("md", "Torque outliers fail ~97% of the time: they are a real high-stress regime and are kept."),
        ("md", "## Which conditions differ between normal and failed machines?"),
        ("code", "d = pd.DataFrame({k: {'normal_median': v['normal_median'], 'failed_median': v['failed_median'],\n"
                 "                   'SMD': v['standardized_mean_difference']} for k, v in eda['distributions'].items()}).T\n"
                 "d.sort_values('SMD', key=abs, ascending=False)"),
        ("code", "from IPython.display import Image\nImage(str(config.FIGURES_DIR / '02_distributions_by_failure.png'))"),
        ("md", "## Where does each failure mode occur?"),
        ("code", "Image(str(config.FIGURES_DIR / '04_failure_mode_regions.png'))"),
        ("md", "Each mode occupies a distinct region: HDF at low speed with a small temperature gap, PWF at both power extremes, "
               "OSF at high wear x torque, TWF at high tool wear. This motivates the engineered features in notebook 02."),
        ("code", "pd.DataFrame(eda['failure_rate_by_type']).T"),
    ],
    "02_feature_engineering.ipynb": [
        ("md", "# 02 · Feature engineering\nThree domain features, each tested by 5-fold CV ablation on the training split."),
        ("code", SETUP),
        ("code", "from ml.features.engineering import ENGINEERED_FEATURES, FeatureBuilder\npd.DataFrame(ENGINEERED_FEATURES).T"),
        ("code", "from ml.preprocessing.data import load_split, feature_target\nX, y = feature_target(load_split('train'))\n"
                 "FeatureBuilder().transform(X).head()"),
        ("md", "## Leakage check\nIdentifiers and the five failure-mode flags never reach the model."),
        ("code", "forbidden = {config.TARGET, *config.FAILURE_MODES, *config.IDENTIFIER_COLUMNS}\n"
                 "assert forbidden.isdisjoint(FeatureBuilder().feature_names)\nFeatureBuilder().feature_names"),
        ("md", "## Ablation (CV PR-AUC, default hyper-parameters)"),
        ("code", "abl = report('feature_ablation.json')\n"
                 "pd.DataFrame({m: {s: v['cv_pr_auc_mean'] for s, v in r.items()} for m, r in abl['results'].items()})"),
        ("md", "## Marginal value for the selected model (train CV and validation)"),
        ("code", "pd.DataFrame(abl['selected_model_marginal']).T[['cv_pr_auc_mean','cv_pr_auc_std','validation_pr_auc']]"),
    ],
    "03_model_comparison.ipynb": [
        ("md", "# 03 · Model comparison\nEach family tuned by 5-fold CV on train (PR-AUC); compared on validation; "
               "thresholds from out-of-fold training predictions."),
        ("code", SETUP),
        ("code", "mc = report('model_comparison.json')\nrows = []\nfor r in mc['models']:\n"
                 "    v = r['validation']\n    rows.append({'model': r['display_name'], 'cv_pr_auc': r['tuning']['cv_pr_auc_mean'], "
                 "'val_pr_auc': v['pr_auc'], 'val_roc_auc': v['roc_auc'], 'val_precision': v['precision'], 'val_recall': v['recall'], "
                 "'val_f1': v['f1'], 'threshold': r['oof_threshold']})\npd.DataFrame(rows).set_index('model')"),
        ("code", "print('Selected:', mc['selected'], '| rule:', mc['selection_metric'])\n"
                 "{r['display_name']: r['tuning']['best_params'] for r in mc['models']}"),
        ("code", "from IPython.display import Image\nImage(str(config.FIGURES_DIR / '07_model_comparison.png'))"),
        ("md", "## Threshold choice (validation)"),
        ("code", "m = report('metrics.json')\nm['thresholds']['policy']"),
        ("code", "pd.DataFrame([{k: v for k, v in r.items() if k != 'confusion_matrix'} for r in m['thresholds']['validation_tradeoff']])"
                 ".set_index('threshold')[['precision','recall','f1','f2']]"),
        ("code", "Image(str(config.FIGURES_DIR / '12_validation_threshold_tradeoff.png'))"),
    ],
    "04_final_evaluation.ipynb": [
        ("md", "# 04 · Final evaluation on the unseen test split\nThe test split is used once. "
               "This notebook **recomputes** the headline metrics from the saved model to show they are not hand-entered."),
        ("code", SETUP),
        ("code", "from ml.inference import Predictor\nfrom ml.preprocessing.data import load_split, feature_target\n"
                 "from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix\n"
                 "pred = Predictor.load()\nX, y = feature_target(load_split('test'))\n"
                 "p = pred.bundle['model'].predict_proba(pred.bundle['feature_builder'].transform(X))[:, 1]\n"
                 "thr = pred.policy.high_threshold\n"
                 "print('PR-AUC ', round(average_precision_score(y, p), 4))\nprint('ROC-AUC', round(roc_auc_score(y, p), 4))\n"
                 "print('threshold', thr)\nconfusion_matrix(y, p >= thr)"),
        ("code", "t = report('metrics.json')['test']\n{k: t[k] for k in ['pr_auc','roc_auc','precision','recall','f1','brier']} | "
                 "{'bootstrap_95ci': t['bootstrap_95ci']}"),
        ("code", "pd.DataFrame(t['risk_bands']).T"),
        ("code", "from IPython.display import Image\nImage(str(config.FIGURES_DIR / '08_test_pr_roc.png'))"),
        ("code", "Image(str(config.FIGURES_DIR / '09_test_confusion_matrix.png'))"),
        ("md", "## Robustness across five random splits"),
        ("code", "r = report('metrics.json')['robustness']\npd.DataFrame(r['per_seed']).set_index('seed')"),
        ("md", "## Explainability"),
        ("code", "e = report('explainability.json')\nprint(e['method'], '|', e['global_sample'])\n"
                 "pd.DataFrame(e['shap_global_importance_test']).set_index('feature')"),
        ("code", "Image(str(config.FIGURES_DIR / '10_feature_importance.png'))"),
        ("md", "## Prediction on a new, unseen record"),
        ("code", "new = {'type': 'L', 'air_temp_k': 301.2, 'process_temp_k': 310.4, 'rotational_speed_rpm': 1325, "
                 "'torque_nm': 61.0, 'tool_wear_min': 212}\nres = pred.predict([new])[0]\n"
                 "print(res['failure_probability'], res['risk_level'])\nprint(res['explanation'])"),
        ("md", "## Failure-mode model (conditional on failure)"),
        ("code", "fm = report('failure_mode.json')\nprint(fm['scope'])\npd.DataFrame(fm['test']['per_class']).T"),
    ],
}


def _display_data(value) -> dict | None:
    if value is None:
        return None
    try:
        from IPython.display import Image  # type: ignore
    except Exception:  # pragma: no cover
        Image = None  # noqa: N806
    if Image is not None and isinstance(value, Image):
        return {"image/png": base64.b64encode(value.data).decode(), "text/plain": "<Image>"}
    if hasattr(value, "_repr_html_") and hasattr(value, "to_string"):
        return {"text/html": value._repr_html_(), "text/plain": value.to_string()}
    return {"text/plain": repr(value)}


def run_cell(src: str, ns: dict) -> list[dict]:
    outputs: list[dict] = []
    buf = io.StringIO()
    tree = ast.parse(src)
    last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    body = tree.body[:-1] if last else tree.body
    value = None
    with contextlib.redirect_stdout(buf):
        exec(compile(ast.Module(body=body, type_ignores=[]), "<cell>", "exec"), ns)
        if last:
            value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), ns)
    if buf.getvalue():
        outputs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue()})
    for num in plt.get_fignums():
        fig = plt.figure(num)
        b = io.BytesIO()
        fig.savefig(b, format="png", bbox_inches="tight")
        outputs.append({"output_type": "display_data", "metadata": {},
                        "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": "<Figure>"}})
        plt.close(fig)
    data = _display_data(value)
    if data:
        outputs.append({"output_type": "execute_result", "metadata": {}, "data": data, "execution_count": None})
    return outputs


def main() -> None:
    try:
        import IPython  # noqa: F401
    except Exception:
        # Minimal stand-in so `from IPython.display import Image` works where IPython is absent.
        import types

        class Image:  # noqa: D401
            def __init__(self, filename: str) -> None:
                self.data = Path(filename).read_bytes()

        mod = types.ModuleType("IPython")
        disp = types.ModuleType("IPython.display")
        disp.Image = Image
        mod.display = disp
        sys.modules["IPython"] = mod
        sys.modules["IPython.display"] = disp

    out_dir = ROOT / "notebooks"
    out_dir.mkdir(exist_ok=True)
    import os

    os.chdir(out_dir)
    for name, cells in NOTEBOOKS.items():
        ns: dict = {}
        nb_cells = []
        count = 0
        for kind, src in cells:
            if kind == "md":
                nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
                continue
            count += 1
            outs = run_cell(src, ns)
            for o in outs:
                if o["output_type"] == "execute_result":
                    o["execution_count"] = count
            nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": count, "source": src, "outputs": outs})
        nb = {
            "cells": nb_cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        (out_dir / name).write_text(json.dumps(nb, indent=1))
        print(f"executed {name}: {count} code cells")


if __name__ == "__main__":
    main()
