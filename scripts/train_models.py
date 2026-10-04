"""Phases 4-5 - feature evaluation, model experiments, selection and final evaluation.

    python scripts/train_models.py

Protocol (the test split is touched exactly once, in step 6):

1. Feature-set ablation      5-fold stratified CV on TRAIN, default hyper-parameters.
2. Hyper-parameter tuning    5-fold stratified CV on TRAIN (scoring: PR-AUC).
3. Model comparison          fit on TRAIN, scored on VALIDATION.
4. Model selection           highest VALIDATION PR-AUC.
5. Thresholds                out-of-fold TRAIN predictions -> decision + watch thresholds,
                             then checked on VALIDATION.
6. Final evaluation          TEST, once, with bootstrap 95% CIs and risk-band outcomes.
7. Robustness                the selected recipe re-run on 5 different random splits.
8. Failure-mode model        conditional classifier on failed records.
9. Explainability            permutation importance (validation) + SHAP (test).

Outputs: models/predictivex_model.joblib and reports/*.json + reports/figures/*.png.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)

from ml import config  # noqa: E402
from ml.evaluation import metrics as M  # noqa: E402
from ml.evaluation import plots  # noqa: E402
from ml.explainability import shap_values as SV  # noqa: E402
from ml.features.engineering import ENGINEERED_FEATURES, FeatureBuilder  # noqa: E402
from ml.models import candidates as C  # noqa: E402
from ml.models.failure_mode import train_failure_mode_model  # noqa: E402
from ml.models.risk import RiskPolicy  # noqa: E402
from ml.preprocessing.data import feature_target, file_sha256, load_raw, load_split, split_data  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

SEED = config.RANDOM_SEED
WATCH_RECALL = 0.95
SHAP_SAMPLE = 600
F_BETA = 2.0


def cv(seed: int = SEED) -> StratifiedKFold:
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def feature_sets() -> dict[str, list[str]]:
    eng = list(ENGINEERED_FEATURES)
    sets: dict[str, list[str]] = {"raw_only": []}
    for f in eng:
        sets[f"raw+{f}"] = [f]
    sets["raw+all_engineered"] = eng
    return sets


def ablation(train_raw: pd.DataFrame, y: np.ndarray, families: list[str]) -> tuple[dict, dict]:
    """Step 1: CV PR-AUC of every family on every feature set (default hyper-parameters)."""
    results: dict[str, dict] = {}
    chosen: dict[str, list[str]] = {}
    for fam in families:
        results[fam] = {}
        for set_name, eng in feature_sets().items():
            X = FeatureBuilder(engineered=eng).transform(train_raw)
            scores = cross_val_score(C.default_model(fam, SEED), X, y, cv=cv(), scoring="average_precision")
            results[fam][set_name] = {
                "cv_pr_auc_mean": round(float(scores.mean()), 4),
                "cv_pr_auc_std": round(float(scores.std()), 4),
                "engineered": eng,
            }
        best = max(results[fam], key=lambda s: results[fam][s]["cv_pr_auc_mean"])
        chosen[fam] = results[fam][best]["engineered"]
        log(f"ablation {fam}: " + ", ".join(f"{k}={v['cv_pr_auc_mean']:.3f}" for k, v in results[fam].items())
            + f" -> {best}")
    return results, chosen


def tune(fam: str, X: pd.DataFrame, y: np.ndarray, ratio: float) -> tuple[object, dict]:
    space = C.search_space(fam, ratio)
    n_combos = int(np.prod([len(v) for v in space.values()]))
    if n_combos <= 20:
        search = GridSearchCV(C.default_model(fam, SEED), space, cv=cv(), scoring="average_precision", n_jobs=1)
    else:
        search = RandomizedSearchCV(
            C.default_model(fam, SEED), space, n_iter=12, cv=cv(), scoring="average_precision",
            random_state=SEED, n_jobs=1,
        )
    search.fit(X, y)
    i = search.best_index_
    info = {
        "search": type(search).__name__,
        "candidates_evaluated": int(len(search.cv_results_["params"])),
        "best_params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in search.best_params_.items()},
        "cv_pr_auc_mean": round(float(search.cv_results_["mean_test_score"][i]), 4),
        "cv_pr_auc_std": round(float(search.cv_results_["std_test_score"][i]), 4),
    }
    return search.best_estimator_, info


def drop_one(fam: str, model: object, train_raw, y, val_raw, y_val, eng: list[str]) -> dict:
    """Marginal value of each engineered feature for the selected model (CV on train + validation)."""
    out = {}
    variants = {"selected_set": eng, "raw_only": []}
    for f in eng:
        variants[f"without_{f}"] = [e for e in eng if e != f]
    for f in ENGINEERED_FEATURES:
        if f not in eng:
            variants[f"with_{f}"] = [*eng, f]
    for name, feats in variants.items():
        fb = FeatureBuilder(engineered=feats)
        Xtr, Xva = fb.transform(train_raw), fb.transform(val_raw)
        scores = cross_val_score(clone(model), Xtr, y, cv=cv(), scoring="average_precision")
        m = clone(model).fit(Xtr, y)
        out[name] = {
            "engineered": feats,
            "cv_pr_auc_mean": round(float(scores.mean()), 4),
            "cv_pr_auc_std": round(float(scores.std()), 4),
            "validation_pr_auc": round(float(average_precision_score(y_val, m.predict_proba(Xva)[:, 1])), 4),
        }
    return out


def risk_band_outcomes(y: np.ndarray, p: np.ndarray, policy: RiskPolicy) -> dict:
    bands = np.array([policy.classify(v) for v in p])
    out = {}
    for b in ("LOW", "MEDIUM", "HIGH"):
        mask = bands == b
        n, k = int(mask.sum()), int(y[mask].sum())
        out[b] = {
            "records": n,
            "actual_failures": k,
            "observed_failure_rate": round(k / n, 4) if n else None,
            "wilson_95ci": M.wilson_interval(k, n) if n else None,
            "share_of_all_failures": round(k / int(y.sum()), 4) if y.sum() else None,
        }
    return out


def robustness(fam: str, model: object, eng: list[str], seeds: list[int]) -> dict:
    """Step 7: re-run the selected recipe (same hyper-parameters) on different random splits."""
    df = load_raw()
    rows = []
    for s in seeds:
        sp = split_data(df, seed=s)
        fb = FeatureBuilder(engineered=eng)
        Xtr, ytr = feature_target(sp.train)
        Xte, yte = feature_target(sp.test)
        Xtr, Xte = fb.transform(Xtr), fb.transform(Xte)
        mdl = clone(model)
        if hasattr(mdl, "random_state"):
            mdl.set_params(random_state=s)
        oof = cross_val_predict(clone(mdl), Xtr, ytr, cv=cv(s), method="predict_proba")[:, 1]
        thr = M.best_fbeta_threshold(ytr.to_numpy(), oof, F_BETA)
        mdl.fit(Xtr, ytr)
        p = mdl.predict_proba(Xte)[:, 1]
        rows.append({"seed": s, **M.full_metrics(yte.to_numpy(), p, thr)})
    keys = ["pr_auc", "roc_auc", "precision", "recall", "f1"]
    return {
        "seeds": seeds,
        "per_seed": [{k: r[k] for k in ["seed", *keys, "threshold"]} for r in rows],
        "mean": {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys},
        "std": {k: round(float(np.std([r[k] for r in rows])), 4) for k in keys},
    }


def main() -> None:
    t0 = time.time()
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_split("train"), load_split("validation"), load_split("test")
    train_raw, y_train = feature_target(train_df)
    val_raw, y_val = feature_target(val_df)
    test_raw, y_test = feature_target(test_df)
    ytr, yva, yte = y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy()
    ratio = float((ytr == 0).sum() / (ytr == 1).sum())
    families = C.available_families()
    log(f"train={len(ytr)} ({ytr.sum()} failures) val={len(yva)} test={len(yte)}; families={families}")

    # 1. Feature ablation -----------------------------------------------------------------
    ablation_results, chosen_sets = ablation(train_raw, ytr, families)

    # 2-3. Tuning on train CV, comparison on validation ---------------------------------------
    comparison = []
    tuned: dict[str, tuple[object, FeatureBuilder]] = {}
    for fam in families:
        fb = FeatureBuilder(engineered=chosen_sets[fam])
        Xtr, Xva = fb.transform(train_raw), fb.transform(val_raw)
        t = time.time()
        model, info = tune(fam, Xtr, ytr, ratio)
        p_val = model.predict_proba(Xva)[:, 1]
        oof = cross_val_predict(clone(model), Xtr, ytr, cv=cv(), method="predict_proba")[:, 1]
        thr = M.best_fbeta_threshold(ytr, oof, F_BETA)
        row = {
            "model": fam,
            "display_name": C.DISPLAY_NAMES[fam],
            "features": fb.feature_names,
            "engineered": chosen_sets[fam],
            "tuning": info,
            "oof_threshold": round(thr, 4),
            "validation": M.full_metrics(yva, p_val, thr),
            "validation_at_0_5": M.threshold_metrics(yva, p_val, 0.5),
            "train_seconds": round(time.time() - t, 1),
        }
        comparison.append(row)
        tuned[fam] = (model, fb)
        v = row["validation"]
        log(f"{fam}: cv={info['cv_pr_auc_mean']:.3f} val PR-AUC={v['pr_auc']:.3f} ROC={v['roc_auc']:.3f} "
            f"P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} @thr={thr:.3f} {info['best_params']}")

    # 4. Selection ---------------------------------------------------------------------------
    best_row = max(comparison, key=lambda r: r["validation"]["pr_auc"])
    fam = best_row["model"]
    model, fb = tuned[fam]
    log(f"SELECTED: {fam} (validation PR-AUC {best_row['validation']['pr_auc']})")

    # 5. Thresholds from out-of-fold train predictions ---------------------------------------
    Xtr, Xva, Xte = fb.transform(train_raw), fb.transform(val_raw), fb.transform(test_raw)
    oof = cross_val_predict(clone(model), Xtr, ytr, cv=cv(), method="predict_proba")[:, 1]
    high = M.best_fbeta_threshold(ytr, oof, F_BETA)
    medium = min(M.threshold_for_recall(ytr, oof, WATCH_RECALL), high)
    f1_thr = M.best_fbeta_threshold(ytr, oof, 1.0)
    policy = RiskPolicy(
        medium_threshold=round(medium, 4),
        high_threshold=round(high, 4),
        method=(f"HIGH = F{F_BETA:g}-optimal threshold on out-of-fold training predictions; "
                f"MEDIUM = highest threshold keeping out-of-fold training recall >= {WATCH_RECALL:.0%}"),
    )
    p_val = model.predict_proba(Xva)[:, 1]
    threshold_report = {
        "policy": policy.to_dict(),
        "f1_optimal_threshold_oof": round(f1_thr, 4),
        "oof_train": M.full_metrics(ytr, oof, high),
        "validation_at_high": M.threshold_metrics(yva, p_val, high),
        "validation_at_medium": M.threshold_metrics(yva, p_val, medium),
        "validation_at_f1_optimal": M.threshold_metrics(yva, p_val, f1_thr),
        "validation_at_0_5": M.threshold_metrics(yva, p_val, 0.5),
        "validation_risk_bands": risk_band_outcomes(yva, p_val, policy),
        "validation_tradeoff": [
            {"threshold": t, **{k: v for k, v in M.threshold_metrics(yva, p_val, t).items() if k != "threshold"}}
            for t in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ],
    }
    log(f"thresholds: medium={medium:.4f} high={high:.4f} (F1-opt {f1_thr:.4f})")

    # 6. Final test evaluation (once) ----------------------------------------------------------
    p_test = model.predict_proba(Xte)[:, 1]
    test_report = {
        "note": "Test split untouched until this step; model fit on training split only.",
        **M.full_metrics(yte, p_test, high),
        "at_medium_threshold": M.threshold_metrics(yte, p_test, medium),
        "at_0_5": M.threshold_metrics(yte, p_test, 0.5),
        "bootstrap_95ci": M.bootstrap_ci(yte, p_test, high, n_boot=1000, seed=SEED),
        "risk_bands": risk_band_outcomes(yte, p_test, policy),
        "calibration": M.calibration_table(yte, p_test),
        "curves": M.curves(yte, p_test),
        "missed_failures_by_mode": {
            m: int(((p_test < high) & (yte == 1) & (test_df[m].to_numpy() == 1)).sum()) for m in config.FAILURE_MODES
        },
        "failures_by_mode": {m: int(((yte == 1) & (test_df[m].to_numpy() == 1)).sum()) for m in config.FAILURE_MODES},
    }
    # Per-model test numbers are reported for transparency only; they did not drive selection.
    for row in comparison:
        mdl, fbr = tuned[row["model"]]
        pt = mdl.predict_proba(fbr.transform(test_raw))[:, 1]
        row["test"] = M.full_metrics(yte, pt, row["oof_threshold"])
    t = test_report
    log(f"TEST: PR-AUC={t['pr_auc']} ROC-AUC={t['roc_auc']} P={t['precision']} R={t['recall']} F1={t['f1']} "
        f"CM={t['confusion_matrix']}")

    # Drop-one ablation for the selected model (train CV + validation only).
    marginal = drop_one(fam, model, train_raw, ytr, val_raw, yva, chosen_sets[fam])

    # 7. Robustness ------------------------------------------------------------------------------
    robust = robustness(fam, model, chosen_sets[fam], seeds=[7, 21, 42, 99, 2026])
    log(f"robustness mean={robust['mean']} std={robust['std']}")

    # 8. Failure-mode model ---------------------------------------------------------------------
    mode_bundle, mode_report = train_failure_mode_model(train_df, val_df, test_df, fb, SEED)
    log(f"failure-mode: {mode_report['selected']} val macroF1={mode_report['validation']['macro_f1']} "
        f"test macroF1={mode_report['test']['macro_f1']}")

    # 9. Explainability -----------------------------------------------------------------------------
    perm = permutation_importance(model, Xva, yva, scoring="average_precision", n_repeats=20,
                                  random_state=SEED, n_jobs=1)
    perm_imp = sorted(
        [{"feature": f, "label": config.FEATURE_LABELS[f], "mean": round(float(m), 4), "std": round(float(s), 4)}
         for f, m, s in zip(fb.feature_names, perm.importances_mean, perm.importances_std, strict=True)],
        key=lambda r: -r["mean"],
    )
    explainer = SV.make_explainer(model, Xtr)
    # Exact TreeSHAP is ~0.3 s/row for the forest in pure Python, so global importance uses a
    # fixed random sample of test rows (computed in parallel). Per-prediction SHAP is always exact.
    shap_idx = np.sort(np.random.default_rng(SEED).choice(len(Xte), size=min(SHAP_SAMPLE, len(Xte)), replace=False))
    Xs = Xte.iloc[shap_idx]
    chunks = np.array_split(np.arange(len(Xs)), 8)
    shap_test = np.vstack(joblib.Parallel(n_jobs=-1)(
        joblib.delayed(explainer.shap_values)(Xs.iloc[c]) for c in chunks))
    shap_imp = sorted(
        [{"feature": f, "label": config.FEATURE_LABELS[f], "mean_abs_shap": round(float(v), 4)}
         for f, v in zip(fb.feature_names, np.abs(shap_test).mean(axis=0), strict=True)],
        key=lambda r: -r["mean_abs_shap"],
    )
    additivity_err = float(np.max(np.abs(explainer.expected_value + shap_test.sum(axis=1) - explainer.margin(Xs))))
    shap_beeswarm = {
        "features": fb.feature_names,
        "sample": [
            {"shap": [round(float(v), 4) for v in shap_test[i]],
             "value": [round(float(v), 3) for v in Xs.iloc[i].to_numpy()],
             "failed": int(yte[shap_idx[i]])}
            for i in range(len(Xs))
        ],
    }
    log(f"SHAP additivity max error={additivity_err:.2e}; top={[r['feature'] for r in shap_imp[:4]]}")

    # Figures -----------------------------------------------------------------------------------------
    plots.model_comparison(comparison, config.FIGURES_DIR / "07_model_comparison.png")
    plots.pr_roc(yte, p_test, high, config.FIGURES_DIR / "08_test_pr_roc.png")
    plots.confusion(test_report["confusion_matrix"], high, config.FIGURES_DIR / "09_test_confusion_matrix.png")
    plots.importance(shap_imp, perm_imp, config.FIGURES_DIR / "10_feature_importance.png")
    plots.calibration(test_report["calibration"], config.FIGURES_DIR / "11_test_calibration.png")
    plots.threshold_tradeoff(yva, p_val, policy, config.FIGURES_DIR / "12_validation_threshold_tradeoff.png")

    # Training ranges for input-plausibility warnings at inference time.
    ranges = {c: {"min": float(train_raw[c].min()), "max": float(train_raw[c].max())} for c in config.NUMERIC_INPUTS}

    metadata = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_sha256": file_sha256(config.DATA_RAW),
        "random_seed": SEED,
        "model_family": fam,
        "model_display_name": C.DISPLAY_NAMES[fam],
        "hyperparameters": best_row["tuning"]["best_params"],
        "features": fb.feature_names,
        "engineered_features": chosen_sets[fam],
        "training_rows": int(len(ytr)),
        "training_failures": int(ytr.sum()),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": __import__("xgboost").__version__ if C.HAS_XGBOOST else None,
        },
    }
    bundle = {
        "model": model,
        "feature_builder": fb,
        "risk_policy": policy,
        "explainer": explainer,
        "failure_mode": mode_bundle,
        "training_ranges": ranges,
        "metadata": metadata,
    }
    joblib.dump(bundle, config.MODEL_BUNDLE_PATH, compress=3)

    def dump(name: str, obj: object) -> None:
        (config.REPORTS_DIR / name).write_text(json.dumps(obj, indent=1, default=str))

    dump("feature_ablation.json", {
        "protocol": "5-fold stratified CV on the training split, default hyper-parameters",
        "results": ablation_results,
        "chosen_per_model": chosen_sets,
        "selected_model_marginal": marginal,
        "definitions": ENGINEERED_FEATURES,
    })
    dump("model_comparison.json", {
        "selection_metric": "validation PR-AUC (average precision)",
        "selected": fam,
        "models": comparison,
    })
    dump("metrics.json", {
        "model": metadata,
        "thresholds": threshold_report,
        "test": test_report,
        "robustness": robust,
    })
    dump("explainability.json", {
        "method": explainer.description,
        "output_space": explainer.output_space,
        "expected_value": round(float(explainer.expected_value), 5),
        "global_sample": f"{len(Xs)} randomly sampled test rows (seed {SEED})",
        "shap_additivity_max_abs_error": additivity_err,
        "shap_global_importance_test": shap_imp,
        "permutation_importance_validation": perm_imp,
        "shap_beeswarm_test_sample": shap_beeswarm,
    })
    dump("failure_mode.json", mode_report)
    log(f"done in {time.time() - t0:.0f}s -> {config.MODEL_BUNDLE_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
