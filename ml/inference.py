"""Inference pipeline: input -> validation -> features -> model -> probability -> risk -> explanation.

Everything comes from the saved bundle produced by `scripts/train_models.py`; nothing is
retrained or hard-coded here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml import config
from ml.features.engineering import ENGINEERED_FEATURES
from ml.models.risk import RiskPolicy

INPUT_LABELS = {
    "type": "Product type",
    "air_temp_k": "Air temperature",
    "process_temp_k": "Process temperature",
    "rotational_speed_rpm": "Rotational speed",
    "torque_nm": "Torque",
    "tool_wear_min": "Tool wear",
}


class ModelUnavailableError(RuntimeError):
    pass


def _fmt_value(feature: str, value: float) -> str:
    if feature == "type_ordinal":
        return ["L", "M", "H"][int(round(value))]
    unit = config.FEATURE_UNITS.get(feature, "")
    if feature in ("power_w", "strain_nm_min"):
        text = f"{value:,.0f}"
    elif feature in ("rotational_speed_rpm", "tool_wear_min"):
        text = f"{value:.0f}"
    else:
        text = f"{value:.1f}"
    return f"{text} {unit}".strip()


@dataclass
class Predictor:
    bundle: dict[str, Any]
    reports_dir: Path = config.REPORTS_DIR

    @classmethod
    def load(cls, path: Path = config.MODEL_BUNDLE_PATH) -> Predictor:
        if not Path(path).exists():
            raise ModelUnavailableError(
                f"Model bundle not found at {path}. Run `python scripts/train_models.py`."
            )
        try:
            bundle = joblib.load(path)
        except Exception as exc:  # corrupted / incompatible file
            raise ModelUnavailableError(f"Model bundle could not be loaded: {type(exc).__name__}") from exc
        required = {"model", "feature_builder", "risk_policy", "explainer", "metadata", "training_ranges"}
        if missing := required - set(bundle):
            raise ModelUnavailableError(f"Model bundle is incomplete (missing {sorted(missing)})")
        return cls(bundle=bundle)

    # ------------------------------------------------------------------------------------------
    @property
    def policy(self) -> RiskPolicy:
        return self.bundle["risk_policy"]

    @property
    def metadata(self) -> dict:
        return self.bundle["metadata"]

    @property
    def feature_names(self) -> list[str]:
        return self.bundle["feature_builder"].feature_names

    def _band_outcomes(self) -> dict:
        path = self.reports_dir / "metrics.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text())["test"]["risk_bands"]

    def range_warnings(self, record: dict) -> list[str]:
        out = []
        for col, rng in self.bundle["training_ranges"].items():
            v = record[col]
            if v < rng["min"] or v > rng["max"]:
                out.append(
                    f"{INPUT_LABELS[col]} = {v:g} is outside the training range "
                    f"[{rng['min']:g}, {rng['max']:g}]; the prediction is an extrapolation and less reliable."
                )
        if record["process_temp_k"] < record["air_temp_k"]:
            out.append("Process temperature is below air temperature, which never occurs in the training data.")
        return out

    # ------------------------------------------------------------------------------------------
    def predict(self, records: list[dict], policy: RiskPolicy | None = None, explain: bool = True) -> list[dict]:
        policy = policy or self.policy
        raw = pd.DataFrame(records)[config.RAW_INPUTS]
        X = self.bundle["feature_builder"].transform(raw)
        proba = self.bundle["model"].predict_proba(X)[:, 1]
        shap = self.bundle["explainer"].shap_values(X) if explain else None
        modes = self._failure_modes(X)
        bands = self._band_outcomes()
        results = []
        for i, rec in enumerate(records):
            p = float(proba[i])
            risk = policy.classify(p)
            res: dict[str, Any] = {
                "failure_probability": round(p, 4),
                "risk_level": risk,
                "predicted_failure": bool(p >= policy.high_threshold),
                "thresholds": {"medium": policy.medium_threshold, "high": policy.high_threshold},
                "risk_band_evidence": self._band_evidence(bands.get(risk), risk, policy),
                "failure_mode": modes[i] if modes else None,
                "warnings": self.range_warnings(rec),
                "engineered_features": {
                    f: round(float(X.iloc[i][f]), 3) for f in self.bundle["feature_builder"].engineered
                },
            }
            if shap is not None:
                contribs = self._contributions(X.iloc[i], shap[i])
                res["contributions"] = contribs
                res["explanation"] = self._explanation(p, risk, contribs)
                res["shap_base_value"] = round(float(self.bundle["explainer"].expected_value), 4)
                res["shap_output_space"] = self.bundle["explainer"].output_space
            results.append(res)
        return results

    def _band_evidence(self, band: dict | None, risk: str, policy: RiskPolicy) -> dict | None:
        default = (policy.medium_threshold, policy.high_threshold) == (
            self.policy.medium_threshold, self.policy.high_threshold)
        if not band or not default:
            return None
        return {
            "description": f"Observed failure rate among held-out test records the model placed in {risk}",
            "test_records_in_band": band["records"],
            "test_failures_in_band": band["actual_failures"],
            "observed_failure_rate": band["observed_failure_rate"],
            "wilson_95ci": band["wilson_95ci"],
        }

    def _contributions(self, x: pd.Series, shap_row: np.ndarray) -> list[dict]:
        rows = []
        for f, v in zip(self.feature_names, shap_row, strict=True):
            rows.append(
                {
                    "feature": f,
                    "label": config.FEATURE_LABELS[f],
                    "value": round(float(x[f]), 4),
                    "display_value": _fmt_value(f, float(x[f])),
                    "shap": round(float(v), 4),
                    "direction": "increases" if v > 0 else "decreases" if v < 0 else "neutral",
                    "formula": ENGINEERED_FEATURES.get(f, {}).get("formula"),
                }
            )
        rows.sort(key=lambda r: -abs(r["shap"]))
        return rows

    @staticmethod
    def _explanation(p: float, risk: str, contribs: list[dict]) -> str:
        level = {"HIGH": "a high", "MEDIUM": "an elevated", "LOW": "a low"}[risk]
        ups = [c for c in contribs[:4] if c["direction"] == "increases"][:2]
        downs = [c for c in contribs[:4] if c["direction"] == "decreases"][:2]

        def names(cs: list[dict]) -> str:
            parts = [f"{c['label'].lower()} ({c['display_value']})" for c in cs]
            return " and ".join(parts)

        shown = "below 0.1%" if p < 0.001 else f"of {p:.1%}"
        text = f"The model estimates {level} failure probability {shown} ({risk} risk)."
        if ups:
            text += f" The factors contributing most toward a higher prediction are {names(ups)}."
        if downs:
            lead = names(downs)
            if ups:
                verb = "pulls" if len(downs) == 1 else "pull"
                text += f" {lead[0].upper() + lead[1:]} {verb} the prediction lower."
            else:
                one = len(downs) == 1
                noun, be, verb = ("input", "is", "pulls") if one else ("inputs", "are", "pull")
                text += f" The {noun} contributing most to this estimate {be} {lead}, which {verb} the prediction lower."
        text += (" These are associations the model has learned from historical (synthetic) data, "
                 "not proof of cause.")
        return text

    def _failure_modes(self, X: pd.DataFrame) -> list[dict] | None:
        fm = self.bundle.get("failure_mode")
        if not fm:
            return None
        proba = fm["model"].predict_proba(X)
        out = []
        for row in proba:
            ranked = sorted(zip(fm["classes"], row, strict=True), key=lambda t: -t[1])
            out.append(
                {
                    "note": "Conditional on a failure occurring; not a separate failure probability.",
                    "most_likely": ranked[0][0],
                    "most_likely_name": fm["names"][ranked[0][0]],
                    "probabilities": [
                        {"mode": m, "name": fm["names"][m], "probability": round(float(v), 4),
                         "test_recall": fm["test_recall_per_class"].get(m)}
                        for m, v in ranked
                    ],
                }
            )
        return out

    # ------------------------------------------------------------------------------------------
    def simulate(self, base: dict, changes: dict, policy: RiskPolicy | None = None) -> dict:
        after_input = {**base, **{k: v for k, v in changes.items() if v is not None}}
        before, after = self.predict([base, after_input], policy=policy)
        delta = after["failure_probability"] - before["failure_probability"]
        changed = [
            {"parameter": k, "label": INPUT_LABELS[k], "before": base[k], "after": after_input[k]}
            for k in config.RAW_INPUTS
            if base[k] != after_input[k]
        ]
        before_c = {c["feature"]: c for c in before.get("contributions", [])}
        shap_shift = sorted(
            [
                {"feature": c["feature"], "label": c["label"],
                 "before": before_c[c["feature"]]["shap"], "after": c["shap"],
                 "change": round(c["shap"] - before_c[c["feature"]]["shap"], 4)}
                for c in after.get("contributions", [])
            ],
            key=lambda r: -abs(r["change"]),
        )
        return {
            "label": "Model simulation - shows how the model's output responds to changed inputs. "
                     "It does not guarantee a real-world outcome.",
            "before": before,
            "after": after,
            "after_input": after_input,
            "changed_parameters": changed,
            "probability_change": round(delta, 4),
            "risk_change": f"{before['risk_level']} -> {after['risk_level']}",
            "prediction_changed": before["predicted_failure"] != after["predicted_failure"],
            "contribution_shift": shap_shift,
        }
