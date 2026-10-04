"""Pick demo machines from the held-out TEST split (records the model never trained on).

    python scripts/select_demo_cases.py

Only the inputs, the true label and the selection rule are stored. Predictions shown in the
dashboard are always computed live by the API from these inputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ml import config  # noqa: E402
from ml.inference import Predictor  # noqa: E402
from ml.preprocessing.data import feature_target, load_split  # noqa: E402


def record(row) -> dict:
    return {
        "type": row["type"],
        "air_temp_k": float(row["air_temp_k"]),
        "process_temp_k": float(row["process_temp_k"]),
        "rotational_speed_rpm": float(row["rotational_speed_rpm"]),
        "torque_nm": float(row["torque_nm"]),
        "tool_wear_min": float(row["tool_wear_min"]),
    }


def main() -> None:
    pred = Predictor.load()
    test = load_split("test").reset_index(drop=True)
    X, y = feature_target(test)
    p = pred.bundle["model"].predict_proba(pred.bundle["feature_builder"].transform(X))[:, 1]
    pol = pred.policy
    test["p"] = p

    # Low risk: a typical normal machine (closest to the median operating point) in the LOW band.
    normal_low = test[(test[config.TARGET] == 0) & (test.p < pol.medium_threshold)]
    num = config.NUMERIC_INPUTS
    med, scale = test[num].median(), test[num].std()
    low = normal_low.loc[((normal_low[num] - med) / scale).pow(2).sum(axis=1).idxmin()]

    # High risk: an actual overstrain failure the model flags HIGH, with high tool wear so the
    # what-if demo (reduce tool wear) is meaningful.
    cand = test[(test[config.TARGET] == 1) & (test.osf == 1) & (test.p >= pol.high_threshold)]
    if cand.empty:
        cand = test[(test[config.TARGET] == 1) & (test.p >= pol.high_threshold)]
    high = cand.sort_values(["tool_wear_min", "p"], ascending=False).iloc[0]

    # Borderline: the record whose probability is closest to the HIGH threshold from below.
    below = test[(test.p < pol.high_threshold) & (test.p >= pol.medium_threshold)]
    border = below.loc[(pol.high_threshold - below.p).idxmin()]

    extras = []
    for mode, title in (("hdf", "Heat dissipation failure (actual)"), ("pwf", "Power failure (actual)")):
        c = test[(test[mode] == 1) & (test[config.TARGET] == 1)]
        if not c.empty:
            r = c.sort_values("p", ascending=False).iloc[0]
            extras.append((mode, title, r))

    def case(cid: str, title: str, rule: str, row) -> dict:
        modes = [m for m in config.FAILURE_MODES if int(row[m]) == 1]
        return {
            "id": cid,
            "title": title,
            "selection_rule": rule,
            "source": f"Test split, UDI {int(row['udi'])}, product {row['product_id']}",
            "actual_failure": bool(row[config.TARGET]),
            "actual_failure_modes": modes,
            "input": record(row),
        }

    cases = [
        case("low", "Low-risk machine", "Normal test record nearest the median operating point, scored LOW", low),
        case("high", "High-risk machine", "Actual overstrain failure in the test split, scored HIGH, highest tool wear", high),
        case("borderline", "Borderline machine", "Test record with probability closest to (below) the HIGH threshold", border),
        {
            "id": "invalid",
            "title": "Invalid input",
            "selection_rule": "Negative rotational speed and non-numeric torque; must be rejected by validation",
            "source": "Constructed to test input validation (not a dataset record)",
            "actual_failure": None,
            "actual_failure_modes": [],
            "input": {"type": "L", "air_temp_k": 300.0, "process_temp_k": 310.0,
                      "rotational_speed_rpm": -1200, "torque_nm": "forty", "tool_wear_min": 50},
        },
    ]
    for mode, title, r in extras:
        cases.append(case(mode, title, f"Actual {mode.upper()} failure in the test split with the highest score", r))

    out = {"note": "Inputs from the unseen test split; predictions are computed live.", "cases": cases}
    (config.REPORTS_DIR / "demo_cases.json").write_text(json.dumps(out, indent=1))
    for c in cases:
        print(c["id"], c["source"], c["actual_failure"], c["actual_failure_modes"], c["input"])
    print("check (not stored):", {c["id"]: round(float(np.round(pred.predict([c['input']], explain=False)[0]['failure_probability'], 3)), 3)
                                  for c in cases if c["id"] != "invalid"})


if __name__ == "__main__":
    main()
