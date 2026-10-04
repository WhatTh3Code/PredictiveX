"""Feature engineering shared by training and inference.

The same `FeatureBuilder` instance is pickled inside the model bundle, so the API applies
exactly the transformation the model was trained with.

Engineered features (each is evaluated by ablation in `scripts/train_models.py`):

| Feature        | Formula                                   | Why                                                    |
|----------------|-------------------------------------------|--------------------------------------------------------|
| temp_diff_k    | process_temp_k - air_temp_k               | Heat must leave the process through the air; a small    |
|                |                                           | gap means poor heat dissipation.                        |
| power_w        | torque_nm * rotational_speed_rpm * 2π/60  | Mechanical power. Both very low and very high power     |
|                |                                           | are physically stressful; a tree can split on both ends.|
| strain_nm_min  | tool_wear_min * torque_nm                 | Accumulated load on a worn tool (overstrain proxy).     |

These relationships come from the physical failure mechanisms described in the dataset's
documentation (Matzka, 2020). They are hypotheses about what is informative; the training
script measures whether each one actually improves validation performance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml import config

ENGINEERED_FEATURES: dict[str, dict[str, str]] = {
    "temp_diff_k": {
        "formula": "process_temp_k - air_temp_k",
        "reason": "Small process-air temperature gap indicates poor heat dissipation.",
    },
    "power_w": {
        "formula": "torque_nm * rotational_speed_rpm * 2*pi/60",
        "reason": "Mechanical power; failures are associated with both very low and very high power.",
    },
    "strain_nm_min": {
        "formula": "tool_wear_min * torque_nm",
        "reason": "Accumulated load on a worn tool, a proxy for overstrain.",
    },
}

TYPE_ORDINAL = {"L": 0, "M": 1, "H": 2}


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]
    out["power_w"] = out["torque_nm"] * out["rotational_speed_rpm"] * 2 * math.pi / 60
    out["strain_nm_min"] = out["tool_wear_min"] * out["torque_nm"]
    return out


@dataclass
class FeatureBuilder:
    """Turns raw machine inputs into the model's numeric feature matrix.

    `engineered` lists which engineered features to include (chosen by ablation).
    Product type is ordinal-encoded (L < M < H quality); this is safe for trees and
    standardised before logistic regression.
    """

    engineered: list[str] = field(default_factory=lambda: list(ENGINEERED_FEATURES))

    @property
    def feature_names(self) -> list[str]:
        return ["type_ordinal", *config.NUMERIC_INPUTS, *self.engineered]

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in config.RAW_INPUTS if c not in raw.columns]
        if missing:
            raise ValueError(f"Missing input columns: {missing}")
        unknown = set(raw["type"].unique()) - set(TYPE_ORDINAL)
        if unknown:
            raise ValueError(f"Unknown product type(s): {sorted(unknown)}")
        df = add_engineered(raw[config.RAW_INPUTS].astype({c: float for c in config.NUMERIC_INPUTS}))
        df["type_ordinal"] = raw["type"].map(TYPE_ORDINAL).astype(float)
        X = df[self.feature_names]
        if not np.isfinite(X.to_numpy()).all():
            raise ValueError("Non-finite value produced during feature engineering")
        return X
