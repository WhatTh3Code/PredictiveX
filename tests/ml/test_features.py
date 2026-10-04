from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ml.features.engineering import FeatureBuilder


def _row(**kw):
    base = {"type": "M", "air_temp_k": 300.0, "process_temp_k": 310.0, "rotational_speed_rpm": 1500.0,
            "torque_nm": 40.0, "tool_wear_min": 100.0}
    base.update(kw)
    return pd.DataFrame([base])


def test_engineered_formulas():
    X = FeatureBuilder().transform(_row())
    assert X["temp_diff_k"].iloc[0] == pytest.approx(10.0)
    assert X["power_w"].iloc[0] == pytest.approx(40.0 * 1500.0 * 2 * math.pi / 60)
    assert X["strain_nm_min"].iloc[0] == pytest.approx(4000.0)
    assert X["type_ordinal"].iloc[0] == 1


def test_feature_order_is_stable():
    fb = FeatureBuilder()
    X = fb.transform(_row())
    assert list(X.columns) == fb.feature_names


def test_unknown_type_rejected():
    with pytest.raises(ValueError):
        FeatureBuilder().transform(_row(type="Z"))


def test_missing_column_rejected():
    with pytest.raises(ValueError):
        FeatureBuilder().transform(_row().drop(columns=["torque_nm"]))


def test_batch_equals_single_rows():
    rows = pd.concat([_row(torque_nm=t) for t in (10.0, 40.0, 70.0)], ignore_index=True)
    fb = FeatureBuilder()
    batch = fb.transform(rows).to_numpy()
    single = np.vstack([fb.transform(rows.iloc[[i]]).to_numpy() for i in range(3)])
    assert np.allclose(batch, single)
