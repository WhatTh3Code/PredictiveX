from __future__ import annotations

import pandas as pd
import pytest

from ml import config
from ml.features.engineering import FeatureBuilder
from ml.preprocessing.data import (
    SchemaError,
    feature_target,
    file_sha256,
    load_raw,
    quality_report,
    split_data,
    validate_schema,
)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return load_raw()


def test_raw_file_is_the_recorded_dataset():
    assert file_sha256(config.DATA_RAW) == config.RAW_SHA256


def test_schema_and_shape(raw):
    assert raw.shape == (10000, 14)
    assert set(raw["type"].unique()) == {"L", "M", "H"}


def test_no_missing_values(raw):
    assert quality_report(raw)["missing_values_total"] == 0


def test_no_duplicate_rows(raw):
    q = quality_report(raw)
    assert q["duplicate_rows"] == 0
    assert q["duplicate_udi"] == 0


def test_invalid_types_are_rejected(raw):
    bad = raw.copy()
    bad["torque_nm"] = bad["torque_nm"].astype(str)
    with pytest.raises(SchemaError):
        validate_schema(bad)


def test_unknown_product_type_is_rejected(raw):
    bad = raw.copy()
    bad.loc[0, "type"] = "X"
    with pytest.raises(SchemaError):
        validate_schema(bad)


def test_non_binary_target_is_rejected(raw):
    bad = raw.copy()
    bad.loc[0, config.TARGET] = 2
    with pytest.raises(SchemaError):
        validate_schema(bad)


def test_no_target_leakage_in_features(raw):
    X, y = feature_target(raw)
    forbidden = {config.TARGET, *config.FAILURE_MODES, *config.IDENTIFIER_COLUMNS}
    assert forbidden.isdisjoint(X.columns)
    names = FeatureBuilder().feature_names
    assert forbidden.isdisjoint(names)
    assert y.name == config.TARGET


def test_split_is_disjoint_complete_and_reproducible(raw):
    a, b = split_data(raw), split_data(raw)
    ids = [set(p.udi) for p in (a.train, a.validation, a.test)]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    assert sum(len(i) for i in ids) == len(raw)
    assert list(a.test.udi) == list(b.test.udi)
    assert (len(a.train), len(a.validation), len(a.test)) == (6000, 2000, 2000)


def test_split_is_stratified(raw):
    s = split_data(raw)
    overall = raw[config.TARGET].mean()
    for part in (s.train, s.validation, s.test):
        assert abs(part[config.TARGET].mean() - overall) < 0.002
        for m in ("hdf", "pwf", "osf"):
            assert part[m].sum() >= 15


def test_processed_files_match_pipeline(raw):
    s = split_data(raw)
    for name, part in (("train", s.train), ("validation", s.validation), ("test", s.test)):
        path = config.DATA_PROCESSED_DIR / f"{name}.csv"
        if not path.exists():
            pytest.skip("processed data not generated")
        assert list(pd.read_csv(path).udi) == list(part.udi)


def test_raw_file_is_never_modified():
    before = file_sha256(config.DATA_RAW)
    load_raw()
    assert file_sha256(config.DATA_RAW) == before
