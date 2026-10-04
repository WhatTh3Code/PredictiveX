"""Dataset loading, schema validation, data-quality checks and the reproducible split."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml import config

EXPECTED_RAW_COLUMNS = list(config.RAW_TO_CLEAN.keys())

EXPECTED_DTYPES = {
    "udi": "integer",
    "product_id": "string",
    "type": "string",
    "air_temp_k": "float",
    "process_temp_k": "float",
    "rotational_speed_rpm": "integer",
    "torque_nm": "float",
    "tool_wear_min": "integer",
    "machine_failure": "integer",
    "twf": "integer",
    "hdf": "integer",
    "pwf": "integer",
    "osf": "integer",
    "rnf": "integer",
}


class SchemaError(ValueError):
    """Raised when the raw file does not match the expected AI4I 2020 schema."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: Path = config.DATA_RAW) -> pd.DataFrame:
    """Read the raw CSV (read-only) and return it with clean snake_case column names."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. See data/raw/README.md for acquisition steps."
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in EXPECTED_RAW_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_RAW_COLUMNS]
    if missing or extra:
        raise SchemaError(f"Schema mismatch. Missing: {missing}. Unexpected: {extra}.")
    df = df.rename(columns=config.RAW_TO_CLEAN)[list(config.RAW_TO_CLEAN.values())]
    validate_schema(df)
    return df


def _kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    return "string"


def validate_schema(df: pd.DataFrame) -> None:
    """Check column dtypes, categorical levels and binary label columns."""
    problems: list[str] = []
    for col, kind in EXPECTED_DTYPES.items():
        actual = _kind(df[col])
        ok = actual == kind or (kind == "float" and actual == "integer")
        if not ok:
            problems.append(f"{col}: expected {kind}, got {actual}")
    bad_types = set(df["type"].dropna().unique()) - set(config.PRODUCT_TYPES)
    if bad_types:
        problems.append(f"type: unexpected levels {sorted(bad_types)}")
    for col in [config.TARGET, *config.FAILURE_MODES]:
        if not set(df[col].dropna().unique()) <= {0, 1}:
            problems.append(f"{col}: not binary")
    if problems:
        raise SchemaError("; ".join(problems))


def iqr_outliers(series: pd.Series, k: float = 1.5) -> dict[str, float | int]:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    mask = (series < lo) | (series > hi)
    return {
        "lower_fence": round(float(lo), 3),
        "upper_fence": round(float(hi), 3),
        "n_outliers": int(mask.sum()),
        "pct_outliers": round(float(mask.mean() * 100), 3),
    }


def quality_report(df: pd.DataFrame) -> dict:
    """Data-quality facts computed from the data (nothing hard-coded)."""
    modes = df[config.FAILURE_MODES]
    n_modes = modes.sum(axis=1)
    feature_cols = config.RAW_INPUTS
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_values_by_column": {c: int(v) for c, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_feature_rows": int(df.duplicated(subset=feature_cols).sum()),
        "duplicate_udi": int(df["udi"].duplicated().sum()),
        "duplicate_product_id": int(df["product_id"].duplicated().sum()),
        "target_counts": {str(k): int(v) for k, v in df[config.TARGET].value_counts().items()},
        "failure_rate": round(float(df[config.TARGET].mean()), 5),
        "imbalance_ratio": round(
            float((df[config.TARGET] == 0).sum() / max((df[config.TARGET] == 1).sum(), 1)), 2
        ),
        "failure_mode_counts": {m: int(df[m].sum()) for m in config.FAILURE_MODES},
        "type_counts": {k: int(v) for k, v in df["type"].value_counts().items()},
        "label_consistency": {
            "failure_without_mode": int(((df[config.TARGET] == 1) & (n_modes == 0)).sum()),
            "mode_without_failure": int(((df[config.TARGET] == 0) & (n_modes > 0)).sum()),
            "mode_without_failure_by_mode": {
                m: int(((df[config.TARGET] == 0) & (df[m] == 1)).sum()) for m in config.FAILURE_MODES
            },
            "failures_with_multiple_modes": int(((df[config.TARGET] == 1) & (n_modes > 1)).sum()),
        },
        "outliers_iqr": {c: iqr_outliers(df[c]) for c in config.NUMERIC_INPUTS},
        "feature_ranges": {
            c: {
                "min": float(df[c].min()),
                "max": float(df[c].max()),
                "mean": round(float(df[c].mean()), 3),
                "std": round(float(df[c].std()), 3),
                "p01": round(float(df[c].quantile(0.01)), 3),
                "p99": round(float(df[c].quantile(0.99)), 3),
            }
            for c in config.NUMERIC_INPUTS
        },
    }


def stratification_label(df: pd.DataFrame) -> pd.Series:
    """Stratify on failure *mode* so that rare modes are spread across all three splits.

    Rows with multiple modes are grouped as "multi"; failures without a mode as "unlabelled".
    """
    modes = df[config.FAILURE_MODES]
    n_modes = modes.sum(axis=1)
    label = pd.Series("none", index=df.index)
    single = n_modes == 1
    label[single] = modes[single].idxmax(axis=1)
    label[n_modes > 1] = "multi"
    label[(df[config.TARGET] == 1) & (n_modes == 0)] = "unlabelled"
    # Mode flags without the target are their own stratum so they don't cluster in one split.
    label[(df[config.TARGET] == 0) & (n_modes > 0)] = "mode_no_failure"
    return label


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_data(df: pd.DataFrame, seed: int = config.RANDOM_SEED) -> Splits:
    """Stratified 60/20/20 split. The test set is never used until the final evaluation."""
    strat = stratification_label(df)
    train_val, test = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=seed, stratify=strat
    )
    val_fraction = config.VALIDATION_SIZE / (1 - config.TEST_SIZE)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        random_state=seed,
        stratify=strat.loc[train_val.index],
    )
    return Splits(train=train.sort_index(), validation=val.sort_index(), test=test.sort_index())


def feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs from the target. Identifiers and mode flags are dropped here."""
    X = df[config.RAW_INPUTS].copy()
    y = df[config.TARGET].astype(int).copy()
    return X, y


def load_split(name: str) -> pd.DataFrame:
    path = config.DATA_PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python scripts/prepare_data.py` first.")
    return pd.read_csv(path)


def split_summary(splits: Splits) -> dict:
    out = {}
    for name, part in (("train", splits.train), ("validation", splits.validation), ("test", splits.test)):
        out[name] = {
            "rows": int(len(part)),
            "failures": int(part[config.TARGET].sum()),
            "failure_rate": round(float(part[config.TARGET].mean()), 5),
            "failure_modes": {m: int(part[m].sum()) for m in config.FAILURE_MODES},
        }
    overlap = (
        len(set(splits.train.udi) & set(splits.validation.udi))
        + len(set(splits.train.udi) & set(splits.test.udi))
        + len(set(splits.validation.udi) & set(splits.test.udi))
    )
    out["udi_overlap_between_splits"] = int(overlap)
    return out


__all__ = [
    "SchemaError",
    "Splits",
    "feature_target",
    "file_sha256",
    "iqr_outliers",
    "load_raw",
    "load_split",
    "quality_report",
    "split_data",
    "split_summary",
    "stratification_label",
    "validate_schema",
]
