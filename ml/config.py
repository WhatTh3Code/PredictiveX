"""Central configuration for the PredictiveX ML pipeline.

Every script and the API import paths, column names and the random seed from here,
so training and inference cannot silently drift apart.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw" / "ai4i2020.csv"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_BUNDLE_PATH = MODELS_DIR / "predictivex_model.joblib"

# SHA-256 of the raw CSV that every reported number was produced from.
RAW_SHA256 = "59db4f1d9c34c58136d89e5a006ec190dcea19e9dbea74f6b3b0c6f22a44d183"

RANDOM_SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20  # fraction of the full dataset
CV_FOLDS = 5

# Raw column name -> clean snake_case name.
RAW_TO_CLEAN = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

TARGET = "machine_failure"
FAILURE_MODES = ["twf", "hdf", "pwf", "osf", "rnf"]
FAILURE_MODE_NAMES = {
    "twf": "Tool wear failure",
    "hdf": "Heat dissipation failure",
    "pwf": "Power failure",
    "osf": "Overstrain failure",
    "rnf": "Random failure",
}

# Identifiers carry no physical information; failure-mode flags are produced by the same
# process as the target (any flag => failure), so using them would be target leakage.
IDENTIFIER_COLUMNS = ["udi", "product_id"]
LEAKAGE_COLUMNS = FAILURE_MODES

CATEGORICAL_INPUTS = ["type"]
NUMERIC_INPUTS = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
RAW_INPUTS = CATEGORICAL_INPUTS + NUMERIC_INPUTS

PRODUCT_TYPES = ["L", "M", "H"]

# Human-readable labels used by reports, the API and the UI.
FEATURE_LABELS = {
    "type": "Product quality type",
    "type_L": "Type L (low quality)",
    "type_M": "Type M (medium quality)",
    "type_H": "Type H (high quality)",
    "type_ordinal": "Product quality type",
    "air_temp_k": "Air temperature",
    "process_temp_k": "Process temperature",
    "rotational_speed_rpm": "Rotational speed",
    "torque_nm": "Torque",
    "tool_wear_min": "Tool wear",
    "temp_diff_k": "Process-air temperature gap",
    "power_w": "Mechanical power",
    "strain_nm_min": "Tool strain (wear x torque)",
}

FEATURE_UNITS = {
    "air_temp_k": "K",
    "process_temp_k": "K",
    "rotational_speed_rpm": "rpm",
    "torque_nm": "Nm",
    "tool_wear_min": "min",
    "temp_diff_k": "K",
    "power_w": "W",
    "strain_nm_min": "Nm·min",
}
