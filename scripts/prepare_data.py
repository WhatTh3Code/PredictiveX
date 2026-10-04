"""Phase 2 - reproducible data pipeline.

    python scripts/prepare_data.py

1. Verifies the raw file checksum (the raw file is never modified).
2. Validates schema and dtypes.
3. Computes data-quality facts (missing values, duplicates, outliers, label consistency).
4. Creates a stratified 60/20/20 train/validation/test split with a fixed seed.
5. Writes data/processed/{train,validation,test}.csv and reports/data_quality.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import config  # noqa: E402
from ml.preprocessing.data import (  # noqa: E402
    file_sha256,
    load_raw,
    quality_report,
    split_data,
    split_summary,
)


def main() -> None:
    digest = file_sha256(config.DATA_RAW)
    checksum_ok = digest == config.RAW_SHA256
    print(f"raw sha256: {digest} ({'matches' if checksum_ok else 'DOES NOT MATCH'} recorded checksum)")
    if not checksum_ok:
        print("WARNING: raw data differs from the file the reported results were produced with.")

    df = load_raw()
    report = quality_report(df)
    splits = split_data(df)
    summary = split_summary(splits)
    assert summary["udi_overlap_between_splits"] == 0, "Split leakage: rows shared between splits"

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", splits.train), ("validation", splits.validation), ("test", splits.test)):
        part.to_csv(config.DATA_PROCESSED_DIR / f"{name}.csv", index=False)

    out = {
        "source_file": str(config.DATA_RAW.relative_to(config.ROOT)),
        "sha256": digest,
        "checksum_matches_recorded": checksum_ok,
        "random_seed": config.RANDOM_SEED,
        "split_strategy": "Stratified by failure mode; 60% train / 20% validation / 20% test",
        "quality": report,
        "splits": summary,
    }
    (config.REPORTS_DIR / "data_quality.json").write_text(json.dumps(out, indent=2))

    q = report
    print(f"rows={q['n_rows']} missing={q['missing_values_total']} duplicates={q['duplicate_rows']}")
    print(f"failure rate={q['failure_rate']:.4f} (imbalance {q['imbalance_ratio']}:1)")
    print(f"label consistency: {q['label_consistency']}")
    for name in ("train", "validation", "test"):
        s = summary[name]
        print(f"{name:<10} rows={s['rows']:<5} failures={s['failures']:<4} rate={s['failure_rate']:.4f}")


if __name__ == "__main__":
    main()
