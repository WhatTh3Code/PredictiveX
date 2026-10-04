"""Guards against stale or invented numbers: key figures in the docs must match the reports."""

from __future__ import annotations

import json

import pytest

from ml import config

DOCS = [config.ROOT / "README.md", config.ROOT / "docs" / "presentation.md"]


@pytest.fixture(scope="module")
def metrics():
    path = config.REPORTS_DIR / "metrics.json"
    if not path.exists():
        pytest.skip("metrics not generated")
    return json.loads(path.read_text())


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_headline_test_metrics_match_reports(doc, metrics):
    if not doc.exists():
        pytest.skip(f"{doc.name} missing")
    text = doc.read_text()
    t = metrics["test"]
    for key in ("pr_auc", "roc_auc"):
        assert f"{t[key]:.3f}" in text, f"{doc.name}: {key} {t[key]:.3f} not found"
    for key in ("precision", "recall", "f1"):
        assert f"{t[key]:.3f}" in text or f"{t[key] * 100:.1f}%" in text, f"{doc.name}: {key} missing"
    cm = t["confusion_matrix"]
    assert f"{cm['tp']}" in text and f"{cm['fp']}" in text and f"{cm['fn']}" in text


def test_readme_thresholds_and_bands_match(metrics):
    text = (config.ROOT / "README.md").read_text()
    pol = metrics["thresholds"]["policy"]
    assert f"{pol['high_threshold']:.3f}" in text
    assert f"{pol['medium_threshold']:.3f}" in text
    for band in ("HIGH", "MEDIUM", "LOW"):
        b = metrics["test"]["risk_bands"][band]
        assert f"{b['observed_failure_rate'] * 100:.1f}%" in text, band


def test_readme_model_comparison_matches():
    mc = json.loads((config.REPORTS_DIR / "model_comparison.json").read_text())
    text = (config.ROOT / "README.md").read_text()
    for row in mc["models"]:
        assert f"{row['validation']['pr_auc']:.3f}" in text, row["model"]
        assert f"{row['tuning']['cv_pr_auc_mean']:.3f}" in text, row["model"]
