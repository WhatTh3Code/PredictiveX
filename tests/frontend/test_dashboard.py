"""End-to-end dashboard tests in a real browser (Playwright + Chromium).

Starts the FastAPI server (which serves frontend/dist) on a free port. Skipped automatically if
Playwright, a browser, the built frontend or the trained model is missing.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
playwright = pytest.importorskip("playwright.sync_api")

if not (ROOT / "frontend" / "dist" / "index.html").exists():
    pytest.skip("frontend not built (cd frontend && npm run build)", allow_module_level=True)
if not (ROOT / "models" / "predictivex_model.joblib").exists():
    pytest.skip("model not trained", allow_module_level=True)

PAGES = ["/", "/predict", "/explain", "/what-if", "/performance", "/data", "/about"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    env = {**os.environ, "PREDICTIVEX_CORS_ORIGINS": f"http://127.0.0.1:{port}"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.15)
    yield url
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"Chromium unavailable: {exc}")
        yield b
        b.close()


@pytest.fixture
def page(browser):
    ctx = browser.new_context(viewport={"width": 1366, "height": 900})
    pg = ctx.new_page()
    pg.console_errors = []  # type: ignore[attr-defined]
    pg.on("console", lambda m: m.type == "error" and pg.console_errors.append(m.text))  # type: ignore[attr-defined]
    pg.on("pageerror", lambda e: pg.console_errors.append(str(e)))  # type: ignore[attr-defined]
    yield pg
    ctx.close()


def test_every_page_renders_without_errors(server, page):
    for path in PAGES:
        page.goto(f"{server}/#{path}")
        page.wait_for_selector("h1")
        page.wait_for_load_state("networkidle")
        assert page.locator("[data-testid=error-state]").count() == 0, path
    assert page.console_errors == []


def test_overview_shows_real_metrics(server, page):
    page.goto(server)
    page.wait_for_selector(".readout-value")
    text = page.locator(".readouts").inner_text()
    assert "10,000" in text and "2,000" in text
    assert page.locator("[data-testid=fleet-table] tbody tr").count() == 3


def test_predict_empty_state(server, page):
    page.goto(f"{server}/#/predict")
    page.wait_for_selector("[data-testid=empty-state]")
    assert "No prediction yet" in page.inner_text("[data-testid=empty-state]")


def test_demo_flow_high_risk_explain_whatif(server, page):
    page.goto(server)
    page.click("[data-testid=overview-demos] [data-case=high]")
    page.wait_for_selector("[data-testid=prediction-result]")
    assert "HIGH" in page.inner_text("[data-testid=risk-badge]")
    prob = float(page.inner_text("[data-testid=probability]").replace("%", ""))
    assert prob >= 0
    assert "contribut" in page.inner_text("[data-testid=explanation]").lower()
    assert "failed" in page.inner_text("[data-testid=actual-outcome]")

    page.click("[data-testid=go-explain]")
    page.wait_for_selector("[data-testid=contributions] rect.mark")
    assert page.locator("[data-testid=contributions] rect.mark").count() == 9

    page.click("a[href='#/what-if']")
    page.wait_for_selector("[data-testid=whatif-idle]")
    page.fill("#w-tool_wear_min", "20")
    page.wait_for_selector("[data-testid=simulation]")
    page.wait_for_function("document.querySelector('[data-testid=sim-changes]')?.innerText.includes('Tool wear')")
    before = float(page.inner_text("[data-testid=sim-before]").replace("%", ""))
    after = float(page.inner_text("[data-testid=sim-after]").replace("%", ""))
    assert before != after
    assert "Model simulation" in page.inner_text("[data-testid=simulation]")
    assert page.console_errors == []


def test_low_risk_demo(server, page):
    page.goto(f"{server}/#/predict")
    page.click("[data-testid=demo-picker] [data-case=low]")
    page.wait_for_selector("[data-testid=prediction-result]")
    assert "LOW" in page.inner_text("[data-testid=risk-badge]")


def test_invalid_input_case_is_rejected(server, page):
    page.goto(f"{server}/#/predict")
    page.click("[data-testid=demo-picker] [data-case=invalid]")
    page.wait_for_selector("[data-testid=validation-errors]")
    txt = page.inner_text("[data-testid=validation-errors]")
    assert "Rotational speed" in txt and "Torque" in txt
    assert page.locator("[data-testid=prediction-result]").count() == 0
    assert page.locator(".field.has-error").count() == 2


def test_loading_state_while_predicting(server, page):
    def slow(route):
        time.sleep(1.2)
        route.continue_()

    page.route("**/predict", slow)
    page.goto(f"{server}/#/predict")
    page.click("[data-testid=predict-button]")
    page.wait_for_selector("[data-testid=predict-button]:disabled")
    assert "Predicting" in page.inner_text("[data-testid=predict-button]")
    page.wait_for_selector("[data-testid=prediction-result]", timeout=10000)


def test_error_state_when_server_unreachable(server, page):
    page.route("**/metrics", lambda route: route.abort())
    page.goto(f"{server}/#/performance")
    page.wait_for_selector("[data-testid=error-state]")
    assert "Server not reachable" in page.inner_text("[data-testid=error-state]")
    page.unroute("**/metrics")
    page.click("[data-testid=error-state] button")
    page.wait_for_selector("[data-testid=model-table]")


def test_error_state_when_model_unavailable(server, page):
    page.route("**/predict", lambda route: route.fulfill(
        status=503, content_type="application/json",
        body='{"error":"model_unavailable","message":"Model bundle not found."}'))
    page.goto(f"{server}/#/predict")
    page.click("[data-testid=predict-button]")
    page.wait_for_selector("[data-testid=error-state]")
    assert "Model not available" in page.inner_text("[data-testid=error-state]")


def test_whatif_empty_state(server, page):
    page.goto(f"{server}/#/what-if")
    page.wait_for_selector("[data-testid=whatif-empty]")


@pytest.mark.parametrize("width", [390, 768])
def test_responsive_layout_has_no_horizontal_scroll(server, browser, width):
    ctx = browser.new_context(viewport={"width": width, "height": 900})
    pg = ctx.new_page()
    for path in PAGES:
        pg.goto(f"{server}/#{path}")
        pg.wait_for_selector("h1")
        pg.wait_for_load_state("networkidle")
        overflow = pg.evaluate("document.documentElement.scrollWidth - window.innerWidth")
        assert overflow <= 1, f"{path} overflows by {overflow}px at {width}px"
    ctx.close()
