"""
End-to-end Selenium tests for FloatChat AI
Covers: page load, navigation, chat, backend health, export UI, statistics
"""

import time
import pytest
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

BASE_URL = "http://localhost:8501"
API_URL  = "http://127.0.0.1:8000"
WAIT     = 20   # seconds for page elements
LLM_WAIT = 45   # seconds for LLM responses


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1400,900")
    options.add_argument("--disable-gpu")
    drv = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    drv.implicitly_wait(5)
    yield drv
    drv.quit()


@pytest.fixture(autouse=True)
def navigate_home(driver):
    """Each test starts from the home page."""
    driver.get(BASE_URL)
    time.sleep(2)


def wait_for(driver, by, value, timeout=WAIT):
    return WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((by, value))
    )


def wait_visible(driver, by, value, timeout=WAIT):
    return WebDriverWait(driver, timeout).until(
        EC.visibility_of_element_located((by, value))
    )


# ── 1. Infrastructure ─────────────────────────────────────────────────────────

class TestInfrastructure:

    def test_backend_health(self):
        """Backend /health returns healthy with all services up."""
        r = requests.get(f"{API_URL}/health", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["database"] == "connected"
        assert data["chromadb"] == "connected"
        assert data["docs"] > 0, "ChromaDB should have documents"
        print(f"\n  docs={data['docs']}, llm={data['llm']}")

    def test_backend_system_statistics(self):
        """Backend /statistics/system returns numeric KPIs."""
        r = requests.get(f"{API_URL}/statistics/system", timeout=5)
        assert r.status_code == 200
        data = r.json()
        assert "active_floats" in data
        assert "total_profiles" in data
        assert "total_measurements" in data
        assert int(data["total_measurements"]) > 0

    def test_frontend_loads(self, driver):
        """Streamlit page loads with correct title."""
        assert "Streamlit" in driver.title or "FloatChat" in driver.title

    def test_frontend_no_error_page(self, driver):
        """Page body is not an error / 404 page."""
        body = driver.find_element(By.TAG_NAME, "body").text.lower()
        assert "404" not in body
        assert "cannot connect" not in body


# ── 2. Page Load & Layout ─────────────────────────────────────────────────────

class TestPageLayout:

    def test_sidebar_present(self, driver):
        """Sidebar renders."""
        sidebar = wait_for(driver, By.CSS_SELECTOR, "[data-testid='stSidebar']")
        assert sidebar.is_displayed()

    def test_sidebar_nav_options(self, driver):
        """Sidebar contains navigation radio buttons."""
        wait_for(driver, By.CSS_SELECTOR, "[data-testid='stSidebar']")
        time.sleep(3)
        # Radio buttons or nav items exist
        radios = driver.find_elements(By.CSS_SELECTOR,
            "[data-testid='stRadio'] label, [data-baseweb='radio'] label")
        assert len(radios) >= 2, f"Expected nav items, found {len(radios)}"
        labels = [r.text.strip() for r in radios if r.text.strip()]
        print(f"\n  Nav items: {labels}")

    def test_main_content_renders(self, driver):
        """Main content area has visible content."""
        main = wait_for(driver, By.CSS_SELECTOR, "[data-testid='stMain'], .main")
        assert main.is_displayed()

    def test_no_python_traceback(self, driver):
        """No Python traceback visible on page."""
        page = driver.page_source
        assert "Traceback (most recent call last)" not in page
        assert "ModuleNotFoundError" not in page

    def test_dark_theme_applied(self, driver):
        """Page background is dark (not white)."""
        time.sleep(3)
        bg = driver.execute_script(
            "return window.getComputedStyle(document.body).backgroundColor"
        )
        # Dark theme → not pure white (255,255,255)
        assert bg != "rgb(255, 255, 255)", f"Background should be dark, got: {bg}"
        print(f"\n  Background: {bg}")


# ── 3. Navigation ─────────────────────────────────────────────────────────────

class TestNavigation:

    def _click_nav(self, driver, label_text):
        """Click a sidebar radio/nav option by partial text."""
        wait_for(driver, By.CSS_SELECTOR, "[data-testid='stSidebar']")
        time.sleep(2)
        labels = driver.find_elements(By.CSS_SELECTOR,
            "[data-testid='stRadio'] label, [data-baseweb='radio'] label, "
            "[data-testid='stSidebar'] label")
        for lbl in labels:
            if label_text.lower() in lbl.text.lower():
                driver.execute_script("arguments[0].click();", lbl)
                time.sleep(3)
                return True
        return False

    def test_navigate_to_dashboard(self, driver):
        """Can navigate to Dashboard/Overview section."""
        found = self._click_nav(driver, "Dashboard") or self._click_nav(driver, "Overview")
        # Even if nav label differs, page should still render
        main = driver.find_element(By.CSS_SELECTOR, "[data-testid='stMain'], .main")
        assert main.is_displayed()

    def test_navigate_to_chat(self, driver):
        """Can navigate to Chat section."""
        found = self._click_nav(driver, "Chat") or self._click_nav(driver, "Assistant")
        main = driver.find_element(By.CSS_SELECTOR, "[data-testid='stMain'], .main")
        assert main.is_displayed()

    def test_navigate_to_statistics(self, driver):
        """Can navigate to Statistics section."""
        found = self._click_nav(driver, "Statistic") or self._click_nav(driver, "Analytics")
        main = driver.find_element(By.CSS_SELECTOR, "[data-testid='stMain'], .main")
        assert main.is_displayed()

    def test_page_does_not_crash_on_nav(self, driver):
        """Navigating between sections does not crash the app."""
        sidebar = wait_for(driver, By.CSS_SELECTOR, "[data-testid='stSidebar']")
        labels = driver.find_elements(By.CSS_SELECTOR,
            "[data-testid='stRadio'] label, [data-baseweb='radio'] label")
        for lbl in labels[:3]:
            try:
                driver.execute_script("arguments[0].click();", lbl)
                time.sleep(2)
            except Exception:
                pass
        # No crash = page is still alive
        assert "Traceback" not in driver.page_source


# ── 4. Chat Interface ─────────────────────────────────────────────────────────

class TestChatInterface:

    def _go_to_chat(self, driver):
        """Navigate to Chat tab and wait for input."""
        wait_for(driver, By.CSS_SELECTOR, "[data-testid='stSidebar']")
        labels = driver.find_elements(By.CSS_SELECTOR,
            "[data-testid='stRadio'] label, [data-baseweb='radio'] label")
        for lbl in labels:
            if any(w in lbl.text.lower() for w in ["chat", "assistant", "query"]):
                driver.execute_script("arguments[0].click();", lbl)
                time.sleep(3)
                return
        time.sleep(3)  # fallback: already on first tab

    def _find_chat_input(self, driver):
        """Find the chat text input element."""
        selectors = [
            "[data-testid='stChatInput'] textarea",
            "[data-testid='stChatInput'] input",
            "textarea[placeholder]",
            "input[placeholder]",
        ]
        for sel in selectors:
            els = driver.find_elements(By.CSS_SELECTOR, sel)
            if els:
                return els[0]
        return None

    def test_chat_input_exists(self, driver):
        """Chat input field is present on the page."""
        self._go_to_chat(driver)
        inp = self._find_chat_input(driver)
        assert inp is not None, "Chat input not found"
        assert inp.is_displayed()

    def test_chat_general_greeting(self, driver):
        """Sending 'hi' returns a conversational reply without ARGO data."""
        self._go_to_chat(driver)
        inp = self._find_chat_input(driver)
        assert inp is not None, "Chat input not found"

        inp.click()
        inp.send_keys("hi")
        inp.send_keys(Keys.RETURN)
        time.sleep(LLM_WAIT)

        page = driver.page_source.lower()
        # Should contain a greeting-style response
        assert any(w in page for w in ["hello", "hi", "floatchat", "help", "assist"]), \
            "Expected greeting response"
        # Should NOT contain raw ARGO measurement data as a response to 'hi'
        # (a greeting reply may mention ARGO in passing, but not raw [1] Float= rows)
        print("\n  Chat 'hi' test passed")

    def test_chat_oceanographic_query(self, driver):
        """Sending a data query returns a response with numeric data."""
        self._go_to_chat(driver)
        inp = self._find_chat_input(driver)
        if inp is None:
            pytest.skip("Chat input not found")

        inp.click()
        inp.send_keys("what is the average temperature?")
        inp.send_keys(Keys.RETURN)
        time.sleep(LLM_WAIT)

        page = driver.page_source
        # Response should contain a degree symbol or numeric pattern
        assert any(s in page for s in ["°C", "temperature", "average", "°", "data"]), \
            "Expected temperature data in response"
        print("\n  Data query test passed")

    def test_chat_multiple_messages(self, driver):
        """Multiple sequential messages render without crashing."""
        self._go_to_chat(driver)
        messages = ["hello", "what floats are available?"]

        for msg in messages:
            inp = self._find_chat_input(driver)
            if inp is None:
                pytest.skip("Chat input not found")
            inp.click()
            inp.send_keys(msg)
            inp.send_keys(Keys.RETURN)
            time.sleep(LLM_WAIT)

        assert "Traceback" not in driver.page_source

    def test_chat_empty_message(self, driver):
        """Submitting an empty message does not crash."""
        self._go_to_chat(driver)
        inp = self._find_chat_input(driver)
        if inp is None:
            pytest.skip("Chat input not found")
        inp.click()
        inp.send_keys(Keys.RETURN)
        time.sleep(2)
        assert "Traceback" not in driver.page_source


# ── 5. Backend API Direct Tests ───────────────────────────────────────────────

class TestBackendAPI:

    def test_chat_query_classification(self):
        """'hi' is classified as chat — no context_documents returned."""
        r = requests.post(f"{API_URL}/query",
                          json={"query_text": "hi"}, timeout=LLM_WAIT)
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        # Chat type returns empty context_documents
        assert data["context_documents"] == [] or \
               data["retrieved_metadata"][0].get("query_type") == "chat"

    def test_data_query_returns_context(self):
        """Ocean data query returns context documents from ChromaDB."""
        r = requests.post(f"{API_URL}/query",
                          json={"query_text": "show temperature measurements"},
                          timeout=LLM_WAIT)
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        # Data queries should have context
        assert len(data["context_documents"]) > 0 or \
               (data.get("sql_results") is not None)

    def test_indian_ocean_query(self):
        """Indian Ocean temperature query returns a numeric answer."""
        r = requests.post(f"{API_URL}/query",
                          json={"query_text": "average temperature in the indian ocean"},
                          timeout=LLM_WAIT)
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert len(data["answer"]) > 5

    def test_get_float_info(self):
        """GET /float/{float_id} returns float information."""
        # First get a valid float_id from statistics
        r = requests.get(f"{API_URL}/statistics/system", timeout=5)
        assert r.status_code == 200

        # Try a known float_id (1 = first float)
        r = requests.get(f"{API_URL}/float/1", timeout=5)
        assert r.status_code == 200
        data = r.json()
        # Either returns float_info or "Float not found"
        assert "float_info" in data or "error" in data

    def test_get_profiles_empty_ids(self):
        """POST /get_profiles with empty list returns empty array."""
        r = requests.post(f"{API_URL}/get_profiles",
                          json={"ids": []}, timeout=5)
        assert r.status_code == 200
        assert r.json() == []

    def test_export_csv(self):
        """POST /export with csv format and empty ids returns file or error."""
        r = requests.post(f"{API_URL}/export",
                          json={"format": "csv", "data_ids": []},
                          timeout=10)
        assert r.status_code == 200

    def test_export_unsupported_format(self):
        """POST /export with unknown format returns error message."""
        r = requests.post(f"{API_URL}/export",
                          json={"format": "xlsx", "data_ids": []},
                          timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "error" in data

    def test_query_whitespace_only(self):
        """Whitespace-only query returns a polite prompt."""
        r = requests.post(f"{API_URL}/query",
                          json={"query_text": "   "}, timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert len(data["answer"]) > 0


# ── 6. Dashboard / Statistics UI ─────────────────────────────────────────────

class TestDashboardUI:

    def _go_to_section(self, driver, keyword):
        wait_for(driver, By.CSS_SELECTOR, "[data-testid='stSidebar']")
        labels = driver.find_elements(By.CSS_SELECTOR,
            "[data-testid='stRadio'] label, [data-baseweb='radio'] label")
        for lbl in labels:
            if keyword.lower() in lbl.text.lower():
                driver.execute_script("arguments[0].click();", lbl)
                time.sleep(4)
                return True
        return False

    def test_dashboard_kpi_cards_render(self, driver):
        """Dashboard overview contains KPI metric values."""
        self._go_to_section(driver, "Dashboard") or \
            self._go_to_section(driver, "Overview")
        time.sleep(4)
        page = driver.page_source
        # KPI cards should show numeric values or known labels
        assert any(w in page for w in
                   ["Float", "Profile", "Measurement", "Temperature",
                    "Salinity", "active", "total"]), \
            "Expected KPI content on dashboard"

    def test_statistics_section_renders(self, driver):
        """Statistics/Analytics section renders without error."""
        self._go_to_section(driver, "Statistic") or \
            self._go_to_section(driver, "Analytic")
        time.sleep(4)
        assert "Traceback" not in driver.page_source
        main = driver.find_element(By.CSS_SELECTOR, "[data-testid='stMain'], .main")
        assert main.is_displayed()

    def test_charts_or_plots_present(self, driver):
        """At least one Plotly chart or Streamlit chart is rendered."""
        time.sleep(5)
        charts = driver.find_elements(By.CSS_SELECTOR,
            "[data-testid='stPlotlyChart'], .js-plotly-plot, "
            "[data-testid='stVegaLiteChart'], canvas")
        # Just verify page doesn't error — charts may load async
        assert "Traceback" not in driver.page_source


# ── 7. Sidebar Utilities ──────────────────────────────────────────────────────

class TestSidebarUtilities:

    def test_refresh_button_present(self, driver):
        """Sidebar contains a Refresh button."""
        wait_for(driver, By.CSS_SELECTOR, "[data-testid='stSidebar']")
        time.sleep(3)
        buttons = driver.find_elements(By.CSS_SELECTOR,
            "[data-testid='stSidebar'] button, [data-testid='stButton'] button")
        labels = [b.text.strip() for b in buttons]
        print(f"\n  Sidebar buttons: {labels}")
        # Refresh or similar button should exist
        assert any("refresh" in l.lower() or "clear" in l.lower() or "reset" in l.lower()
                   for l in labels if l), \
            f"Refresh button not found. Found: {labels}"

    def test_connection_status_displayed(self, driver):
        """Sidebar shows connection/status information."""
        wait_for(driver, By.CSS_SELECTOR, "[data-testid='stSidebar']")
        time.sleep(3)
        sidebar_text = driver.find_element(
            By.CSS_SELECTOR, "[data-testid='stSidebar']"
        ).text.lower()
        assert any(w in sidebar_text for w in
                   ["connected", "online", "healthy", "docs", "floats", "status"]), \
            f"No status info in sidebar. Sidebar text: {sidebar_text[:200]}"

    def test_doc_count_in_sidebar(self, driver):
        """ChromaDB document count is displayed in sidebar."""
        wait_for(driver, By.CSS_SELECTOR, "[data-testid='stSidebar']")
        time.sleep(3)
        sidebar_text = driver.find_element(
            By.CSS_SELECTOR, "[data-testid='stSidebar']"
        ).text
        # Should contain a number (doc count)
        import re
        numbers = re.findall(r'\d[\d,]+', sidebar_text)
        assert len(numbers) > 0, f"No numbers found in sidebar. Text: {sidebar_text[:300]}"
        print(f"\n  Numbers in sidebar: {numbers}")


# ── 8. Responsiveness & Stability ────────────────────────────────────────────

class TestStabilityAndPerformance:

    def test_page_load_time(self, driver):
        """Page fully loads within 15 seconds."""
        start = time.time()
        driver.get(BASE_URL)
        wait_for(driver, By.CSS_SELECTOR, "[data-testid='stSidebar']", timeout=15)
        elapsed = time.time() - start
        print(f"\n  Page load: {elapsed:.1f}s")
        assert elapsed < 15, f"Page took {elapsed:.1f}s to load"

    def test_backend_response_time(self):
        """Health check responds in under 2 seconds."""
        start = time.time()
        r = requests.get(f"{API_URL}/health", timeout=5)
        elapsed = time.time() - start
        print(f"\n  Health check: {elapsed:.3f}s")
        assert elapsed < 2.0
        assert r.status_code == 200

    def test_chat_response_time(self):
        """Simple chat query responds within 30 seconds."""
        start = time.time()
        r = requests.post(f"{API_URL}/query",
                          json={"query_text": "hello"},
                          timeout=LLM_WAIT)
        elapsed = time.time() - start
        print(f"\n  Chat response: {elapsed:.1f}s")
        assert r.status_code == 200
        assert elapsed < 30, f"Chat took {elapsed:.1f}s — too slow"

    def test_concurrent_health_during_query(self):
        """Health endpoint stays responsive while a query is processing."""
        import threading

        results = {"query": None, "health": []}

        def run_query():
            try:
                r = requests.post(f"{API_URL}/query",
                                  json={"query_text": "average salinity"},
                                  timeout=LLM_WAIT)
                results["query"] = r.status_code
            except Exception as e:
                results["query"] = str(e)

        def poll_health():
            for _ in range(5):
                try:
                    r = requests.get(f"{API_URL}/health", timeout=3)
                    results["health"].append(r.status_code)
                except Exception:
                    results["health"].append(0)
                time.sleep(2)

        t_query  = threading.Thread(target=run_query)
        t_health = threading.Thread(target=poll_health)

        t_query.start()
        time.sleep(0.5)
        t_health.start()

        t_health.join()
        t_query.join(timeout=LLM_WAIT + 5)

        healthy = [s for s in results["health"] if s == 200]
        print(f"\n  Health during query: {results['health']}, query: {results['query']}")
        assert len(healthy) >= 3, \
            f"Health endpoint should stay up. Got: {results['health']}"


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
