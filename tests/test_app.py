from __future__ import annotations

import sys
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satquery.historical import DEMO_AOI_LABEL, DEMO_BOUNDS  # noqa: E402


class GeoLensInterfaceTests(unittest.TestCase):
    def _app(self) -> AppTest:
        app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=20)
        app.run()
        self.assertEqual([], list(app.exception))
        return app

    def test_all_workflows_remain_available_and_historical_is_default(self) -> None:
        app = self._app()
        workflow = app.sidebar.radio[0]
        self.assertEqual(
            ["Construction changes", "Vegetation comparison", "Historical vegetation"],
            workflow.options,
        )
        self.assertEqual("Historical vegetation", workflow.value)
        self.assertEqual("Historical vegetation", app.header[0].value)
        self.assertIn("Manual structured request (fallback)", [item.label for item in app.expander])

        workflow.set_value("Construction changes").run()
        self.assertEqual([], list(app.exception))
        self.assertEqual("Construction changes", app.header[0].value)

        app.sidebar.radio[0].set_value("Vegetation comparison").run()
        self.assertEqual([], list(app.exception))
        self.assertEqual("Vegetation comparison", app.header[0].value)

    def test_demo_aoi_is_confirmed_once_and_survives_unchanged_rerun(self) -> None:
        app = self._app()
        confirmed = app.session_state["historical_confirmed_aoi"]
        self.assertEqual(DEMO_BOUNDS, confirmed.bounds)
        self.assertEqual(DEMO_AOI_LABEL, app.session_state["historical_aoi_label"])
        self.assertEqual("demo_preset", app.session_state["historical_aoi_source"])

        app.run()
        self.assertEqual([], list(app.exception))
        self.assertEqual(DEMO_BOUNDS, app.session_state["historical_confirmed_aoi"].bounds)

    def test_reset_invalidates_the_confirmed_aoi_without_reloading_it(self) -> None:
        app = self._app()
        reset = next(button for button in app.button if button.label == "Reset boundary and result")
        reset.click().run()
        self.assertEqual([], list(app.exception))
        with self.assertRaises(KeyError):
            app.session_state["historical_confirmed_aoi"]
        self.assertTrue(any("No area is confirmed" in item.value for item in app.warning))


if __name__ == "__main__":
    unittest.main()
