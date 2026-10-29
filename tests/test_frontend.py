from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "frontend"


class ImportedFrontendTests(unittest.TestCase):
    @staticmethod
    def _css_rule(source: str, selector: str) -> str:
        match = re.search(rf"{re.escape(selector)}\s*\{{([^}}]*)\}}", source)
        if match is None:
            raise AssertionError(f"missing CSS rule: {selector}")
        return match.group(1)

    def test_active_app_uses_real_api_contract_without_simulated_results(self) -> None:
        source = (FRONTEND / "src" / "App.tsx").read_text(encoding="utf-8")
        for forbidden in (
            "setTimeout",
            "Simulated execution",
            "No live model",
            "Sample result",
            "infrastructureImages",
        ):
            self.assertNotIn(forbidden, source)
        for endpoint in (
            "/api/infrastructure/plans",
            "/api/vegetation/plans",
            "/api/jobs/",
            "/followups",
            "/report",
        ):
            self.assertIn(endpoint, source)

    def test_page_uses_one_main_scroll_area_at_desktop_and_narrow_widths(self) -> None:
        css = (FRONTEND / "src" / "index.css").read_text(encoding="utf-8")
        workspace = self._css_rule(css, ".workspace")
        chat = self._css_rule(css, ".chat-scroll")
        evidence = self._css_rule(css, ".evidence-panel")
        sidebar = self._css_rule(css, ".sidebar")
        self.assertIn("min-height: 100vh", workspace)
        self.assertIn("overflow: visible", workspace)
        self.assertIsNone(re.search(r"(?:^|;)\s*height:\s*100vh", workspace))
        self.assertIn("overflow: visible", chat)
        self.assertNotIn("overflow-y: auto", chat)
        self.assertIn("overflow: visible", evidence)
        self.assertNotIn("overflow-y: auto", evidence)
        self.assertIn("position: sticky", sidebar)
        self.assertIn("overflow-y: auto", sidebar)
        self.assertNotIn("min-height: 720px", css)
        self.assertNotIn("min-height: 600px", css)

    def test_vegetation_entry_uses_backend_preset_and_map_fallback(self) -> None:
        source = (FRONTEND / "src" / "App.tsx").read_text(encoding="utf-8")
        css = (FRONTEND / "src" / "index.css").read_text(encoding="utf-8")
        self.assertIn('/api/vegetation/preset', source)
        self.assertIn("function GeographicMap", source)
        self.assertIn("tile.openstreetmap.org", source)
        self.assertNotIn("const DEMO_BOUNDS", source)
        self.assertIn("Basemap tiles are unavailable", source)
        self.assertIn("Previous saved observation", source)
        self.assertIn("preset.preview.label", source)
        self.assertIn(".confirmed-rectangle", css)
        self.assertIn(".map-tile-error", css)

    def test_vite_configuration_has_no_figma_runtime_plugins(self) -> None:
        config = (FRONTEND / "vite.config.ts").read_text(encoding="utf-8")
        self.assertIn("127.0.0.1", config)
        self.assertIn("port: 5173", config)
        self.assertNotIn("figma", config.casefold())
        self.assertNotIn("siteConfiguration", config)
        html = (FRONTEND / "index.html").read_text(encoding="utf-8")
        self.assertIn("<title>GeoLens</title>", html)
        self.assertNotIn("figma:", html)

    def test_package_contract_is_local_and_reproducible(self) -> None:
        package = json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))
        self.assertEqual(package["name"], "geolens-frontend")
        self.assertEqual(package["engines"]["node"], ">=22.12")
        self.assertEqual(package["engines"]["npm"], ">=10")
        self.assertIn("127.0.0.1", package["scripts"]["dev"])
        self.assertTrue((FRONTEND / "package-lock.json").is_file())
        self.assertFalse((FRONTEND / "pnpm-lock.yaml").exists())


if __name__ == "__main__":
    unittest.main()
