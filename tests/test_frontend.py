from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "frontend"


class ImportedFrontendTests(unittest.TestCase):
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
