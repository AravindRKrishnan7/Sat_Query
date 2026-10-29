from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / ".streamlit" / "config.toml"
APP_PATH = ROOT / "app.py"


def _relative_luminance(hex_colour: str) -> float:
    channels = [int(hex_colour[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(first: str, second: str) -> float:
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


class GeoLensThemeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.app_source = APP_PATH.read_text(encoding="utf-8")
        match = re.search(r"def _inject_geolens_theme\(\).*?<style>(.*?)</style>", cls.app_source, re.S)
        if match is None:
            raise AssertionError("GeoLens theme style block is missing")
        cls.css = match.group(1)

    def test_native_light_theme_uses_required_palette_and_visible_borders(self) -> None:
        theme = self.config["theme"]
        self.assertEqual("light", theme["base"])
        self.assertEqual("#F8FAFC", theme["backgroundColor"])
        self.assertEqual("#FFFFFF", theme["secondaryBackgroundColor"])
        self.assertEqual("#172033", theme["textColor"])
        self.assertEqual("#475569", theme["grayTextColor"])
        self.assertEqual("#0F766E", theme["primaryColor"])
        self.assertEqual("#CBD5E1", theme["borderColor"])
        self.assertTrue(theme["showWidgetBorder"])
        self.assertTrue(theme["showSidebarBorder"])

    def test_sidebar_inherits_a_complete_light_widget_palette(self) -> None:
        sidebar = self.config["theme"]["sidebar"]
        self.assertEqual("#FFFFFF", sidebar["backgroundColor"])
        self.assertEqual("#F8FAFC", sidebar["secondaryBackgroundColor"])
        self.assertEqual("#172033", sidebar["textColor"])
        self.assertEqual("#475569", sidebar["grayTextColor"])
        self.assertEqual("#0F766E", sidebar["primaryColor"])
        self.assertEqual("#CBD5E1", sidebar["borderColor"])
        self.assertTrue(sidebar["showWidgetBorder"])

    def test_required_text_and_primary_button_pairs_meet_wcag_aa(self) -> None:
        for foreground, background in (
            ("#172033", "#F8FAFC"),
            ("#172033", "#FFFFFF"),
            ("#475569", "#F8FAFC"),
            ("#475569", "#FFFFFF"),
            ("#FFFFFF", "#0F766E"),
        ):
            with self.subTest(foreground=foreground, background=background):
                self.assertGreaterEqual(_contrast_ratio(foreground, background), 4.5)

    def test_injected_css_avoids_widget_wide_text_and_colour_overrides(self) -> None:
        for forbidden in (
            ".stApp",
            "stSidebar",
            "data-baseweb",
            ".stButton",
            "button[kind",
            "* { color",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.css)
        stable_test_ids = set(re.findall(r'data-testid="([^"]+)"', self.css))
        self.assertEqual(
            {"stMainBlockContainer", "stExpander", "stChatMessage"},
            stable_test_ids,
        )
        for colour in ("#172033", "#475569", "#0f766e", "#ffffff", "#cbd5e1"):
            self.assertIn(colour, self.css)


if __name__ == "__main__":
    unittest.main()
