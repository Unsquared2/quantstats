"""The report renders end to end.

Every other test here is a statistic in isolation. This one is the only thing
that puts the template and the figures together, which is where a chart that
was removed from one and not the other shows up.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from conftest import sample_report

CHARTS = {
    "cumulative_returns",
    "dd_plot",
    "vol_returns",
    "eoy_returns",
    "rolling_net_exposure",
    "rolling_gross_exposure",
    "liquidity_tilt",
    "monthly_dist",
    "rolling_beta",
    "rolling_vol",
    "rolling_sharpe",
    "monthly_heatmap",
}


def test_a_report_renders_every_chart(tmp_path: Path):
    page = Path(sample_report(str(tmp_path / "tearsheet.html"))).read_text()
    rendered = set(re.findall(r'<div id="([^"]+)"><\?xml', page))
    assert rendered == CHARTS


def test_no_template_placeholder_survives(tmp_path: Path):
    page = Path(sample_report(str(tmp_path / "tearsheet.html"))).read_text()
    assert "{{" not in page


def test_the_report_carries_the_exposure_and_liquidity_rows(tmp_path: Path):
    page = Path(sample_report(str(tmp_path / "tearsheet.html"))).read_text()
    for row in (
        "Average Net Exposure",
        "Average Gross Exposure",
        "Daily Turnover",
        "Weighted Liquidity Pctile",
        "Gross in Least-Liquid 5th",
    ):
        assert row in page


def test_text_is_not_embedded_as_vector_outlines(tmp_path: Path):
    """`svg.fonttype = "none"` was 43% of the file. A glyph `<defs>` block
    coming back means the rcParam stopped being applied."""
    page = Path(sample_report(str(tmp_path / "tearsheet.html"))).read_text()
    assert "<text" in page
    assert len(page) < 700_000
