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

import pandas as pd
import pytest
from conftest import sample_report

from quantstats.reports import html

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


def test_turnover_is_skipped_without_components(tmp_path: Path):
    """Optional: absent `components`, the turnover chart renders nothing."""
    page = Path(sample_report(str(tmp_path / "tearsheet.html"))).read_text()
    assert '<div id="stacked_turnover_weekly"></div>' in page


def test_components_add_the_stacked_turnover_chart(tmp_path: Path):
    page = Path(
        sample_report(str(tmp_path / "tearsheet.html"), with_components=True)
    ).read_text()
    rendered = set(re.findall(r'<div id="([^"]+)"><\?xml', page))
    assert rendered == CHARTS | {"stacked_turnover_weekly"}
    assert "{{" not in page
    # Each sub-strategy's own name reaches the legend of a stacked chart.
    assert "alpha" in page
    assert "beta" in page
    assert "gamma" in page


def test_a_dark_report_renders_with_no_leftover_placeholder(tmp_path: Path):
    page = Path(
        sample_report(str(tmp_path / "tearsheet.html"), with_components=True, dark=True)
    ).read_text()
    assert "{{" not in page
    assert "#141b2cff" in page


def test_a_report_renders_with_no_benchmark_at_all(tmp_path: Path):
    """`benchmark=None` is a documented call shape (ensembler has none) --
    every chart and metric that only makes sense against a benchmark must
    drop out cleanly rather than crash on `None.name`."""
    page = Path(
        sample_report(
            str(tmp_path / "tearsheet.html"),
            with_components=True,
            with_benchmark=False,
        )
    ).read_text()
    assert "{{" not in page
    rendered = set(re.findall(r'<div id="([^"]+)"><\?xml', page))
    assert "vol_returns" not in rendered
    assert "rolling_beta" not in rendered
    assert "stacked_turnover_weekly" in rendered
    assert "rolling_gross_exposure" in rendered


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


def test_cagr_is_calendar_based_not_sampling_rate_based(tmp_path: Path):
    """CAGR must not blow up for a report scored on hourly returns.

    `periods_per_year=8760` is *sampling rate* -- how many rows make a
    year -- correct for Sharpe/volatility, which scale with sqrt(periods).
    CAGR's own year-length is real elapsed calendar days regardless: passing
    the sampling rate there raises an already-compounded return to a wildly
    wrong power. A constant hourly rate makes both the true answer and the
    bug's answer computable by hand, so this pins the right one.
    """
    hours = 24 * 400  # a bit over a year of hourly bars
    rate = 0.0002  # a mild, realistic constant hourly return
    index = pd.date_range("2024-01-01", periods=hours, freq="h")
    returns = pd.Series(rate, index=index, name="strategy")

    report = html(
        returns=returns,
        benchmark=None,
        weights=None,
        title="cagr sanity",
        periods_per_year=24 * 365,
    ).source_code

    match = re.search(r"<tr><td>CAGR﹪</td><td>(-?[\d,]+\.\d+)%</td></tr>", report)
    assert match, "CAGR row not found in the rendered metrics table"
    rendered_cagr = float(match.group(1).replace(",", "")) / 100

    total_return = (1 + rate) ** hours - 1
    calendar_days = (index[-1] - index[0]).days
    expected_cagr = abs(total_return + 1.0) ** (365 / calendar_days) - 1

    assert rendered_cagr == pytest.approx(expected_cagr, rel=1e-3)
    # The bug's answer (raising to 8760/calendar_days instead of 365/it) is
    # off by many orders of magnitude -- a loose sanity bound catches it even
    # if the exact formula above ever drifts.
    assert abs(rendered_cagr) < 100  # under 10,000%


def test_text_is_not_embedded_as_vector_outlines(tmp_path: Path):
    """`svg.fonttype = "none"` was 43% of the file. A glyph `<defs>` block
    coming back means the rcParam stopped being applied."""
    page = Path(sample_report(str(tmp_path / "tearsheet.html"))).read_text()
    assert "<text" in page
    assert len(page) < 700_000
