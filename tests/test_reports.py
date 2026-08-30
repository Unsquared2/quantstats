"""Pure-function helpers in `reports.py`, in isolation from a full render."""

from __future__ import annotations

import pytest

from quantstats.reports import _rows_per_day


@pytest.mark.parametrize(
    ("periods_per_year", "expected"),
    [
        (365, 1),  # daily data: one row already is one day
        (24 * 365, 24),  # hourly data: 24 rows make a day
        (12, 1),  # monthly data: never round below one row
    ],
)
def test_rows_per_day_matches_the_sampling_rate(
    periods_per_year: int, expected: int
) -> None:
    assert _rows_per_day(periods_per_year) == expected
