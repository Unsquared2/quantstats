"""This package against the `finml_utils.quantstats` it was forked from.

The fork exists to drop what nobody calls, to carry the liquidity statistics,
and to run the hot paths in numpy. None of that is allowed to move a number,
so every statistic still reachable from `reports.html` is asserted against the
original -- in the same process on the same inputs, rather than against stored
values that drift.
"""

import numpy as np
import pandas as pd
import pytest
from finml_utils.quantstats import stats as original

from quantstats import one_sided
from quantstats import stats as fork

PERIODS = 365


ONE_ARGUMENT = [
    "comp",
    "compsum",
    "to_drawdown_series",
    "max_drawdown",
    "volatility",
    "sharpe",
    "sortino",
    "calmar",
    "skew",
    "kurtosis",
    "value_at_risk",
    "conditional_value_at_risk",
    "best",
    "worst",
    "win_rate",
    "avg_win",
    "avg_loss",
    "consecutive_wins",
    "consecutive_losses",
    "recovery_factor",
    "ulcer_index",
    "serenity_index",
    "cagr",
    "expected_return",
]


@pytest.mark.parametrize("name", [n for n in ONE_ARGUMENT if hasattr(fork, n)])
def test_one_argument_statistics_match_the_original(name, returns):
    forked, source = getattr(fork, name), getattr(original, name)
    np.testing.assert_allclose(
        np.asarray(forked(returns), dtype=float),
        np.asarray(source(returns), dtype=float),
        rtol=0,
        atol=0,
        err_msg=name,
    )


@pytest.mark.parametrize("quantile", [0.5, 0.95, 0.99])
def test_remove_outliers_matches_the_original(returns, quantile):
    pd.testing.assert_series_equal(
        fork.remove_outliers(returns, quantile),
        original.remove_outliers(returns, quantile),
    )


def test_comp_matches_the_original_with_missing_values(returns):
    holed = returns.copy()
    holed.iloc[[3, 40, 900]] = np.nan
    assert fork.comp(holed) == original.comp(holed)


def test_drawdown_details_match_the_original(returns):
    """The numpy inner loop is the largest single speed-up in the fork and the
    easiest place to move a number, so it is compared column by column."""
    series = fork.to_drawdown_series(returns)
    forked = fork.drawdown_details(series)
    source = original.drawdown_details(series)
    assert len(forked) == len(source) > 5
    pd.testing.assert_frame_equal(
        forked.reset_index(drop=True), source.reset_index(drop=True)
    )


def test_greeks_match_the_original(returns, benchmark):
    pd.testing.assert_series_equal(
        fork.greeks(returns, benchmark, PERIODS),
        original.greeks(returns, benchmark, PERIODS),
    )


def test_rolling_statistics_match_the_original(returns, benchmark):
    pd.testing.assert_series_equal(
        fork.rolling_volatility(returns, 60, PERIODS),
        original.rolling_volatility(returns, 60, PERIODS),
    )
    pd.testing.assert_series_equal(
        fork.rolling_sharpe(returns, rolling_period=60, periods_per_year=PERIODS),
        original.rolling_sharpe(returns, rolling_period=60, periods_per_year=PERIODS),
    )


def _weights_and_liquidity() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2024-01-01", periods=200, freq="D")
    tickers = [f"T{i}" for i in range(20)]
    rng = np.random.default_rng(2)
    weights = pd.DataFrame(rng.normal(size=(200, 20)), index=index, columns=tickers)
    weights = weights.div(weights.abs().sum(axis=1), axis=0)
    liquidity = pd.DataFrame(
        rng.lognormal(18, 1.2, size=(200, 20)), index=index, columns=tickers
    )
    return weights, liquidity


def test_liquidity_tilt_is_neutral_when_weight_ignores_liquidity():
    """Gross spread without regard to liquidity ranks at the middle of the
    cross-section, which is what makes 0.5 the reference line."""
    weights, liquidity = _weights_and_liquidity()
    assert fork.liquidity_tilt(weights, liquidity).mean() == pytest.approx(
        0.5, abs=0.02
    )
    assert fork.illiquid_quintile_share(weights, liquidity) == pytest.approx(
        0.2, abs=0.02
    )


def test_liquidity_tilt_falls_when_gross_moves_to_the_thin_names():
    weights, liquidity = _weights_and_liquidity()
    thinnest = liquidity.rank(axis=1) <= 4
    tilted = weights.where(~thinnest, weights * 6.0)
    tilted = tilted.div(tilted.abs().sum(axis=1), axis=0)

    assert (
        fork.liquidity_tilt(tilted, liquidity).mean()
        < fork.liquidity_tilt(weights, liquidity).mean()
    )
    assert fork.illiquid_quintile_share(tilted, liquidity) > 0.4


def test_liquidity_tilt_ignores_names_the_liquidity_frame_does_not_cover():
    weights, liquidity = _weights_and_liquidity()
    assert fork.liquidity_tilt(weights, liquidity.drop(columns=["T0"])).notna().all()


def test_liquidity_tilt_is_missing_where_too_few_names_are_held():
    """Five names cannot rank each other into quintiles; the row says nothing
    rather than saying something arbitrary."""
    weights, liquidity = _weights_and_liquidity()
    weights.iloc[10, 3:] = 0.0
    assert np.isnan(fork.liquidity_tilt(weights, liquidity).iloc[10])


ONE_SIDED = [
    "beta",
    "downside_beta",
    "upside_beta",
    "weighted_downside_beta",
    "downside_correlation",
    "upside_correlation",
]


@pytest.mark.parametrize("name", ONE_SIDED)
def test_the_one_sided_statistics_match_finml_utils(name, returns, benchmark):
    """Ported rather than imported, so a plotting package does not depend on a
    process pool and a dotenv reader for six lines of covariance."""
    from finml_utils import stats as source

    assert getattr(one_sided, name)(returns, benchmark) == pytest.approx(
        getattr(source, name)(returns, benchmark)
    )
