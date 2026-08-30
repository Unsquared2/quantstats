"""Shared fixtures, and the one end-to-end render CI runs as its own job."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _returns() -> pd.Series:
    rng = np.random.default_rng(0)
    index = pd.date_range("2021-01-01", periods=1500, freq="D")
    values = rng.normal(0.0015, 0.03, len(index))
    # A run of losses deep enough to make the drawdown table interesting.
    values[400:460] -= 0.02
    return pd.Series(values, index=index, name="strategy")


def _benchmark(series: pd.Series) -> pd.Series:
    rng = np.random.default_rng(1)
    return pd.Series(
        series.to_numpy() * 0.6 + rng.normal(0, 0.02, len(series)),
        index=series.index,
        name="benchmark",
    )


def _weights_and_liquidity(
    index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(2)
    tickers = [f"T{i}" for i in range(20)]
    weights = pd.DataFrame(
        rng.normal(size=(len(index), 20)), index=index, columns=tickers
    )
    weights = weights.div(weights.abs().sum(axis=1), axis=0)
    liquidity = pd.DataFrame(
        rng.lognormal(18, 1.2, size=(len(index), 20)), index=index, columns=tickers
    )
    return weights, liquidity


def _components(index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Three sub-strategies whose weights sum to roughly one combined book."""
    rng = np.random.default_rng(3)
    tickers = [f"T{i}" for i in range(20)]
    components = {}
    for name in ("alpha", "beta", "gamma"):
        raw = rng.normal(size=(len(index), 20))
        scaled = raw / (np.abs(raw).sum(axis=1, keepdims=True) * 3)
        components[name] = pd.DataFrame(scaled, index=index, columns=tickers)
    return components


@pytest.fixture(scope="module")
def returns() -> pd.Series:
    return _returns()


@pytest.fixture(scope="module")
def benchmark(returns: pd.Series) -> pd.Series:
    return _benchmark(returns)


def sample_report(
    path: str,
    *,
    with_components: bool = False,
    dark: bool = False,
    with_benchmark: bool = True,
) -> str:
    """Render one report to `path`. Called by the tests and by CI's own job."""
    from quantstats.reports import html

    series = _returns()
    weights, liquidity = _weights_and_liquidity(series.index)
    html(
        returns=series,
        benchmark=_benchmark(series) if with_benchmark else None,
        weights=weights,
        liquidity=liquidity,
        title="Sample",
        subtitle="synthetic",
        periods_per_year=365,
        components=_components(series.index) if with_components else None,
        background_dark=dark,
    ).write_html(path)
    return path
