# original code: QuantStats: Portfolio analytics for quants
# https://github.com/ranaroussi/quantstats Copyright 2019-2023 Ran Aroussi
# Licensed originally under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0

from warnings import warn

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm
from weightcraft import gross, held, row_counts, row_rank_pct, row_sums

from . import utils as _utils

# ======== STATS ========


def compsum(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Calculates rolling compounded returns"""
    return returns.add(1).cumprod() - 1


def comp(returns: pd.DataFrame) -> pd.Series:
    """Calculates total compounded returns"""
    if isinstance(returns, pd.Series):
        # `groupby(...).apply(comp)` calls this once per group, so the pandas
        # arithmetic around a product of a few dozen numbers dominates it.
        # `nanprod`, not `prod`: `Series.prod` skips missing values.
        return np.nanprod(returns.to_numpy(dtype=float) + 1.0) - 1.0
    return returns.add(1).prod(axis=0) - 1


def expected_return(
    returns: pd.Series | pd.DataFrame,
    aggregate=None,
    compounded: bool = True,
):
    """
    Returns the expected return for a given period
    by calculating the geometric holding period return
    """
    returns = _utils.aggregate_returns(returns, aggregate, compounded)
    return np.prod(1 + returns, axis=0) ** (1 / len(returns)) - 1


def remove_outliers(returns, quantile=0.95):
    """Returns series of returns without the outliers"""
    if isinstance(returns, pd.Series):
        values = returns.to_numpy(dtype=float)
        # `Series.quantile` drops missing values first and interpolates
        # linearly, which is `np.nanquantile`'s default too.
        keep = values < np.nanquantile(values, quantile)
        return returns[keep]
    return returns[returns < returns.quantile(quantile)]


def best(returns, aggregate=None, compounded=True):
    """Returns the best day/month/week/quarter/year's return"""
    return _utils.aggregate_returns(returns, aggregate, compounded).max()


def worst(returns, aggregate=None, compounded=True):
    """Returns the worst day/month/week/quarter/year's return"""
    return _utils.aggregate_returns(returns, aggregate, compounded).min()


def win_rate(
    returns: pd.Series | pd.DataFrame,
    aggregate=None,
    compounded: bool = True,
):
    """Calculates the win ratio for a period"""

    def _win_rate(series):
        try:
            return len(series[series > 0]) / len(series[series != 0])
        except Exception:
            return 0.0

    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)

    if isinstance(returns, pd.DataFrame):
        _df = {}
        for col in returns.columns:
            _df[col] = _win_rate(returns[col])

        return pd.Series(_df)

    return _win_rate(returns)


def volatility(returns, periods=252, annualize=True):
    """Calculates the volatility of returns for a period"""
    std = returns.std()
    if annualize:
        return std * np.sqrt(periods)

    return std


def rolling_volatility(returns, rolling_period=126, periods_per_year=252):
    return returns.rolling(rolling_period).std() * np.sqrt(periods_per_year)


# ======= METRICS =======


def sharpe(returns, rf=0.0, periods=252, annualize=True, smart=False):
    """
    Calculates the sharpe ratio of access returns

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms

    Args:
        * returns (Series, DataFrame): Input return series
        * rf (float): Risk-free rate expressed as a yearly (annualized) return
        * periods (int): Freq. of returns (252/365 for daily, 12 for monthly)
        * annualize: return annualize sharpe?
    """
    if rf != 0 and periods is None:
        raise Exception("Must provide periods if rf != 0")

    res = returns.mean() / returns.std(ddof=1)

    if annualize:
        return res * np.sqrt(1 if periods is None else periods)

    return res


def rolling_sharpe(
    returns,
    rf=0.0,
    rolling_period=126,
    annualize=True,
    periods_per_year=252,
):
    if rf != 0 and rolling_period is None:
        raise Exception("Must provide periods if rf != 0")

    res = returns.rolling(rolling_period).mean() / returns.rolling(rolling_period).std()

    if annualize:
        return res * np.sqrt(1 if periods_per_year is None else periods_per_year)
    return res


def sortino(returns, rf=0, periods=252, annualize=True, smart=False):
    """
    Calculates the sortino ratio of access returns

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms

    Calculation is based on this paper by Red Rock Capital
    http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf
    """
    if rf != 0 and periods is None:
        raise Exception("Must provide periods if rf != 0")

    downside = np.sqrt((returns[returns < 0] ** 2).sum() / len(returns))
    res = returns.mean() / downside

    if annualize:
        return res * np.sqrt(1 if periods is None else periods)

    return res


def cagr(returns, rf=0.0, compounded=True, periods=252):
    """
    Calculates the communicative annualized growth return
    (CAGR%) of access returns

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms
    """
    total = returns
    total = comp(total) if compounded else np.sum(total)

    years = (returns.index[-1] - returns.index[0]).days / periods

    res = abs(total + 1.0) ** (1.0 / years) - 1

    if isinstance(returns, pd.DataFrame):
        res = pd.Series(res)
        res.index = returns.columns

    return res


def skew(returns):
    """
    Calculates returns' skewness
    (the degree of asymmetry of a distribution around its mean)
    """
    return returns.skew()


def kurtosis(returns):
    """
    Calculates returns' kurtosis
    (the degree to which a distribution peak compared to a normal distribution)
    """
    return returns.kurtosis()


def calmar(returns):
    """Calculates the calmar ratio (CAGR% / MaxDD%)"""
    cagr_ratio = cagr(returns)
    max_dd = max_drawdown(returns)
    return cagr_ratio / abs(max_dd)


def ulcer_index(returns):
    """Calculates the ulcer index score (downside risk measurment)"""
    dd = to_drawdown_series(returns)
    return np.sqrt(np.divide((dd**2).sum(), returns.shape[0] - 1))


def serenity_index(returns, rf=0):
    """
    Calculates the serenity index score
    (https://www.keyquant.com/Download/GetFile?Filename=%5CPublications%5CKeyQuant_WhitePaper_APT_Part1.pdf)
    """
    dd = to_drawdown_series(returns)
    pitfall = -cvar(dd) / returns.std()
    return (returns.sum() - rf) / (ulcer_index(returns) * pitfall)


def risk_of_ruin(returns):
    """
    Calculates the risk of ruin
    (the likelihood of losing all one's investment capital)
    """
    wins = win_rate(returns)
    return ((1 - wins) / (1 + wins)) ** len(returns)


def value_at_risk(returns, sigma=1, confidence=0.95):
    """
    Calculats the daily value-at-risk
    (variance-covariance calculation with confidence n)
    """
    mu = returns.mean()
    sigma *= returns.std()

    if confidence > 1:
        confidence = confidence / 100

    return _norm.ppf(1 - confidence, mu, sigma)


def var(returns, sigma=1, confidence=0.95):
    """Shorthand for value_at_risk()"""
    return value_at_risk(returns, sigma, confidence)


def conditional_value_at_risk(returns, sigma=1, confidence=0.95):
    """
    Calculats the conditional daily value-at-risk (aka expected shortfall)
    quantifies the amount of tail risk an investment
    """
    var = value_at_risk(returns, sigma, confidence)
    c_var = returns[returns < var].to_numpy().mean()
    return c_var if ~np.isnan(c_var) else var


def cvar(returns, sigma=1, confidence=0.95):
    """Shorthand for conditional_value_at_risk()"""
    return conditional_value_at_risk(returns, sigma, confidence)


def recovery_factor(returns, rf=0.0):
    """Measures how fast the strategy recovers from drawdowns"""
    total_returns = returns.sum() - rf
    max_dd = max_drawdown(returns)
    return abs(total_returns) / abs(max_dd)


def max_drawdown(prices):
    """Calculates the maximum drawdown"""
    prices = _utils._prepare_prices(prices)
    return (prices / prices.expanding(min_periods=0).max()).min() - 1


def to_drawdown_series(returns):
    """Convert returns series to drawdown series"""
    prices = _utils._prepare_prices(returns)
    dd = prices / np.maximum.accumulate(prices) - 1.0
    return dd.replace([np.inf, -np.inf, -0], 0)


def drawdown_details(drawdown):
    """
    Calculates drawdown details, including start/end/valley dates,
    duration, max drawdown and max dd for 99% of the dd period
    for every drawdown period
    """

    def _drawdown_details(drawdown):
        # mark no drawdown
        no_dd = drawdown == 0

        # extract dd start dates, first date of the drawdown
        starts = ~no_dd & no_dd.shift(1)
        starts = list(starts[starts.to_numpy()].index)

        # extract end dates, last date of the drawdown
        ends = no_dd & (~no_dd).shift(1)
        ends = ends.shift(-1, fill_value=False)
        ends = list(ends[ends.to_numpy()].index)

        # no drawdown :)
        if not starts:
            return pd.DataFrame(
                index=[],
                columns=(
                    "start",
                    "valley",
                    "end",
                    "days",
                    "max drawdown",
                    "99% max drawdown",
                ),
            )

        # drawdown series begins in a drawdown
        if ends and starts[0] > ends[0]:
            starts.insert(0, drawdown.index[0])

        # series ends in a drawdown fill with last date
        if not ends or starts[-1] > ends[-1]:
            ends.append(drawdown.index[-1])

        # build dataframe from results
        #
        # One numpy slice per period rather than a pandas label slice: the
        # periods are short and there are hundreds of them, so the pandas
        # indexing around each one costs more than the arithmetic in it.
        index = drawdown.index
        values = drawdown.to_numpy(dtype=float)
        first = index.searchsorted(starts, side="left")
        last = index.searchsorted(ends, side="right")

        data = []
        for i, _ in enumerate(starts):
            window = values[first[i] : last[i]]
            # `-remove_outliers(-dd, 0.99).min()` is the smallest value left
            # once the deepest 1% are dropped -- `>` here because the sign
            # flips twice.
            cutoff = -np.nanquantile(-window, 0.99)
            kept = window[window > cutoff]
            data.append(
                (
                    starts[i],
                    index[first[i] + int(np.nanargmin(window))],
                    ends[i],
                    (ends[i] - starts[i]).days + 1,
                    np.nanmin(window) * 100,
                    (np.nanmin(kept) if kept.size else np.nan) * 100,
                )
            )

        df = pd.DataFrame(
            data=data,
            columns=(
                "start",
                "valley",
                "end",
                "days",
                "max drawdown",
                "99% max drawdown",
            ),
        )
        df["days"] = df["days"].astype(int)
        df["max drawdown"] = df["max drawdown"].astype(float)
        df["99% max drawdown"] = df["99% max drawdown"].astype(float)

        df["start"] = df["start"].dt.strftime("%Y-%m-%d")
        df["end"] = df["end"].dt.strftime("%Y-%m-%d")
        df["valley"] = df["valley"].dt.strftime("%Y-%m-%d")

        return df

    if isinstance(drawdown, pd.DataFrame):
        _dfs = {}
        for col in drawdown.columns:
            _dfs[col] = _drawdown_details(drawdown[col])
        return pd.concat(_dfs, axis=1, sort=True)

    return _drawdown_details(drawdown)


# ==== VS. BENCHMARK ====


def greeks(returns, benchmark, periods=252.0):
    """Calculates alpha and beta of the portfolio"""
    # find covariance
    matrix = np.cov(returns, benchmark)
    beta = matrix[0, 1] / matrix[1, 1]
    alpha = returns.mean() - beta * benchmark.mean()
    alpha = alpha * periods
    return pd.Series(
        {
            "beta": beta,
            "alpha": alpha,
        }
    ).fillna(0)


def rolling_greeks(returns, benchmark, periods=252):
    """Calculates rolling alpha and beta of the portfolio"""
    df = pd.DataFrame(
        data={
            "returns": returns,
            "benchmark": benchmark,
        }
    )
    df = df.fillna(0)
    corr = df.rolling(int(periods)).corr().unstack()["returns"]["benchmark"]  # noqa: PD010 -- reshape only; pivot_table would aggregate duplicates
    std = df.rolling(int(periods)).std()
    beta = corr * std["returns"] / std["benchmark"]
    alpha = df["returns"].mean() - beta * df["benchmark"].mean()
    return pd.DataFrame(index=returns.index, data={"beta": beta, "alpha": alpha})


def compare(
    returns,
    benchmark,
    aggregate=None,
    compounded=True,
    round_vals=None,
):
    """
    Compare returns to benchmark on a
    day/week/month/quarter/year basis
    """
    if isinstance(returns, pd.Series):
        data = pd.DataFrame(
            data={
                "Benchmark": _utils.aggregate_returns(benchmark, aggregate, compounded)
                * 100,
                "Returns": _utils.aggregate_returns(returns, aggregate, compounded)
                * 100,
            }
        )
        data["Multiplier"] = data["Returns"] / data["Benchmark"]
        data["Won"] = np.where(data["Returns"] >= data["Benchmark"], "+", "-")
    elif isinstance(returns, pd.DataFrame):
        bench = {
            "Benchmark": _utils.aggregate_returns(benchmark, aggregate, compounded)
            * 100
        }
        strategy = {
            "Returns_" + str(i): _utils.aggregate_returns(
                returns[col], aggregate, compounded
            )
            * 100
            for i, col in enumerate(returns.columns)
        }
        data = pd.DataFrame(data={**bench, **strategy})
    if round_vals is not None:
        return np.round(data, round_vals)
    return data


def monthly_returns(returns, eoy=True, compounded=True):
    """Calculates monthly returns"""
    if isinstance(returns, pd.DataFrame):
        warn(  # noqa
            "Pandas DataFrame was passed (Series expected). "
            "Only first column will be used."
        )
        returns = returns.copy()
        returns.columns = map(str.lower, returns.columns)
        if len(returns.columns) > 1 and "close" in returns.columns:
            returns = returns["close"]
        else:
            returns = returns[returns.columns[0]]

    original_returns = returns.copy()

    returns = pd.DataFrame(
        _utils.group_returns(returns, returns.index.strftime("%Y-%m-01"), compounded)
    )

    returns.columns = ["Returns"]
    returns.index = pd.to_datetime(returns.index)

    # get returnsframe
    returns["Year"] = returns.index.strftime("%Y")
    returns["Month"] = returns.index.strftime("%b")

    # make pivot table
    returns = returns.pivot(index="Year", columns="Month", values="Returns").fillna(0)  # noqa: PD010 -- reshape only; pivot_table would aggregate duplicates

    # handle missing months
    for month in [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]:
        if month not in returns.columns:
            returns.loc[:, month] = 0

    # order columns by month
    returns = returns[
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    ]

    if eoy:
        returns["eoy"] = _utils.group_returns(
            original_returns, original_returns.index.year, compounded=compounded
        ).to_numpy()

    returns.columns = (str(x).upper() for x in returns.columns)
    returns.index.name = None

    return returns


_MINIMUM_RANKABLE = 5


def _liquidity_percentile(
    weights: pd.DataFrame, liquidity: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Per row: each held name's share of gross, and where it ranks on liquidity.

    The ranking is within the row's own held set, so 0.5 is the neutral answer
    whatever the universe is that day. Names the liquidity frame does not cover
    are dropped from both sides rather than treated as illiquid.
    """
    shared = [column for column in weights.columns if column in liquidity.columns]
    held_weights = weights[shared].to_numpy(dtype=float)
    held_liquidity = (
        liquidity[shared].reindex(index=weights.index).ffill().to_numpy(dtype=float)
    )

    holdings = held(held_weights) & np.isfinite(held_liquidity) & (held_liquidity > 0.0)
    ranked = np.where(holdings, held_liquidity, np.nan)

    counts = row_counts(ranked)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        # `row_rank_pct` puts the most liquid name at 1.0; half a step back
        # centres the row on 0.5, which is what makes neutral readable.
        percentile = row_rank_pct(ranked) - 0.5 / counts
    # A handful of names cannot rank each other into quintiles.
    percentile = np.where(counts >= _MINIMUM_RANKABLE, percentile, np.nan)

    absolute = np.where(holdings, np.abs(held_weights), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = absolute / gross(absolute)
    return share, percentile


def liquidity_tilt(weights: pd.DataFrame, liquidity: pd.DataFrame) -> pd.Series:
    """Gross-weighted mean liquidity percentile, one value per row.
    0.5 is neutral; below it the book leans on the thinner half of what it holds.
    A row with nothing rankable is missing, not zero -- `nansum` would answer
    zero, and the chart would read that as the book going wholly illiquid.
    """
    share, percentile = _liquidity_percentile(weights, liquidity)
    contribution = share * percentile
    ranked = np.isfinite(contribution).any(axis=1)
    return pd.Series(
        np.where(ranked, row_sums(contribution)[:, 0], np.nan), index=weights.index
    )


def illiquid_quintile_share(weights: pd.DataFrame, liquidity: pd.DataFrame) -> float:
    """Mean share of gross sitting in the least-liquid fifth of the row.
    20% is neutral."""
    share, percentile = _liquidity_percentile(weights, liquidity)
    ranked = np.isfinite(share * percentile).any(axis=1)
    bottom = row_sums(np.where(percentile < 0.2, share, 0.0))[:, 0]
    return float(np.nanmean(np.where(ranked, bottom, np.nan)))
