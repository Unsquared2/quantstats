# original code: QuantStats: Portfolio analytics for quants
# https://github.com/ranaroussi/quantstats Copyright 2019-2023 Ran Aroussi
# Licensed originally under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0


import io as _io

import numpy as np
import pandas as pd

from . import stats as _stats


def to_prices(
    returns: pd.Series | pd.DataFrame, base: float = 1e5
) -> pd.Series | pd.DataFrame:
    """Converts returns series to price data"""
    returns = returns.copy().fillna(0).replace([np.inf, -np.inf], float("NaN"))

    return base + base * _stats.compsum(returns)


def group_returns(
    returns: pd.Series | pd.DataFrame, groupby: str, compounded: bool = False
) -> pd.Series | pd.DataFrame:
    """Summarize returns
    group_returns(df, df.index.year)
    group_returns(df, [df.index.year, df.index.month])
    """
    if compounded:
        return returns.groupby(groupby).apply(_stats.comp)
    return returns.groupby(groupby).sum()


def aggregate_returns(
    returns: pd.Series | pd.DataFrame, period=None, compounded: bool = True
) -> pd.Series | pd.DataFrame:
    """Aggregates returns based on date periods"""
    if period is None or "day" in period:
        return returns
    index = returns.index

    if "month" in period:
        return group_returns(returns, index.month, compounded=compounded)

    if "quarter" in period:
        return group_returns(returns, index.quarter, compounded=compounded)

    if period == "A" or any(x in period for x in ["year", "eoy", "yoy"]):
        return group_returns(returns, index.year, compounded=compounded)

    if "week" in period:
        return group_returns(returns, index.week, compounded=compounded)

    if "eow" in period or period == "W":
        return group_returns(returns, [index.year, index.week], compounded=compounded)

    if "eom" in period or period == "M":
        return group_returns(returns, [index.year, index.month], compounded=compounded)

    if "eoq" in period or period == "Q":
        return group_returns(
            returns, [index.year, index.quarter], compounded=compounded
        )

    if not isinstance(period, str):
        return group_returns(returns, period, compounded)

    return returns


def _prepare_prices(data: pd.Series | pd.DataFrame, base: float = 1.0):
    """Converts return data into prices + cleanup"""
    data = data.copy()
    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            if data[col].dropna().min() <= 0 or data[col].dropna().max() < 1:
                data[col] = to_prices(data[col], base)

    # is it returns?
    # elif data.min() < 0 and data.max() < 1:
    elif data.min() < 0 or data.max() < 1:
        data = to_prices(data, base)

    if isinstance(data, pd.DataFrame | pd.Series):
        data = data.fillna(0).replace([np.inf, -np.inf], float("NaN"))

    return data


def _file_stream():
    """Returns a file stream"""
    return _io.BytesIO()
