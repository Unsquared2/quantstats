"""Beta and correlation measured on one side of the benchmark's moves.

Ported from `finml_utils.stats` rather than imported: these five functions and
their two helpers are all this package needed from it, and depending on the
whole library for them pulls in a process pool, a progress bar and a dotenv
reader that a plotting package has no use for.

`tests/test_parity.py` asserts every one of them against the original.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Side = Literal["upside", "downside"]
Method = Literal["pearson", "kendall", "spearman"]


def _one_side(
    returns: pd.Series, underlying: pd.Series, side: Side
) -> tuple[pd.Series, pd.Series]:
    """Both series restricted to the dates the benchmark moved that way."""
    if side == "upside":
        moved = underlying[underlying > 0]
    elif side == "downside":
        moved = underlying[underlying < 0]
    else:
        raise ValueError("side must be either 'upside' or 'downside'")
    shared = moved.index.intersection(returns.index)
    return returns.loc[shared], underlying.loc[shared]


def beta(returns: pd.Series, underlying: pd.Series) -> float:
    """Covariance over the benchmark's variance."""
    matrix = np.cov(returns, underlying)
    return float(matrix[0, 1] / matrix[1, 1])


def downside_beta(returns: pd.Series, underlying: pd.Series) -> float:
    return beta(*_one_side(returns, underlying, "downside"))


def upside_beta(returns: pd.Series, underlying: pd.Series) -> float:
    return beta(*_one_side(returns, underlying, "upside"))


def weighted_downside_beta(returns: pd.Series, underlying: pd.Series) -> float:
    """`downside_beta` with each date weighted by how far the benchmark fell."""
    down_returns, down_underlying = _one_side(returns, underlying, "downside")
    weights = down_underlying / down_underlying.sum()
    matrix = np.cov(down_returns, down_underlying, aweights=weights)
    return float(matrix[0, 1] / matrix[1, 1])


def downside_correlation(
    returns: pd.Series, underlying: pd.Series, method: Method = "pearson"
) -> float:
    down_returns, down_underlying = _one_side(returns, underlying, "downside")
    return float(down_returns.corr(down_underlying, method=method))


def upside_correlation(
    returns: pd.Series, underlying: pd.Series, method: Method = "pearson"
) -> float:
    up_returns, up_underlying = _one_side(returns, underlying, "upside")
    return float(up_returns.corr(up_underlying, method=method))
