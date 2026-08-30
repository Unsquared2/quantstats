# original code: QuantStats: Portfolio analytics for quants
# https://github.com/ranaroussi/quantstats Copyright 2019-2023 Ran Aroussi
# Licensed originally under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0

import contextlib
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from pandas import DataFrame as _df

from quantstats import stats as _stats

from . import core as _core


def plot_returns(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series | pd.DataFrame,
    grayscale: bool = False,
    figsize: tuple[int, int] = (10, 6),
    lw: float = 1.5,
    match_volatility: bool = False,
    compound: bool = True,
    cumulative: bool = True,
    resample=None,
    ylabel: str = "Cumulative Returns",
    subtitle: bool = True,
    savefig: dict | None = None,
    shade_periods: list[tuple[object, object]] | None = None,
    dark: bool = False,
):
    title = "Cumulative Returns" if compound else "Returns"
    if benchmark is not None:
        if isinstance(benchmark, str):
            title += f" vs {benchmark.upper()}"
        else:
            title += " vs Benchmark"
        if match_volatility:
            title += " (Volatility Matched)"

    return _core.plot_timeseries(
        returns,
        benchmark,
        title,
        ylabel=ylabel,
        match_volatility=match_volatility,
        resample=resample,
        compound=compound,
        cumulative=cumulative,
        shade_periods=shade_periods,
        lw=lw,
        figsize=figsize,
        grayscale=grayscale,
        subtitle=subtitle,
        savefig=savefig,
        dark=dark,
    )


def plot_yearly_returns(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series | pd.DataFrame,
    grayscale: bool = False,
    hlw: float = 1.5,
    hlcolor: str = "red",
    hllabel: str = "",
    match_volatility: bool = False,
    figsize: tuple[int, int] = (10, 5),
    ylabel: bool = True,
    subtitle: bool = True,
    compounded: bool = True,
    savefig: dict | None = None,
    dark: bool = False,
) -> Figure:
    title = "EOY Returns"
    if benchmark is not None:
        title += "  vs Benchmark"
        benchmark = benchmark.resample("YE").apply(_stats.comp).resample("YE").last()

    if compounded:
        returns = returns.resample("YE").apply(_stats.comp)
    else:
        returns = returns.resample("YE").apply(_df.sum)
    returns = returns.resample("YE").last()

    return _core.plot_returns_bars(
        returns,
        benchmark,
        hline=returns.mean(),
        hlw=hlw,
        hllabel=hllabel,
        hlcolor=hlcolor,
        match_volatility=match_volatility,
        resample=None,
        title=title,
        figsize=figsize,
        grayscale=grayscale,
        ylabel=ylabel,
        subtitle=subtitle,
        savefig=savefig,
        dark=dark,
    )


def plot_histogram(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series | pd.DataFrame,
    resample: Literal["W", "M", "Q", "A"] = "ME",
    grayscale: bool = False,
    figsize: tuple[int, int] = (10, 5),
    ylabel: bool = True,
    subtitle: bool = True,
    compounded: bool = True,
    savefig: dict | None = None,
    dark: bool = False,
) -> Figure:
    if resample == "W":
        title = "Weekly "
    elif resample == "ME":
        title = "Monthly "
    elif resample == "QE":
        title = "Quarterly "
    elif resample == "YE":
        title = "Annual "
    else:
        title = ""

    return _core.plot_histogram(
        returns,
        benchmark,
        resample=resample,
        grayscale=grayscale,
        title=f"Distribution of {title}Returns",
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        compounded=compounded,
        savefig=savefig,
        dark=dark,
    )


def drawdown(
    returns: pd.Series | pd.DataFrame,
    grayscale: bool = False,
    figsize: tuple[int, int] = (10, 5),
    lw: float = 1,
    match_volatility: bool = False,
    compound: bool = False,
    ylabel: str = "Drawdown",
    resample: bool | None = None,
    subtitle: bool = True,
    title: str | None = "Underwater Plot",
    savefig: dict | None = None,
    dark: bool = False,
) -> Figure:
    dd = _stats.to_drawdown_series(returns)

    # Weekly, and by the week's *worst* value: a filled daily series is the
    # largest single figure in the report, and the trough is the whole point.
    dd = dd.resample("W").min().dropna()
    return _core.plot_timeseries(
        dd,
        title=title,
        hline=dd.mean(),
        hlw=2,
        hllabel="Average",
        returns_label="Drawdown",
        compound=compound,
        match_volatility=match_volatility,
        resample=resample,
        fill=True,
        lw=lw,
        figsize=figsize,
        ylabel=ylabel,
        grayscale=grayscale,
        subtitle=subtitle,
        savefig=savefig,
        dark=dark,
    )


def plot_rolling_beta(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series | pd.DataFrame,
    window1: int = 126,
    window1_label: str = "6-Months",
    window2: int = 252,
    window2_label: str = "12-Months",
    lw: float = 1.5,
    grayscale: bool = False,
    figsize: tuple[int, int] = (10, 3),
    ylabel: bool = True,
    subtitle: bool = True,
    savefig: dict | None = None,
    dark: bool = False,
) -> Figure:
    return _core.plot_rolling_beta(
        returns,
        benchmark,
        window1=window1,
        window1_label=window1_label,
        window2=window2,
        window2_label=window2_label,
        title="Rolling Beta to Benchmark",
        grayscale=grayscale,
        lw=lw,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        savefig=savefig,
        dark=dark,
    )


def _component_gross(weights: pd.DataFrame) -> pd.Series:
    return weights.abs().sum(axis="columns")


def _component_turnover(weights: pd.DataFrame) -> pd.Series:
    return weights.diff(1).abs().sum(axis="columns")


def plot_rolling_exposure(
    weights: pd.DataFrame,
    period: int = 5,
    period_label: str = "1 Week",
    lw: float = 1.5,
    grayscale: bool = False,
    figsize: tuple[int, int] = (10, 3),
    ylabel: str = "Rolling Exposure",
    subtitle: bool = True,
    savefig: dict | None = None,
    net_or_gross: Literal["net", "gross"] = "net",
    reference_line: float = 0.0,
    components: dict[str, pd.DataFrame] | None = None,
    dark: bool = False,
) -> Figure:
    assert net_or_gross in ["net", "gross"], (
        "net_or_gross must be either 'net' or 'gross'"
    )
    if net_or_gross == "gross":
        ylabel = "Rolling Gross Exposure"
        # Each sub-strategy's own gross, stacked, rather than the combined
        # book's -- a component can be added or retired without redrawing.
        if components:
            title = f"{ylabel} ({period_label})"
            stacked = {
                name: _component_gross(component).rolling(period).mean()
                for name, component in components.items()
            }
            return _core.plot_stacked(
                stacked,
                title=title,
                ylabel=ylabel,
                grayscale=grayscale,
                figsize=figsize,
                subtitle=subtitle,
                savefig=savefig,
                dark=dark,
            )
        weights = weights.abs()
    else:
        ylabel = "Rolling Net Exposure"

    rolling_weights = weights.sum(axis="columns").rolling(period).mean()

    return _core.plot_rolling_stats(
        rolling_weights,
        hline=weights.sum(axis="columns").mean(),
        hlw=1.5,
        ylabel=ylabel,
        returns_label=ylabel,
        title=f"{ylabel} ({period_label})",
        grayscale=grayscale,
        lw=lw,
        figsize=figsize,
        subtitle=subtitle,
        savefig=savefig,
        reference_line=reference_line,
        dark=dark,
    )


def plot_turnover(
    components: dict[str, pd.DataFrame],
    smooth: Literal["W", "3M"] = "W",
    grayscale: bool = False,
    figsize: tuple[int, int] = (10, 4),
    ylabel: str = "Turnover",
    subtitle: bool = True,
    savefig: dict | None = None,
    dark: bool = False,
) -> Figure:
    """Stacked turnover by sub-strategy: each area is that book's own share.

    `smooth="W"` resamples daily turnover to a weekly mean; `smooth="3M"`
    takes a trailing 90-day mean instead, still drawn at one point a week --
    raw daily turnover is too jagged to read once several areas are stacked.
    """
    if smooth == "W":
        title = "Turnover (Weekly Smoothed)"
        stacked = {
            name: _component_turnover(frame).resample("W").mean()
            for name, frame in components.items()
        }
    else:
        title = "Turnover (3-Month Smoothed)"
        stacked = {
            name: _component_turnover(frame)
            .rolling("90D", min_periods=1)
            .mean()
            .resample("W")
            .last()
            for name, frame in components.items()
        }

    return _core.plot_stacked(
        stacked,
        title=title,
        ylabel=ylabel,
        grayscale=grayscale,
        figsize=figsize,
        subtitle=subtitle,
        savefig=savefig,
        dark=dark,
    )


def plot_liquidity_tilt(
    weights: pd.DataFrame,
    liquidity: pd.DataFrame,
    lw: float = 1.5,
    grayscale: bool = False,
    figsize: tuple[int, int] = (10, 3),
    subtitle: bool = True,
    savefig: dict | None = None,
    dark: bool = False,
) -> Figure:
    """Where the book's gross sits in its own cross-section's liquidity ranking.

    0.5 is liquidity-neutral: gross spread with no regard for how much of each
    name trades. Below it the book leans on the thinner half of what it holds.
    """
    tilt = _stats.liquidity_tilt(weights, liquidity)
    return _core.plot_rolling_stats(
        tilt,
        hline=tilt.mean(),
        hlw=1.5,
        ylabel="Liquidity Tilt",
        returns_label="Liquidity Tilt",
        title="Gross-Weighted Liquidity Percentile (1 Day)",
        grayscale=grayscale,
        lw=lw,
        figsize=figsize,
        subtitle=subtitle,
        savefig=savefig,
        reference_line=0.5,
        dark=dark,
    )


def plot_rolling_volatility(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series | pd.DataFrame,
    period: int = 126,
    period_label: str = "6-Months",
    periods_per_year: int = 252,
    lw: float = 1.5,
    grayscale: int = False,
    figsize: tuple[int, int] = (10, 3),
    ylabel: str = "Volatility",
    subtitle: bool = True,
    savefig: dict | None = None,
    dark: bool = False,
) -> Figure:
    returns = _stats.rolling_volatility(returns, period, periods_per_year)

    if benchmark is not None:
        benchmark = _stats.rolling_volatility(benchmark, period, periods_per_year)

    return _core.plot_rolling_stats(
        returns,
        benchmark,
        hline=returns.mean(),
        hlw=1.5,
        ylabel=ylabel,
        title=f"Rolling Volatility ({period_label})",
        grayscale=grayscale,
        lw=lw,
        figsize=figsize,
        subtitle=subtitle,
        savefig=savefig,
        dark=dark,
    )


def plot_rolling_sharpe(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series | pd.DataFrame,
    rf: float = 0.0,
    period: int = 126,
    period_label: str = "6-Months",
    periods_per_year: int = 252,
    lw: float = 1.25,
    grayscale: bool = False,
    figsize: tuple[int, int] = (10, 3),
    ylabel: str = "Sharpe",
    subtitle: bool = True,
    savefig: dict | None = None,
    dark: bool = False,
) -> Figure:
    returns = _stats.rolling_sharpe(
        returns,
        rf,
        period,
        True,
        periods_per_year,
    )

    if benchmark is not None:
        benchmark = _stats.rolling_sharpe(benchmark, rf, period, True, periods_per_year)

    return _core.plot_rolling_stats(
        returns,
        benchmark,
        hline=returns.mean(),
        hlw=1.5,
        ylabel=ylabel,
        title=f"Rolling Sharpe ({period_label})",
        grayscale=grayscale,
        lw=lw,
        figsize=figsize,
        subtitle=subtitle,
        savefig=savefig,
        dark=dark,
    )


def plot_monthly_heatmap(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series | pd.DataFrame,
    annot_size: int = 10,
    figsize: tuple[int, int] = (10, 5),
    cbar: bool = True,
    square: bool = False,
    returns_label: str = "Strategy",
    compounded: bool = True,
    eoy: bool = False,
    grayscale: bool = False,
    ylabel: bool = True,
    savefig: dict | None = None,
    active: bool = False,
    dark: bool = False,
) -> Figure:
    # colors, ls, alpha = _core._get_colors(grayscale)
    cmap = "gray" if grayscale else "RdYlGn"

    returns = _stats.monthly_returns(returns, eoy=eoy, compounded=compounded) * 100

    fig_height = len(returns) / 2.5

    if figsize is None:
        size = list(plt.gcf().get_size_inches())
        figsize = (size[0], size[1])

    figsize = (figsize[0], max([fig_height, figsize[1]]))

    if cbar:
        figsize = (figsize[0] * 1.051, max([fig_height, figsize[1]]))

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.set_facecolor(_core._bg(dark))
    ax.set_facecolor(_core._bg(dark))

    # _sns.set(font_scale=.9)
    if active and benchmark is not None:
        ax.set_title(
            f"{returns_label} - Monthly Active Returns (%)\n",
            fontsize=14,
            y=0.995,
            fontweight="bold",
            color=_core._ink(dark),
        )
        benchmark = (
            _stats.monthly_returns(benchmark, eoy=eoy, compounded=compounded) * 100
        )
        active_returns = returns - benchmark

        ax = sns.heatmap(
            active_returns,
            ax=ax,
            annot=True,
            center=0,
            annot_kws={"size": annot_size},
            fmt="0.2f",
            linewidths=0.5,
            square=square,
            cbar=cbar,
            cmap=cmap,
            cbar_kws={"format": "%.0f%%"},
        )
    else:
        ax.set_title(
            f"{returns_label} - Monthly Returns (%)\n",
            fontsize=14,
            y=0.995,
            fontweight="bold",
            color=_core._ink(dark),
        )
        ax = sns.heatmap(
            returns,
            ax=ax,
            annot=True,
            center=0,
            annot_kws={"size": annot_size},
            fmt="0.2f",
            linewidths=0.5,
            square=square,
            cbar=cbar,
            cmap=cmap,
            cbar_kws={"format": "%.0f%%"},
        )
    # _sns.set(font_scale=1)

    # align plot to match other
    if ylabel:
        ax.set_ylabel("Years", fontweight="bold", fontsize=12, color=_core._ink(dark))
        ax.yaxis.set_label_coords(-0.1, 0.5)

    ax.tick_params(colors="#808080")
    plt.xticks(rotation=0, fontsize=annot_size * 1.2)
    plt.yticks(rotation=0, fontsize=annot_size * 1.2)

    with contextlib.suppress(Exception):
        plt.subplots_adjust(hspace=0, bottom=0, top=1)
    with contextlib.suppress(Exception):
        fig.tight_layout(w_pad=0, h_pad=0)

    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig)

    plt.close()
    return fig
