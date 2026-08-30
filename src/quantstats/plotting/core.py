# original code: QuantStats: Portfolio analytics for quants
# https://github.com/ranaroussi/quantstats Copyright 2019-2023 Ran Aroussi
# Licensed originally under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0

import contextlib

import matplotlib.dates as _mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter as _FormatStrFormatter
from matplotlib.ticker import FuncFormatter as _FuncFormatter

from quantstats import stats as _stats

sns.set_theme(
    font_scale=1.1,
    rc={
        "figure.figsize": (10, 6),
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "grid.color": "#dddddd",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "text.color": "#333333",
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        # Text as <text>, not as filled outlines. matplotlib's default embeds
        # every character's geometry and then references it, which was 43% of
        # the finished report -- more than the chart data.
        "svg.fonttype": "none",
    },
)

_FLATUI_COLORS = [
    "#FEDD78",
    "#348DC1",
    "#BA516B",
    "#4FA487",
    "#9B59B6",
    "#613F66",
    "#84B082",
    "#DC136C",
    "#559CAD",
    "#4A5899",
]
_GRAYSCALE_COLORS = [
    "#000000",
    "#222222",
    "#555555",
    "#888888",
    "#AAAAAA",
    "#CCCCCC",
    "#EEEEEE",
    "#333333",
    "#666666",
    "#999999",
]


def _get_colors(grayscale):
    colors = _FLATUI_COLORS
    ls = "-"
    alpha = 0.8
    if grayscale:
        colors = _GRAYSCALE_COLORS
        ls = "-"
        alpha = 0.5
    return colors, ls, alpha


_DARK_BG = "#141b2cff"
_DARK_INK = "#e8e8e8"
_DARK_GRID = "#333c52"


def _bg(dark: bool) -> str:
    return _DARK_BG if dark else "white"


def _ink(dark: bool) -> str:
    return _DARK_INK if dark else "black"


def _dark_rc(dark: bool) -> dict:
    """rcParams overridden for the lifetime of one figure, not the process.

    A caller renders both a light and a dark report from the same import, so
    the theme cannot live in the module-level `sns.set_theme` call above --
    only whatever is active when a given `plot_*` call opens its figure.
    """
    if not dark:
        return {}
    return {
        "axes.facecolor": _DARK_BG,
        "figure.facecolor": _DARK_BG,
        "grid.color": _DARK_GRID,
        "text.color": _DARK_INK,
        "xtick.color": _DARK_INK,
        "ytick.color": _DARK_INK,
        "axes.edgecolor": _DARK_INK,
        "axes.labelcolor": _DARK_INK,
    }


def plot_returns_bars(
    returns,
    benchmark=None,
    returns_label="Strategy",
    hline=None,
    hlw=None,
    hlcolor="red",
    hllabel="",
    resample="A",
    title="Returns",
    match_volatility=False,
    figsize=(10, 6),
    grayscale=False,
    ylabel=True,
    subtitle=True,
    savefig=None,
    dark=False,
):
    if match_volatility and benchmark is None:
        raise ValueError("match_volatility requires passing of benchmark.")
    if match_volatility and benchmark is not None:
        bmark_vol = benchmark.loc[returns.index].std()
        returns = (returns / returns.std()) * bmark_vol

    # ---------------
    colors, _, _ = _get_colors(grayscale)
    df = pd.DataFrame(index=returns.index, data={returns.name: returns})

    if isinstance(benchmark, pd.Series):
        df[benchmark.name] = benchmark[benchmark.index.isin(returns.index)]
        df = df[[benchmark.name, returns.name]]

    df = df.dropna()
    if resample is not None:
        df = df.resample(resample).apply(_stats.comp).resample(resample).last()
    # ---------------

    with plt.rc_context(_dark_rc(dark)):
        fig, ax = plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # use a more precise date string for the x axis locations in the toolbar
    if title:
        fig.suptitle(title, y=0.94, fontweight="bold", fontsize=14, color=_ink(dark))

    if subtitle:
        ax.set_title(
            "{} - {}           \n".format(
                df.index.date[:1][0].strftime("%Y"),
                df.index.date[-1:][0].strftime("%Y"),
            ),
            fontsize=12,
            color="gray",
        )

    if benchmark is None:
        colors = colors[1:]
    df.plot(kind="bar", ax=ax, color=colors)

    fig.set_facecolor(_bg(dark))
    ax.set_facecolor(_bg(dark))

    try:
        ax.set_xticklabels(df.index.year)
        years = sorted(set(df.index.year))
    except AttributeError:
        ax.set_xticklabels(df.index)
        years = sorted(set(df.index))

    # ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')
    # years = sorted(list(set(df.index.year)))
    if len(years) > 10:
        mod = int(len(years) / 10)
        plt.xticks(
            np.arange(len(years)),
            [str(year) if not i % mod else "" for i, year in enumerate(years)],
        )

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    if hline is not None and not isinstance(hline, pd.Series):
        if grayscale:
            hlcolor = "gray"
        ax.axhline(hline, ls="--", lw=hlw, color=hlcolor, label=hllabel, zorder=2)

    ax.axhline(0, ls="--", lw=1, color=_ink(dark), zorder=2)

    # if isinstance(benchmark, _pd.Series) or hline:
    ax.legend(fontsize=11)

    ax.set_xlabel("")
    if ylabel:
        ax.set_ylabel("Returns", fontweight="bold", fontsize=12, color=_ink(dark))
        ax.yaxis.set_label_coords(-0.1, 0.5)

    ax.yaxis.set_major_formatter(_FuncFormatter(format_pct_axis))

    if benchmark is None and len(pd.DataFrame(returns).columns) == 1:
        ax.get_legend().remove()

    with contextlib.suppress(Exception):
        plt.subplots_adjust(hspace=0, bottom=0, top=1)

    with contextlib.suppress(Exception):
        fig.tight_layout()

    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig)

    plt.close()
    return fig


def plot_timeseries(
    returns,
    benchmark=None,
    title="Returns",
    compound=False,
    cumulative=True,
    fill=False,
    returns_label="Strategy",
    hline=None,
    hlw=None,
    hlcolor="red",
    hllabel="",
    percent=True,
    match_volatility=False,
    resample=None,
    shade_periods=None,
    lw=1.5,
    figsize=(10, 6),
    ylabel: str = "",
    xlabel: str = "",
    grayscale=False,
    subtitle=True,
    savefig=None,
    marker: str | None = None,
    dark=False,
):
    colors, ls, alpha = _get_colors(grayscale)

    returns = returns.fillna(0)
    if isinstance(benchmark, pd.Series):
        # The benchmark's price history usually starts years before the book
        # does, and every one of those bars plots as a flat zero.
        benchmark = benchmark.loc[returns.index[0] : returns.index[-1]].fillna(0)

    if match_volatility and benchmark is None:
        raise ValueError("match_volatility requires passing of benchmark.")
    if match_volatility and benchmark is not None:
        bmark_vol = benchmark.std()
        returns = (returns / returns.std()) * bmark_vol

    # ---------------
    if compound is True:
        if cumulative:
            returns = _stats.compsum(returns)
            if isinstance(benchmark, pd.Series):
                benchmark = _stats.compsum(benchmark)
        else:
            returns = returns.cumsum()
            if isinstance(benchmark, pd.Series):
                benchmark = benchmark.cumsum()

    if resample:
        returns = returns.resample(resample)
        returns = returns.last() if compound is True else returns.sum()
        if isinstance(benchmark, pd.Series):
            benchmark = benchmark.resample(resample)
            benchmark = benchmark.last() if compound is True else benchmark.sum()
    # ---------------

    with plt.rc_context(_dark_rc(dark)):
        fig, ax = plt.subplots(figsize=figsize)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # The worst drawdown periods are shaded behind the curve rather than
    # repeated on a chart of their own.
    for start, end in shade_periods or ():
        ax.axvspan(
            *_mdates.datestr2num([str(start), str(end)]),
            color="black" if grayscale else "red",
            alpha=0.1,
        )

    if title:
        fig.suptitle(title, y=0.94, fontweight="bold", fontsize=14, color=_ink(dark))

    if subtitle:
        ax.set_title(
            "{} - {}            \n".format(
                returns.index.date[:1][0].strftime("%e %b '%y"),
                returns.index.date[-1:][0].strftime("%e %b '%y"),
            ),
            fontsize=12,
            color="gray",
        )

    fig.set_facecolor(_bg(dark))
    ax.set_facecolor(_bg(dark))

    if isinstance(benchmark, pd.Series):
        ax.plot(
            benchmark,
            lw=lw,
            ls=ls,
            label=benchmark.name,
            color=colors[0],
            marker=marker,
        )

    alpha = 0.25 if grayscale else 1
    ax.plot(
        returns, lw=lw, label=returns.name, color=colors[1], alpha=alpha, marker=marker
    )

    if fill:
        ax.fill_between(returns.index, 0, returns, color=colors[1], alpha=0.25)

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    # ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')

    if hline is not None and not isinstance(hline, pd.Series):
        if grayscale:
            hlcolor = "black"
        ax.axhline(hline, ls="--", lw=hlw, color=hlcolor, label=hllabel, zorder=2)

    ax.axhline(0, ls="-", lw=1, color="gray", zorder=1)
    ax.axhline(0, ls="--", lw=1, color="white" if grayscale else _ink(dark), zorder=2)

    # if isinstance(benchmark, _pd.Series) or hline is not None:
    ax.legend(fontsize=11)

    if percent:
        ax.yaxis.set_major_formatter(_FuncFormatter(format_pct_axis))
        # ax.yaxis.set_major_formatter(_plt.FuncFormatter(
        #     lambda x, loc: "{:,}%".format(int(x*100))))

    ax.set_xlabel(xlabel, fontweight="bold", fontsize=12, color=_ink(dark))
    if isinstance(returns.index[0], int):
        ax.xticks(returns.index)
    if ylabel:
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=12, color=_ink(dark))
    ax.yaxis.set_label_coords(-0.1, 0.5)

    if benchmark is None and len(pd.DataFrame(returns).columns) == 1:
        ax.get_legend().remove()

    with contextlib.suppress(Exception):
        plt.subplots_adjust(hspace=0, bottom=0, top=1)

    with contextlib.suppress(Exception):
        fig.tight_layout()

    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig)

    plt.close()

    return fig


def plot_histogram(
    returns,
    benchmark,
    resample="ME",
    bins=20,
    grayscale=False,
    title="Returns",
    kde=True,
    figsize=(10, 6),
    ylabel=True,
    subtitle=True,
    compounded=True,
    savefig=None,
    dark=False,
):
    # colors = ['#348dc1', '#003366', 'red']
    # if grayscale:
    #     colors = ['silver', 'gray', 'black']

    colors, _, _ = _get_colors(grayscale)

    apply_fnc = _stats.comp if compounded else np.sum
    if benchmark is not None:
        benchmark = (
            benchmark.fillna(0)
            .resample(resample)
            .apply(apply_fnc)
            .resample(resample)
            .last()
        )

    returns = (
        returns.fillna(0).resample(resample).apply(apply_fnc).resample(resample).last()
    )

    figsize = (0.995 * figsize[0], figsize[1])
    with plt.rc_context(_dark_rc(dark)):
        fig, ax = plt.subplots(figsize=figsize)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.suptitle(title, y=0.94, fontweight="bold", fontsize=14, color=_ink(dark))

    if subtitle:
        ax.set_title(
            "{} - {}           \n".format(
                returns.index.date[:1][0].strftime("%Y-%m-%d"),
                returns.index.date[-1:][0].strftime("%Y-%m-%d"),
            ),
            fontsize=10,
            color="gray",
        )

    fig.set_facecolor(_bg(dark))
    ax.set_facecolor(_bg(dark))

    if isinstance(returns, pd.DataFrame) and len(returns.columns) == 1:
        returns = returns[returns.columns[0]]

    pallete = colors[1:2] if benchmark is None else colors[:2]
    alpha = 0.7

    if benchmark is not None:
        combined_returns = (
            benchmark.to_frame()
            .join(returns.to_frame())
            # Long-form for one histogram, not a reshape by key: `melt` would
            # drop the index the next step groups on.
            .stack()
            .reset_index()
            .rename(columns={"level_1": "", 0: "Returns"})
        )

        sns.histplot(
            data=combined_returns,
            x="Returns",
            bins=bins,
            alpha=alpha,
            kde=kde,
            stat="density",
            hue="",
            palette=pallete,
            ax=ax,
        )

    else:
        combined_returns = returns.copy()
        if kde:
            sns.kdeplot(data=combined_returns, color=_ink(dark), ax=ax)
        sns.histplot(
            data=combined_returns,
            bins=bins,
            alpha=alpha,
            kde=False,
            stat="density",
            color=colors[1],
            ax=ax,
        )

    # Why do we need average?
    if isinstance(combined_returns, pd.Series) or len(combined_returns.columns) == 1:
        ax.axvline(
            combined_returns.mean(),
            ls="--",
            lw=1.5,
            zorder=2,
            label="Average",
            color="red",
        )

    # _plt.setp(x.get_legend().get_texts(), fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x * 100):,}%"))

    # Removed static lines for clarity
    # ax.axhline(0.01, lw=1, color="#000000", zorder=2)
    # ax.axvline(0, lw=1, color="#000000", zorder=2)

    ax.set_xlabel("")
    ax.set_ylabel("Occurrences", fontweight="bold", fontsize=12, color=_ink(dark))
    ax.yaxis.set_label_coords(-0.1, 0.5)

    # fig.autofmt_xdate()

    with contextlib.suppress(Exception):
        plt.subplots_adjust(hspace=0, bottom=0, top=1)

    with contextlib.suppress(Exception):
        fig.tight_layout()

    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig)

    plt.close()
    return fig


def _weekly(series: pd.Series) -> pd.Series:
    """One point a week instead of one a day.

    Every rolling statistic here moves over months, so the daily samples cost
    an order of magnitude in embedded SVG and show nothing the weekly ones do
    not. `last`, not `mean`: a rolling value is already an average.
    """
    return series.resample("W").last().dropna()


def plot_rolling_stats(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "",
    returns_label: str = "Strategy",
    hline=None,
    hlw=None,
    hlcolor: str = "red",
    hllabel: str = "",
    lw: float = 1.5,
    figsize: tuple[int, int] = (10, 6),
    ylabel: str = "",
    grayscale: bool = False,
    subtitle: bool = True,
    savefig: bool | None = None,
    reference_line: float = 0.0,
    dark: bool = False,
):
    colors, _, _ = _get_colors(grayscale)

    with plt.rc_context(_dark_rc(dark)):
        fig, ax = plt.subplots(figsize=figsize)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    returns = _weekly(returns)
    if isinstance(benchmark, pd.Series):
        benchmark = _weekly(benchmark)
    df = pd.DataFrame(index=returns.index, data={returns_label: returns})

    # A derived series (a rolling exposure, a liquidity tilt) usually has no
    # `.name` of its own; `returns_label` is what the caller meant this line
    # to be called, so it is the fallback rather than an unlabelled legend
    # entry.
    label = returns.name or returns_label
    if isinstance(benchmark, pd.Series):
        df["Benchmark"] = benchmark[benchmark.index.isin(returns.index)]
        df = df[["Benchmark", returns_label]].dropna()
        ax.plot(df[returns_label].dropna(), lw=lw, label=label, color=colors[1])
        ax.plot(
            df["Benchmark"], lw=lw, label=benchmark.name, color=colors[0], alpha=0.8
        )
    else:
        df = df[[returns_label]].dropna()
        ax.plot(df[returns_label].dropna(), lw=lw, label=label, color=colors[1])

    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    # ax.fmt_xdata = _mdates.DateFormatter('%Y-%m-%d')\
    fig.suptitle(title, y=0.94, fontweight="bold", fontsize=14, color=_ink(dark))

    if subtitle:
        ax.set_title(
            "{} - {}           \n".format(
                df.index.date[:1][0].strftime("%e %b '%y"),
                df.index.date[-1:][0].strftime("%e %b '%y"),
            ),
            fontsize=12,
            color="gray",
        )

    if hline is not None and not isinstance(hline, pd.Series):
        if grayscale:
            hlcolor = "black"
        ax.axhline(hline, ls="--", lw=hlw, color=hlcolor, label=hllabel, zorder=2)

    ax.axhline(reference_line, ls="--", lw=1, color=_ink(dark), zorder=2)

    if ylabel:
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=12, color=_ink(dark))
        ax.yaxis.set_label_coords(-0.1, 0.5)

    ax.yaxis.set_major_formatter(_FormatStrFormatter("%.2f"))

    ax.legend(fontsize=11)

    if benchmark is None and len(pd.DataFrame(returns).columns) == 1:
        ax.get_legend().remove()

    with contextlib.suppress(Exception):
        plt.subplots_adjust(hspace=0, bottom=0, top=1)

    with contextlib.suppress(Exception):
        fig.tight_layout()

    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig)
    plt.close()
    return fig


def plot_stacked(
    components: dict[str, pd.Series],
    title: str = "",
    ylabel: str = "",
    figsize: tuple[int, int] = (10, 6),
    grayscale: bool = False,
    subtitle: bool = True,
    savefig: dict | None = None,
    percent: bool = True,
    dark: bool = False,
):
    """Each series stacked on the ones before it; the top edge is their sum."""
    colors, _, _ = _get_colors(grayscale)

    with plt.rc_context(_dark_rc(dark)):
        fig, ax = plt.subplots(figsize=figsize)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    frame = pd.DataFrame(components).sort_index().fillna(0.0)

    fig.suptitle(title, y=0.94, fontweight="bold", fontsize=14, color=_ink(dark))
    if subtitle:
        ax.set_title(
            "{} - {}           \n".format(
                frame.index.date[:1][0].strftime("%e %b '%y"),
                frame.index.date[-1:][0].strftime("%e %b '%y"),
            ),
            fontsize=12,
            color="gray",
        )

    palette = (colors * (len(frame.columns) // len(colors) + 1))[: len(frame.columns)]
    ax.stackplot(
        frame.index,
        *[frame[column] for column in frame.columns],
        labels=frame.columns,
        colors=palette,
        alpha=0.85,
    )

    fig.autofmt_xdate()

    if percent:
        ax.yaxis.set_major_formatter(_FuncFormatter(format_pct_axis))

    if ylabel:
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=12, color=_ink(dark))
        ax.yaxis.set_label_coords(-0.1, 0.5)

    ax.legend(fontsize=9, ncol=min(len(frame.columns), 4), loc="upper left")

    with contextlib.suppress(Exception):
        plt.subplots_adjust(hspace=0, bottom=0, top=1)

    with contextlib.suppress(Exception):
        fig.tight_layout()

    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig)
    plt.close()
    return fig


def plot_rolling_beta(
    returns,
    benchmark,
    window1=126,
    window1_label="",
    window2=None,
    window2_label="",
    title="",
    hlcolor="red",
    figsize=(10, 6),
    grayscale=False,
    lw=1.5,
    ylabel=True,
    subtitle=True,
    savefig=None,
    dark=False,
):
    colors, _, _ = _get_colors(grayscale)

    with plt.rc_context(_dark_rc(dark)):
        fig, ax = plt.subplots(figsize=figsize)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.suptitle(title, y=0.94, fontweight="bold", fontsize=14, color=_ink(dark))

    if subtitle:
        ax.set_title(
            "{} - {}           \n".format(
                returns.index.date[:1][0].strftime("%e %b '%y"),
                returns.index.date[-1:][0].strftime("%e %b '%y"),
            ),
            fontsize=12,
            color="gray",
        )

    beta = _weekly(_stats.rolling_greeks(returns, benchmark, window1)["beta"].fillna(0))
    ax.plot(beta, lw=lw, label=window1_label, color=colors[1])

    if window2:
        lw = lw - 0.5
        ax.plot(
            _weekly(_stats.rolling_greeks(returns, benchmark, window2)["beta"]),
            lw=lw,
            label=window2_label,
            color="gray",
            alpha=0.8,
        )

    beta_min = (
        beta.min()
        if isinstance(returns, pd.Series)
        else min([b.min() for b in beta.to_numpy()()])
    )
    beta_max = (
        beta.max()
        if isinstance(returns, pd.Series)
        else max([b.max() for b in beta.to_numpy()()])
    )
    mmin = min([-100, int(beta_min * 100)])
    mmax = max([100, int(beta_max * 100)])
    step = 50 if (mmax - mmin) >= 200 else 100
    ax.set_yticks([x / 100 for x in list(range(mmin, mmax, step))])

    if isinstance(returns, pd.Series):
        hlcolor = "black" if grayscale else hlcolor
        ax.axhline(beta.mean(), ls="--", lw=1.5, color=hlcolor, zorder=2)

    ax.axhline(0, ls="--", lw=1, color=_ink(dark), zorder=2)

    fig.autofmt_xdate()

    # use a more precise date string for the x axis locations in the toolbar
    ax.fmt_xdata = _mdates.DateFormatter("%Y-%m-%d")

    if ylabel:
        ax.set_ylabel("Beta", fontweight="bold", fontsize=12, color=_ink(dark))
        ax.yaxis.set_label_coords(-0.1, 0.5)

    ax.legend(fontsize=11)
    if benchmark is None and len(pd.DataFrame(returns).columns) == 1:
        ax.get_legend().remove()

    with contextlib.suppress(Exception):
        plt.subplots_adjust(hspace=0, bottom=0, top=1)

    with contextlib.suppress(Exception):
        fig.tight_layout()

    if savefig:
        if isinstance(savefig, dict):
            plt.savefig(**savefig)
        else:
            plt.savefig(savefig)

    plt.close()
    return fig


def format_pct_axis(x, _):
    x *= 100  # lambda x, loc: "{:,}%".format(int(x * 100))
    if x >= 1e12:
        res = "%1.1fT%%" % (x * 1e-12)
        return res.replace(".0T%", "T%")
    if x >= 1e9:
        res = "%1.1fB%%" % (x * 1e-9)
        return res.replace(".0B%", "B%")
    if x >= 1e6:
        res = "%1.1fM%%" % (x * 1e-6)
        return res.replace(".0M%", "M%")
    if x >= 1e3:
        res = "%1.1fK%%" % (x * 1e-3)
        return res.replace(".0K%", "K%")
    res = f"{x:1.0f}%"
    return res.replace(".0%", "%")
