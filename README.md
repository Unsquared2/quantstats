# quantstats

Portfolio tearsheets: metrics, drawdowns, exposure and liquidity, rendered to
one self-contained HTML page.

A fork of the `quantstats` inside `finml-utils`, which is itself a fork of
[ranaroussi/quantstats](https://github.com/ranaroussi/quantstats). It exists
because the report needed changes the upstream has no reason to carry, and
because it is consumed by more than one caller.

## Use

```python
from quantstats.reports import html

html(
    returns=returns,  # a daily return series
    benchmark=benchmark,  # the series to measure against
    weights=weights,  # dates x assets, missing filled with zero
    liquidity=adv,  # dates x assets, optional
    components={  # optional: one weights frame per sub-strategy
        "spectra_private": spectra_weights,
        "unsquared_v3_daily_okx": unsquared_weights,
    },
    title="spectra_private.40",
    subtitle="2021-01-01 to 2026-08-28, top 40",
    periods_per_year=365,
    background_dark=True,  # optional: dark chrome and dark chart backgrounds
).write_html("tearsheet.html")
```

`weights` must have **zero, not NaN**, where a name is not held. `sum` treats
a null as zero, so net and gross come out right either way, but `diff().abs()`
does not: a position opening or closing reads as missing and registers no
trade, which understates turnover. Do not forward-fill it -- a null means the
name is not held that day, and filling carries every delisted name's last
weight forever, so gross exposure climbs with the length of the sample instead
of staying at the book's leverage.

`components` carries the same convention, one frame per sub-strategy. Passing
it draws two additional charts -- turnover stacked by sub-strategy, smoothed
weekly and again over a trailing 3 months -- right after "Cumulative Returns
vs Benchmark (Volatility Matched)", and switches the gross exposure chart from
the combined book's single line to a stack of each sub-strategy's own gross.
Omit it and both charts behave exactly as before: no turnover chart, gross
exposure as one line.

## What it does differently

- **Twelve charts, not twenty.** Everything the one entry point does not reach
  is gone: the in-sample/out-of-sample split, component statistics, delayed
  sharpes, comparison bars, the returns distribution, daily returns. 4043
  lines became 2795.
- **No log scale.** Nothing switches axis on its own.
- **The worst five drawdowns shade the first chart** rather than getting a
  chart of their own that redraws the same equity line, and the underwater
  plot sits beneath it at a fifth the height.
- **Liquidity.** `stats.liquidity_tilt` ranks each row's held names on their
  own trailing volume and takes the gross-weighted mean percentile; 0.5 is
  neutral. It shows as a chart and as two metrics rows against their neutral
  values, so a book that quietly leans on its thinnest names says so.
- **A tenth the bytes.** Text is emitted as `<text>` rather than as filled
  vector outlines, which was 43% of the file, and path coordinates are rounded
  to a tenth of a point. A report went from 540 KB to 240 KB, and 132 KB to
  43 KB gzipped, without a visible difference.
- **Faster.** `comp`, `remove_outliers` and `drawdown_details` run in numpy
  rather than per-slice pandas: 4.89s to 2.61s for a full render.
- **Sub-strategy composition.** Pass `components` and the turnover chart and
  the gross exposure chart both stack by sub-strategy instead of drawing the
  combined book alone -- each area is one component's own contribution, so
  the total is still the combined book's own figure.
- **Dark mode.** `background_dark=True` matches the existing dark-chrome HTML
  option: every chart's own background, gridlines, axis text and reference
  lines flip too, not just the page around them.

## Tests

`tests/test_parity.py` asserts every statistic still reachable from
`reports.html` against the `finml_utils` implementation it was forked from, in
the same process on the same inputs, at zero tolerance. That is what makes the
speed-ups and the strip safe: none of them is allowed to move a number.

`tests/test_render.py` renders a report end to end, which is the only place
the template and the figures meet -- a chart removed from one and not the
other shows up there and nowhere else.

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
```
