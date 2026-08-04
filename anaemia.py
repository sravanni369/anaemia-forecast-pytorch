"""Forecasting anaemia in women - and why the forecast is precision theatre.

WHO Global Health Observatory, indicator NUTRITION_ANAEMIA_REPRODUCTIVEAGE_PREV:
prevalence of anaemia in women aged 15-49, 194 countries, 2000-2023.
Anaemia affects roughly a third of women of reproductive age worldwide and
drives maternal mortality, so forecasting it is a real policy question.

The standard exercise: train on 2000-2015, forecast 2016-2023, report error.
This script does that - a PyTorch per-country linear trend against a naive
"last value carried forward" baseline - and then does the thing the standard
exercise skips.

WHO publishes an uncertainty interval with EVERY estimate, so this script asks
how often each forecast lands inside that interval. If both models sit inside
it nearly always, then ranking them by a fraction of a percentage point is
reporting noise with a decimal point.

It also verifies its own arithmetic: every gradient-descent fit is checked
against the closed-form least-squares solution for the same series. An earlier
version of this script silently underfit (300 epochs, unstandardised inputs)
and produced a convincing but false result - "trend extrapolation fails on
health data" - which was really just a line that had not finished training.

Note on the data: every one of the 194 countries has a value for all 24 years,
including countries that ran no national survey in most of them. These are
modelled estimates from a Bayesian hierarchical model, not measurements - so
this is a model trained on another model's output.

Sources: linear regression + train/test split by time - Dive into Deep Learning
ch. 3 (d2l.ai); ISLP ch. 3. stdlib + torch only.

Run:  python anaemia.py   (CPU, seconds; needs raw.json - see README)
"""

import json
import statistics

import torch

SPLIT_YEAR = 2015  # train 2000-2015, forecast 2016-2023
EPOCHS = 500


def load(path="raw.json"):
    """Country -> {year: (value, ci_low, ci_high)}, countries only.

    The API returns a second dimension the obvious loader misses: every
    (country, year) appears three times, once per PREGNANCYSTATUS. Without the
    filter below each series silently mixes pregnant, non-pregnant and total
    women, and since the rows are unsorted, which one survives is arbitrary.
    """
    rows = [r for r in json.load(open(path))["value"]
            if r["SpatialDimType"] == "COUNTRY" and r["NumericValue"] is not None
            and r["Dim2"] == "PREGNANCYSTATUS_TOTAL"]
    out = {}
    for r in rows:
        out.setdefault(r["SpatialDim"], {})[r["TimeDim"]] = (
            r["NumericValue"], r["Low"], r["High"])

    # One row per cell, or a dimension is colliding again.
    kept = sum(len(s) for s in out.values())
    assert kept == len(rows), f"{len(rows)} rows collapsed into {kept} cells"
    return out


def exact_trend(x, y):
    """Closed-form least squares. The answer gradient descent should reach."""
    xm, ym = x.mean(), y.mean()
    a = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum()
    return float(a), float(ym - a * xm)


def fit_trend(years, values):
    """Fit value = a*(year - SPLIT_YEAR) + b by gradient descent.

    Standardising both axes first is what makes convergence reliable: raw
    prevalence sits near 30-60, so a zero-initialised bias needs hundreds of
    steps just to reach the data, and an underfit line is indistinguishable
    from a failed method. Returns (a, b, gap) where gap is the distance from
    the closed-form solution - the caller uses it to verify convergence
    instead of trusting the epoch count.
    """
    x = torch.tensor(years, dtype=torch.float32) - SPLIT_YEAR
    y = torch.tensor(values, dtype=torch.float32)

    xs, ys = x.std(), y.std()
    if xs == 0 or ys == 0:  # degenerate series - closed form handles it
        a, b = exact_trend(x, y)
        return a, b, 0.0
    xn = ((x - x.mean()) / xs).unsqueeze(1)
    yn = ((y - y.mean()) / ys).unsqueeze(1)

    torch.manual_seed(0)
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(EPOCHS):
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(xn), yn).backward()
        opt.step()

    with torch.no_grad():
        w, c = float(model.weight), float(model.bias)
    a = w * float(ys) / float(xs)                    # undo the scaling
    b = float(y.mean()) + c * float(ys) - a * float(x.mean())

    a_exact, b_exact = exact_trend(x, y)
    gap = max(abs(a - a_exact), abs(b - b_exact))
    return a, b, gap


def evaluate(data):
    """Per country: trend forecast error, naive error, and WHO CI width."""
    trend_err, naive_err, ci_width, gaps = [], [], [], []
    inside = {"trend": 0, "naive": 0, "total": 0}

    for country, series in data.items():
        train = sorted((y, v) for y, (v, _, _) in series.items() if y <= SPLIT_YEAR)
        test = sorted((y, v) for y, (v, _, _) in series.items() if y > SPLIT_YEAR)
        if len(train) < 5 or not test:
            continue

        a, b, gap = fit_trend([y for y, _ in train], [v for _, v in train])
        gaps.append(gap)
        last = train[-1][1]

        for year, actual in test:
            forecast = a * (year - SPLIT_YEAR) + b
            trend_err.append(abs(forecast - actual))
            naive_err.append(abs(last - actual))
            _, lo, hi = series[year]
            if lo is not None and hi is not None:
                ci_width.append(hi - lo)
                inside["total"] += 1
                inside["trend"] += lo <= forecast <= hi
                inside["naive"] += lo <= last <= hi

    return trend_err, naive_err, ci_width, gaps, inside


def summary(name, errors):
    print(f"  {name:28s} median {statistics.median(errors):5.2f} pp   "
          f"mean {statistics.mean(errors):5.2f} pp")


if __name__ == "__main__":
    data = load()
    print(f"WHO anaemia prevalence, women 15-49 | {len(data)} countries, 2000-2023")
    print(f"train {min(min(s) for s in data.values())}-{SPLIT_YEAR}, "
          f"forecast {SPLIT_YEAR+1}-2023\n")

    trend_err, naive_err, ci, gaps, inside = evaluate(data)

    # Convergence is checked, not assumed: every gradient-descent fit is compared
    # against the closed-form least-squares solution for the same series.
    print(f"convergence check vs closed-form least squares:")
    print(f"  worst coefficient gap across {len(gaps)} country fits: {max(gaps):.2e}")
    print(f"  (an underfit line disagrees by whole units - this is the check that")
    print(f"   an earlier version of this script failed silently)\n")

    print(f"Forecast error over {len(trend_err)} country-years:")
    summary("PyTorch linear trend", trend_err)
    summary("naive last-value-carried", naive_err)

    print(f"\nWHO's own uncertainty on the SAME estimates:")
    print(f"  {'published CI width':28s} median {statistics.median(ci):5.2f} pp   "
          f"mean {statistics.mean(ci):5.2f} pp")

    t = inside["total"]
    print(f"\nForecasts landing INSIDE WHO's published interval ({t} country-years):")
    print(f"  {'PyTorch linear trend':28s} {inside['trend'] / t * 100:5.1f}%")
    print(f"  {'naive last-value-carried':28s} {inside['naive'] / t * 100:5.1f}%")

    ratio = statistics.median(ci) / statistics.median(naive_err)
    worse = statistics.median(trend_err) / statistics.median(naive_err)
    print(f"\nCarrying the last value forward is {worse:.1f}x more accurate than the")
    print(f"fitted trend, and sits inside WHO's own interval {inside['naive']/t*100:.1f}% of the time.")
    print(f"That interval is {ratio:.0f}x wider than the naive error itself.")

    changes = [abs(s[2023][0] - s[2000][0]) for s in data.values()
               if 2000 in s and 2023 in s]
    print(f"\nFor scale: median change across 23 years is "
          f"{statistics.median(changes):.1f} pp,")
    print(f"against a {statistics.median(ci):.1f} pp uncertainty band on any single year.")
    print("The quantity being forecast is smaller than the error bars around it.")
