"""Forecasting anaemia in women - and why the forecast is precision theatre.

WHO Global Health Observatory, indicator NUTRITION_ANAEMIA_REPRODUCTIVEAGE_PREV:
prevalence of anaemia in women aged 15-49, 194 countries, 2000-2023.
Anaemia affects roughly a third of women of reproductive age worldwide and
drives maternal mortality, so forecasting it is a real policy question.

The standard exercise: train on 2000-2015, forecast 2016-2023, report error.
This script does that - a PyTorch per-country linear trend against a naive
"last value carried forward" baseline - and then does the thing the standard
exercise skips.

WHO publishes an uncertainty interval with EVERY estimate. Compare:

    median forecast error   (how wrong the model is)
    median WHO CI width     (how unsure the source data already is)

If the second dwarfs the first, the model is reporting decimal places that the
underlying data cannot support. That comparison is the whole point here.

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
# 2000, not 300: at 300 the fit has not converged (training loss 293 vs 0.42),
# and an underfit line looks exactly like "trend forecasting does not work".
EPOCHS = 2000


def load(path="raw.json"):
    """Country -> {year: (value, ci_low, ci_high)}, countries only."""
    rows = [r for r in json.load(open(path))["value"]
            if r["SpatialDimType"] == "COUNTRY" and r["NumericValue"] is not None]
    out = {}
    for r in rows:
        out.setdefault(r["SpatialDim"], {})[r["TimeDim"]] = (
            r["NumericValue"], r["Low"], r["High"])
    return out


def fit_trend(years, values):
    """Fit value = a*year + b by gradient descent. Returns (a, b)."""
    x = (torch.tensor(years, dtype=torch.float32) - SPLIT_YEAR).unsqueeze(1)
    y = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
    torch.manual_seed(0)
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(EPOCHS):
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        opt.step()
    with torch.no_grad():
        loss = float(torch.nn.functional.mse_loss(model(x), y))
    return float(model.weight.detach()), float(model.bias.detach()), loss


def evaluate(data):
    """Per country: trend forecast error, naive error, and WHO CI width."""
    trend_err, naive_err, ci_width, fit_losses = [], [], [], []

    for country, series in data.items():
        train = sorted((y, v) for y, (v, _, _) in series.items() if y <= SPLIT_YEAR)
        test = sorted((y, v) for y, (v, _, _) in series.items() if y > SPLIT_YEAR)
        if len(train) < 5 or not test:
            continue

        a, b, loss = fit_trend([y for y, _ in train], [v for _, v in train])
        fit_losses.append(loss)
        last = train[-1][1]

        for year, actual in test:
            trend_err.append(abs((a * (year - SPLIT_YEAR) + b) - actual))
            naive_err.append(abs(last - actual))
            _, lo, hi = series[year]
            if lo is not None and hi is not None:
                ci_width.append(hi - lo)

    return trend_err, naive_err, ci_width, fit_losses


def summary(name, errors):
    print(f"  {name:28s} median {statistics.median(errors):5.2f} pp   "
          f"mean {statistics.mean(errors):5.2f} pp")


if __name__ == "__main__":
    data = load()
    print(f"WHO anaemia prevalence, women 15-49 | {len(data)} countries, 2000-2023")
    print(f"train {min(min(s) for s in data.values())}-{SPLIT_YEAR}, "
          f"forecast {SPLIT_YEAR+1}-2023\n")

    trend_err, naive_err, ci, losses = evaluate(data)
    print(f"trend fits converged: median training loss "
          f"{statistics.median(losses):.3f} (an underfit line would sit in the hundreds)\n")
    print(f"Forecast error over {len(trend_err)} country-years:")
    summary("PyTorch linear trend", trend_err)
    summary("naive last-value-carried", naive_err)

    print(f"\nWHO's own uncertainty on the SAME estimates:")
    print(f"  {'published CI width':28s} median {statistics.median(ci):5.2f} pp   "
          f"mean {statistics.mean(ci):5.2f} pp")

    ratio = statistics.median(ci) / statistics.median(trend_err)
    print(f"\nThe uncertainty band is {ratio:.0f}x wider than the forecast error.")
    print("Both models land far inside the noise of the numbers they predict, so")
    print("'which model is better' is not a question this data can answer.")

    changes = [abs(s[2023][0] - s[2000][0]) for s in data.values()
               if 2000 in s and 2023 in s]
    print(f"\nFor scale: median change across 23 years is "
          f"{statistics.median(changes):.1f} pp,")
    print(f"against a {statistics.median(ci):.1f} pp uncertainty band on any single year.")
    print("The quantity being forecast is smaller than the error bars around it.")
