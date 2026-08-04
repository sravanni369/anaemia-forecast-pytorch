# Anaemia in Women — forecasting a number smaller than its own error bar 🩸

**WHO's uncertainty band on these estimates is 27.7 points wide. The total change
being forecast is 3.9 points across 23 years.** The quantity is smaller than the
error bars around it, and no model fixes that.

| | Median error | Mean error | Lands inside WHO's own interval |
|---|---|---|---|
| PyTorch linear trend | 2.34 pp | 2.95 pp | 96.8% of forecasts |
| naive "assume nothing changed" | **0.90 pp** | **1.31 pp** | 99.8% of forecasts |
| **WHO's published uncertainty on the same estimates** | **27.70 pp** | 27.59 pp | — |

Carrying the last value forward is **2.6× more accurate** than the trained
model, and lands inside WHO's own published interval 99.8% of the time — against
an interval **31× wider** than its own error.

> **Correction (4 Aug 2026).** An earlier version of this README reported 2.45 pp
> vs 2.20 pp and concluded the two models were indistinguishable. That was wrong,
> and the cause was a loader bug described below. The corrected result is
> stronger, not weaker: the trend model is decisively worse, not tied.

## The problem

WHO Global Health Observatory, indicator `NUTRITION_ANAEMIA_REPRODUCTIVEAGE_PREV`:
anaemia prevalence in women aged 15–49, 194 countries, 2000–2023. Anaemia affects
roughly a third of women of reproductive age worldwide and contributes to maternal
mortality, so forecasting it is a real policy question.

Standard exercise: train on 2000–2015, forecast 2016–2023, report the error. This
repo does that, then does the part the standard exercise skips — **it compares the
forecast error against the uncertainty WHO publishes alongside every single
estimate.**

## The bug I shipped into my own first result — and the fix

My first run reported the trend model at **14.83 pp error**, six times worse than
the naive baseline. It looked like a clean finding: "trend extrapolation fails on
health data."

It was my bug. Prevalence values sit around 30–60, so a zero-initialised bias needs
hundreds of steps just to reach the data; at 300 epochs the fit had not converged
(training loss 293 versus 0.42). **An underfit line is indistinguishable from a
failed method.**

Raising the epoch count made the number correct but the code no problem-free — it
still depended on a hardcoded guess being large enough for every country. Two
structural fixes replaced it:

1. **Standardise both axes before fitting**, then rescale the coefficients back.
   Convergence no longer depends on the units of the series, and 500 epochs now
   does what 2000 did before.
2. **Check the answer instead of trusting the loop.** Every gradient-descent fit is
   compared against the closed-form least-squares solution for the same series.
   The script prints the worst disagreement across all 194 fits:

```
worst coefficient gap across 194 country fits: 4.43e-06
```

An underfit line disagrees by whole units. This check is the one the earlier
version failed silently, and it now runs on every execution — the reader verifies
convergence rather than taking my word for it.

Fixing convergence did not move the trend error at all — the number that finally
moved everything was the loader bug below, found later by an automated audit.
Two separate bugs, both found after publishing, both documented here rather than
quietly patched.

## The second bug: a dimension I never looked at

The WHO endpoint returns a second dimension the obvious loader misses.
**Every `(country, year)` appears three times** — once for `PREGNANT`, once for
`NONPREGNANT`, once for `TOTAL`:

```
IND 2015  PREGNANCYSTATUS_PREGNANT      48.5
IND 2015  PREGNANCYSTATUS_NONPREGNANT   50.9
IND 2015  PREGNANCYSTATUS_TOTAL         50.8
```

My loader wrote all three into the same slot, so the last one won. The rows
aren't sorted, so which population survived each year was arbitrary. Every
country's "time series" was a mix of three different populations — and re-running
the download could have produced different headline numbers.

13,968 rows were collapsing into 4,656 cells without a word of complaint. The fix
is one filter plus an assertion so the failure can't recur silently:

```python
and r["Dim2"] == "PREGNANCYSTATUS_TOTAL"
...
assert kept == len(rows), f"{len(rows)} rows collapsed into {kept} cells"
```

This barely moved the trend model — line-fitting averages the jitter away — but
it inflated the naive baseline by 2.4×, because carrying 2015 forward fails badly
when 2016 is drawn from a different population. That manufactured the near-tie
the original write-up was built on.

## What the comparison shows

- The trained model **loses to doing nothing, decisively** (2.34 vs 0.90 pp).
  When a series barely moves, "assume no change" is very hard to beat.
- **99.8% of naive forecasts land inside WHO's published interval**, against an
  interval 31× wider than the naive error. By the source's own accounting, doing
  nothing is indistinguishable from correct.
- **These are modelled estimates, not measurements.** All 194 countries have a
  value for all 24 years — including countries that ran no national survey in most
  of them. The wide intervals are WHO being honest about that. Training on this
  data is modelling a model's output, and the error bars are the tell.

Previous entries in this series found leaks
([duplicate rows](https://github.com/sravanni369/maternal-risk-pytorch),
[the label in the feature table](https://github.com/sravanni369/birthweight-leakage-pytorch))
and a [model that lost to a majority baseline](https://github.com/sravanni369/cervical-screening-pytorch).
This one is different: the modelling is fine, the *precision* is fake.

## Run it

```bash
pip install torch
curl -o raw.json "https://ghoapi.azureedge.net/api/NUTRITION_ANAEMIA_REPRODUCTIVEAGE_PREV"
python anaemia.py
```

stdlib + PyTorch only, CPU, a couple of minutes for 194 country fits. Full unedited
output in [`run_log.txt`](run_log.txt); the run captured live in VS Code:

![VS Code run](vscode_run.png)

## Honest scope

A per-country straight line is a deliberately simple model; ARIMA or a hierarchical
model pooling across countries would likely fit better, and it would not change the
conclusion, because the limit here is the uncertainty in the source estimates rather
than the flexibility of the forecaster. Country-level prevalence says nothing about
any individual woman, and aggregate trends hide within-country inequality — the
women worst affected are not the average. Nothing here is a health projection; WHO
publishes those with proper uncertainty propagation, which this does not attempt.

**Source:** WHO Global Health Observatory (accessed 4 Aug 2026), indicator
`NUTRITION_ANAEMIA_REPRODUCTIVEAGE_PREV`. Linear regression and time-based splits —
[Dive into Deep Learning](https://d2l.ai) ch. 3, [ISLP](https://www.statlearning.com)
ch. 3. Adapted, not transcribed.
