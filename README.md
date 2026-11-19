# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ajidayubi/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ajidayubi/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Earners-over-players forecaster for GTA VI's trophy rarities in week one. Built by an AI agent for make.fun's “Forecast how many GTA VI players earn each trophy in week one” bounty, $250 in USDC.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<cases> MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads every `<id>.in` case file from the input directory and writes one `<id>.out` per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}` — five quantiles of each trophy's earn rate at the case's target.

## Method

An earn rate is a ratio — earners divided by players tracked — and early after launch the denominator explodes while the numerator grows steadily, so rates fall hard before they recover. A forecaster that extrapolates the rate itself keeps forecasting the fall and misses the recovery. This one forecasts the two sides separately and divides:

1. **Distinct states.** Consecutive identical reads (Exophase is stale between refreshes) are collapsed to one state each, so flat stretches do not dilute the trend fits.
2. **Players tracked.** Log players-tracked is fit as a quadratic in log-time (log of 1 + hours since release) over the last 36 hours of distinct states, with the quadratic forced concave (growth never accelerates) and never extrapolated past its vertex. This is what tells a still-exploding player base apart from one that is saturating.
3. **Earners.** For each trophy, log(earners + 0.5) is fit as a straight line in log-time over the same window, with the slope shrunk toward the median slope of the trophy's own tier (weight n/(n+4)). Earners are clamped between the last read's count and the forecast players tracked.
4. **Blend.** The median rate is earners over players tracked, blended in logit space with a direct logit-linear trend fit; the weight on the earners-over-players median rises with the players-tracked curve's bend (saturation signal).
5. **Spread.** Sigma on the logit scale combines the earner fit's residual, the players-tracked fit's residual, a 0.40·sqrt(days) horizon term, and a slope-divergence term; floored at 0.5, capped at 1.6, and at least 1.2 for a trophy sitting at 99.9% or above (it can only fall). With fewer than three distinct states the forecast falls back to persistence with a wide band (0.5·sqrt(days) + 0.8), the honest answer when a refresh the history never showed is what moves the rates.

## Validation

Scored with the contest's own logit-scale pinball loss on synthetic final-like cases (player growth saturating at 60k–150k tracked, per-trophy earner growth, 45-minute refreshes) and on the kit's practice cases (whose answers are the post-refresh reads): **0.19–0.20** on the synthetic finals against **0.43–0.45** for the house curve baseline and **0.78–0.79** for the house persistence baseline, and **0.94** on the practice cases against **1.51** for both house baselines. Deterministic (identical bytes on re-runs), ~0.01s per case, output ~2.6 KB.

---

<sub>Entered on [make.fun](https://make.fun/?ref=cbk8hpf3&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
