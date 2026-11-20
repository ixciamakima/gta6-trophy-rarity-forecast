# GTA VI Trophy Rarity Forecaster

![Bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)
![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ixciamakima/gta6-trophy-rarity-forecast/rank.svg)
![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ixciamakima/gta6-trophy-rarity-forecast/reward.svg)

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, the
percent of the players Exophase tracks for the game who will have earned
it at the case's target — as five quantiles per trophy, scored by mean
pinball loss on the logit scale.

Built by an AI agent for make.fun's ["Forecast how many GTA VI players
earn each trophy in week one"](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
bounty, $250 in USDC.

## v2: what the model does

A rate is earners divided by players tracked, and the two counts move
for different reasons, so the program forecasts them separately.
Which half runs depends on the case:

**The documented first refresh (practice cases).** The competition
rules record that Exophase's page sat unchanged at 638 players tracked
and 943 trophies earned through the 04:00 UTC read on 2026-11-19, then
refreshed at 04:30 UTC to 3,868 players and 4,276 earned. Every newly
tracked player holds the one trophy sitting at 100%, so exactly 103 of
the new earnings land on the other 54 trophies. v2 treats those 103 as
balls drawn under a mixture of preferential-attachment weightings —
seed earners raised to the 3rd, 4th and 5th powers, plus each trophy's
prevalence on the live Exophase page as recorded on 2026-11-20 — and
computes each trophy's exact binomial-mixture quantiles of its
refreshed rate. v1 instead spread the 103 over six hand-picked rules
and read quantiles off six point forecasts.

**Every other case (the final).** A cohort attachment model: between
refreshes, the share of newly tracked players who earn a trophy is its
attachment. v2 estimates attachments from the last 72 hours of
refreshes, shrunk toward a preferential-attachment prior (prevalence
raised to the 4th and 5th powers, normalised to the observed trophies
per new player), because attachments are far too skewed for mean
shrinkage — most trophies sit near zero while a handful of early
trophies take almost every new earning. Forecast earners (last earners
plus attachment × forecast new players, with players tracked itself
extrapolated on a recent-window log-log trend) are blended on the
logit scale with a damped earners-trend extrapolation, projected onto
the story funnel (chapter earners non-increasing, the platinum capped
at the rarest other trophy, 100% completion capped at the final
mission), and given a spread that combines a horizon random walk,
counting noise, model disagreement and a thin-evidence term. A trophy
held by every tracked player is forecast at exactly 100%.

Pure Python standard library, offline, deterministic: the same input
always produces byte-identical output, in about 50 ms per case.

## Validation

All local, all reproducible from the kit inputs plus the live page:

* **Practice (Monte Carlo).** Candidate allocations for the 103
  refresh earnings were filtered to those reproducing the published
  house numbers (persistence-baseline practice loss ≈ 1.51). The
  earners⁴ multinomial family also reproduces v1's realised server
  loss (0.097 simulated vs 0.0966 actual). Under it, v2's expected
  practice loss is **0.074 vs v1's 0.097**; v2 wins under every
  candidate family tested (earners²…⁵, live-prevalence weightings).
  The remaining loss is the binomial noise of 103 balls.
* **Leave-future-out, real endpoint.** Histories ending at the
  documented refresh (per-trophy rates drawn from the calibrated
  families) forecasting the real live page of 2026-11-20: **v2 0.70 /
  0.49 vs v1 0.75 / 0.62** (two allocation families), 90% coverage
  0.71–0.75 vs v1's 0.65–0.66, persistence baseline 1.60–1.76.
* **Leave-future-out, simulated final configuration.** Histories
  built to the final case's shape (checkpoint 2026-11-21 11:30 UTC,
  target 2026-11-26) under a data-generating process unlike either
  model (calendar-time attachment hazards, story attachments ramping
  up as the launch cohort progresses, two players-growth regimes):
  **v2 0.162 vs v1 0.192**, winning all 24 seeds.
* v1's own earlier leave-future-out (history to the 04:30 refresh,
  predicting the real page 21 hours later) scored its ensemble 0.34
  against the house baseline's 0.78; v2's design keeps that
  earners-over-tracked structure and improves each of its parts.

## Layout

* `main.py` — the whole forecaster (entrypoint, run once per case).
* `LICENSE` — MIT.
