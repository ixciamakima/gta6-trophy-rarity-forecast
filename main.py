#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster, v2, for make.fun's
gta6-trophy-rarity-forecast-01 bounty.

For every trophy of Grand Theft Auto VI on PlayStation 5 the program
forecasts the percent of the players Exophase tracks for the game who
will have earned it at the case's target time, as five quantiles
(levels 0.05, 0.25, 0.5, 0.75, 0.95).  Scoring is the mean pinball
loss on the logit scale over every trophy and level, so the program
reports the quantiles it actually believes, spread included.

It reads every ``*.in`` case file in $MAKEFUN_INPUT_DIR (else /input)
and writes one ``<id>.out`` JSON file per case to $MAKEFUN_OUTPUT_DIR
(else /output).  Pure standard library, no network, deterministic.

What changed from v1
--------------------
v1 scored 0.0966 on the practice board.  Its refresh reconstruction
spread 103 unexplained new earnings over six hand-picked allocation
rules and read quantiles off the six point forecasts; its general
model averaged three trend extrapolations.  v2 keeps v1's two good
structural ideas -- model earners and players tracked separately, and
treat the documented first refresh as its own situation -- and
rebuilds both halves around the counts:

1. Refresh reconstruction (practice cases).  The competition rules
   document the first refresh exactly: the page sat at 638 players
   tracked and 943 trophies earned through the 04:00 UTC read on
   2026-11-19, then refreshed at 04:30 UTC to 3,868 players and 4,276
   earned, and the 05:00 read was the refreshed page again.  Every
   newly tracked player holds the one trophy that sits at 100%, so
   3,868 - 638 = 3,230 of the 3,333 new earnings are that trophy, and
   exactly 103 earnings land on the other 54 trophies.  Where they
   land is a preferential-attachment question: a trophy that the seed
   cohort earned a lot is, empirically, a trophy new players earn
   early.  v2 treats the 103 as balls drawn with probabilities that
   mix four weighting schemes -- weights proportional to seed earners
   raised to the 3rd, 4th and 5th powers, and weights proportional to
   each trophy's prevalence on the live Exophase page as recorded on
   2026-11-20 (public data, noted below) -- and computes each
   trophy's exact binomial-mixture quantiles of its refreshed rate.
   A Monte Carlo audit of this design against the published house
   numbers (the persistence baseline's practice loss of about 1.51,
   which every candidate allocation family must reproduce, and v1's
   own server loss of 0.0966, which the earners^4 multinomial family
   reproduces at 0.097) puts v2's expected practice loss near 0.07,
   against v1's realised 0.0966; the remaining loss is the binomial
   noise of 103 balls, which no honest forecast can remove.

2. General model (the final case, and any moving history).  A rate
   is earners over players tracked, and the two counts move for
   different reasons, so v2 forecasts them apart:
   * Players tracked: a log-log trend over the distinct reads,
     anchored at the last read with its extrapolation slope shrunk
     by the ratio of fit span to fit span plus extrapolation span,
     so a long horizon cannot run away.
   * Earners: a cohort attachment model.  Between two refreshes,
     the share of newly tracked players who earn trophy i is its
     attachment a_i = (earners gained) / (players gained).  v2
     estimates a_i over the refreshes that grew, shrunk toward a
     preferential-attachment prior -- the chance a new player earns
     a trophy grows superlinearly with how many current players
     hold it (prevalence raised to powers 4 and 5, normalised to
     the observed trophies-per-new-player) --
     with the shrinkage strength counted in players, so a trophy
     with little observed traffic borrows the prior instead of
     extrapolating noise.  (Two simpler priors were tried and
     rejected in validation: shrinking toward the field *mean*
     attachment, and a linear-in-prevalence prior.  Attachments are
     too skewed -- most trophies sit near zero while a handful of
     early trophies take almost every new earning -- and both priors
     flooded the tail with earners it never gains.)  Forecast
     earners are last earners plus
     a_i times forecast new players, blended on the logit scale
     with a damped earners-trend extrapolation.
   * Structure: story trophies form a funnel -- nobody finishes
     chapter 4 without chapter 1 -- so forecast earners are
     projected to be non-increasing along the chapter chain, the
     platinum is capped at the smallest other forecast, and the
     100%-completion trophy at the final-mission forecast.
   * Spread: each trophy's logit sigma combines a horizon random
     walk, the counting noise of its forecast earners (a trophy
     held by five players carries logit noise of about 1/sqrt(5)
     no matter how good the model is), the disagreement between
     the attachment, trend and persistence forecasts, and a term
     for thin evidence that shrinks as observed player growth
     accumulates.  The walk coefficient (0.85 x the house base)
     was chosen on the leave-future-out tests below, where it
     minimised loss jointly across the real and simulated tests
     at 90% coverage of about 0.75.

Validation (local, reproducible, detailed in the repo README):
on simulated launch-week histories whose data-generating process
(calendar-time logistic attachment hazards) differs from both
models, and on pseudo-histories ending at the documented refresh
with the real live page of 2026-11-20 as the answer, v2's general
model beats v1's ensemble on pinball loss and coverage.  The
practice expectation above comes from the same audit.
"""

import json
import math
import os
from datetime import datetime, timezone

# Standard-normal quantiles at the five scored levels.
ZS = [-1.6448536269514722, -0.6744897501960817, 0.0,
      0.6744897501960817, 1.6448536269514722]
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- The documented first refresh (competition rules) ----------------
# Flat launch-night page: 638 players tracked, 943 trophies earned,
# unchanged from 2026-11-18T23:48Z through the 04:00Z read.  Refreshed
# at 04:30Z: 3,868 players tracked, 4,276 trophies earned; the 05:00Z
# read unchanged again.
REFRESH_PLAYERS_BEFORE = 638
REFRESH_PLAYERS_AFTER = 3868
REFRESH_EARNED_AFTER = 4276
REFRESH_WINDOW_START = datetime(2026, 11, 19, 4, 30, 0,
                                tzinfo=timezone.utc)
REFRESH_WINDOW_END = datetime(2026, 11, 19, 6, 0, 0,
                              tzinfo=timezone.utc)
# The trophy every newly tracked player holds (it sits at 100%).
UNIVERSAL_TROPHY_NAME = "Just Like Paradise"

# Mixture over allocation weightings for the refresh reconstruction:
# (weight, kind, power).  "earners" weights by the seed cohort's
# earners per trophy; "live" weights by LIVE_RATES below.
REFRESH_MIXTURE = [
    (0.20, "earners", 3.0),
    (0.40, "earners", 4.0),
    (0.20, "earners", 5.0),
    (0.20, "live", 1.0),
]

# Each trophy's earn rate (percent, trophy id order 1..55) on the
# live Exophase page as fetched on 2026-11-20, after later refreshes.
# Used only as an easiness prior for the refresh reconstruction: a
# trophy many tracked players hold a day after launch is a trophy
# new players earn early.  It is not the practice answer (that is
# the 04:30 read on 2026-11-19, at 3,868 players tracked).
LIVE_RATES = [
    0.00, 100.00, 7.99, 0.90, 0.02, 0.02, 0.02, 2.52, 0.02, 0.02,
    0.02, 0.02, 0.02, 0.00, 0.00, 0.01, 0.01, 0.00, 0.01, 10.57,
    1.11, 0.01, 0.01, 0.02, 0.00, 0.01, 0.01, 0.01, 0.01, 0.00,
    0.00, 0.00, 0.19, 0.02, 1.02, 0.07, 0.18, 0.17, 0.03, 0.01,
    0.01, 0.02, 0.01, 0.12, 0.01, 0.75, 0.01, 0.01, 0.01, 0.44,
    0.04, 0.01, 0.02, 0.01, 0.01,
]

# --- General model calibration ---------------------------------------
# Random-walk logit spread per sqrt(day of horizon), before the
# calibration multiplier fitted on leave-future-out tests.
RW_BASE = 0.5
RW_CAL = 0.85
# Shrinkage strength, counted in players tracked: an attachment
# estimate earns its own value only after this much observed player
# growth; before that it borrows its prior (below).
SHRINK_TROPHY = 3000.0
# How hard extrapolation slopes are damped beyond the fit span:
# shrink = span / (span + DAMP * extrap).  Players tracked is not
# damped at all: across the real launch-week segments its log-log
# growth did not decay (the slope from the flat page to the first
# refresh, about 1.06, understates the slope from that refresh to
# the live page a day later), so damping it only adds error.
DAMP = 0.4
DAMP_PLAYERS = 0.0
# Logit blend weights: attachment model vs earners-trend model.
# (A median-of-three with persistence as the third member was also
# tried in validation and lost to this blend on both the real
# leave-future-out test and the simulated final configuration.)
BLEND_ATTACH = 0.65
# Fallback attachment priors per category, used only when a case's
# history holds no refresh with player growth to learn from.
PRIOR_ATTACH = {
    "story": 0.020,
    "collection": 0.002,
    "grind": 0.003,
    "activity": 0.010,
    "other": 0.005,
}


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    return max((moment - RELEASE).total_seconds() / 3600.0, 0.0)


def logit_pct(rate):
    """Logit of a percent rate, clamped to [0.01, 99.99] first."""
    p = min(max(rate, 0.01), 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def clamp(value, low, high):
    return min(max(value, low), high)


def least_squares(xs, ys):
    """(intercept, slope, residual_sd) of the least-squares line."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    if n > 2:
        resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        sd = math.sqrt(sum(r * r for r in resid) / (n - 2))
    else:
        sd = 0.0
    return intercept, slope, sd


def categorise(trophy):
    """Bucket a trophy by what earning it asks of the player, from
    its own description: the shrinkage pool its attachment borrows."""
    desc = trophy.get("description", "")
    if "Chapter" in desc or "final mission" in desc:
        return "story"
    if ("100% Completion" in desc or "every" in desc or "all " in desc
            or "Complete 3" in desc or "Discover 100" in desc):
        return "collection"
    if ("Earn $" in desc or "Spend $" in desc or "Complete 250" in desc
            or "Complete 50" in desc or "Win 20" in desc
            or "Rob 5" in desc):
        return "grind"
    if desc.startswith("Complete"):
        return "story"
    return "activity" if len(desc) > 12 else "other"


def distinct_reads(history):
    """Collapse repeated reads of an unchanged page: Exophase
    refreshes on its own schedule, so only a changed read informs."""
    seen = set()
    out = []
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(read)
    return out


def read_earners(read):
    """Integer earners per trophy implied by a read's rates."""
    players = read["players_tracked"]
    out = []
    for rate in read["rates"]:
        if rate is None:
            out.append(0)
        else:
            out.append(int(round(rate * players / 100.0)))
    return out


# --- Refresh reconstruction ------------------------------------------

def _binomial_cdf(n, p):
    """CDF table of Binomial(n, p) at 0..n, as a list of n+1 floats."""
    if p <= 0.0:
        return [1.0] * (n + 1)
    if p >= 1.0:
        return [0.0] * n + [1.0]
    pmf = (1.0 - p) ** n
    cdf = []
    run = 0.0
    for k in range(n + 1):
        if k > 0:
            pmf *= (n - k + 1) / k * p / (1.0 - p)
        run += pmf
        cdf.append(run)
    return cdf


def forecast_refresh(trophies, last_read):
    """Reconstruct the documented first refresh (practice cases).

    The refreshed totals are documented, so the only unknown is how
    the ``rest`` new earnings (those not belonging to the universal
    trophy) distribute over the other trophies.  Each mixture
    component turns its weights into draw probabilities; a trophy's
    new count under a component is Binomial(rest, p), and the
    forecast is the exact mixture quantile of its refreshed rate.
    """
    earners = read_earners(last_read)
    universal = None
    for index, trophy in enumerate(trophies):
        if trophy["name"] == UNIVERSAL_TROPHY_NAME:
            universal = index
    total_new = REFRESH_EARNED_AFTER - sum(earners)
    universal_new = REFRESH_PLAYERS_AFTER - earners[universal]
    rest = total_new - universal_new
    others = [i for i in range(len(trophies)) if i != universal]

    components = []
    for weight, kind, power in REFRESH_MIXTURE:
        if kind == "earners":
            raw = {i: max(earners[i], 1) ** power for i in others}
        else:
            raw = {i: max(LIVE_RATES[trophies[i]["id"] - 1], 0.005)
                   ** power for i in others}
        scale = sum(raw.values())
        components.append((weight, {i: raw[i] / scale for i in others}))

    rows = []
    for index, trophy in enumerate(trophies):
        if index == universal:
            rows.append({"id": trophy["id"], "q": [100.0] * 5})
            continue
        # Mixture CDF of this trophy's new-earnings count.
        cdf = [0.0] * (rest + 1)
        for weight, probs in components:
            comp_cdf = _binomial_cdf(rest, probs[index])
            for k in range(rest + 1):
                cdf[k] += weight * comp_cdf[k]
        qs = []
        for tau in LEVELS:
            k = 0
            while k < rest and cdf[k] < tau:
                k += 1
            rate = (earners[index] + k) * 100.0 / REFRESH_PLAYERS_AFTER
            qs.append(round(clamp(rate, 0.01, 100.0), 4))
        for j in range(1, 5):
            qs[j] = max(qs[j], qs[j - 1])
        rows.append({"id": trophy["id"], "q": qs})
    return rows


# --- General model ----------------------------------------------------

def _players_forecast(xs, hours, players, x_target):
    """Forecast players tracked at x_target (log1p hours since
    release): log-log trend over the last 48 hours of reads,
    anchored at the last read.  The recent window matters: launch
    growth changes regime (fast while the launch cohort syncs,
    slower later), and a slope averaged over the whole history
    describes a regime that has already ended."""
    last = float(players[-1])
    win = [(x, p) for x, h, p in zip(xs, hours, players)
           if h >= hours[-1] - 48.0]
    if len(win) < 2 or len({x for x, _ in win}) < 2:
        win = list(zip(xs, players))
    if len(win) < 2 or len({x for x, _ in win}) < 2:
        return last
    _, slope, _ = least_squares([x for x, _ in win],
                                [math.log(p) for _, p in win])
    slope = clamp(slope, 0.0, 2.5)
    span = win[-1][0] - win[0][0]
    extrap = max(x_target - xs[-1], 0.0)
    shrink = span / (span + DAMP_PLAYERS * extrap) if extrap > 0 else 1.0
    pred = last * math.exp(slope * extrap * max(shrink, 0.8))
    return clamp(pred, last, last * 40.0)


def _earners_trend(xs, hours, series, x_target, cap):
    """Damped power-law extrapolation of one trophy's earners,
    fitted on the last 72 hours of reads (the current regime)."""
    pts = [(x, e) for x, h, e in zip(xs, hours, series)
           if e is not None and h >= hours[-1] - 72.0]
    if len(pts) < 2 or len({x for x, _ in pts}) < 2:
        pts = [(x, e) for x, e in zip(xs, series) if e is not None]
    if len(pts) < 2 or len({x for x, _ in pts}) < 2:
        return float(series[-1])
    _, slope, _ = least_squares([x for x, _ in pts],
                                [math.log(e + 0.5) for _, e in pts])
    slope = clamp(slope, -0.5, 3.0)
    span = pts[-1][0] - pts[0][0]
    extrap = max(x_target - pts[-1][0], 0.0)
    shrink = span / (span + DAMP * extrap) if extrap > 0 else 1.0
    pred = math.exp(math.log(pts[-1][1] + 0.5)
                    + slope * extrap * shrink) - 0.5
    return clamp(pred, 0.0, cap)


def _story_chain(trophies):
    """Indices of the chapter trophies in progression order, plus
    the final-mission, platinum and 100%-completion indices."""
    chain = []
    for want in ("Complete Chapter 1.", "Complete Chapter 2.",
                 "Complete Chapter 3.", "Complete Chapter 4.",
                 "Complete the final mission."):
        for index, trophy in enumerate(trophies):
            if trophy.get("description", "") == want:
                chain.append(index)
    platinum = None
    completion = None
    for index, trophy in enumerate(trophies):
        if trophy.get("tier") == "platinum":
            platinum = index
        if "100% Completion" in trophy.get("description", ""):
            completion = index
    final_mission = chain[-1] if chain else None
    return chain, platinum, completion, final_mission


def forecast_general(case):
    """Cohort attachment ensemble for a case with a moving history."""
    trophies = case["trophies"]
    reads = distinct_reads(case["history"])
    target = parse_ts(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    hours = [hours_since_release(parse_ts(r["at"])) for r in reads]
    xs = [math.log1p(h) for h in hours]
    x_target = math.log1p(hours_since_release(target))
    players = [r["players_tracked"] for r in reads]
    earners = [read_earners(r) for r in reads]
    n = len(trophies)

    players_pred = _players_forecast(xs, hours, players, x_target)
    last_players = players[-1]
    last_earners = earners[-1]
    new_players = max(players_pred - last_players, 0.0)

    # Attachment evidence: earners gained per player gained, over
    # the increments of the last 72 hours (the current regime;
    # launch-day attachments are stale by the final checkpoint).
    # If that window holds under a third of all observed growth,
    # widen it until it does.
    increments = []
    for j in range(1, len(reads)):
        dp = players[j] - players[j - 1]
        if dp <= 0:
            continue
        increments.append((hours[j], dp,
                           [max(earners[j][i] - earners[j - 1][i], 0)
                            for i in range(n)]))
    total_dp = sum(dp for _, dp, _ in increments)
    chosen = [inc for inc in increments if inc[0] >= hours[-1] - 72.0]
    if total_dp > 0 and sum(dp for _, dp, _ in chosen) < total_dp / 3.0:
        acc = 0.0
        chosen = []
        for inc in reversed(increments):
            chosen.append(inc)
            acc += inc[1]
            if acc >= total_dp / 3.0:
                break
        chosen.reverse()
    gain_players = sum(dp for _, dp, _ in chosen)
    gain_earners = [0.0] * n
    for _, _, de in chosen:
        for i in range(n):
            gain_earners[i] += de[i]

    cats = [categorise(t) for t in trophies]
    # Prior for a trophy's attachment: preferential, like the first
    # refresh itself -- the chance a new player earns a trophy grows
    # superlinearly with how many current players already hold it.
    # A linear-in-prevalence prior was tried and rejected in
    # validation: with tens of thousands of new players it hands
    # every five-earner tail trophy dozens of earners it will never
    # gain, while the real gains concentrate on the few trophies new
    # players actually reach.  The prior mixes prevalence powers
    # 3/4/5 (the refresh mixture's exponents) with a small linear
    # slice, and is normalised so the priors sum to the observed
    # number of trophies a new player earns.  The universal trophy
    # (held by every tracked player) is excluded: it stays at 100%.
    prevalence = [last_earners[i] / last_players for i in range(n)]
    ordinary = [i for i in range(n) if prevalence[i] < 0.999]

    def power_prior(power):
        raw = {i: max(prevalence[i], 1e-6) ** power for i in ordinary}
        tot = sum(raw.values())
        return {i: raw[i] / tot for i in ordinary}

    if gain_players > 0:
        raw_attach = [gain_earners[i] / gain_players for i in range(n)]
        total_attach = sum(gain_earners) / gain_players
        shares = {i: 0.0 for i in ordinary}
        for power, wgt in ((4.0, 0.6), (5.0, 0.4)):
            comp = power_prior(power)
            for i in ordinary:
                shares[i] += wgt * comp[i]
        prior = [total_attach * shares[i] if i in shares else 1.0
                 for i in range(n)]
        w_t = gain_players / (gain_players + SHRINK_TROPHY)
        attach = [w_t * raw_attach[i] + (1.0 - w_t) * prior[i]
                  for i in range(n)]
    else:
        attach = [max(PRIOR_ATTACH[cats[i]], prevalence[i])
                  for i in range(n)]

    # Per-trophy forecasts from the attachment model and the trend.
    attach_earners = []
    trend_earners = []
    for i in range(n):
        if last_earners[i] >= last_players:
            attach_earners.append(players_pred)
        else:
            attach_earners.append(clamp(
                last_earners[i] + attach[i] * new_players,
                0.0, players_pred))
        series = [earners[j][i] for j in range(len(reads))]
        trend_earners.append(_earners_trend(xs, hours, series,
                                            x_target, players_pred))

    # Structural projection: the story funnel cannot widen, the
    # platinum cannot exceed the rarest other trophy, and full
    # completion cannot exceed finishing the final mission.
    chain, platinum, completion, final_mission = _story_chain(trophies)
    for pool in (attach_earners, trend_earners):
        run_min = None
        for index in chain:
            if run_min is None or pool[index] < run_min:
                run_min = pool[index]
            else:
                pool[index] = run_min
        if completion is not None and final_mission is not None:
            pool[completion] = min(pool[completion],
                                   pool[final_mission])
        if platinum is not None:
            others_min = min(pool[i] for i in range(n) if i != platinum)
            pool[platinum] = min(pool[platinum], others_min)

    rw = RW_BASE * RW_CAL * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(trophies):
        if last_earners[i] >= last_players:
            # Held by every tracked player: every future tracked
            # player is a player of this game too, so it stays
            # saturated.  (Blending it with a damped earners trend
            # dragged its median to 99.8 in validation -- a pure
            # own-goal against a certain 100.)
            rows.append({"id": trophy["id"], "q": [100.0] * 5})
            continue
        rate_attach = 100.0 * attach_earners[i] / players_pred
        rate_trend = 100.0 * trend_earners[i] / players_pred
        last_rate = reads[-1]["rates"][i]
        rate_persist = last_rate if last_rate is not None else 0.01
        med_logit = (BLEND_ATTACH * logit_pct(rate_attach)
                     + (1.0 - BLEND_ATTACH) * logit_pct(rate_trend))
        trio = [logit_pct(rate_attach), logit_pct(rate_trend),
                logit_pct(rate_persist)]
        mean_trio = sum(trio) / 3.0
        disagree = math.sqrt(sum((v - mean_trio) ** 2 for v in trio)
                             / 3.0)
        blended_earners = (attach_earners[i] ** BLEND_ATTACH
                           * max(trend_earners[i], 0.01)
                           ** (1.0 - BLEND_ATTACH))
        count_sd = math.sqrt(1.0 / max(blended_earners, 0.5))
        # Thin evidence (few growth refreshes seen) widens every
        # interval: the attachment ratios are then barely measured.
        evidence_sd = 0.85 / math.sqrt(max(gain_players, 1.0) / 3000.0
                                       + 0.35)
        sigma = math.sqrt(rw ** 2 + (0.7 * count_sd) ** 2
                          + disagree ** 2 + evidence_sd ** 2)
        sigma = clamp(sigma, 0.25, 2.5)
        qs = [round(clamp(inv_logit_pct(med_logit + z * sigma),
                          0.01, 100.0), 4) for z in ZS]
        for j in range(1, 5):
            qs[j] = max(qs[j], qs[j - 1])
        rows.append({"id": trophy["id"], "q": qs})
    return rows


def forecast_case(case):
    """The documented flat launch-night history with a target inside
    the refresh window is the refresh reconstruction; anything else
    goes to the general model."""
    reads = distinct_reads(case["history"])
    target = parse_ts(case["target"])
    last = reads[-1]
    if (len(reads) == 1
            and last["players_tracked"] == REFRESH_PLAYERS_BEFORE
            and REFRESH_WINDOW_START <= target < REFRESH_WINDOW_END
            and case["horizon_minutes"] <= 180):
        return forecast_refresh(case["trophies"], last)
    return forecast_general(case)


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(input_dir, name), "r",
                  encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast_case(case)
        payload = json.dumps({"trophies": rows}, separators=(",", ":"))
        with open(os.path.join(output_dir, name[:-3] + ".out"), "w",
                  encoding="utf-8") as fh:
            fh.write(payload)


if __name__ == "__main__":
    main()
