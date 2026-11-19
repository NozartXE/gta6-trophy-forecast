#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for each of the game's 55 PlayStation 5 trophies, five quantiles
(5/25/50/75/95%) of its Exophase earn rate at the case's target time.

Method: earners-over-players decomposition.
  rate_i(t) = Earners_i(t) / Players(t)

Exophase refreshes its counts on its own schedule (hours apart), not with
every player's trophy, so the history a case carries is a staircase: long
flat stretches, then a jump in players tracked and a drop in most rates as
fresh, early-progress players dilute the pool. A forecaster that only
extends the last read therefore misses every refresh. This program models
the refresh explicitly:

* Trend branch (the history moved): log-linear fits of log Players and of
  each trophy's log Earners against time, per-trophy slopes shrunk toward
  the pooled slope and floored at zero (earner counts never fall), then
  projected to the target. The rate is the projected ratio.
* Flat branch (the history never moved, as on launch night): no trend is
  estimable, so the refresh is priced from its hazard. Refreshes arrive
  about one per cadence C (median gap between observed changes, else the
  launch-night prior of 4.5 h); the expected number by the target is
  n = (hours since the last change + horizon hours) / C. Each refresh
  multiplies players by G (observed median per-refresh growth, else the
  published launch refresh, 3868 / 638 = 6.06x). New players each bring
  about one trophy (the published launch refresh: 3333 new trophies over
  3230 new players); trophies already at 100% are earned by every new
  player and stay at 100%, the platinum earns none from new players, and
  the small residual of extra earns spreads in proportion to each
  trophy's current earners.

Spread grows with the horizon (a random-walk term in sqrt(days)) plus a
base term and a refresh-count term, so a 90-minute practice forecast is
narrow and the 6,510-minute final forecast is wide. Everything is
deterministic: no randomness, no wall-clock reads, no network, standard
library only. Reads every *.in file in $MAKEFUN_INPUT_DIR (else /input/)
in name order and writes one *.out per case to $MAKEFUN_OUTPUT_DIR
(else /output/).
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Priors, used only when the case history cannot estimate them. All come
# from the launch-night refresh the competition rules publish (page flat
# at 638 players / 943 trophies earned, then refreshed to 3868 / 4276):
# one refresh per ~4.5 h, players x6.06 per refresh, 3333/3230 = 1.0317
# trophies per new player.
PRIOR_CADENCE_HOURS = 4.5
PRIOR_PLAYER_GROWTH = 3868.0 / 638.0          # per refresh
PRIOR_TROPHIES_PER_NEW_PLAYER = 3333.0 / 3230.0
MAX_LOG_PLAYER_GROWTH = math.log(200.0)      # sanity cap on total growth
BASE_SPREAD = 0.30                            # logit units
RW_SPREAD_PER_SQRT_DAY = 0.5                 # random-walk spread
RW_SPREAD_CAP = 0.6


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (as JavaScript Math.round does)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def ols_slope(xs, ys):
    """Least-squares slope of ys on xs; 0.0 when it cannot be estimated."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = case["horizon_minutes"] / 1440.0
    n_tr = len(trophies)

    if not history:
        # No reads at all: emit a valid, maximally uninformative forecast.
        return {"trophies": [
            {"id": t["id"], "q": [0.01, 0.1, 1.0, 10.0, 50.0]}
            for t in trophies]}

    # ---- last non-null rate and earners per trophy -----------------------
    last = history[-1]
    p_last = last.get("players_tracked") or 0
    if p_last <= 0:
        p_last = max((r.get("players_tracked") or 0) for r in history)
    last_rates = []
    for i in range(n_tr):
        rate = None
        for read in reversed(history):
            v = read["rates"][i]
            if v is not None:
                rate = v
                break
        last_rates.append(rate if rate is not None else 0.0)
    earners = [r * p_last / 100.0 for r in last_rates]

    # ---- did the history ever move? when was the last change? -----------
    def signature(read):
        return (read.get("players_tracked"),
                tuple(read["rates"]))

    last_sig = signature(last)
    moved = any(signature(r) != last_sig for r in history)
    checkpoint = parse_time(case["at"])
    # Start of the trailing run of reads identical to the last one: the
    # last change happened there (or before the recording, if the whole
    # history is one run).
    idx = len(history) - 1
    while idx > 0 and signature(history[idx - 1]) == last_sig:
        idx -= 1
    last_change_t = parse_time(history[idx]["at"])
    # If nothing ever changed, the last change predates the recording;
    # time since the first read is a lower bound on the flat spell.
    flat_hours = (checkpoint - last_change_t).total_seconds() / 3600.0

    # ---- refresh cadence from observed changes (else prior) -------------
    change_times = [parse_time(history[0]["at"])]
    prev = signature(history[0])
    for read in history[1:]:
        sig = signature(read)
        if sig != prev:
            change_times.append(parse_time(read["at"]))
            prev = sig
    gaps = [(b - a).total_seconds() / 3600.0
            for a, b in zip(change_times, change_times[1:]) if b > a]
    cadence = median(gaps) if gaps else PRIOR_CADENCE_HOURS
    if cadence <= 0:
        cadence = PRIOR_CADENCE_HOURS

    # ---- per-refresh player growth from observed jumps (else prior) -----
    growths = []
    prev_p = None
    for read in history:
        p = read.get("players_tracked") or 0
        if prev_p and p > prev_p:
            growths.append(math.log(p / prev_p))
        if p:
            prev_p = p
    log_growth = median(growths) if growths else math.log(PRIOR_PLAYER_GROWTH)

    medians = [0.0] * n_tr

    if not moved:
        # ---- flat branch: price the refresh(es) expected by the target --
        n_exp = (flat_hours + horizon_h) / cadence
        total_log_growth = min(n_exp * log_growth, MAX_LOG_PLAYER_GROWTH)
        p_target = p_last * math.exp(total_log_growth)
        new_players = max(0.0, p_target - p_last)
        new_trophies = new_players * PRIOR_TROPHIES_PER_NEW_PLAYER
        residual = max(0.0, new_trophies - new_players)
        others = [i for i, t in enumerate(trophies)
                  if t["tier"] != "platinum" and last_rates[i] < 100.0]
        denom = sum(earners[i] for i in others) or 1.0
        for i, t in enumerate(trophies):
            if last_rates[i] >= 100.0:
                medians[i] = 100.0                      # every new player earns it
            elif t["tier"] == "platinum":
                medians[i] = 100.0 * earners[i] / p_target
            else:
                extra = residual * earners[i] / denom
                medians[i] = 100.0 * (earners[i] + extra) / p_target
        refresh_term = 0.15 * math.sqrt(max(0.0, n_exp - 1.0))
    else:
        # ---- trend branch: log-linear projection of players and earners -
        t0 = parse_time(history[0]["at"])
        xs = [(parse_time(r["at"]) - t0).total_seconds() / 86400.0
              for r in history]
        log_p = [math.log(max(1, r.get("players_tracked") or 1))
                 for r in history]
        g_players = max(0.0, ols_slope(xs, log_p))
        p_target = p_last * math.exp(g_players * horizon_d)
        slopes = []
        for i in range(n_tr):
            ys = []
            xx = []
            for r, x in zip(history, xs):
                v = r["rates"][i]
                if v is None or v <= 0:
                    continue
                e = v * (r.get("players_tracked") or p_last) / 100.0
                if e > 0:
                    xx.append(x)
                    ys.append(math.log(e))
            slopes.append(ols_slope(xx, ys))
        pooled = median(slopes)
        for i in range(n_tr):
            if last_rates[i] >= 100.0:
                medians[i] = 100.0
                continue
            # Shrink the per-trophy slope halfway to the pooled slope,
            # floor at zero: earners never decrease.
            b = max(0.0, 0.5 * slopes[i] + 0.5 * pooled)
            e_target = earners[i] * math.exp(b * horizon_d)
            medians[i] = 100.0 * e_target / p_target
        refresh_term = 0.0

    # ---- spread: base + random walk in sqrt(days) + refresh uncertainty -
    rw = min(RW_SPREAD_CAP, RW_SPREAD_PER_SQRT_DAY * math.sqrt(horizon_d))
    spread = math.sqrt(BASE_SPREAD ** 2 + rw ** 2 + refresh_term ** 2)

    rows = []
    for i, t in enumerate(trophies):
        m = medians[i]
        if m >= 100.0:
            rows.append({"id": t["id"], "q": [100, 100, 100, 100, 100]})
            continue
        lg = logit(m)
        qs = []
        prev_q = 0.0
        for z in Z:
            q = round4(clamp(inv_logit(lg + z * spread)))
            q = max(q, prev_q)      # keep quantiles non-decreasing
            qs.append(q)
            prev_q = q
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
