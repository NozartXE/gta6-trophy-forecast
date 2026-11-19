# GTA VI Trophy Rarity Forecaster

Forecasts, for every Grand Theft Auto VI PlayStation 5 trophy, five
quantiles (5 / 25 / 50 / 75 / 95%) of its Exophase earn rate at the
case's target time. Entry for make.fun competition
`gta6-trophy-rarity-forecast-01`.

## Run

```sh
python3 main.py            # reads $MAKEFUN_INPUT_DIR (else /input),
                          # writes $MAKEFUN_OUTPUT_DIR (else /output)
```

Python 3.13, standard library only. Deterministic, offline, ~0.1 s
per case. One file, `main.py`, at the repository root.

## Method: earners-over-players decomposition

`rate(t) = Earners(t) / Players(t)`. Exophase refreshes its counts on
its own schedule (hours apart), so the history is a staircase: long
flat stretches, then a jump in players tracked and a drop in most
rates as fresh, early-progress players dilute the pool.

* **Trend branch** (history moved): log-linear fits of log Players and
  of each trophy's log Earners against time, per-trophy slopes shrunk
  halfway to the pooled slope and floored at zero, projected to the
  target; the rate is the projected ratio.
* **Flat branch** (history never moved, as on launch night): the
  refresh is priced from its hazard. Expected refreshes by the target
  `n = (hours since last change + horizon hours) / cadence` (cadence
  from observed change gaps, else the 4.5 h launch-night prior). Each
  refresh multiplies players by the observed per-refresh growth (else
  the published launch refresh, 3868/638 = 6.06x). New players each
  bring ~1.03 trophies (published launch refresh: 3333 new trophies
  over 3230 new players); trophies at 100% are earned by every new
  player and stay at 100%, the platinum earns none from new players,
  and the small residual of extra earns spreads in proportion to each
  trophy's current earners.

Spread is `sqrt(0.30^2 + (0.5*sqrt(days))^2 + refresh term^2)` in logit
units, so it widens with the horizon: narrow over the 90-minute
practice, wide over the 6,510-minute final.

## Backtest (practice cases p00/p01)

The kit ships no practice answers, so the answers were reconstructed
from the refresh aggregates the rules publish (players 638 → 3868,
trophies earned 943 → 4276, trophy #2 stays 100%): integer earners
recovered exactly from the 2-decimal rates (sum 943 = the
`trophies_earned` field). The only unknown is how the 103 residual new
earns spread; three scenarios span it (top-15 concentrated, calibrated
so the house starter scores its stated ~1.51 / miss ~0.68 / coverage 0;
proportional; uniform). Mean pinball loss over the three scenarios,
both practice cases:

| forecaster | mean loss |
|---|---|
| this entry (built `main.py`) | **~0.10** |
| persistence, σ=0.5 (house baseline) | 1.44 |
| curve-trend baseline | 1.44 |

The house AI forecast is not reproducible locally (one recorded
superbot call), so it is not in the table. Practice mainly checks
valid output and refresh-coping; one 90-minute refresh is little
guide to the final week's movement.

## Files

* `main.py` — the whole forecaster.
* `LICENSE` — MIT.
