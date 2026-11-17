# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/cmalf/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/cmalf/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A scale-conditioned forecaster for GTA VI's launch-day Twitch viewers: it reads the
category's hour-1 print and the 38 past big-launch comparables in the case file, and
forecasts hours 2–24 as five quantiles each, on the log scale.

## How it works

Every past launch has its own intraday shape, and that shape depends strongly on how
big hour 1 already is: launches whose hour 1 was tiny (a game that unlocked later in
the day) exploded 100× by evening, while launches whose hour 1 was already huge
stayed almost flat. So instead of one average shape, the program weights each
comparable launch by how close its hour-1 size is to this case's hour-1 size
(Gaussian kernel, bandwidth 1.0 in log space) and takes the weighted empirical
quantiles of the log-ratios `ln(viewers_h / viewers_1)` at the 5 / 25 / 50 / 75 /
95% levels. The median follows the typical diurnal curve — a dip through the
night hours, a rise into the evening — and the spread is the honest cross-launch
spread, which is what the pinball loss on the log scale rewards. Two refinements
improved the leave-one-out score on every slice (overall 0.4004 → 0.3938, big
hour-1 launches 0.2061 → 0.2045, console-only 0.2196 → 0.2149): a shrunk
local-linear slope that corrects each comparable's log-ratio toward the target's
scale (bigger launches decay faster late in the day, a stable pattern in every
big held-out launch), and a very gentle kernel on the pre-release dip
(`ln(hour-1 / max viewers in the 6 hours before release)`). If a case ever
arrived with no usable comparables, it falls back to the flat persistence
forecast. Standard library only, no network, deterministic, runs in well under
a second.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 -P -s /bot/main.py` (Python 3.13, standard library only):
it reads every `<id>.in` in `/input` and writes `<id>.out` to `/output` with
23 rows (hours 2–24), five quantiles each.

---

<sub>Entered on [make.fun](https://make.fun/?ref=cmalf&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
