"""gta6-twitch-ccu-forecast-01: scale-conditioned launch-day Twitch forecaster.

Reads one case and forecasts, for each of hours 2..24 of the release day (UTC),
five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's concurrent Twitch viewers
in the Grand Theft Auto VI category, given only the first hour's print.

Model (log scale throughout, because the score is pinball loss on log scale):
every past launch has its own intraday shape, and that shape depends strongly on
how big hour 1 already is. A launch whose hour 1 was tiny -- a game that only
unlocked later in the day -- could explode 100x by evening, while a launch whose
hour 1 was already huge stayed almost flat. So instead of one average shape, each
comparable launch in the case file is weighted by how close its own hour-1 size
is to this case's hour-1 size (Gaussian kernel, bandwidth 1.0 in log space), and
for each hour the forecast quantiles are the weighted empirical quantiles of the
comparables' log-ratios ln(viewers_h / viewers_1). The median follows the typical
diurnal curve -- a dip through the night hours, a rise into the evening -- and the
spread is the honest cross-launch spread, which is what pinball loss rewards.
Three refinements, each adopted only after it improved leave-one-out
cross-validation over the kit's past launches on every slice (overall, big
hour-1, console-only, big+console):
  - a shrunk local-linear slope: per hour, the log-ratio's slope on
    ln(hour-1 size) is fitted by weighted least squares and shrunk toward 0
    (effective-sample-size shrinkage, k=20), then each comparable's ratio is
    corrected toward the target's scale. The slope is a stable structural
    pattern -- near 0 in hours 2-5, about -0.15 to -0.40 at late hours in every
    big held-out launch, so bigger launches decay faster late in the day --
    and the shrinkage keeps a noisy slope from running the forecast.
  - a very gentle dip kernel (bandwidth 3.0 in log space) on the pre-release
    dip, ln(hour-1 / max viewers in the 6 hours before release): comparables
    whose midnight dip resembles the target's count slightly more. It helps
    console-only launches most and is ~neutral elsewhere; narrower bandwidths
    hurt the big regime, so it stays gentle.
  - a log-ratio spread multiplier of 0.95 around the median: leave-one-out
    improved on all slices (overall 0.3938 to 0.3937, big 0.2045 to 0.2042,
    console 0.2149 to 0.2118, big+console 0.1698 to 0.1671). The console slice
    over-covers (0.951 vs 0.90 nominal), so a slightly narrower spread is
    better calibrated. Hours without comparables keep the fallback spread.
What was tried and rejected by the same leave-one-out test: a momentum kernel
on the pre-release slope, shifting the median by the dip, a viewer/channel
ratio kernel, console-only filtering or upweighting, wider spreads and the
coarse 0.90 spread step in the first grid, and changing the scale bandwidth.
The finer 0.93-0.97 spread band improves all slices, with 0.95 adopted.
Comparables without a usable release day (no hour-1 print, or fewer than 20
of the 24 release-day hours) are skipped; with
no usable comparable at all, the forecast falls back to flat persistence with
a widening lognormal spread.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. -P keeps
this file's directory OFF sys.path, so a module shipped beside main.py imports
only after the sys.path line below.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case at a time,
    with only that case's file there): JSON {"schema": "makefun/twitch-ccu-case/v1",
    "id", "game", "day_start", "checkpoint", "levels", "known", "forecast",
    "history", "history_hourly", "streams", "categories_hourly", "comparables",
    "notes"}, where `known` holds hour 1's print {hour: 1, from, until, viewers,
    channels, reads} and `forecast` lists the hours to forecast, 2..24, and each
    comparable holds {"day_start", "hourly": [{"from", "viewers", "channels",
    "reads"}], "name", "platforms", "unlock"};
  - write <id>.out to the output directory: {"hours": [{"hour": 2, "q": [q05, q25,
    q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24 each once, no other key,
    each q five finite numbers above 0 and at most 1e9 (viewers; fractions
    allowed), non-decreasing; at most 65,536 bytes. Anything else is an invalid
    case, scored as if every quantile equalled hour 1's print, and an entry with
    an invalid case takes no prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand
in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Gaussian kernel bandwidth on log hour-1 size: one e-fold of weight per factor
# of e in hour-1 viewers. Calibrated by leave-one-out cross-validation over the
# kit's 37 usable past launches (mean pinball loss 0.3937, vs 0.9297 for the
# flat house baseline; 0.2042 vs 0.2509 on the big hour-1 launches that
# resemble GTA VI; 0.2118 on console-only launches; 0.1671 on big+console).
BANDWIDTH = 1.0
# Local-linear slope shrinkage: the slope's weight is neff / (neff + k), so a
# slope fitted on a thin effective sample is pulled hard toward zero.
SLOPE_SHRINK = 20.0
# Dip kernel bandwidth on ln(hour-1 / pre-release 6h peak): deliberately wide,
# so the dip only tilts the weights instead of selecting on them.
DIP_BANDWIDTH = 3.0
# Log-ratio spread multiplier around the median; adopted after LOO improved
# all slices: overall 0.3938 to 0.3937, big 0.2045 to 0.2042, console 0.2149
# to 0.2118, big+console 0.1698 to 0.1671. Console over-covers (0.951 vs 0.90
# nominal), so a slightly narrower spread is better calibrated.
SPREAD = 0.95
# A comparable only counts if its release day is nearly complete.
MIN_RELEASE_HOURS = 20
# Fallback spread (flat persistence), as in the house baseline: log-sigma per
# square-root hour.
FALLBACK_SIGMA = 0.15
# Standard normal quantiles, used only by the fallback.
Z = [-1.645, -0.674, 0, 0.674, 1.645]


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    """An ISO-8601 UTC timestamp from the case files."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def pre_release_dip(hourly, day_start, p1):
    """ln(hour-1 print / max viewers in the 6 hours before release), or None.

    `hourly` is the raw point list [{from, viewers, ...}]. The hour-1 point
    itself (from == day_start) is not pre-release and is excluded. Needs at
    least 3 pre-release points, else there is no dip to measure.
    """
    pre = []
    for point in hourly or []:
        try:
            at = parse_time(point["from"])
            viewers = point["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None or at >= day_start:
            continue
        pre.append((at, max(1, viewers)))
    pre.sort()
    last6 = [viewers for _, viewers in pre[-6:]]
    if len(last6) < 3:
        return None
    return math.log(p1 / max(last6))


def comparable_days(case: dict) -> list:
    """Usable comparables: (hour-1 viewers, {hour: viewers}, dip or None).

    A comparable is usable when its release day has an hour-1 print and at
    least MIN_RELEASE_HOURS of the 24 hours present.
    """
    usable = []
    for comp in case.get("comparables") or []:
        try:
            day_start = parse_time(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        by_hour = {}
        for point in comp.get("hourly") or []:
            try:
                at = parse_time(point["from"])
                viewers = point["viewers"]
            except (KeyError, TypeError, ValueError):
                continue
            if viewers is None:
                continue
            hour = int((at - day_start).total_seconds() // 3600) + 1
            if 1 <= hour <= 24:
                by_hour[hour] = max(1, viewers)
        if 1 in by_hour and len(by_hour) >= MIN_RELEASE_HOURS:
            dip = pre_release_dip(comp.get("hourly"), day_start, by_hour[1])
            usable.append((by_hour[1], by_hour, dip))
    return usable


def case_weights(p1: float, target_dip, records: list) -> list:
    """One kernel weight per comparable: scale closeness, tilted by the dip.

    The dip factor applies only when both dips exist; a comparable or target
    without a measurable dip keeps its scale weight unchanged.
    """
    log_p1 = math.log(p1)
    weights = []
    for comp_p1, _, comp_dip in records:
        weight = math.exp(-0.5 * ((math.log(comp_p1) - log_p1) / BANDWIDTH) ** 2)
        if target_dip is not None and comp_dip is not None:
            weight *= math.exp(
                -0.5 * ((target_dip - comp_dip) / DIP_BANDWIDTH) ** 2)
        weights.append(weight)
    return weights


def weighted_quantile(pairs: list, tau: float) -> float:
    """The tau-quantile of values weighted by `pairs` [(value, weight)].

    Linear interpolation on cumulative weight: each value sits at the midpoint
    of its weight block, and the quantile is interpolated between the two
    bracketing midpoints. Smoother than a stepwise rule, and it scored better
    in leave-one-out cross-validation over the kit's past launches (mean
    pinball loss 0.400 vs 0.405 stepwise; 0.206 vs 0.208 on the big hour-1
    launches). Deterministic: ties keep file order.
    """
    ordered = sorted(pairs, key=lambda pair: pair[0])
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    midpoints = []
    acc = 0.0
    for value, weight in ordered:
        midpoints.append((acc + weight / 2, value))
        acc += weight
    target = tau * total
    if target <= midpoints[0][0]:
        return midpoints[0][1]
    if target >= midpoints[-1][0]:
        return midpoints[-1][1]
    for (x0, v0), (x1, v1) in zip(midpoints, midpoints[1:]):
        if x0 <= target <= x1:
            frac = (target - x0) / (x1 - x0) if x1 > x0 else 0.0
            return v0 * (1 - frac) + v1 * frac
    return midpoints[-1][1]


def hour_quantiles(p1: float, h: int, levels: list, records: list,
                   weights: list) -> list:
    """The five forecast quantiles for hour h, as viewer counts."""
    log_p1 = math.log(p1)
    pairs = []
    for (comp_p1, day, _), weight in zip(records, weights):
        if h not in day:
            continue
        pairs.append((math.log(day[h] / comp_p1), weight, math.log(comp_p1)))
    if pairs:
        ratios = [ratio for ratio, _, _ in pairs]
        # Shrunk local-linear slope of the log-ratio on ln(hour-1 size):
        # each comparable's ratio is corrected toward the target's scale,
        # which removes the local average's boundary bias when the target is
        # bigger than most comparables. The shrinkage factor is the
        # effective sample size over itself plus SLOPE_SHRINK, so a slope
        # fitted on a thin effective sample barely moves the forecast.
        ws = [w for _, w, _ in pairs]
        xs = [x for _, _, x in pairs]
        total_w = sum(ws)
        mean_x = sum(w * x for w, x in zip(ws, xs)) / total_w
        mean_r = sum(w * r for w, r in zip(ws, ratios)) / total_w
        denom = sum(w * (x - mean_x) ** 2 for w, x in zip(ws, xs))
        beta = (sum(w * (x - mean_x) * (r - mean_r)
                    for w, x, r in zip(ws, xs, ratios)) / denom) if denom > 0 else 0.0
        neff = total_w * total_w / sum(w * w for w in ws) if total_w > 0 else 0.0
        beta *= neff / (neff + SLOPE_SHRINK) if neff > 0 else 0.0
        corrected = [(r + beta * (log_p1 - x), w) for r, w, x in pairs]
        ratios = [weighted_quantile(corrected, tau) for tau in levels]
        med = ratios[2]
        if SPREAD != 1.0:
            ratios = [med + (r - med) * SPREAD for r in ratios]
        values = [p1 * math.exp(ratio) for ratio in ratios]
    else:
        # No comparable has this hour: flat persistence, spread widening with
        # the square root of the horizon, as in the house baseline.
        values = [p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1)) for z in Z]
    # Round half-up, clamp to the valid range, keep non-decreasing.
    out = []
    floor = 0.0
    for value in values:
        rounded = round2(min(max(value, 0.01), 1e9))
        rounded = max(rounded, floor)
        floor = rounded
        out.append(rounded)
    return out


def first_hour_print(case: dict) -> float:
    """Hour 1's viewer print, or a fallback if the case somehow lacks one."""
    for row in case.get("known") or []:
        if row.get("hour") == 1 and row.get("viewers") is not None:
            return max(1, row["viewers"])
    history = case.get("history") or case.get("history_hourly") or []
    if history:
        # Newest point by time, not by file order, which the case does not
        # guarantee.
        def _at(point):
            try:
                return parse_time(point["from"])
            except (KeyError, TypeError, ValueError):
                return datetime.min.replace(tzinfo=timezone.utc)

        last = max(history, key=_at)
        if last.get("viewers") is not None:
            return max(1, last["viewers"])
    return 1


def forecast_case(case: dict, name: str) -> dict:
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    if len(levels) != 5 or any(not 0 < tau < 1 for tau in levels):
        raise ValueError(f"{name} asks for levels {levels}, not five quantiles")
    p1 = first_hour_print(case)
    records = comparable_days(case)
    target_dip = None
    try:
        day_start = parse_time(case["day_start"])
    except (KeyError, TypeError, ValueError):
        day_start = None
    if day_start is not None:
        # history_hourly ends at hour 1; pre_release_dip excludes that point
        # and measures the 6 hours before release from the rest.
        target_dip = pre_release_dip(case.get("history_hourly"), day_start, p1)
    weights = case_weights(p1, target_dip, records)
    hours = [
        {"hour": row["hour"],
         "q": hour_quantiles(p1, row["hour"], levels, records, weights)}
        for row in case["forecast"]
    ]
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
