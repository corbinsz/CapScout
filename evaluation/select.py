"""Pure selection rules.

These functions are the protocol. The evaluation records the development
MAEs and calls them. The test file calls the same functions on the recorded
MAEs, so a hand-edited winner in the JSON fails the test.
"""

from __future__ import annotations

from evaluation.protocol import (
    ADDON_MUST_BEAT,
    BLEND_MUST_BEAT,
    INTERVAL_BAND_GAIN,
    INTERVAL_MEAN_COVERAGE_FLOOR,
    INTERVAL_TAIL_GAIN,
    INTERVAL_WIDTH_CAP,
    INTERVAL_WORST_SLACK,
    LIBRARY_MUST_BEAT,
    POINT_INSIDE_FLOOR,
    SIMPLER_WITHIN,
)


def choose_simpler(options: list[tuple[str, float]]) -> str:
    """Earliest option within 1% MAE of the best. `options` is in simplicity order."""
    if not options:
        raise ValueError("no options")
    best = min(mae for _name, mae in options)
    for name, mae in options:
        if mae <= best * SIMPLER_WITHIN:
            return name
    return options[-1][0]


def choose_addon(base_mae: float, addons: list[tuple[str, float]]) -> str | None:
    """An add-on must beat the base by 1%. Earlier names win a 1% tie.

    `addons` is in simplicity order. Pass the smoothed encoding before the
    nearest-neighbour encoding.
    """
    earned = [(name, mae) for name, mae in addons if mae < base_mae * ADDON_MUST_BEAT]
    if not earned:
        return None
    best = min(mae for _name, mae in earned)
    near = [(name, mae) for name, mae in earned if mae <= best * SIMPLER_WITHIN]
    return near[0][0]


def choose_quality(base_mae: float, quality_mae: float) -> bool:
    return quality_mae < base_mae * ADDON_MUST_BEAT


def choose_depth(depths: list[tuple[int, float]]) -> int:
    """Shallower depth within 1% of the best. `depths` is sorted shallow to deep."""
    ordered = [(str(depth), mae) for depth, mae in sorted(depths)]
    return int(choose_simpler(ordered))


def choose_library(scores: dict[str, float]) -> str:
    """XGBoost unless another library is at least 2% better than XGBoost.

    Among libraries that clear that bar, the lowest MAE wins. A library that
    beats XGBoost by 1% does not replace it: that was the previous study's
    bar, and it is applied to CatBoost as well.
    """
    if "xgboost" not in scores:
        raise KeyError("xgboost")
    baseline = scores["xgboost"]
    winner = "xgboost"
    winner_mae = baseline
    for name in ("lightgbm", "catboost"):
        mae = scores[name]
        if mae < baseline * LIBRARY_MUST_BEAT and mae < winner_mae:
            winner = name
            winner_mae = mae
    return winner


def choose_monotonic(unconstrained: float, monotonic: float) -> bool:
    """Keep the constraint when it is not materially worse."""
    return monotonic <= unconstrained * SIMPLER_WITHIN


def choose_blend(winner_mae: float, blend_mae: float) -> bool:
    """Equal weights have to beat the winner by 1%. Otherwise they are decoration."""
    return blend_mae < winner_mae * BLEND_MUST_BEAT


def feature_label(stage1: str, addon: str | None, quality: bool) -> str:
    parts = [stage1]
    if addon:
        parts.append(addon)
    if quality:
        parts.append("listing_quality")
    return "+".join(parts)


def choose_sequential(base_mae: float, steps: list[tuple[str, float]]) -> list[str]:
    """Keep each step that beats the stack so far by 1%.

    `steps` is the order the add-ons were tried. A step's MAE is the stack
    with that add-on on top of whatever earlier steps were kept, not the
    add-on on its own. Alone-on-the-base numbers are an ablation, and this
    function does not see them.
    """
    kept: list[str] = []
    current = float(base_mae)
    for name, mae in steps:
        if float(mae) < current * ADDON_MUST_BEAT:
            kept.append(name)
            current = float(mae)
    return kept


def choose_pooled(per_city_mae: float, pooled_mae: float) -> bool:
    """One model for all three cities has to beat the per-city blend by 1%."""
    return float(pooled_mae) < float(per_city_mae) * ADDON_MUST_BEAT


def choose_confirmed(default_mae: float, candidate_mae: float) -> bool:
    """A change fit on some development seeds must still clear 1% on the others."""
    return float(candidate_mae) < float(default_mae) * ADDON_MUST_BEAT


def choose_correction(base_mae: float, options: list[tuple[str, float]]) -> str:
    """None, unless a correction beats it by 1%. The simpler correction wins a tie.

    `options` is in simplicity order. Smearing is one number. Isotonic is a
    curve. Pass smearing first.
    """
    earned = [
        (name, mae) for name, mae in options if float(mae) < float(base_mae) * ADDON_MUST_BEAT
    ]
    if not earned:
        return "none"
    best = min(mae for _name, mae in earned)
    near = [(name, mae) for name, mae in earned if mae <= best * SIMPLER_WITHIN]
    return near[0][0]


def choose_band_interval(candidates: list[dict]) -> str:
    """The scaled log band, unless a challenger covers the worst price band better.

    The floors are the v3 ones: calibration, width, and the point estimate
    staying inside the band. The gain that matters this round is the worst
    city's worst predicted-price band, which is where Austin's cheap listings
    were missed. A challenger has to move that by the pre-set amount. If two
    do, the one with the better worst band wins, and a half-point tie keeps
    the earlier, simpler method.
    """
    if not candidates:
        raise ValueError("no interval candidates")
    baseline = candidates[0]
    best = baseline
    for challenger in candidates[1:]:
        if not _interval_ok(
            challenger,
            baseline,
            INTERVAL_MEAN_COVERAGE_FLOOR,
            INTERVAL_WIDTH_CAP,
            INTERVAL_WORST_SLACK,
            POINT_INSIDE_FLOOR,
        ):
            continue
        if challenger.get("worst_band_coverage") is None:
            continue
        if challenger["worst_band_coverage"] < baseline["worst_band_coverage"] + INTERVAL_BAND_GAIN:
            continue
        if challenger["worst_band_coverage"] > best["worst_band_coverage"] + 0.005:
            best = challenger
    return str(best["name"])


def choose_interval(candidates: list[dict]) -> str:
    """Pick an interval method. The first candidate is the incumbent.

    A challenger has to stay calibrated (mean coverage, and it must not
    open a hole in the worst city), stay about as tight, and actually
    contain the point estimate. It then has to cover the expensive
    listings better, or be clearly tighter at the same tail coverage.
    The incumbent does not have to pass those floors. It is what we
    already ship.
    """
    if not candidates:
        raise ValueError("no interval candidates")
    baseline = candidates[0]
    best = baseline
    for challenger in candidates[1:]:
        if not _interval_ok(
            challenger,
            baseline,
            INTERVAL_MEAN_COVERAGE_FLOOR,
            INTERVAL_WIDTH_CAP,
            INTERVAL_WORST_SLACK,
            POINT_INSIDE_FLOOR,
        ):
            continue
        if _interval_better(challenger, best, INTERVAL_TAIL_GAIN):
            best = challenger
    return str(best["name"])


def _interval_ok(challenger, baseline, floor, width_cap, slack, point_floor) -> bool:
    required = (
        "mean_coverage",
        "point_inside",
        "worst_coverage",
        "mean_median_width",
        "worst_upper_coverage",
    )
    if any(challenger.get(key) is None for key in required):
        return False
    if challenger["mean_coverage"] < floor:
        return False
    if challenger["point_inside"] < point_floor:
        return False
    if challenger["worst_coverage"] < baseline["worst_coverage"] - slack:
        return False
    if challenger["mean_median_width"] > baseline["mean_median_width"] * width_cap:
        return False
    upper = challenger["worst_upper_coverage"]
    return upper == upper  # reject NaN


def _interval_better(challenger, best, tail_gain: float) -> bool:
    if challenger["worst_upper_coverage"] >= best["worst_upper_coverage"] + tail_gain:
        return True
    narrower = challenger["mean_median_width"] < best["mean_median_width"] * 0.90
    tail_holds = challenger["worst_upper_coverage"] >= best["worst_upper_coverage"] - 0.005
    return bool(narrower and tail_holds)
