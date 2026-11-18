"""Pure selection rules.

These functions are the protocol. The evaluation records the development
MAEs and calls them. The test file calls the same functions on the recorded
MAEs, so a hand-edited winner in the JSON fails the test.
"""

from __future__ import annotations

from evaluation.protocol import (
    ADDON_MUST_BEAT,
    BLEND_MUST_BEAT,
    LIBRARY_MUST_BEAT,
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
