"""Overall figures for the three-city holdout.

The city rows stay the numbers to quote. Two summaries sit next to them,
and they are not the same number.

Each city once
    Average the three city scores. This is how the model was chosen, so
    Austin's larger holdout does not outvote Seattle.

Every home once
    Stack the holdout rows. Austin then pulls the dollar error up, because
    it has more homes and more expensive misses.

The intervals resample holdout cells inside each city, 1,000 times, with
the same seed as the city intervals. The point is the full holdout, not
the center of the bootstrap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.metrics import regression_metrics
from evaluation.study_format import money, pct, r2
from evaluation.study_protocol import BOOTSTRAP_SEED, N_BOOTSTRAP

CITY_ORDER = ("austin", "san-diego", "seattle")


def _score(y: np.ndarray, pred: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    model = regression_metrics(y, pred)
    base = regression_metrics(y, baseline)
    removed = (base["mae"] - model["mae"]) / base["mae"] if base["mae"] > 0 else float("nan")
    return {
        "r2": model["r2"],
        "mae": model["mae"],
        "median_ape": model["median_ape"],
        "baseline_mae": base["mae"],
        "error_removed": float(removed),
        "n": float(model["n"]),
    }


def _city_parts(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {city: predictions[predictions["city"] == city] for city in CITY_ORDER}


def prediction_column(frame: pd.DataFrame) -> str:
    """Shipped prediction column. Older files named it after ridge."""
    if "price_model" in frame.columns:
        return "price_model"
    return "price_ridge"


def _equal_weight(parts: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Mean of the three city scores. Each city is one vote.

    The percent less error is computed from those two averaged misses.
    Averaging the three city percentages is a different number, because
    the cities do not have the same usual rate.
    """
    rows = [
        _score(
            part["y_price"].to_numpy(),
            part[prediction_column(part)].to_numpy(),
            part["price_area_median"].to_numpy(),
        )
        for part in parts.values()
    ]
    out = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("r2", "mae", "median_ape", "baseline_mae")
    }
    out["error_removed"] = (out["baseline_mae"] - out["mae"]) / out["baseline_mae"]
    return out


def _every_home(parts: dict[str, pd.DataFrame]) -> dict[str, float]:
    """One score on the stacked holdout. A city with more homes counts more."""
    frame = pd.concat(parts.values(), ignore_index=True)
    return _score(
        frame["y_price"].to_numpy(),
        frame[prediction_column(frame)].to_numpy(),
        frame["price_area_median"].to_numpy(),
    )


def _resample(part: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    groups: dict[int, list[int]] = {}
    for index, cell in enumerate(part["cell"].to_numpy()):
        groups.setdefault(int(cell), []).append(index)
    keys = list(groups)
    draw = rng.integers(0, len(keys), size=len(keys))
    chosen = np.concatenate([np.asarray(groups[keys[int(item)]], dtype=int) for item in draw])
    return part.iloc[chosen]


def bootstrap_overall(
    predictions: pd.DataFrame, *, n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED
) -> dict[str, dict[str, dict[str, float]]]:
    """95% intervals for both overalls. Cells are drawn inside each city."""
    parts = _city_parts(predictions)
    point = {"each_city_once": _equal_weight(parts), "every_home_once": _every_home(parts)}
    rng = np.random.default_rng(int(seed))
    bags = {
        "each_city_once": {key: [] for key in point["each_city_once"]},
        "every_home_once": {key: [] for key in point["every_home_once"]},
    }
    for _ in range(int(n_boot)):
        drawn = {city: _resample(part, rng) for city, part in parts.items()}
        for name, scores in (
            ("each_city_once", _equal_weight(drawn)),
            ("every_home_once", _every_home(drawn)),
        ):
            for key, value in scores.items():
                bags[name][key].append(value)
    packed: dict[str, dict[str, dict[str, float]]] = {}
    for name, scores in point.items():
        packed[name] = {}
        for key, value in scores.items():
            low, high = np.nanpercentile(np.asarray(bags[name][key], dtype=float), [2.5, 97.5])
            packed[name][key] = {"point": float(value), "low": float(low), "high": float(high)}
    return packed


def _show(value: float, kind: str) -> str:
    if kind == "r2":
        return r2(value)
    if kind == "money":
        return money(value)
    if kind == "pct":
        return pct(value)
    # error_removed is a fraction. The city tables print it as a percent.
    return pct(100.0 * value)


def _span(metric: dict[str, float], kind: str) -> str:
    """Point and interval, formatted the same way as the city tables."""
    return f"{_show(metric['point'], kind)} ({_show(metric['low'], kind)} to {_show(metric['high'], kind)})"


def _summary_table(each: dict, homes: dict) -> list[str]:
    header = "| Summary | R² | Model miss | Local median rate | Median % error | Less error |"
    rule = "| --- | ---: | ---: | ---: | ---: | ---: |"

    def row(label: str, block: dict) -> str:
        return (
            f"| {label} | "
            + " | ".join(
                [
                    _span(block["r2"], "r2"),
                    _span(block["mae"], "money"),
                    _span(block["baseline_mae"], "money"),
                    _span(block["median_ape"], "pct"),
                    _span(block["error_removed"], "removed"),
                ]
            )
            + " |"
        )

    return [header, rule, row("Every home once", homes), row("Each city once", each)]


def findings_overall(predictions: pd.DataFrame) -> str:
    """Selection-time overall, before the added features. Quoted by the report test."""
    stats = bootstrap_overall(predictions)
    each = stats["each_city_once"]
    homes = stats["every_home_once"]
    lines = [
        "Counting every holdout home once is the pooled result on these columns. "
        f"The miss is {money(homes['mae']['point'])} a night, against {money(homes['baseline_mae']['point'])} for the local median rate, "
        f"which is {pct(100.0 * homes['error_removed']['point'])} less error. "
        f"R² is {r2(homes['r2']['point'])}, and the median error is {pct(homes['median_ape']['point'])}. "
        "The holdout is 3,648 homes. "
        "Resampling cells inside each city 1,000 times, then stacking the draws, gives a 95% interval of "
        f"R² {r2(homes['r2']['low'])} to {r2(homes['r2']['high'])}, "
        f"a miss of {money(homes['mae']['low'])} to {money(homes['mae']['high'])}, "
        f"a local-median miss of {money(homes['baseline_mae']['low'])} to {money(homes['baseline_mae']['high'])}, "
        f"a median error of {pct(homes['median_ape']['low'])} to {pct(homes['median_ape']['high'])}, "
        f"and {pct(100.0 * homes['error_removed']['low'])} to {pct(100.0 * homes['error_removed']['high'])} less error. "
        "Seattle's part of that interval resamples only 8 held-out cells.",
        "",
        "The city rows stay beside that pool. Austin, San Diego, and Seattle are different markets: "
        "the median advertised rate on this holdout is $146 in Austin, $153 in San Diego, and $111 in Seattle, "
        "and the holdouts are 2,101, 817, and 730 homes. "
        "One score for all three would let Austin's prices, and its larger sample, speak for the other two cities.",
        "",
        "Counting each city once is the overall the model was chosen on. Ridge's average miss is "
        f"{money(each['mae']['point'])} a night, against {money(each['baseline_mae']['point'])} for the local median rate, "
        f"which is {pct(100.0 * each['error_removed']['point'])} less error. "
        f"The average of the three R² values is {r2(each['r2']['point'])}, and the average of the three median percent errors is {pct(each['median_ape']['point'])}. "
        "Resampling each city's holdout cells 1,000 times, then averaging the three cities, gives a 95% interval of "
        f"R² {r2(each['r2']['low'])} to {r2(each['r2']['high'])}, a miss of {money(each['mae']['low'])} to {money(each['mae']['high'])}, "
        f"a local-median miss of {money(each['baseline_mae']['low'])} to {money(each['baseline_mae']['high'])}, "
        f"a median percent error of {pct(each['median_ape']['low'])} to {pct(each['median_ape']['high'])}, "
        f"and {pct(100.0 * each['error_removed']['low'])} to {pct(100.0 * each['error_removed']['high'])} less error.",
        "",
        *_summary_table(each, homes),
    ]
    return "\n".join(lines)


def readme_overall_table(predictions: pd.DataFrame) -> str:
    """Point estimates. Every home once is the first row."""
    parts = _city_parts(predictions)
    each = _equal_weight(parts)
    homes = _every_home(parts)
    labels = {"austin": "Austin, TX", "san-diego": "San Diego, CA", "seattle": "Seattle, WA"}
    lines = [
        "| City | R² | Average miss | Median error | Less error than the local median rate |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| Every home once | "
        f"{r2(homes['r2'])} | {money(homes['mae'])} | "
        f"{pct(homes['median_ape'])} | {pct(100.0 * homes['error_removed'])} |",
    ]
    for city, part in parts.items():
        score = _score(
            part["y_price"].to_numpy(),
            part[prediction_column(part)].to_numpy(),
            part["price_area_median"].to_numpy(),
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    labels[city],
                    r2(score["r2"]),
                    money(score["mae"]),
                    pct(score["median_ape"]),
                    pct(100.0 * score["error_removed"]),
                ]
            )
            + " |"
        )
    lines.append(
        "| Each city once | "
        f"{r2(each['r2'])} | {money(each['mae'])} | "
        f"{pct(each['median_ape'])} | {pct(100.0 * each['error_removed'])} |"
    )
    return "\n".join(lines)
