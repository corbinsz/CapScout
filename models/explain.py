"""TreeSHAP contributions in log-price units.

The bias column is dropped. A positive value pushed the prediction up
relative to the model's baseline. For a single row the app turns that into
an approximate percent, exp(shap) - 1, which is the right scale because the
model was trained on log1p(price).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def shap_matrix(model, frame: pd.DataFrame, library: str) -> np.ndarray:
    frame = frame.reset_index(drop=True)
    if library == "xgboost":
        import xgboost as xgb

        contribs = model.get_booster().predict(xgb.DMatrix(frame), pred_contribs=True)
    elif library == "lightgbm":
        contribs = model.predict(frame, pred_contrib=True)
    elif library == "catboost":
        from catboost import Pool

        contribs = model.get_feature_importance(Pool(frame), type="ShapValues")
    else:
        raise ValueError(library)
    values = np.asarray(contribs, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values[:, :-1]


def top_reasons(
    names: list[str],
    values: np.ndarray,
    k: int = 5,
    *,
    observed: np.ndarray | None = None,
    feature_values: np.ndarray | None = None,
) -> list[dict]:
    """Largest contributions. Skip columns the caller left missing.

    A missing review count is not "this host has few listings." Showing that
    SHAP bar would describe the hole, not the home.
    """
    order = np.argsort(-np.abs(values))
    rows = []
    for index in order:
        if observed is not None and not bool(observed[index]):
            continue
        shap = float(values[index])
        if shap == 0.0:
            continue
        entry = {
            "feature": names[int(index)],
            "shap": shap,
            "approx_percent": float((np.exp(shap) - 1.0) * 100.0),
        }
        if feature_values is not None:
            entry["value"] = float(feature_values[index])
        rows.append(entry)
        if len(rows) >= k:
            break
    return rows
