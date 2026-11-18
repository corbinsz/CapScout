"""Price encodings that are fit on the training fold only.

Two features use other listings' prices:

* A smoothed neighbourhood mean of log price. Small neighbourhoods are
  pulled toward the city mean (prior strength 20) so a block with three
  listings cannot memorize those three prices.
* The median log price of the 40 nearest *other* listings. Trees split on
  latitude and longitude as axis-aligned cuts, which is a poor way to
  represent "the places next door". A distance-based comparable is the
  feature a pricing analyst would actually compute.

A third feature is the same smoothing on (neighbourhood × bedroom bin). A
one-bedroom and a four-bedroom in the same neighbourhood are different
markets; the pair is the interaction, and it is still fit only on training
rows. An unseen pair falls back to the neighbourhood mean, then the city mean.

The scoring artifact is fit on the development listings only. The locked
test listings are not in the index. `transform_train` also drops a row's
own index when another listing shares its coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0


def _smooth(
    keys: np.ndarray, log_price: np.ndarray, prior: float, global_mean: float
) -> dict[str, float]:
    frame = pd.DataFrame({"k": np.asarray(keys).astype(str), "y": log_price})
    grouped = frame.groupby("k")["y"].agg(["sum", "count"])
    smooth = (grouped["sum"] + prior * global_mean) / (grouped["count"] + prior)
    return {str(key): float(val) for key, val in smooth.items()}


def _bed_keys(neighbourhood: np.ndarray, bedrooms: np.ndarray) -> np.ndarray:
    """Bedroom bins are clipped so a 10-bedroom outlier is not its own market."""
    bins = np.clip(np.rint(np.asarray(bedrooms, dtype=float)), 0, 6).astype(int)
    names = np.asarray(neighbourhood).astype(str)
    return np.array([f"{name}|{bed}" for name, bed in zip(names, bins, strict=True)])


def _radians(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.radians(np.column_stack([np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)]))


@dataclass
class FoldEncoder:
    """Neighbourhood smoothing and a haversine comparable index."""

    k: int = 40
    prior: float = 20.0
    global_log_price: float = 0.0
    neigh_map: dict[str, float] = field(default_factory=dict)
    bed_map: dict[str, float] = field(default_factory=dict)
    _train_log: np.ndarray | None = field(default=None, repr=False)
    _tree: BallTree | None = field(default=None, repr=False)

    def fit(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        neighbourhood: np.ndarray,
        log_price: np.ndarray,
        bedrooms: np.ndarray,
    ) -> FoldEncoder:
        log_price = np.asarray(log_price, dtype=float)
        self.global_log_price = float(np.mean(log_price))
        names = np.asarray(neighbourhood).astype(str)
        self.neigh_map = _smooth(names, log_price, self.prior, self.global_log_price)
        self.bed_map = _smooth(
            _bed_keys(names, bedrooms), log_price, self.prior, self.global_log_price
        )
        self._train_log = log_price
        self._tree = BallTree(_radians(lat, lon), metric="haversine")
        return self

    def neighbourhood_feature(self, neighbourhood: np.ndarray) -> np.ndarray:
        global_mean = self.global_log_price
        return np.array(
            [self.neigh_map.get(str(name), global_mean) for name in np.asarray(neighbourhood)],
            dtype=float,
        )

    def bedroom_interaction(self, neighbourhood: np.ndarray, bedrooms: np.ndarray) -> np.ndarray:
        """Smoothed (neighbourhood, bedroom) mean, with neighbourhood fallback."""
        keys = _bed_keys(neighbourhood, bedrooms)
        neigh = self.neighbourhood_feature(neighbourhood)
        return np.array(
            [self.bed_map.get(key, neigh[i]) for i, key in enumerate(keys)],
            dtype=float,
        )

    def knn_feature(
        self, lat: np.ndarray, lon: np.ndarray, *, row_index: np.ndarray | None
    ) -> np.ndarray:
        """Median neighbour log price.

        Pass `row_index` (positions into the fitted training arrays) when the
        query rows are themselves in the tree, so a row cannot be its own
        comparable. Pass None for new coordinates.
        """
        if self._tree is None or self._train_log is None:
            raise RuntimeError("FoldEncoder.fit must be called before knn_feature")

        n_train = len(self._train_log)
        points = _radians(lat, lon)
        drop_self = row_index is not None
        k_query = min(self.k + (1 if drop_self else 0), n_train)
        _dist, idx = self._tree.query(points, k=k_query)
        idx = np.atleast_2d(idx)
        values = self._train_log[idx].astype(float)

        if drop_self:
            own = np.asarray(row_index, dtype=int)[:, None]
            values = np.where(idx == own, np.nan, values)

        with np.errstate(all="ignore"):
            med = np.nanmedian(values, axis=1)
        return np.where(np.isfinite(med), med, self.global_log_price)

    def transform_train(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        neighbourhood: np.ndarray,
        bedrooms: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encodings for the rows that were just fit, in that same order."""
        knn = self.knn_feature(lat, lon, row_index=np.arange(len(lat)))
        return (
            self.neighbourhood_feature(neighbourhood),
            self.bedroom_interaction(neighbourhood, bedrooms),
            knn,
        )

    def transform_new(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        neighbourhood: np.ndarray,
        bedrooms: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encodings for coordinates that were not in the fit."""
        knn = self.knn_feature(lat, lon, row_index=None)
        return (
            self.neighbourhood_feature(neighbourhood),
            self.bedroom_interaction(neighbourhood, bedrooms),
            knn,
        )
