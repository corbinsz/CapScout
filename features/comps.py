"""Comparable listings, fit on the training fold only.

v2's nearest-neighbour median mixed a private room with the house next
door, and it looked only 40 listings out, which under a 1 km buffer is
still the edge of the same neighbourhood. This version asks for the same
product: the same room type and the same bedroom bin, inside 15 km, which
is far enough to clear a held-out cell. A row never contributes its own
price. If the match is too thin, the feature falls back to the same room
type and then to the training-set median, rather than inventing a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from evaluation.protocol import COMP_MAX_KM, COMP_MIN_MATCHES, COMP_NEIGHBORS, COMP_QUERY
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0


def _radians(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.radians(np.column_stack([np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)]))


def bedroom_bin(bedrooms: np.ndarray) -> np.ndarray:
    """0, 1, 2, 3, and 4-plus. A 10-bedroom outlier is not its own market."""
    return np.clip(np.rint(np.asarray(bedrooms, dtype=float)), 0, 4).astype(int)


def _masked_median(values: np.ndarray, mask: np.ndarray, min_count: int) -> np.ndarray:
    """Median of the closest matches. `values` are already in distance order.

    Matches past the neighbour cap are dropped. Fewer than `min_count`
    matches becomes NaN so the caller can fall back.
    """
    order = np.cumsum(mask, axis=1)
    picked = np.where(mask & (order <= COMP_NEIGHBORS), values, np.nan)
    counts = np.sum(np.isfinite(picked), axis=1)
    med = np.full(picked.shape[0], np.nan, dtype=float)
    enough = counts > 0
    if np.any(enough):
        med[enough] = np.nanmedian(picked[enough], axis=1)
    med = np.where(counts >= min_count, med, np.nan)
    return med


@dataclass
class MatchedComps:
    """Median log price of similar training listings within a fixed radius."""

    max_km: float = COMP_MAX_KM
    min_matches: int = COMP_MIN_MATCHES
    global_log_price: float = 0.0
    _log_price: np.ndarray | None = field(default=None, repr=False)
    _rooms: np.ndarray | None = field(default=None, repr=False)
    _bins: np.ndarray | None = field(default=None, repr=False)
    _tree: BallTree | None = field(default=None, repr=False)

    def fit(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        log_price: np.ndarray,
        room_type: np.ndarray,
        bedrooms: np.ndarray,
    ) -> MatchedComps:
        log_price = np.asarray(log_price, dtype=float)
        self.global_log_price = float(np.mean(log_price))
        self._log_price = log_price
        self._rooms = np.asarray(room_type).astype(str)
        self._bins = bedroom_bin(bedrooms)
        self._tree = BallTree(_radians(lat, lon), metric="haversine")
        return self

    def transform(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        room_type: np.ndarray,
        bedrooms: np.ndarray,
        *,
        row_index: np.ndarray | None,
    ) -> np.ndarray:
        if self._tree is None or self._log_price is None:
            raise RuntimeError("MatchedComps.fit must be called before transform")
        n_train = len(self._log_price)
        rooms = np.asarray(room_type).astype(str)
        bins = bedroom_bin(bedrooms)
        if n_train == 0:
            return np.full(len(rooms), self.global_log_price)
        k_query = min(COMP_QUERY + (1 if row_index is not None else 0), n_train)
        dist, idx = self._tree.query(_radians(lat, lon), k=max(k_query, 1))
        dist = np.atleast_2d(np.asarray(dist, dtype=float))
        idx = np.atleast_2d(np.asarray(idx, dtype=int))
        within = dist * EARTH_RADIUS_KM <= self.max_km
        if row_index is not None:
            own = np.asarray(row_index, dtype=int)[:, None]
            within = within & (idx != own)
        train_rooms = self._rooms[idx]
        train_bins = self._bins[idx]
        values = self._log_price[idx]
        same_segment = within & (train_rooms == rooms[:, None]) & (train_bins == bins[:, None])
        same_room = within & (train_rooms == rooms[:, None])
        segment = _masked_median(values, same_segment, self.min_matches)
        room = _masked_median(values, same_room, self.min_matches)
        chosen = np.where(np.isfinite(segment), segment, room)
        return np.where(np.isfinite(chosen), chosen, self.global_log_price)
