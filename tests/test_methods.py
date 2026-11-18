"""Guards for the parts of the study that are easy to get subtly wrong."""

from __future__ import annotations

import numpy as np
from data.occupancy import san_francisco_occupancy
from evaluation.blocking import buffer_train, purge_hosts
from evaluation.metrics import regression_metrics
from evaluation.splits import iter_group_folds, spatial_group_ids
from features.fold import FoldEncoder


def test_occupancy_formula_and_cap():
    # 1 review/month, 50% review rate -> 2 bookings. Stay is the city average
    # of 3 nights. 6 nights / 30 = 0.20.
    occ = san_francisco_occupancy(np.array([1.0]), np.array([1.0]), 3.0)
    assert abs(occ[0] - 0.20) < 1e-9

    # A longer minimum stay replaces the city average.
    longer = san_francisco_occupancy(np.array([1.0]), np.array([7.0]), 3.0)
    assert abs(longer[0] - 14 / 30) < 1e-9

    # The published cap is 70%, even when reviews imply a full calendar.
    capped = san_francisco_occupancy(np.array([10.0]), np.array([1.0]), 3.0)
    assert capped[0] == 0.70

    # No reviews is "not observed", not 0% occupied.
    missing = san_francisco_occupancy(np.array([0.0, np.nan]), np.array([2.0, 2.0]), 3.0)
    assert np.isnan(missing).all()


def test_encoder_ignores_prices_it_was_not_fit_on():
    lat = np.array([30.0, 30.01, 30.02, 30.5, 30.51])
    lon = np.array([-97.0, -97.01, -97.02, -97.5, -97.51])
    neigh = np.array(["a", "a", "a", "b", "b"])
    beds = np.array([1, 1, 2, 1, 1], dtype=float)
    train_log = np.log1p(np.array([100, 110, 300, 200, 210], dtype=float))

    encoder = FoldEncoder(k=2, prior=1.0).fit(lat[:4], lon[:4], neigh[:4], train_log[:4], beds[:4])
    before = encoder.transform_new(lat[4:], lon[4:], neigh[4:], beds[4:])

    # Re-fitting with a different held-out price must not be required for the
    # transform, and changing an unseen price cannot change a fitted encoder.
    encoder_again = FoldEncoder(k=2, prior=1.0).fit(
        lat[:4], lon[:4], neigh[:4], train_log[:4], beds[:4]
    )
    after = encoder_again.transform_new(lat[4:], lon[4:], neigh[4:], beds[4:])
    for left, right in zip(before, after, strict=True):
        np.testing.assert_allclose(left, right)

    # The training row with log-price log1p(300) must not be its own comparable.
    _neigh, _bed, knn = encoder.transform_train(lat[:4], lon[:4], neigh[:4], beds[:4])
    assert knn[2] < np.log1p(300)


def test_spatial_folds_do_not_share_a_cell():
    rng = np.random.default_rng(0)
    lat = 30.2 + rng.normal(0, 0.05, 80)
    lon = -97.7 + rng.normal(0, 0.05, 80)
    groups = spatial_group_ids(lat, lon, step=0.03)
    for train_idx, test_idx in iter_group_folds(groups, n_splits=4):
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))


def test_host_purge_and_buffer():
    hosts = np.array(["h1", "h1", "h2", "h3"])
    train, removed = purge_hosts(hosts, np.array([0, 2, 3]), np.array([1]))
    assert removed == 1
    assert list(train) == [2, 3]

    lat = np.array([0.0, 0.0, 0.0, 10.0])
    lon = np.array([0.0, 0.01, 0.05, 10.0])
    kept, n_close = buffer_train(lat, lon, np.array([1, 2, 3]), np.array([0]), buffer_km=5.0)
    assert n_close >= 1
    assert 3 in kept
    assert 1 not in kept


def test_metrics_are_on_the_dollar_scale():
    metrics = regression_metrics(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
    assert abs(metrics["mae"] - 15.0) < 1e-9
    assert metrics["r2"] < 1.0
    # |10|/100 and |20|/200 -> 10% both, so MAPE and median APE agree.
    assert abs(metrics["mape"] - 10.0) < 1e-9
    assert abs(metrics["median_ape"] - 10.0) < 1e-9


def test_bedroom_interaction_falls_back_when_the_pair_is_new():
    lat = np.array([30.0, 30.0, 30.0])
    lon = np.array([-97.0, -97.0, -97.0])
    neigh = np.array(["east", "east", "east"])
    beds = np.array([1.0, 1.0, 1.0])
    prices = np.log1p(np.array([100.0, 120.0, 140.0]))
    encoder = FoldEncoder(k=2, prior=0.0).fit(lat, lon, neigh, prices, beds)
    # A 4-bedroom in the same neighbourhood has no pair of its own.
    value = encoder.bedroom_interaction(np.array(["east"]), np.array([4.0]))
    neighbourhood = encoder.neighbourhood_feature(np.array(["east"]))
    np.testing.assert_allclose(value, neighbourhood)
