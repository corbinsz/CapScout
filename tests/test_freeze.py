"""The frozen listing ids are the holdout, and the buffer rules still hold."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from data.ingest import load_listings
from evaluation.audit import split_audit
from evaluation.blocking import locked_split
from evaluation.freeze import PATH, listing_ids
from evaluation.protocol import TEST_SEED


def test_frozen_ids_match_the_seed_and_block_hosts_and_neighbours():
    saved = json.loads(PATH.read_text())
    assert saved["test_seed"] == TEST_SEED
    assert "price" not in json.dumps(saved["cities"])
    for city, frozen in saved["cities"].items():
        frame, _stats = load_listings(city)
        split = locked_split(
            frame["latitude"].to_numpy(dtype=float),
            frame["longitude"].to_numpy(dtype=float),
            frame["host_id"].to_numpy(),
        )
        computed = sorted(listing_ids(frame.iloc[split["test_idx"]]["id"]), key=int)
        assert computed == frozen
        audit = split_audit(frame, split["development_idx"], split["test_idx"])
        assert audit["passed"], audit
        assert audit["min_train_test_km"] >= 1.0 - 1e-6
        # Every listing in a test cell is in the test. A partial cell would
        # let the model train on the other half of the block.
        from evaluation.splits import spatial_group_ids

        groups = spatial_group_ids(
            frame["latitude"].to_numpy(dtype=float),
            frame["longitude"].to_numpy(dtype=float),
        )
        test_groups = set(groups[split["test_idx"]])
        assert set(groups[np.isin(groups, list(test_groups))]).isdisjoint(
            set(groups[split["development_idx"]])
        )


def test_freeze_file_lives_with_the_protocol():
    assert Path(PATH).name == "locked_test_ids.json"


def test_confirmation_ids_are_disjoint_from_the_locked_test():
    from evaluation.confirm import PATH as CONFIRM_PATH
    from evaluation.confirm import computed_ids

    saved = json.loads(CONFIRM_PATH.read_text())
    assert saved["confirm_seed"] == 11
    assert "price" not in json.dumps(saved["cities"])
    recomputed = computed_ids(tuple(saved["cities"]))
    locked = json.loads(PATH.read_text())
    for city, ids in saved["cities"].items():
        assert ids == recomputed[city]
        assert set(ids).isdisjoint(locked["cities"][city])
