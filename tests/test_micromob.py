from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from micromob.api.app import create_app
from micromob.engine import Interaction, Road, brute_pairs, neighbour_pairs, queue
from micromob.experiments import discharge


def test_spatial_hash_finds_all_close_pairs() -> None:
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 60, (400, 2))
    i, j = neighbour_pairs(pos)
    got = {(min(a, b), max(a, b)) for a, b in zip(i, j)}
    bi, bj = brute_pairs(pos)
    d = np.linalg.norm(pos[bi] - pos[bj], axis=1)
    want = {(a, b) for a, b, dd in zip(bi, bj, d) if dd < 6.0}
    assert want <= got


def test_red_holds_the_queue() -> None:
    tr = queue({"car": 0.5, "motorcycle": 0.5}, 60, Road(), np.random.default_rng(0), Interaction())
    for _ in range(100):
        tr.step()
    assert np.all(tr.pos[:, 0] + tr.half_l <= 0.0 + 1e-6)
    assert np.isnan(tr.crossed_at).all()


def test_no_overlap_and_inside_road() -> None:
    tr = queue({"car": 0.3, "motorcycle": 0.7}, 80, Road(), np.random.default_rng(2), Interaction())
    tr.red = False
    for _ in range(150):
        tr.step()
    assert np.all(tr.pos[:, 1] - tr.half_w >= 0.0) and np.all(tr.pos[:, 1] + tr.half_w <= 7.0)
    i, j = neighbour_pairs(tr.pos)
    d = np.abs(tr.pos[i] - tr.pos[j])
    overlap = (d[:, 0] < (tr.half_l[i] + tr.half_l[j]) - 0.2) & (d[:, 1] < (tr.half_w[i] + tr.half_w[j]) - 0.2)
    assert overlap.mean() < 0.01


def test_discharge_and_dynamic_pcu() -> None:
    r = discharge({"car": 0.5, "motorcycle": 0.5}, 100, 0)
    assert r["uncleared"] == 0 and r["veh_per_h"] > 2000
    assert 0.05 < r["pcu_motorcycle"] < 0.6  # dynamic PCU of a motorcycle is a fraction of a car
    assert r["pcu_car"] == 1.0


def test_api() -> None:
    c = TestClient(create_app())
    assert c.post("/v1/discharge", json={"mix": {"car": 1.0}, "vehicles": 40}).status_code == 200
    assert c.post("/v1/discharge", json={"mix": {"tank": 1.0}}).status_code == 422
