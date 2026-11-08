"""Vectorised 2-D mixed-traffic physics.

Every road user ``i`` is a disc of radius ``r_i`` (half its width) with an effective longitudinal
half-length ``l_i``. Its acceleration is::

    a_i = (v0_i e_x - v_i) / tau_i                                     driving term
        + sum_j w(phi_ij) A_i exp((r_ij^eff - d_ij) / B) n_ij           anisotropic repulsion
        + wall terms (road edges)                                        lateral confinement
        + lateral seeking term towards the larger side gap               filtering behaviour

* ``w(phi) = lambda + (1 - lambda)(1 + cos phi)/2`` weights users *ahead* more than users behind
  (Helbing & Molnar's anisotropy), so traffic is forward-looking.
* ``r_ij^eff`` grows with the longitudinal component of the separation (``l_i + l_j`` along x,
  ``r_i + r_j`` across), so a car needs more space ahead than beside it, which lets narrow vehicles
  filter between cars.
* Hard contact is prevented by a projection step that separates overlapping pairs (position-based
  dynamics), so the dynamics stay stable at dt = 0.1 s without stiff contact forces.
* Neighbour search uses a uniform spatial hash (cell 6 m) with a half-stencil: O(N) for bounded density.

A red signal is a wall at the stop line for everyone; on green it disappears.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CELL = 6.0
_HALF = ((1, -1), (1, 0), (1, 1), (0, 1))


@dataclass(frozen=True)
class VehicleType:
    name: str
    length: float
    width: float
    v0_mean: float
    v0_sd: float
    accel: float
    color: str


TYPES: dict[str, VehicleType] = {
    "car": VehicleType("car", 4.5, 1.8, 12.0, 1.0, 2.0, "#495057"),
    "motorcycle": VehicleType("motorcycle", 2.0, 0.8, 12.5, 1.5, 2.8, "#e03131"),
    "scooter": VehicleType("scooter", 1.2, 0.6, 6.5, 0.8, 1.5, "#1c7ed6"),
    "cargo_bike": VehicleType("cargo_bike", 2.5, 1.0, 5.0, 0.7, 0.9, "#2b8a3e"),
}


@dataclass(frozen=True)
class Interaction:
    a: float = 3.0  # m/s^2 repulsion strength
    b: float = 0.6  # m range
    lam: float = 0.15  # anisotropy (0 = only ahead matters)
    tau: float = 0.8  # s relaxation
    lateral_gain: float = 0.6
    dt: float = 0.1


def neighbour_pairs(pos: np.ndarray, cell: float = CELL) -> tuple[np.ndarray, np.ndarray]:
    n = pos.shape[0]
    if n < 2:
        return np.empty(0, int), np.empty(0, int)
    keys = np.floor(pos / cell).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv, minlength=uniq.shape[0])
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    buckets = {(int(uniq[c, 0]), int(uniq[c, 1])): order[starts[c]:starts[c] + counts[c]] for c in range(uniq.shape[0])}
    ii, jj = [], []
    for (kx, ky), m in buckets.items():
        if m.size > 1:
            a, b = np.triu_indices(m.size, 1)
            ii.append(m[a])
            jj.append(m[b])
        for dx, dy in _HALF:
            o = buckets.get((kx + dx, ky + dy))
            if o is not None:
                ii.append(np.repeat(m, o.size))
                jj.append(np.tile(o, m.size))
    if not ii:
        return np.empty(0, int), np.empty(0, int)
    return np.concatenate(ii), np.concatenate(jj)


def brute_pairs(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    i, j = np.triu_indices(pos.shape[0], 1)
    return i, j


@dataclass
class Road:
    width: float = 7.0
    stop_line: float = 0.0


@dataclass
class Traffic:
    road: Road
    pos: np.ndarray
    vel: np.ndarray
    v0: np.ndarray
    half_w: np.ndarray
    half_l: np.ndarray
    accel_max: np.ndarray
    kind: np.ndarray  # type index
    inter: Interaction = field(default_factory=Interaction)
    t: float = 0.0
    red: bool = True
    crossed_at: np.ndarray = field(default_factory=lambda: np.empty(0))

    def step(self, pair_fn=neighbour_pairs) -> None:  # type: ignore[no-untyped-def]
        p, rd = self.inter, self.road
        n = self.pos.shape[0]
        drive = np.zeros_like(self.vel)
        drive[:, 0] = (self.v0 - self.vel[:, 0]) / p.tau
        drive[:, 1] = -self.vel[:, 1] / p.tau
        f = drive
        i, j = pair_fn(self.pos)
        if i.size:
            d = self.pos[i] - self.pos[j]  # from j to i
            dist = np.linalg.norm(d, axis=1) + 1e-9
            near = dist < 15.0
            i, j, d, dist = i[near], j[near], d[near], dist[near]
            nrm = d / dist[:, None]
            # effective contact distance: longitudinal extent along x, lateral extent along y
            reff = np.abs(nrm[:, 0]) * (self.half_l[i] + self.half_l[j]) + np.abs(nrm[:, 1]) * (self.half_w[i] + self.half_w[j])
            mag = p.a * np.exp((reff + 0.5 - dist) / p.b)
            # anisotropy: i reacts strongly to j ahead of it (j to the +x side of i)
            cos_i = -nrm[:, 0]  # direction of travel is +x; j ahead of i => nrm_x < 0
            w_i = p.lam + (1 - p.lam) * (1 + cos_i) / 2
            w_j = p.lam + (1 - p.lam) * (1 - cos_i) / 2
            np.add.at(f, i, (w_i * mag)[:, None] * nrm)
            np.add.at(f, j, -(w_j * mag)[:, None] * nrm)
        # road edges
        y = self.pos[:, 1]
        f[:, 1] += p.a * np.exp((self.half_w + 0.2 - y) / p.b) - p.a * np.exp((self.half_w + 0.2 - (rd.width - y)) / p.b)
        # lateral seeking: drift away from a slower user directly ahead towards the freer side
        if i.size:
            ahead = (self.pos[j, 0] > self.pos[i, 0]) & (np.abs(d[:, 1]) < self.half_w[i] + self.half_w[j] + 0.3) & (dist < 8.0)
            side = np.sign(self.pos[i, 1] - self.pos[j, 1] + 1e-6)
            room = np.where(side > 0, rd.width - self.pos[i, 1], self.pos[i, 1]) > (2 * self.half_w[i] + 0.4)
            push = np.where(ahead & room, p.lateral_gain * side, 0.0)
            np.add.at(f[:, 1], i, push)
        # signal: red = wall at the stop line for anyone still upstream
        if self.red:
            gap = rd.stop_line - (self.pos[:, 0] + self.half_l)
            upstream = gap > -0.5
            f[:, 0] -= np.where(upstream, 6.0 * np.exp(-np.maximum(gap, 0.0) / 1.0) + np.maximum(self.vel[:, 0], 0) ** 2 / (2 * np.maximum(gap, 0.3)), 0.0)
        acc = np.clip(f, -6.0, self.accel_max[:, None])
        self.vel = self.vel + acc * p.dt
        self.vel[:, 0] = np.clip(self.vel[:, 0], 0.0, self.v0 * 1.1)
        self.vel[:, 1] = np.clip(self.vel[:, 1], -1.5, 1.5)
        old_x = self.pos[:, 0].copy()
        self.pos = self.pos + self.vel * p.dt
        self.pos[:, 1] = np.clip(self.pos[:, 1], self.half_w + 0.05, rd.width - self.half_w - 0.05)
        self._separate()
        if self.red:
            # hold every user that has not crossed: the contact correction above can push a front
            # vehicle forward, so the stop line is enforced after it (a test caught leaks to x = 63 m)
            waiting = np.isnan(self.crossed_at)
            over = waiting & (self.pos[:, 0] + self.half_l > rd.stop_line)
            self.pos[over, 0] = rd.stop_line - self.half_l[over]
            self.vel[over, 0] = 0.0
        crossed = (old_x < rd.stop_line) & (self.pos[:, 0] >= rd.stop_line) & np.isnan(self.crossed_at)
        self.crossed_at[crossed] = self.t + p.dt
        self.t += p.dt

    def _separate(self) -> None:
        """Position-based contact resolution: push overlapping pairs apart (no interpenetration)."""
        i, j = neighbour_pairs(self.pos)
        if not i.size:
            return
        for _ in range(2):
            d = self.pos[i] - self.pos[j]
            dx, dy = np.abs(d[:, 0]), np.abs(d[:, 1])
            ox = (self.half_l[i] + self.half_l[j]) - dx
            oy = (self.half_w[i] + self.half_w[j]) - dy
            hit = (ox > 0) & (oy > 0)
            if not hit.any():
                return
            ii, jj = i[hit], j[hit]
            # resolve along the axis of smaller overlap (rectangles, axis-aligned)
            along_x = ox[hit] < oy[hit]
            sx = np.where(along_x, np.sign(d[hit, 0] + 1e-9) * ox[hit] / 2, 0.0)
            sy = np.where(~along_x, np.sign(d[hit, 1] + 1e-9) * oy[hit] / 2, 0.0)
            corr = np.zeros_like(self.pos)
            np.add.at(corr, ii, np.stack([sx, sy], axis=1))
            np.add.at(corr, jj, -np.stack([sx, sy], axis=1))
            self.pos += corr
            self.pos[:, 1] = np.clip(self.pos[:, 1], self.half_w + 0.05, self.road.width - self.half_w - 0.05)
            # kill closing velocity along the resolved axis for the follower
            self.vel[ii[along_x], 0] = np.minimum(self.vel[ii[along_x], 0], self.vel[jj[along_x], 0] + 0.5)


def queue(mix: dict[str, float], n: int, road: Road, rng: np.random.Generator, inter: Interaction) -> Traffic:
    """Arrange ``n`` users of the given mix as a dense, randomly packed queue upstream of the stop line."""
    names = list(mix)
    probs = np.array([mix[k] for k in names], float)
    probs /= probs.sum()
    kinds = rng.choice(len(names), size=n, p=probs)
    tt = [TYPES[names[k]] for k in kinds]
    half_w = np.array([t.width / 2 for t in tt])
    half_l = np.array([t.length / 2 for t in tt])
    v0 = np.array([max(rng.normal(t.v0_mean, t.v0_sd), 2.0) for t in tt])
    acc = np.array([t.accel for t in tt])
    pos = np.zeros((n, 2))
    x_front = road.stop_line - 0.5
    lane_y = 0.0
    row_len = 0.0
    for k in range(n):  # fill rows across the width, front to back
        if lane_y + 2 * half_w[k] + 0.2 > road.width:
            x_front -= row_len + 0.6
            lane_y, row_len = 0.0, 0.0
        pos[k] = (x_front - half_l[k], lane_y + half_w[k] + 0.1)
        lane_y += 2 * half_w[k] + 0.3
        row_len = max(row_len, 2 * half_l[k])
    tr = Traffic(road, pos, np.zeros((n, 2)), v0, half_w, half_l, acc, np.array([names.index(names[k]) for k in kinds]), inter)
    tr.crossed_at = np.full(n, np.nan)
    tr.kind_names = [names[k] for k in kinds]  # type: ignore[attr-defined]
    return tr
