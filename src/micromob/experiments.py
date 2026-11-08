"""Queue discharge, dynamic PCU and calibration.

**Saturation flow**: a packed queue waits at red; after green the flow across the stop line between
the 10th and 90th percentile crossings is the saturation flow (veh/h), also expressed in PCU/h.

**Dynamic PCU** (Chandra & Kumar 2003), per type ``k`` relative to cars::

    PCU_k = (V_car / V_k) / (A_car / A_k)

with ``V`` the mean speed while crossing the 30 m downstream of the stop line and ``A`` the plan area.

**Calibration**: Optuna minimises the squared relative error between simulated PCU saturation flow
per metre of approach width and a target (default 630 PCU/h/m, a planning value for mixed-traffic
signalised approaches; replace it with field-measured discharge for a real project), on training seeds,
then reports held-out seeds.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import optuna

from .engine import TYPES, Interaction, Road, neighbour_pairs, queue


def discharge(mix: dict[str, float], n: int = 160, seed: int = 0, inter: Interaction | None = None, road: Road | None = None,
              t_max: float = 180.0, pair_fn=neighbour_pairs) -> dict[str, float]:  # type: ignore[no-untyped-def]
    inter = inter or Interaction()
    road = road or Road()
    rng = np.random.default_rng(seed)
    tr = queue(mix, n, road, rng, inter)
    names = np.array(tr.kind_names)  # type: ignore[attr-defined]
    for _ in range(int(5.0 / inter.dt)):  # settle at red
        tr.step(pair_fn)
    tr.red = False
    t_green = tr.t
    speed_sum = {k: 0.0 for k in TYPES}
    speed_cnt = {k: 0 for k in TYPES}
    while np.isnan(tr.crossed_at).any() and tr.t - t_green < t_max:
        tr.step(pair_fn)
        zone = (tr.pos[:, 0] > 0.0) & (tr.pos[:, 0] < 30.0)
        for k in np.unique(names[zone]):
            m = zone & (names == k)
            speed_sum[k] += float(tr.vel[m, 0].sum())
            speed_cnt[k] += int(m.sum())
    c = np.sort(tr.crossed_at[np.isfinite(tr.crossed_at)])
    lo, hi = int(0.1 * c.size), int(0.9 * c.size)
    veh_h = (hi - lo) / max(c[hi - 1] - c[lo], 1e-6) * 3600.0 if c.size > 10 else float("nan")
    mean_v = {k: speed_sum[k] / speed_cnt[k] for k in TYPES if speed_cnt[k] > 0}
    area = {k: TYPES[k].length * TYPES[k].width for k in TYPES}
    pcu = {}
    if "car" in mean_v:
        for k, v in mean_v.items():
            pcu[k] = (mean_v["car"] / v) / (area["car"] / area[k])
    else:  # no cars in this mix: fall back to static area ratios
        pcu = {k: area[k] / area["car"] for k in mean_v}
    share = {k: float((names == k).mean()) for k in TYPES}
    mean_pcu = sum(share[k] * pcu.get(k, area[k] / area["car"]) for k in TYPES)
    return {"veh_per_h": veh_h, "pcu_per_h": veh_h * mean_pcu, "pcu_per_h_per_m": veh_h * mean_pcu / road.width,
            "mean_pcu_factor": mean_pcu, "uncleared": int(np.isnan(tr.crossed_at).sum()),
            **{f"pcu_{k}": v for k, v in pcu.items()}, **{f"speed_{k}": v for k, v in mean_v.items()}}


BOUNDS = {"a": (1.0, 8.0), "b": (0.2, 1.2), "lam": (0.05, 0.6), "lateral_gain": (0.1, 1.5)}


def calibrate(mix: dict[str, float], target_pcu_per_m: float, *, trials: int = 40, train_seeds: tuple[int, ...] = (0, 1),
              sampler: str = "tpe", seed: int = 0) -> tuple[Interaction, list[float]]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    smp = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10) if sampler == "tpe" else optuna.samplers.RandomSampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=smp)

    def obj(trial: optuna.Trial) -> float:
        inter = replace(Interaction(), **{k: trial.suggest_float(k, *b) for k, b in BOUNDS.items()})
        vals = [discharge(mix, 120, s, inter)["pcu_per_h_per_m"] for s in train_seeds]
        return float(np.mean([(v / target_pcu_per_m - 1.0) ** 2 for v in vals]))

    study.optimize(obj, n_trials=trials)
    best = replace(Interaction(), **study.best_params)
    hist = list(np.minimum.accumulate([t.value for t in study.trials if t.value is not None]))
    return best, hist
