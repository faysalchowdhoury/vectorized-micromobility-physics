"""Joint calibration (TPE vs random), two-wheeler share sweep, dynamic PCU, scaling benchmark."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from .engine import Interaction, Road, brute_pairs, neighbour_pairs, queue
from .experiments import BOUNDS, discharge

CAR_ONLY = {"car": 1.0}
MIXED = {"car": 0.2, "motorcycle": 0.6, "scooter": 0.1, "cargo_bike": 0.1}
# planning targets (PCU/h per metre of approach width) - replace with field discharge counts for a real site
TARGETS = {"car_only": 1900.0 / 3.6, "mixed": 630.0}


def joint_objective(inter: Interaction, seeds: tuple[int, ...]) -> float:
    errs = []
    for s in seeds:
        errs.append((discharge(CAR_ONLY, 100, s, inter)["pcu_per_h_per_m"] / TARGETS["car_only"] - 1) ** 2)
        errs.append((discharge(MIXED, 120, s, inter)["pcu_per_h_per_m"] / TARGETS["mixed"] - 1) ** 2)
    return float(np.mean(errs))


def calibrate(sampler: str, trials: int, seed: int = 0) -> tuple[Interaction, list[float]]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    smp = optuna.samplers.TPESampler(seed=seed, n_startup_trials=8) if sampler == "tpe" else optuna.samplers.RandomSampler(seed=seed)
    st = optuna.create_study(direction="minimize", sampler=smp)
    st.optimize(lambda tr: joint_objective(replace(Interaction(), **{k: tr.suggest_float(k, *b) for k, b in BOUNDS.items()}), (100,)),
                n_trials=trials)
    return replace(Interaction(), **st.best_params), list(np.minimum.accumulate([t.value for t in st.trials]))


def bench(sizes: tuple[int, ...] = (200, 800, 2000, 4000), steps: int = 20) -> list[dict[str, float]]:
    out = []
    for n in sizes:
        rng = np.random.default_rng(0)
        road = Road(width=7.0 * max(1, n // 400))
        tr = queue(MIXED, n, road, rng, Interaction())
        row = {"agents": n}
        for name, fn in (("spatial_hash", neighbour_pairs), ("brute_force", brute_pairs)):
            if name == "brute_force" and n > 2000:
                row[name + "_ms"] = float("nan")
                continue
            t = replace(tr)
            t0 = time.perf_counter()
            for _ in range(steps):
                t.step(fn)
            row[name + "_ms"] = 1000 * (time.perf_counter() - t0) / steps
        out.append(row)
    return out


def run(out_dir: str | Path, trials: int = 40, seeds: int = 5) -> dict[str, Any]:
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    default = Interaction()
    cal = {s: calibrate(s, trials) for s in ("random", "tpe")}
    best_name = min(cal, key=lambda s: cal[s][1][-1])
    best = cal[best_name][0]
    held = tuple(range(200, 200 + seeds))
    val = {}
    for name, inter in (("default", default), ("calibrated", best)):
        val[name] = {
            "car_only_pcu_per_m": float(np.mean([discharge(CAR_ONLY, 100, s, inter)["pcu_per_h_per_m"] for s in held])),
            "mixed_pcu_per_m": float(np.mean([discharge(MIXED, 120, s, inter)["pcu_per_h_per_m"] for s in held])),
        }
    rows = []
    for tw in (0.0, 0.2, 0.4, 0.6, 0.8):
        mix = {"car": 1 - tw, "motorcycle": tw * 0.75, "scooter": tw * 0.125, "cargo_bike": tw * 0.125}
        mix = {k: v for k, v in mix.items() if v > 0}
        for s in held:
            r = discharge(mix, 140, s, best)
            rows.append({"two_wheeler_share": tw, "seed": s, **r})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out / "sweep.csv", index=False)
    agg = sweep.groupby("two_wheeler_share").mean(numeric_only=True).reset_index()
    summary = {"targets": TARGETS, "trials": trials, "best_sampler": best_name, "calibrated": asdict(best), "default": asdict(default),
               "best_objective": {s: cal[s][1][-1] for s in cal}, "convergence": {s: cal[s][1] for s in cal},
               "validation_held_out": val, "sweep": agg.to_dict("records"), "benchmark": bench(),
               "wall_s": time.perf_counter() - t0}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    _plot(agg, cal, summary["benchmark"], out / "figures")
    return summary


def _plot(agg: pd.DataFrame, cal: dict, bench_rows: list[dict[str, float]], fig_dir: Path) -> None:  # type: ignore[type-arg]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    x = 100 * agg["two_wheeler_share"]
    axes[0].plot(x, agg["veh_per_h"], "o-", color="#495057", label="vehicles/h")
    ax2 = axes[0].twinx()
    ax2.plot(x, agg["pcu_per_h"], "s--", color="#1c7ed6", label="PCU/h")
    axes[0].set_xlabel("two-wheeler share (%)")
    axes[0].set_ylabel("vehicles/h (7 m approach)")
    ax2.set_ylabel("PCU/h", color="#1c7ed6")
    for s, col in (("random", "#adb5bd"), ("tpe", "#2b8a3e")):
        axes[1].plot(np.arange(1, len(cal[s][1]) + 1), cal[s][1], color=col, label=s.upper())
    axes[1].set_yscale("log")
    axes[1].set_xlabel("trial")
    axes[1].set_ylabel("best joint objective")
    axes[1].legend()
    b = pd.DataFrame(bench_rows)
    axes[2].plot(b["agents"], b["spatial_hash_ms"], "o-", color="#2b8a3e", label="spatial hash")
    axes[2].plot(b["agents"], b["brute_force_ms"], "o-", color="#e03131", label="brute force O(N²)")
    axes[2].set_xlabel("agents")
    axes[2].set_ylabel("ms per step")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "mixed_traffic.png", dpi=130)
    plt.close(fig)


def write_report(s: dict[str, Any], out: Path) -> Path:
    v = s["validation_held_out"]
    c, d = s["calibrated"], s["default"]
    L = ["# Mixed-traffic results", "", f"Calibration: {s['trials']} trials each (TPE vs random) on seed 100; best = {s['best_sampler'].upper()} "
         f"(objective TPE {s['best_objective']['tpe']:.4f}, random {s['best_objective']['random']:.4f}). Wall time {s['wall_s']:.0f} s.", "",
         "| Parameter | Default | Calibrated |", "|---|---|---|"]
    for k in ("a", "b", "lam", "lateral_gain"):
        L.append(f"| {k} | {d[k]:.3g} | {c[k]:.3g} |")
    L += ["", "## Held-out validation (seeds 200-204)", "", "| Mix | Target (PCU/h/m) | Default | Calibrated |", "|---|---|---|---|",
          f"| Car only | {s['targets']['car_only']:.0f} | {v['default']['car_only_pcu_per_m']:.0f} | {v['calibrated']['car_only_pcu_per_m']:.0f} |",
          f"| Mixed (20/60/10/10) | {s['targets']['mixed']:.0f} | {v['default']['mixed_pcu_per_m']:.0f} | {v['calibrated']['mixed_pcu_per_m']:.0f} |",
          "", "## Two-wheeler share sweep (calibrated, 7 m approach)", "",
          "| Two-wheeler share | Vehicles/h | PCU/h | Mean PCU factor | PCU motorcycle | PCU scooter | PCU cargo bike |", "|---|---|---|---|---|---|---|"]
    for r in s["sweep"]:
        g = lambda k: f"{r[k]:.2f}" if k in r and r[k] == r[k] else "-"  # noqa: E731
        L.append(f"| {100 * r['two_wheeler_share']:.0f}% | {r['veh_per_h']:.0f} | {r['pcu_per_h']:.0f} | {r['mean_pcu_factor']:.2f} | "
                 f"{g('pcu_motorcycle')} | {g('pcu_scooter')} | {g('pcu_cargo_bike')} |")
    L += ["", "## Scaling", "", "| Agents | Spatial hash (ms/step) | Brute force (ms/step) |", "|---|---|---|"]
    for r in s["benchmark"]:
        L.append(f"| {r['agents']} | {r['spatial_hash_ms']:.1f} | {r['brute_force_ms']:.1f} |")
    L += ["", "![Mixed traffic](figures/mixed_traffic.png)", ""]
    md = out / "RESULTS.md"
    md.write_text("\n".join(L), encoding="utf-8")
    return md
