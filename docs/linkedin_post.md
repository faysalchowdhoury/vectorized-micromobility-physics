I calibrated a mixed-traffic model and hit the target within 0.2%. Then it failed on all-car traffic, and that failure was the useful part.

I built a 2-D physics engine for non-lane-based traffic at a signal: cars, motorcycles, e-scooters and cargo bikes, with forward-looking repulsion, gap-seeking that lets two-wheelers filter between cars, and a spatial hash for neighbour search. I used it to measure saturation flow and dynamic PCU factors, then calibrated it with Bayesian optimisation (Optuna).

What came out:
🔹 Mixed traffic (60% motorcycles): 631 PCU/h/m vs a 630 target, on held-out runs
🔹 Bayesian search beat random search: objective 0.031 vs 0.079 in the same 40 trials
🔹 Dynamic PCU: motorcycle 0.17, scooter 0.11, cargo bike 0.45, stable across mixes
🔹 Replacing cars with two-wheelers doubled vehicles/hour (5,298 → 11,873) but PCU/hour peaked near 40% and then fell

The miss: the same parameters overshot car-only traffic by 41%. On a 7 m approach my physics packs cars three abreast; real car traffic keeps two lanes. No parameter value fixes that, because the model has no lane-discipline term that depends on traffic mix.

That's the lesson I'd take into any Vissim or Aimsun job: one behaviour set rarely transfers between very different traffic compositions. If calibration only works for one regime, look for the missing mechanism before tuning harder.

Engineering side: fully vectorised forces, position-based contact resolution for stability, and a spatial hash that's 4.3× faster than brute force at 2,000 agents (and slower below ~400, which is worth knowing).

Caveat: calibration targets are published planning values, not field counts.

Repo: [link]

Anyone modelled lane discipline as a function of composition? I'd like to compare approaches.

#MixedTraffic #Micromobility #TrafficSimulation #TrafficEngineering #Calibration #Optuna #Python
