"""Operationalisation of the scenario-quality objectives (paper Eq. 8).

  R  realism    -- 1 - violation rate over the rollout (speed bounds,
                   off-road, unsafe proximity without collision)
  C  criticality-- collision rate against a fixed NON-REACTIVE reference
                   ego (constant velocity), characterising the adversarial
                   pressure of the scenario distribution itself
  F  fidelity   -- agreement between the JSON spec and the executed rollout
                   (initial placement + interaction occurrence)

Report R, C, F separately AND the weighted composite Q, with a
weight-sensitivity row. Default weights w = (0.4, 0.4, 0.2).
"""
from __future__ import annotations
import numpy as np
from .env_graph import make_env

DEFAULT_W = (0.4, 0.4, 0.2)


def rollout_with_metrics(policy_fn, scenario, seed=0):
    env, obs = make_env(scenario, seed=seed)
    uenv = env.unwrapped
    ego = uenv.vehicle
    ego_spec = next(v for v in scenario.vehicles if v.role == "ego")
    f_init = float(abs(ego.speed - ego_spec.speed) <= 1.0)

    viol = crit = steps = 0
    inter_done = 0.0
    inter = scenario.interactions[0] if scenario.interactions else None
    done = trunc = False
    while not (done or trunc):
        a = policy_fn(obs)
        obs, r, done, trunc, info = env.step(a)
        steps += 1
        v = np.hypot(*ego.velocity)
        offroad = not ego.on_road
        speed_bad = not (15.0 <= v <= 40.0)
        gaps = [np.linalg.norm(ego.position - o.position)
                for o in uenv.road.vehicles if o is not ego]
        ttcs = []
        for o in uenv.road.vehicles:
            if o is ego:
                continue
            rel_p = o.position - ego.position
            rel_v = np.array(o.velocity) - np.array(ego.velocity)
            closing = rel_p @ rel_v < 0
            sp = np.linalg.norm(rel_v)
            if closing and sp > 1e-3:
                ttcs.append(max(0.0, (np.linalg.norm(rel_p) - 5.0) / sp))
        viol += int(speed_bad or offroad or
                    (gaps and min(gaps) < 2.0 and not ego.crashed))
        crit += int(bool(ttcs) and min(ttcs) < 3.0)
        if inter and inter["type"] == "cut_in" and not inter_done:
            adv = next((o for o in uenv.road.vehicles
                        if o is not ego and
                        abs(np.linalg.norm(o.position - ego.position)) < 60), None)
            if adv is not None:
                same_lane = abs(adv.position[1] - ego.position[1]) < 2.0
                ahead = 0 < (adv.position[0] - ego.position[0]) < 40
                if same_lane and ahead:
                    inter_done = 1.0
    crashed = ego.crashed
    env.close()
    R = 1.0 - viol / max(steps, 1)
    C = min(1.0, crit / max(steps, 1) + (1.0 if crashed else 0.0))
    f_parts = [f_init] + ([inter_done] if (inter and inter["type"] == "cut_in") else [])
    Fs = float(np.mean(f_parts))
    return {"R": R, "C": C, "F": Fs, "crashed": crashed, "steps": steps}


def objective(m, w=DEFAULT_W):
    return w[0] * m["R"] + w[1] * m["C"] + w[2] * m["F"]


def summarise(records, weights=(DEFAULT_W, (0.5, 0.3, 0.2), (0.3, 0.5, 0.2))):
    out = {k: float(np.mean([r[k] for r in records])) for k in ("R", "C", "F")}
    out["collision"] = float(np.mean([r["crashed"] for r in records]))
    for w in weights:
        out[f"J_w{w}"] = float(np.mean([objective(r, w) for r in records]))
    return out
