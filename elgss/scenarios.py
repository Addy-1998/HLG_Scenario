"""Scenario generation: LLM-conditioned vs. random-control.

Two generators share identical marginal parameter ranges so the only
difference is the *structure* the language specification imposes (an ego
role, an adversarial role, and a coherent interaction with plausible
geometry). This isolates the effect of language-structured training used
in the intent-conditioning ablation.

- ``gen_llm_conditioned``: an ego plus one adversarial actor with a
  scripted interaction (cut_in / brake / overtake / follow) placed where
  the interaction is geometrically possible, plus background traffic.
  In the paper these JSON specs come from Gemini; the generator here
  draws from the same schema so the pipeline is reproducible offline.
- ``gen_random_control``: same schema, same ranges, but positions,
  lanes, speeds, and behaviours sampled independently with NO interaction
  specified. This is the ablation control distribution.
"""
from __future__ import annotations
import json
import random
from dataclasses import dataclass, field, asdict

SPEED_RANGE = (20.0, 35.0)
POS_RANGE = (0.0, 1000.0)
LANES = 4
MIN_GAP = 10.0
INTERACTIONS = ["cut_in", "brake", "overtake", "follow"]
DENSITIES = {"sparse": (3, 4), "moderate": (5, 6), "dense": (7, 10)}


@dataclass
class Vehicle:
    id: str
    lane: int
    position: float
    speed: float
    behavior: str = "normal"   # cautious | normal | aggressive
    role: str = "background"   # ego | adversarial | background
    type: str = "car"


@dataclass
class Scenario:
    vehicles: list = field(default_factory=list)
    interactions: list = field(default_factory=list)
    density: str = "moderate"
    source: str = "llm_conditioned"  # or random_control

    def to_json(self):
        return json.dumps({"vehicles": [asdict(v) for v in self.vehicles],
                           "interactions": self.interactions,
                           "density": self.density, "source": self.source})


def _spaced_positions(n, rng, lo=350.0, hi=650.0):
    """Positions with a minimum gap, centred so interactions happen early."""
    while True:
        ps = sorted(rng.uniform(lo, hi) for _ in range(n))
        if all(b - a >= MIN_GAP for a, b in zip(ps, ps[1:])):
            return ps


def gen_llm_conditioned(density: str, rng: random.Random) -> Scenario:
    n = rng.randint(*DENSITIES[density])
    ego_lane = rng.randint(0, LANES - 1)
    ego_speed = rng.uniform(*SPEED_RANGE)
    ego = Vehicle("ego", ego_lane, 500.0, ego_speed, "normal", "ego")

    itype = rng.choice(INTERACTIONS)
    if itype == "cut_in":
        adv_lane = max(0, min(LANES - 1, ego_lane + rng.choice([-1, 1])))
        adv_pos = 500.0 + rng.uniform(8, 25)
        adv_speed = min(SPEED_RANGE[1], ego_speed + rng.uniform(2, 8))
    elif itype == "brake":
        adv_lane, adv_pos = ego_lane, 500.0 + rng.uniform(20, 45)
        adv_speed = ego_speed + rng.uniform(-2, 4)
    elif itype == "overtake":
        adv_lane = max(0, min(LANES - 1, ego_lane + rng.choice([-1, 1])))
        adv_pos = 500.0 - rng.uniform(15, 40)
        adv_speed = min(SPEED_RANGE[1], ego_speed + rng.uniform(4, 10))
    else:  # follow
        adv_lane, adv_pos = ego_lane, 500.0 - rng.uniform(10, 25)
        adv_speed = min(SPEED_RANGE[1], ego_speed + rng.uniform(0, 5))
    adv = Vehicle("adv_1", adv_lane, adv_pos, adv_speed, "aggressive", "adversarial")

    bg_positions = _spaced_positions(n - 2, rng)
    bg = [Vehicle(f"bg_{k}", rng.randint(0, LANES - 1), p,
                  rng.uniform(*SPEED_RANGE),
                  rng.choice(["cautious", "normal"]), "background")
          for k, p in enumerate(bg_positions)]

    inter = [{"source": "adv_1", "target": "ego", "type": itype,
              "trigger_distance": rng.uniform(10, 20),
              "severity": rng.choice(["medium", "high"])}]
    return Scenario([ego, adv] + bg, inter, density, "llm_conditioned")


def gen_random_control(density: str, rng: random.Random) -> Scenario:
    """Same marginals as the LLM generator, no structure: ablation control."""
    n = rng.randint(*DENSITIES[density])
    ps = _spaced_positions(n, rng)
    vs = [Vehicle(f"v_{k}", rng.randint(0, LANES - 1), p,
                  rng.uniform(*SPEED_RANGE),
                  rng.choice(["cautious", "normal", "aggressive"]),
                  "ego" if k == 0 else "background")
          for k, p in enumerate(ps)]
    vs[0].position = 500.0
    return Scenario(vs, [], density, "random_control")


def make_dataset(n_total=2500, source="llm_conditioned", seed=0,
                 balanced_test=True):
    """80/20 split. The test set is balanced across densities so that
    reported success is not dominated by easy sparse scenarios."""
    rng = random.Random(seed)
    gen = gen_llm_conditioned if source == "llm_conditioned" else gen_random_control
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    train = [gen(rng.choices(list(DENSITIES), weights=[0.36, 0.44, 0.20])[0], rng)
             for _ in range(n_train)]
    if balanced_test:
        per = n_test // 3
        test = [gen(d, rng) for d in DENSITIES for _ in range(per)]
        test += [gen("dense", rng) for _ in range(n_test - len(test))]
    else:
        test = [gen(rng.choice(list(DENSITIES)), rng) for _ in range(n_test)]
    return train, test
