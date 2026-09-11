"""HighwayEnv wrapper with scenario injection and graph construction.

Observation: Kinematics (presence, x, y, vx, vy, cos_h, sin_h), sorted,
absolute, ``see_behind`` enabled so injected vehicles behind the ego are
visible. Graph: k=5 nearest neighbours within 100 m, bidirectional;
6-dim node features, 8-dim edge features (see the paper, Sec. 4.2).
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
import torch
from torch_geometric.data import Data

from .adversary import AdversaryVehicle

K, DMAX, LANE_W = 5, 100.0, 4.0
ROAD_LEN, VMAX, TTC_MAX = 1000.0, 35.0, 10.0
BEHAV_SPEED_FACTOR = {"cautious": 0.9, "normal": 1.0, "aggressive": 1.15}


class MarginRewardWrapper(gym.Wrapper):
    """Dense comfort term r_margin(d_front) from the paper's reward (Eq. 10):
    rewards a safe following distance, penalises tailgating. Gives gradient
    signal before a crash rather than only at the terminal collision."""
    def __init__(self, env, safe_gap=15.0, danger_gap=8.0, w=0.3):
        super().__init__(env)
        self.safe_gap, self.danger_gap, self.w = safe_gap, danger_gap, w

    def step(self, action):
        obs, r, done, trunc, info = self.env.step(action)
        ego = self.env.unwrapped.vehicle
        d_front = None
        for v in self.env.unwrapped.road.vehicles:
            if v is ego:
                continue
            dx = v.position[0] - ego.position[0]
            dy = abs(v.position[1] - ego.position[1])
            if dx > 0 and dy < 2.0 and (d_front is None or dx < d_front):
                d_front = dx
        if d_front is not None:
            if d_front < self.danger_gap:
                r -= self.w * (1.0 - d_front / self.danger_gap)
            elif d_front > self.safe_gap:
                r += 0.1 * self.w
        return obs, r, done, trunc, info


def make_env(scenario, seed=0, sparse_reward_speed=1.0, use_margin=False):
    n = len(scenario.vehicles)
    cfg = {
        "observation": {"type": "Kinematics", "vehicles_count": 15,
                        "features": ["presence", "x", "y", "vx", "vy",
                                     "cos_h", "sin_h"],
                        "absolute": True, "normalize": False,
                        "see_behind": True, "order": "sorted"},
        "action": {"type": "DiscreteMetaAction"},
        "lanes_count": 4,
        "vehicles_count": n - 1,
        "duration": 40,
        "policy_frequency": 1,
        "simulation_frequency": 15,
        "collision_reward": -1.0,
        "high_speed_reward": 0.4 * sparse_reward_speed,
        "right_lane_reward": 0.1,
        "lane_change_reward": 0.0,
        "normalize_reward": False,
    }
    env = gym.make("highway-v0", config=cfg)
    if use_margin:
        env = MarginRewardWrapper(env)
    obs, info = env.reset(seed=seed)
    _inject(env.unwrapped, scenario)
    obs = env.unwrapped.observation_type.observe()
    return env, obs


def _inject(uenv, scenario):
    """Override default vehicle init with the scenario spec (deterministic
    starts), swapping in the scripted adversary for the interaction actor."""
    road = uenv.road
    ego = uenv.vehicle
    others = [v for v in road.vehicles if v is not ego]
    spec = scenario.vehicles
    ego_spec = next(v for v in spec if v.role == "ego")
    lane = road.network.get_lane(("0", "1", ego_spec.lane))
    ego.position = lane.position(ego_spec.position, 0)
    ego.speed = ego_spec.speed
    ego.heading = lane.heading_at(ego_spec.position)

    rest = [v for v in spec if v.role != "ego"]
    inter = {i["source"]: i for i in getattr(scenario, "interactions", [])}
    for veh, s in zip(list(others), rest):
        ln = road.network.get_lane(("0", "1", s.lane))
        if s.role == "adversarial" and s.id in inter:
            adv = AdversaryVehicle(road, ln.position(s.position, 0),
                                   heading=ln.heading_at(s.position), speed=s.speed)
            spec_i = inter[s.id]
            adv.setup_attack(ego, spec_i["type"], spec_i["trigger_distance"],
                             spec_i.get("severity", "medium"))
            road.vehicles.remove(veh)
            road.vehicles.append(adv)
            continue
        veh.position = ln.position(s.position, 0)
        veh.heading = ln.heading_at(s.position)
        veh.speed = s.speed
        if hasattr(veh, "target_speed"):
            veh.target_speed = s.speed * BEHAV_SPEED_FACTOR.get(s.behavior, 1.0)
    for veh in others[len(rest):]:
        road.vehicles.remove(veh)


def obs_to_graph(obs) -> Data:
    """obs: 15x7 [presence, x, y, vx, vy, cos_h, sin_h], ego row 0, absolute."""
    m = obs[obs[:, 0] > 0.5]
    n = m.shape[0]
    x = np.zeros((n, 6), dtype=np.float32)
    x[:, 0] = m[:, 1] / ROAD_LEN
    x[:, 1] = m[:, 2] / (4 * LANE_W)
    x[:, 2:4] = m[:, 3:5] / VMAX
    x[:, 4:6] = m[:, 5:7]
    pos = m[:, 1:3]
    vel = m[:, 3:5]
    src, dst, ea = [], [], []
    for i in range(n):
        d = np.linalg.norm(pos - pos[i], axis=1)
        d[i] = np.inf
        order = np.argsort(d)
        nbrs = [j for j in order[:K] if d[j] <= DMAX]
        if not nbrs:
            nbrs = [i]
        for j in nbrs:
            dx, dy = pos[j] - pos[i]
            dist = d[j] if j != i else 0.0
            dvx, dvy = vel[j] - vel[i]
            rel_speed = np.hypot(dvx, dvy)
            closing = (dx * dvx + dy * dvy) < 0
            ttc = (dist - 5.0) / rel_speed if (closing and rel_speed > 1e-3) else TTC_MAX
            ttc = float(np.clip(ttc, 0, TTC_MAX))
            dlane = round(dy / LANE_W)
            ea.append([dx / 100, dy / 100, dist / 100,
                       np.arctan2(dy, dx) / np.pi,
                       dvx / 20, dvy / 20, ttc / TTC_MAX, dlane / 3])
            src.append(j)
            dst.append(i)
    return Data(x=torch.tensor(x),
                edge_index=torch.tensor([src, dst], dtype=torch.long),
                edge_attr=torch.tensor(ea, dtype=torch.float32))
