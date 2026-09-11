"""DQN training for ERQ-Net and baselines, with the mitigation flags used
in the policy-collapse study.

Default config (paper): lr 3e-5, batch 64, buffer 10k, target update every
20 episodes, gamma 0.99, eps 1.0->0.05 (decay 0.995), 1000 episodes.

Mitigation / ablation flags:
  double_dqn        -- Double DQN target (van Hasselt et al., 2016)
  target_update_ep  -- target-network sync period
  speed_w           -- scale of the high-speed reward (reward-reweighting test)
  grad_clip         -- gradient-norm clipping
  use_margin        -- enable the dense margin-reward wrapper
"""
from __future__ import annotations
import random
import time
import collections
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from .env_graph import make_env, obs_to_graph
from .models import (GATPolicy, GAT1Head, GCNPolicy, MLPPolicy,
                     SetTransformerPolicy)

MODELS = {"gat": GATPolicy, "gat1": GAT1Head, "gcn": GCNPolicy,
          "mlp": MLPPolicy, "transformer": SetTransformerPolicy}


def _obs_norm_flat(obs):
    o = obs.copy()
    o[:, 1] /= 1000.0
    o[:, 2] /= 16.0
    o[:, 3:5] /= 35.0
    return o


def encode(obs, kind):
    if kind in ("gat", "gat1", "gcn"):
        return obs_to_graph(obs)
    if kind == "mlp":
        return torch.tensor(_obs_norm_flat(obs).flatten(), dtype=torch.float32)
    return torch.tensor(_obs_norm_flat(obs), dtype=torch.float32)  # transformer


def to_batch(items, kind, dev):
    if kind in ("gat", "gat1", "gcn"):
        return Batch.from_data_list(items).to(dev)
    return torch.stack(items).to(dev)


@torch.no_grad()
def evaluate(model, kind, scenarios, dev, seed=0, keep_attention=False):
    model.eval()
    succ = coll = 0
    lens, rews, att_records = [], [], []
    for si, sc in enumerate(scenarios):
        env, obs = make_env(sc, seed=seed + si)
        done = trunc = False
        ep_r = 0
        steps = 0
        while not (done or trunc):
            enc = encode(obs, kind)
            q = model(to_batch([enc], kind, dev), keep_attention=keep_attention) \
                if kind.startswith("ga") else model(to_batch([enc], kind, dev))
            if keep_attention and getattr(model, "last_attention", None):
                att_records.append(_ego_attention(model.last_attention, sc))
            a = int(q.argmax())
            obs, r, done, trunc, info = env.step(a)
            ep_r += r
            steps += 1
        crashed = env.unwrapped.vehicle.crashed
        succ += (not crashed)
        coll += crashed
        lens.append(steps)
        rews.append(ep_r)
        env.close()
    n = len(scenarios)
    model.train()
    out = {"success": succ / n, "collision": coll / n,
           "avg_len": float(np.mean(lens)), "avg_reward": float(np.mean(rews))}
    if keep_attention:
        out["attention"] = att_records
    return out


def _ego_attention(att, scenario):
    """Mean layer-1 attention the ego (node 0) places on the adversarial
    vehicle vs. background vehicles. Adversary is node 1 in sorted spec
    order (ego, adversarial, background...)."""
    ei, a = att
    a = a.mean(dim=1)
    into_ego = ei[1] == 0
    srcs = ei[0][into_ego].tolist()
    ws = a[into_ego].tolist()
    has_adv = any(v.role == "adversarial" for v in scenario.vehicles)
    rec = {"adv": None, "bg_mean": None}
    if srcs:
        bg_ws = [w for s, w in zip(srcs, ws) if s != 1]
        rec["bg_mean"] = float(np.mean(bg_ws)) if bg_ws else None
        if has_adv and 1 in srcs:
            rec["adv"] = float(ws[srcs.index(1)])
    return rec


def train(model_kind="gat", train_scen=None, test_scen=None, episodes=1000,
          lr=3e-5, batch=64, buf_size=10_000, gamma=0.99,
          target_update_ep=20, eps_start=1.0, eps_end=0.05, eps_decay=0.995,
          double_dqn=False, speed_w=1.0, grad_clip=None, use_margin=False,
          eval_every=50, eval_n=100, seed=0, dev=None, log_path=None,
          ckpt_prefix=None):
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    q = MODELS[model_kind]().to(dev)
    qt = MODELS[model_kind]().to(dev)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = collections.deque(maxlen=buf_size)
    eps = eps_start
    history = []
    t0 = time.time()

    for ep in range(1, episodes + 1):
        sc = random.choice(train_scen)
        env, obs = make_env(sc, seed=seed * 100_000 + ep,
                            sparse_reward_speed=speed_w, use_margin=use_margin)
        done = trunc = False
        while not (done or trunc):
            enc = encode(obs, model_kind)
            if random.random() < eps:
                a = random.randrange(5)
            else:
                with torch.no_grad():
                    a = int(q(to_batch([enc], model_kind, dev)).argmax())
            obs2, r, done, trunc, info = env.step(a)
            buf.append((enc, a, float(r), encode(obs2, model_kind), float(done)))
            obs = obs2
            if len(buf) >= 1000:
                s, a_, r_, s2, d_ = zip(*random.sample(buf, batch))
                bs = to_batch(list(s), model_kind, dev)
                bs2 = to_batch(list(s2), model_kind, dev)
                a_t = torch.tensor(a_, device=dev)
                qv = q(bs).gather(1, a_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    if double_dqn:
                        a_star = q(bs2).argmax(1, keepdim=True)
                        nq = qt(bs2).gather(1, a_star).squeeze(1)
                    else:
                        nq = qt(bs2).max(1).values
                    tgt = torch.tensor(r_, device=dev) + \
                        gamma * (1 - torch.tensor(d_, device=dev)) * nq
                loss = F.smooth_l1_loss(qv, tgt)
                opt.zero_grad()
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(q.parameters(), grad_clip)
                opt.step()
        env.close()
        eps = max(eps_end, eps * eps_decay)
        if ep % target_update_ep == 0:
            qt.load_state_dict(q.state_dict())
        if ep % eval_every == 0 or ep == episodes:
            m = evaluate(q, model_kind, test_scen[:eval_n], dev, seed=9999)
            m.update(episode=ep, eps=round(eps, 3),
                     wall_min=round((time.time() - t0) / 60, 1))
            history.append(m)
            print(f"[{model_kind} seed{seed}] ep {ep:4d} "
                  f"succ {m['success']:.1%} coll {m['collision']:.1%} "
                  f"len {m['avg_len']:.1f} rew {m['avg_reward']:.1f} "
                  f"({m['wall_min']} min)")
            if ckpt_prefix:
                torch.save(q.state_dict(), f"{ckpt_prefix}_s{seed}_ep{ep}.pt")
            if log_path:
                with open(log_path, "w") as f:
                    json.dump(history, f, indent=1)
    return q, history
