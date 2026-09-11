"""Recognition-control diagnostics: constant-action policies, untrained
networks, and the 12-policy portfolio union (paper Table 2 / Eq. 6-7).

Usage:
    python scripts/baselines_portfolio.py --n 100
"""
import argparse
import random
import numpy as np
import torch

from elgss.scenarios import make_dataset
from elgss.env_graph import make_env
from elgss.models import GATPolicy
from elgss.train import encode, to_batch


def rollout(policy, sc, seed):
    env, obs = make_env(sc, seed=seed)
    done = trunc = False
    while not (done or trunc):
        obs, r, done, trunc, info = env.step(policy(obs))
    ok = not env.unwrapped.vehicle.crashed
    env.close()
    return ok


def build_policy(pid):
    """0-4: constant actions; 5-7: random seeds; 8-11: untrained GATs."""
    if pid < 5:
        return lambda obs, a=pid: a
    if pid < 8:
        rng = random.Random(pid)
        return lambda obs: rng.randrange(5)
    torch.manual_seed(pid)
    q = GATPolicy()
    q.eval()

    def f(obs):
        with torch.no_grad():
            return int(q(to_batch([encode(obs, "gat")], "gat", "cpu")).argmax())
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _, test = make_dataset(2500, "llm_conditioned", seed=args.seed)
    test = test[:args.n]

    solved = np.zeros(args.n, bool)
    per_policy = []
    for pid in range(12):
        pol = build_policy(pid)
        wins = np.array([rollout(pol, test[i], 9999 + i) for i in range(args.n)])
        per_policy.append(wins.mean())
        solved |= wins
        label = (["LEFT", "IDLE", "RIGHT", "FASTER", "SLOWER"][pid] if pid < 5
                 else f"random-{pid}" if pid < 8 else f"untrained-{pid}")
        print(f"policy {pid:2d} ({label:12s}): {wins.mean():.0%}")
    print(f"\nbest single policy: {max(per_policy):.0%}")
    print(f"portfolio union:    {solved.mean():.0%}")
    print(f"recognition-control gap: "
          f"{solved.mean() - max(per_policy):.0%}")


if __name__ == "__main__":
    main()
