"""Run the paper's experiments from the command line.

Examples:
    # Exp 1: main ERQ-Net, 3 seeds
    python scripts/run_experiments.py --exp main

    # Exp 2: intent-conditioning ablation (random-control training)
    python scripts/run_experiments.py --exp ablation

    # Exp 3: architecture baselines
    python scripts/run_experiments.py --exp baselines

    # Exp 4: reward-shaping mitigations
    python scripts/run_experiments.py --exp mitigations
"""
import argparse
import os

from elgss.scenarios import make_dataset
from elgss.train import train

os.makedirs("results", exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True,
                    choices=["main", "ablation", "baselines", "mitigations"])
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    train_llm, test_llm = make_dataset(2500, "llm_conditioned", seed=42)
    train_rand, _ = make_dataset(2500, "random_control", seed=42)

    if args.exp == "main":
        for s in args.seeds:
            train("gat", train_llm, test_llm, episodes=args.episodes, seed=s,
                  ckpt_prefix="ckpt_main",
                  log_path=f"results/main_gat_s{s}.json")

    elif args.exp == "ablation":
        for s in args.seeds:
            train("gat", train_rand, test_llm, episodes=args.episodes, seed=s,
                  ckpt_prefix="ckpt_rand",
                  log_path=f"results/ablation_rand_s{s}.json")

    elif args.exp == "baselines":
        for kind in ("mlp", "gcn", "gat1", "transformer"):
            train(kind, train_llm, test_llm, episodes=args.episodes, seed=0,
                  log_path=f"results/baseline_{kind}.json")

    elif args.exp == "mitigations":
        configs = {"double": dict(double_dqn=True),
                   "slowtgt": dict(target_update_ep=50),
                   "speed05": dict(speed_w=0.5),
                   "margin": dict(use_margin=True),
                   "gradclip": dict(grad_clip=10.0)}
        for name, kw in configs.items():
            train("gat", train_llm, test_llm, episodes=args.episodes, seed=0,
                  log_path=f"results/mitigation_{name}.json", **kw)


if __name__ == "__main__":
    main()
