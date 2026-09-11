#!/usr/bin/env bash
# Reproduce all experiments from the ECCV 2026 workshop paper in sequence.
# Results are written to results/ (JSON logs) and checkpoints to the CWD.
#
# Usage:
#   ./reproduce.sh                # full run: 1000 episodes, 3 seeds
#   EPISODES=300 ./reproduce.sh   # quick smoke run
#
# Runs inside the uv-managed environment (see README). Override the
# episode budget with the EPISODES env var (default 1000).

set -euo pipefail

EPISODES="${EPISODES:-1000}"
RUN="uv run python scripts/run_experiments.py --episodes ${EPISODES}"

mkdir -p results

echo "=============================================="
echo " ERQ-Net full reproduction  (episodes=${EPISODES})"
echo "=============================================="

echo
echo ">>> [1/5] Main ERQ-Net training (3 seeds)"
${RUN} --exp main

echo
echo ">>> [2/5] Intent-conditioning ablation (random-control training)"
${RUN} --exp ablation

echo
echo ">>> [3/5] Architecture baselines (MLP / GCN / single-head GAT / transformer)"
${RUN} --exp baselines

echo
echo ">>> [4/5] Reward-shaping mitigations"
${RUN} --exp mitigations

echo
echo ">>> [5/5] Recognition-control gap (constant / untrained / portfolio union)"
uv run python scripts/baselines_portfolio.py --n 100

echo
echo "=============================================="
echo " Done. Logs in results/ ; checkpoints as ckpt_*.pt"
echo "=============================================="
