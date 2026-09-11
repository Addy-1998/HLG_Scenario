# ERQ-Net: Language-Structured Relational Q-Learning

Reference implementation for the ECCV 2026 workshop paper
*"Language-Structured Relational Q-Learning for Threat-Aware Control in
Safety-Critical Driving."*

The code trains an Ego-Centric Relational Q-Network (a graph-attention
DQN) on language-structured traffic scenarios in
[HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv), and
reproduces the paper's two central findings:

1. **Emergent threat attention.** Training on LLM-structured scenarios
   (vs. a statistically matched random-control distribution) increases
   both success and the ego's attention on the adversarial vehicle,
   even though semantic role labels are never shown to the policy.
2. **Policy collapse / recognition–control gap.** All seeds converge
   within tens of episodes to a distribution-level attractor matching
   the best constant action, leaving measurable headroom (quantified by
   a 12-policy portfolio union) unexploited.

## Install

This project uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management. If you don't have uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone and sync uv. This creates the virtual environment and installs
everything (including the `elgss` package itself) from `pyproject.toml`:

```bash
git clone <this-repo>
cd elgss
uv sync
```

Run any command inside the managed environment with `uv run`, e.g.
`uv run python scripts/run_experiments.py --exp main`. Requires
Python ≥ 3.9; uv will fetch a suitable interpreter automatically if
needed. The main dependencies are PyTorch, PyTorch Geometric, Gymnasium,
and `highway-env==1.12.0`.

<details>
<summary>Prefer pip?</summary>

```bash
python -m venv venv && source venv/bin/activate
pip install -e .
```
The `requirements.txt` file is also provided for pip-based workflows.
</details>

## Repository layout

```
elgss/
  scenarios.py    scenario generation: LLM-conditioned + random-control
  adversary.py    scripted adversary that enacts the interaction spec
  env_graph.py    HighwayEnv injection, k-NN graph, margin-reward wrapper
  models.py       ERQ-Net (GAT), GCN, single-head GAT, MLP, transformer
  train.py        DQN training + periodic evaluation + mitigation flags
  metrics.py      realism / criticality / fidelity (paper Eq. 8)
scripts/
  run_experiments.py     main / ablation / baselines / mitigations
  baselines_portfolio.py constant-action, untrained, portfolio-union
notebooks/
  run_experiments.ipynb  Colab driver for the full experiment suite
```

## Reproducing the experiments

Run the whole suite in sequence:

```bash
./reproduce.sh              # full run: 1000 episodes, 3 seeds
EPISODES=300 ./reproduce.sh # quick smoke run
```

Or run individual pieces:

| Paper result | Command |
|---|---|
| Training dynamics (3 seeds) | `uv run python scripts/run_experiments.py --exp main` |
| Intent-conditioning ablation | `uv run python scripts/run_experiments.py --exp ablation` |
| Architecture baselines | `uv run python scripts/run_experiments.py --exp baselines` |
| Reward-shaping mitigations | `uv run python scripts/run_experiments.py --exp mitigations` |
| Recognition–control gap | `uv run python scripts/baselines_portfolio.py --n 100` |

Each training run writes a JSON log to `results/` and checkpoints to the
working directory. On a single GPU (or a modern CPU — the graphs are
small) a 1,000-episode run takes roughly half an hour.

### Expected result bands

Because scenarios are sampled at run time, exact percentages vary across
runs; the findings are in the *bands* and *relationships*, not single
numbers. As a guide, the main ERQ-Net settles around **55–58%** success,
random-control training reaches a **similar or slightly different** band
(the paper analyses this comparison), untrained networks score
**45–49%**, the best constant action about **57%**, and the 12-policy
portfolio union about **76%** — the gap between that union and any single
policy is the recognition–control gap the paper reports. Treat these as
target ranges for a sanity check, not guarantees.

### Attention analysis

The threat-attention ratio (paper Eq. 5) is produced by evaluating a
trained checkpoint with `evaluate(..., keep_attention=True)` in
`elgss/train.py`, which returns per-scenario attention on the adversarial
vehicle relative to background vehicles.

## Notes on scope

- Scenarios are generated from a fixed JSON schema. In the paper the
  specs are produced by the Gemini API; the generator here samples from
  the same schema so results are reproducible without an API key. Any
  model emitting schema-valid JSON can be substituted.
- The GAT layers use the standard additive-attention `GATConv`. Swapping
  to `GATv2Conv` (Brody et al., ICLR 2022) is a one-line change in
  `models.py` if you wish to test dynamic attention.
- CARLA transfer in the paper is a qualitative validation; this
  repository covers the HighwayEnv training and analysis.

## Citation

```bibtex
@inproceedings{humnabadkar2026erqnet,
  title     = {Language-Structured Relational Q-Learning for Threat-Aware
               Control in Safety-Critical Driving},
  author    = {Humnabadkar, Aditya and Zhang, Huaizhong and Behera, Ardhendu},
  booktitle = {ECCV Workshops},
  year      = {2026}
}
```
