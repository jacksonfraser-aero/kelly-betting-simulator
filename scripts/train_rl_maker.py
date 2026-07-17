"""RL Day 3: full training run for the Q-learning market maker.

Trains for 20,000 episodes at 30% informed flow, evaluating the frozen
greedy policy every 500 episodes on held-out seeds. Produces:

- figures/rl_learning_curve.png : eval P&L vs training episodes, with
  the hold-still (naive) and Bayesian maker benchmarks as horizontal
  reference lines
- models/qtable.npy, models/qtable_visits.npy : the trained table

Run from the repo root (takes a few minutes):
    python scripts/train_rl_maker.py
    python scripts/train_rl_maker.py --episodes 5000   # quicker look
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.qlearning import QLearningAgent, evaluate, train

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
MODEL_DIR = ROOT / "models"
FIG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Reference points at 30% informed flow (see Days 3-4 README tables).
HOLD_BENCHMARK = 29.0       # never-move policy == naive maker
BAYESIAN_BENCHMARK = 68.2   # Day 4 tournament, knows game structure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--informed", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    agent, hist = train(
        n_episodes=args.episodes,
        alpha=args.alpha,
        eval_every=500,
        eval_games=200,
        seed=args.seed,
        informed_share=args.informed,
    )

    final = evaluate(agent, n_games=1_000, seed_base=10_000,
                     informed_share=args.informed)
    coverage = (agent.visits.sum(axis=-1) > 0).mean()
    print(f"\nfinal eval over 1,000 fresh held-out games: {final:.1f}")
    print(f"state-space coverage: {coverage:.0%} of bins visited")

    np.save(MODEL_DIR / "qtable.npy", agent.q)
    np.save(MODEL_DIR / "qtable_visits.npy", agent.visits)
    print(f"Q-table saved to {MODEL_DIR}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(hist.eval_episodes, hist.eval_mean_pnl, lw=2, marker="o",
            ms=3, label="RL agent (greedy eval, held-out seeds)")
    ax.axhline(HOLD_BENCHMARK, color="tab:red", ls="--",
               label=f"hold-still / naive ({HOLD_BENCHMARK:.0f})")
    ax.axhline(BAYESIAN_BENCHMARK, color="tab:green", ls="--",
               label=f"Bayesian maker ({BAYESIAN_BENCHMARK:.0f})")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("training episodes")
    ax.set_ylabel("mean P&L over 200 evaluation games")
    ax.set_title(
        f"Q-learning market maker, {args.informed:.0%} informed flow "
        f"(alpha = {args.alpha})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rl_learning_curve.png", dpi=150)
    print(f"Learning curve saved to {FIG_DIR / 'rl_learning_curve.png'}")


if __name__ == "__main__":
    main()
