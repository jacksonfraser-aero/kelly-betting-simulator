"""RL Day 6a: dense vs sparse reward.

Trains two agents on identical conditions, differing only in when the
reward signal arrives:

- dense (default so far): the per-fill mispricing every round
- sparse: all rewards withheld until settlement, where the agent
  receives the full episode P&L in one step

Both objectives are IDENTICAL in total (the dense rewards sum exactly
to the sparse payout -- see test_dense_and_sparse_agree_on_total_reward).
The only difference is how much per-step feedback the agent gets, which
tests how much of tabular Q-learning's performance comes from dense
shaping versus the underlying task being learnable at all.

Run from the repo root (a few minutes):
    python scripts/ablate_reward_density.py
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.qlearning import train

FIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

N_EPISODES = 20_000


def main() -> None:
    print("Training DENSE-reward agent...")
    agent_d, hist_d = train(
        n_episodes=N_EPISODES, alpha=0.05, eval_every=500, eval_games=200,
        seed=0, verbose=False, dense_reward=True,
    )
    print(f"  final (best-checkpoint) eval: {max(hist_d.eval_mean_pnl):.1f}")

    print("Training SPARSE-reward agent...")
    agent_s, hist_s = train(
        n_episodes=N_EPISODES, alpha=0.05, eval_every=500, eval_games=200,
        seed=0, verbose=False, dense_reward=False,
    )
    print(f"  final (best-checkpoint) eval: {max(hist_s.eval_mean_pnl):.1f}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(hist_d.eval_episodes, hist_d.eval_mean_pnl, lw=2,
           label="dense reward (per-fill)")
    ax.plot(hist_s.eval_episodes, hist_s.eval_mean_pnl, lw=2,
           label="sparse reward (settlement only)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("training episodes")
    ax.set_ylabel("mean P&L over 200 evaluation games")
    ax.set_title("Reward density ablation: identical objective, "
                 "different per-step feedback")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reward_density_ablation.png", dpi=150)
    print(f"\nFigure written to {FIG_DIR / 'reward_density_ablation.png'}")


if __name__ == "__main__":
    main()
