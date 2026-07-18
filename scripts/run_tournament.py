"""Day 4 experiment: strategy tournament across informed-flow regimes.

Four market makers face identical market conditions as the share of
informed traders sweeps from 0% to 60%:

- naive: fixed spread around the prior, never adapts (Day 3 baseline)
- inventory-aware: skews quotes against its position
- bayesian: full posterior over V updated from every buy, sell, AND pass
- bayesian + skew: posterior tracking plus inventory skew

Also produces a convergence demo: the Bayesian maker's fair-value
estimate homing in on the true V from the flow alone.

Run from the repo root:
    python scripts/run_tournament.py
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.market_game import NaiveMarketMaker, draw_true_value, play_game, tournament
from src.rl_maker import RLMarketMaker
from src.strategies import BayesianMarketMaker, InventoryAwareMarketMaker

SEED = 42
CONV_SEED = 34  # example game with V = 3, far from the prior
N_GAMES = 1_500
N_ROUNDS = 100

FIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)

STRATEGIES = [
    ("naive", lambda: NaiveMarketMaker(1.0)),
    ("inventory-aware", lambda: InventoryAwareMarketMaker(1.0, skew=0.25)),
    ("bayesian", lambda: BayesianMarketMaker(1.0)),
    ("bayesian + skew", lambda: BayesianMarketMaker(1.0, inventory_skew=0.1)),
]

_QTABLE_PATH = pathlib.Path(__file__).resolve().parents[1] / "models" / "qtable.npy"
if _QTABLE_PATH.exists():
    _Q = np.load(_QTABLE_PATH)
    STRATEGIES.append(("RL (tabular Q)", lambda: RLMarketMaker(_Q)))


def convergence_demo() -> None:
    """One game: the Bayesian maker's fair value homing in on V."""
    rng = np.random.default_rng(CONV_SEED)
    probe = np.random.default_rng(CONV_SEED)
    v = draw_true_value(probe)

    estimates = []

    class Recorder(BayesianMarketMaker):
        def on_trade(self, trade):
            super().on_trade(trade)
            estimates.append(self.fair_value)

        def on_pass(self, round_idx):
            super().on_pass(round_idx)
            estimates.append(self.fair_value)

    rec = Recorder(1.0)
    estimates.append(rec.fair_value)
    play_game(rec, n_rounds=100, informed_share=0.3, rng=rng)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(estimates, lw=2)
    ax.axhline(v, color="tab:green", ls="--", label=f"true value V = {v}")
    ax.set_xlabel("event number")
    ax.set_ylabel("posterior mean of V")
    ax.set_title("The Bayesian maker learns V from the flow alone")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bayesian_convergence.png", dpi=150)
    plt.close(fig)
    print(f"convergence demo: V = {v}, final estimate = {estimates[-1]:.2f}")


def main() -> None:
    shares = np.linspace(0.0, 0.6, 7)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name, factory in STRATEGIES:
        means = []
        for phi in shares:
            rng = np.random.default_rng(SEED)  # identical conditions per strategy
            pnls = tournament(
                factory, n_games=N_GAMES, n_rounds=N_ROUNDS,
                informed_share=float(phi), rng=rng,
            )
            means.append(pnls.mean())
        ax.plot(shares, means, lw=2, marker="o", label=name)
        print(f"{name:>18s}: " + "  ".join(f"{m:7.1f}" for m in means))

    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("share of informed traders")
    ax.set_ylabel(f"mean final P&L over {N_ROUNDS} rounds ({N_GAMES} games)")
    ax.set_title("Adaptive quoting neutralises adverse selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "strategy_tournament.png", dpi=150)
    plt.close(fig)
    print(f"\nFigure written to {FIG_DIR / 'strategy_tournament.png'}")


if __name__ == "__main__":
    convergence_demo()
    main()
