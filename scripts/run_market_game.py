"""Day 3 experiment: a naive market maker meets informed order flow.

The maker quotes a fixed spread around the prior mean (10.5) of a hidden
value V (sum of three dice) and never updates. We look at:

(1) A single example game: inventory and mark-to-settlement P&L round by
    round. The chosen seed draws V = 3, far below the prior, so informed
    traders repeatedly sell to the maker at a bid that is far too high.
(2) Final P&L distributions across 2,000 games as the share of informed
    traders rises from 0% to 60% -- spread income turning into
    adverse-selection losses.

Run from the repo root:
    python scripts/run_market_game.py
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.market_game import NaiveMarketMaker, play_game, tournament

EXAMPLE_SEED = 34
SWEEP_SEED = 22
N_ROUNDS = 100
N_GAMES = 2_000
HALF_SPREAD = 1.0

FIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)


def example_game() -> None:
    """One game with 30% informed flow: inventory and P&L paths."""
    rng = np.random.default_rng(EXAMPLE_SEED)
    result = play_game(
        NaiveMarketMaker(HALF_SPREAD), n_rounds=N_ROUNDS,
        informed_share=0.3, rng=rng,
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(result.inventory_path, lw=2)
    ax1.axhline(0, color="grey", lw=0.8)
    ax1.set_ylabel("inventory")
    ax1.set_title(
        f"Example game: true value V = {result.true_value} "
        f"(prior mean 10.5), 30% informed flow"
    )
    ax2.plot(result.pnl_path, lw=2, color="tab:red")
    ax2.axhline(0, color="grey", lw=0.8)
    ax2.set_ylabel("mark-to-settlement P&L")
    ax2.set_xlabel(f"round (of {N_ROUNDS})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "example_game.png", dpi=150)
    plt.close(fig)

    print(
        f"example game: V = {result.true_value}, trades = {result.n_trades} "
        f"({result.n_informed_trades} informed), final P&L = {result.final_pnl:.1f}"
    )


def informed_share_sweep() -> None:
    """Final P&L distribution vs share of informed traders."""
    rng = np.random.default_rng(SWEEP_SEED)
    shares = [0.0, 0.15, 0.3, 0.45, 0.6]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    print(f"\n{'informed share':>15s} {'mean P&L':>10s} {'median':>10s} {'P(loss)':>9s}")

    all_pnls = []
    for phi in shares:
        pnls = tournament(
            lambda: NaiveMarketMaker(HALF_SPREAD),
            n_games=N_GAMES, n_rounds=N_ROUNDS,
            informed_share=phi, rng=rng,
        )
        all_pnls.append(pnls)
        print(
            f"{phi:15.2f} {pnls.mean():10.1f} {np.median(pnls):10.1f} "
            f"{(pnls < 0).mean():9.3f}"
        )

    ax.boxplot(
        all_pnls, tick_labels=[f"{s:.0%}" for s in shares],
        showfliers=False,
    )
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("share of informed traders")
    ax.set_ylabel(f"final P&L over {N_ROUNDS} rounds")
    ax.set_title("A fixed-spread maker bleeds as flow becomes informed")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "informed_share_sweep.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    example_game()
    informed_share_sweep()
    print(f"\nFigures written to {FIG_DIR}")
