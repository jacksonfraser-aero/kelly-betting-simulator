"""RL Day 5: what did the agent actually learn?

Loads the trained Q-table and produces:

1. figures/policy_heatmap.png -- the greedy centre-move over the
   (flow imbalance x centre offset) grid, next to the *empirically
   measured* behaviour of the Bayesian maker on the same grid (its
   mean per-round centre change, snapped to the nearest discrete
   action).

2. figures/policy_visits.png -- how much training data backs each
   cell (log scale). Edge cells with few visits are guesswork.

3. figures/policy_slices.png -- the learned policy at early / mid /
   late game.

Two methodological notes, both discoveries of this inspection:

- In this game the maker is the counterparty to every fill, so flow
  imbalance and inventory are DETERMINISTICALLY linked
  (inventory = -imbalance): the observation has three effective
  degrees of freedom, not four. Grids are therefore sliced along
  that diagonal, pairing each imbalance bin with the inventory bin
  it actually co-occurs with.

- Q-values are dominated by STATE value; in most core cells the gap
  between the best and second-best action is within estimation
  noise. Cells without a decisive preference are masked rather than
  painted with argmax-over-noise.

Run from the repo root:
    python scripts/inspect_policy.py
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.market_game import PRIOR_MEAN, Trade, draw_true_value
from src.rl_env import (
    CENTRE_EDGES,
    CENTRE_MOVES,
    IMBALANCE_EDGES,
    INVENTORY_EDGES,
)
from src.strategies import BayesianMarketMaker

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
Q = np.load(ROOT / "models" / "qtable.npy")
VISITS = np.load(ROOT / "models" / "qtable_visits.npy")

IMB_LABELS = ["<-4", "-4:-3", "-2", "-1:0", "+1", "+2:+3", ">=+4"]
CEN_LABELS = ["<-4", "-4:-3", "-2", "-1:0", "+1", "+2:+3", ">=+4"]
MID_TIME = 1

# Imbalance/inventory are deterministically linked (see module note):
# pair each imbalance bin with the inventory bin it co-occurs with.
IMB_REPRESENTATIVE = np.array([-5, -3, -2, 0, 1, 2.5, 5])
LINKED_INV_BIN = [int(np.digitize(-r, INVENTORY_EDGES)) for r in IMB_REPRESENTATIVE]

# A cell only expresses a real preference if its best action beats the
# runner-up by more than estimation noise. With per-step reward noise
# of order the half-spread and ~10^3 visits per action, gaps below
# ~1.0 are indistinguishable from noise.
MIN_GAP = 1.0
MIN_VISITS = 300


def greedy_move_grid(time_bin: int) -> np.ndarray:
    """Greedy centre-move over imbalance x centre bins, on the
    imbalance/inventory diagonal. Cells are masked (NaN) when
    untrained OR when the action gap is within estimation noise."""
    grid = np.full((len(IMB_LABELS), len(CEN_LABELS)), np.nan)
    for i in range(len(IMB_LABELS)):
        inv_bin = LINKED_INV_BIN[i]
        for c in range(len(CEN_LABELS)):
            q = Q[i, c, inv_bin, time_bin]
            v = VISITS[i, c, inv_bin, time_bin]
            if v.sum() < MIN_VISITS:
                continue
            top2 = np.sort(q)[-2:]
            if top2[1] - top2[0] < MIN_GAP:
                continue
            grid[i, c] = CENTRE_MOVES[int(np.argmax(q))]
    return grid


def bayesian_move_grid(
    n_games: int = 3_000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical mean centre-change of the Bayesian maker per cell."""
    rng = np.random.default_rng(seed)
    sums = np.zeros((len(IMB_LABELS), len(CEN_LABELS)))
    counts = np.zeros_like(sums)

    for _ in range(n_games):
        v = draw_true_value(rng)
        maker = BayesianMarketMaker(1.0)
        imbalance, prev_centre = 0, PRIOR_MEAN
        for t in range(100):
            bid, ask = maker.quote(t)
            centre = 0.5 * (bid + ask)
            i = int(np.digitize(imbalance, IMBALANCE_EDGES))
            c = int(np.digitize(prev_centre - PRIOR_MEAN, CENTRE_EDGES))
            sums[i, c] += centre - prev_centre
            counts[i, c] += 1
            prev_centre = centre

            informed = bool(rng.random() < 0.3)
            if informed:
                signal = v + rng.normal(0.0, 1.0)
                side = 1 if signal > ask else (-1 if signal < bid else 0)
            else:
                side = 1 if rng.random() < 0.5 else -1
            if side != 0:
                maker.on_trade(Trade(t, side, ask if side == 1 else bid, informed))
            else:
                maker.on_pass(t)
            imbalance += side

    mean_move = np.where(counts > 50, sums / np.maximum(counts, 1), np.nan)
    return mean_move, counts


def snap_to_actions(grid: np.ndarray) -> np.ndarray:
    """Round continuous moves to the nearest discrete action value."""
    out = np.full_like(grid, np.nan)
    ok = ~np.isnan(grid)
    idx = np.abs(grid[ok][:, None] - CENTRE_MOVES[None, :]).argmin(axis=1)
    out[ok] = CENTRE_MOVES[idx]
    return out


def _draw_grid(ax, grid, title):
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-1.0, vmax=1.0, origin="lower")
    ax.set_xticks(range(len(CEN_LABELS)), CEN_LABELS, rotation=45)
    ax.set_yticks(range(len(IMB_LABELS)), IMB_LABELS)
    ax.set_xlabel("quote centre offset from prior")
    ax.set_ylabel("flow imbalance (buys - sells)")
    ax.set_title(title)
    for i in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if not np.isnan(grid[i, c]):
                ax.text(c, i, f"{grid[i, c]:+.1f}", ha="center", va="center",
                        fontsize=8)
    return im


def figure_policy_vs_bayesian() -> None:
    rl_grid = greedy_move_grid(MID_TIME)
    bayes_grid, _ = bayesian_move_grid()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    _draw_grid(ax1, rl_grid,
               "RL agent: decisive greedy moves\n(masked where indifferent)")
    im = _draw_grid(ax2, snap_to_actions(bayes_grid),
                    "Bayesian maker: mean centre move\n(empirical, snapped to actions)")
    fig.colorbar(im, ax=[ax1, ax2], label="centre move", shrink=0.8)
    fig.savefig(FIG_DIR / "policy_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Directional correlation over cells where the RL policy expresses
    # a real preference and the Bayesian estimate is well-sampled. (A
    # raw sign-match is misleading: Bayesian per-round moves are small
    # and continuous, RL moves are coarse and discrete.)
    mask = ~np.isnan(rl_grid) & ~np.isnan(bayes_grid)
    x, y = rl_grid[mask], bayes_grid[mask]
    if x.size > 2 and x.std() > 0 and y.std() > 0:
        corr = float(np.corrcoef(x, y)[0, 1])
        print(f"directional correlation (RL move vs Bayesian mean move, "
              f"{x.size} significant cells): {corr:+.2f}")


def figure_visits() -> None:
    v = np.array([[VISITS[i, c, LINKED_INV_BIN[i], MID_TIME].sum()
                   for c in range(len(CEN_LABELS))]
                  for i in range(len(IMB_LABELS))])
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(np.log10(v + 1), cmap="viridis", origin="lower")
    ax.set_xticks(range(len(CEN_LABELS)), CEN_LABELS, rotation=45)
    ax.set_yticks(range(len(IMB_LABELS)), IMB_LABELS)
    ax.set_xlabel("quote centre offset from prior")
    ax.set_ylabel("flow imbalance")
    ax.set_title("Training data per cell (log10 visits)\n"
                 "imbalance/inventory diagonal, mid-game")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "policy_visits.png", dpi=150)
    plt.close(fig)
    print(f"visited cells in reference slice: {(v > 0).sum()} of {v.size}; "
          f"median visits where >0: {np.median(v[v > 0]):.0f}")


def figure_slices() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    for ax, (tim, title) in zip(
        axes.flat, [(0, "early game"), (1, "mid game"), (2, "late game")]
    ):
        im = _draw_grid(ax, greedy_move_grid(tim), title)
    fig.colorbar(im, ax=list(axes.flat), label="centre move", shrink=0.8)
    fig.savefig(FIG_DIR / "policy_slices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figure_policy_vs_bayesian()
    figure_visits()
    figure_slices()
    print(f"Figures written to {FIG_DIR}")
