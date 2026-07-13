"""Day 2 experiments: Kelly sizing when the edge is estimated, not known.

Experiment A (noise): the bettor estimates p from n_history past
outcomes, then bets multiplier * Kelly(p_hat). We sweep the multiplier
for several history lengths and plot realised growth. With a perfect
estimate the optimum is multiplier = 1 (full Kelly); with noisy
estimates the optimal multiplier drops below 1.

Experiment B (bias): the bettor's estimate is systematically optimistic
(+3 percentage points, e.g. believes 55% when truth is 52%). Full Kelly
on the believed edge then destroys wealth, while fractional Kelly
remains robust.

Run from the repo root:
    python scripts/run_estimation_study.py
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.estimation import growth_vs_multiplier
from src.kelly import expected_log_growth, kelly_fraction

TRUE_P = 0.55
B = 1.0
N_BETS = 1_000
N_PATHS = 10_000
SEED = 11

FIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)


def experiment_noise() -> None:
    """Realised growth vs Kelly multiplier for varying estimate quality."""
    rng = np.random.default_rng(SEED)
    multipliers = np.linspace(0.0, 2.0, 41)
    histories = [50, 200, 1_000]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Perfect-information benchmark: g(m * f_star) computed analytically.
    f_star = kelly_fraction(TRUE_P, B)
    perfect = expected_log_growth(
        np.clip(multipliers * f_star, 0.0, 0.99), TRUE_P, B
    )
    ax.plot(multipliers, perfect, "k--", lw=2, label="known edge (theory)")

    print(f"{'estimate':>18s} {'best multiplier':>16s} {'growth at best':>15s}")
    best_m = multipliers[np.argmax(perfect)]
    print(f"{'known edge':>18s} {best_m:16.2f} {perfect.max():15.5f}")

    for n_hist in histories:
        means, _ = growth_vs_multiplier(
            multipliers, TRUE_P, B, n_hist,
            n_bets=N_BETS, n_paths=N_PATHS, rng=rng,
        )
        ax.plot(multipliers, means, lw=2, label=f"estimated from {n_hist} bets")
        best_m = multipliers[np.argmax(means)]
        print(f"{f'{n_hist} bets':>18s} {best_m:16.2f} {means.max():15.5f}")

    ax.axvline(1.0, color="grey", lw=0.8)
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_ylim(-0.01, 0.007)
    ax.set_xlabel("Kelly multiplier m  (bet = m x Kelly on estimated edge)")
    ax.set_ylabel("mean realised log-growth per bet")
    ax.set_title("Noisy edge estimates shift the optimal sizing below full Kelly")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "noisy_edge_multiplier.png", dpi=150)
    plt.close(fig)


def experiment_bias() -> None:
    """Optimistic bias: believes edge is ~3 points better than reality."""
    rng = np.random.default_rng(SEED + 1)
    multipliers = np.linspace(0.0, 2.0, 41)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    print(f"\n{'scenario':>28s} {'best multiplier':>16s} {'growth at m=1':>14s}")
    scenarios = [
        (0.55, 0.00, "unbiased, true p = 0.55"),
        (0.52, 0.03, "believes 0.55, true p = 0.52"),
    ]
    for true_p, bias, label in scenarios:
        means, _ = growth_vs_multiplier(
            multipliers, true_p, B, 200,
            n_bets=N_BETS, n_paths=N_PATHS, bias=bias, rng=rng,
        )
        ax.plot(multipliers, means, lw=2, label=label)
        at_full = means[np.argmin(np.abs(multipliers - 1.0))]
        print(f"{label:>28s} {multipliers[np.argmax(means)]:16.2f} {at_full:14.5f}")

    ax.axvline(1.0, color="grey", lw=0.8)
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("Kelly multiplier m  (sized on the BELIEVED edge)")
    ax.set_ylabel("mean realised log-growth per bet")
    ax.set_title("An optimistic edge estimate makes full Kelly destructive")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "biased_edge_multiplier.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    experiment_noise()
    experiment_bias()
    print(f"\nFigures written to {FIG_DIR}")
