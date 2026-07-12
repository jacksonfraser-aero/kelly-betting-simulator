"""Day 1 experiment: compare betting fractions on a fixed favourable bet.

Scenario: even-money bet (b = 1) with win probability p = 0.55.
Kelly says f* = 2p - 1 = 0.10. We compare full Kelly, half Kelly,
double Kelly (over-betting), and a small fixed fraction, and plot
(1) the expected log-growth curve g(f) and (2) simulated bankroll paths.

Run from the repo root:
    python scripts/run_bankroll_sim.py
Outputs figures to ./figures/.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.kelly import expected_log_growth, kelly_fraction, simulate_bankroll_paths

P = 0.55          # true win probability
B = 1.0           # even-money odds
N_BETS = 1_000
N_PATHS = 20_000
SEED = 7

FIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)


def plot_growth_curve() -> None:
    f_star = kelly_fraction(P, B)
    f_grid = np.linspace(0.0, 0.35, 500)
    g = expected_log_growth(f_grid, P, B)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(f_grid, g, lw=2)
    ax.axvline(f_star, color="tab:green", ls="--", label=f"Kelly f* = {f_star:.2f}")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("bet fraction f")
    ax.set_ylabel("expected log-growth per bet g(f)")
    ax.set_title(f"Log-growth vs bet fraction (p = {P}, even odds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "growth_curve.png", dpi=150)
    plt.close(fig)


def plot_bankroll_paths() -> None:
    f_star = kelly_fraction(P, B)
    strategies = {
        "half Kelly (f = 0.05)": 0.5 * f_star,
        "full Kelly (f = 0.10)": f_star,
        "double Kelly (f = 0.20)": 2.0 * f_star,
        "fixed 2% (f = 0.02)": 0.02,
    }

    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    summary = []
    for label, f in strategies.items():
        paths = simulate_bankroll_paths(
            f, P, B, n_bets=N_BETS, n_paths=N_PATHS, rng=rng
        )
        final = paths[:, -1]
        median_path = np.median(paths, axis=0)
        ax.plot(median_path, lw=2, label=label)
        summary.append(
            (label, np.median(final), final.mean(), float((final < 0.5).mean()))
        )

    ax.set_yscale("log")
    ax.set_xlabel(f"bet number (of {N_BETS})")
    ax.set_ylabel("bankroll (log scale, start = 1)")
    ax.set_title("Median bankroll paths by betting fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bankroll_paths.png", dpi=150)
    plt.close(fig)

    print(f"{'strategy':28s} {'median final':>14s} {'mean final':>14s} {'P(loss>50%)':>12s}")
    for label, med, mean, p_dd in summary:
        print(f"{label:28s} {med:14.2f} {mean:14.2f} {p_dd:12.3f}")


if __name__ == "__main__":
    plot_growth_curve()
    plot_bankroll_paths()
    print(f"\nFigures written to {FIG_DIR}")
