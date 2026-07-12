"""Core Kelly criterion functions for binary bets.

The Kelly criterion (Kelly, 1956) gives the bet fraction that maximises
the long-run exponential growth rate of wealth for a repeated favourable
bet. For a binary bet with win probability p and net fractional odds b
(i.e. a winning 1-unit stake returns b units of profit), the optimal
fraction of bankroll to wager is:

    f* = (b * p - q) / b,   where q = 1 - p

A negative f* means the bet has negative edge and should not be taken.
"""

from __future__ import annotations

import numpy as np


def kelly_fraction(p: float, b: float) -> float:
    """Optimal Kelly bet fraction for a binary bet.

    Parameters
    ----------
    p : float
        Probability of winning, in (0, 1).
    b : float
        Net fractional odds: profit per unit staked on a win.
        e.g. even-money odds -> b = 1.0; "2-to-1" -> b = 2.0.

    Returns
    -------
    float
        Optimal fraction of bankroll to bet. Clipped at 0 (never bet
        when edge is negative).
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if b <= 0.0:
        raise ValueError("b must be positive")
    q = 1.0 - p
    f_star = (b * p - q) / b
    return max(f_star, 0.0)


def expected_log_growth(f: float | np.ndarray, p: float, b: float) -> float | np.ndarray:
    """Expected log-growth rate per bet when betting fraction f.

    g(f) = p * ln(1 + b f) + (1 - p) * ln(1 - f)

    This is the quantity Kelly betting maximises. It is defined for
    f in [0, 1); at f = 1 a single loss is total ruin (g -> -inf).
    """
    f = np.asarray(f, dtype=float)
    if np.any(f < 0.0) or np.any(f >= 1.0):
        raise ValueError("f must be in [0, 1)")
    g = p * np.log1p(b * f) + (1.0 - p) * np.log1p(-f)
    return g if g.ndim else float(g)


def simulate_bankroll_paths(
    f: float,
    p: float,
    b: float,
    n_bets: int = 1_000,
    n_paths: int = 10_000,
    initial_bankroll: float = 1.0,
    ruin_threshold: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate bankroll paths for repeated fractional betting.

    Each bet stakes fraction f of the current bankroll on a binary
    outcome with win probability p and net odds b. Multiplicative
    wealth dynamics: bankroll *= (1 + b f) on a win, (1 - f) on a loss.

    Returns
    -------
    np.ndarray of shape (n_paths, n_bets + 1)
        Bankroll trajectories including the initial value at column 0.
        Paths that fall below ruin_threshold are floored at 0 and stay
        there (absorbing ruin state).
    """
    if not 0.0 <= f < 1.0:
        raise ValueError("f must be in [0, 1)")
    rng = rng or np.random.default_rng()

    wins = rng.random((n_paths, n_bets)) < p
    growth = np.where(wins, 1.0 + b * f, 1.0 - f)

    paths = np.empty((n_paths, n_bets + 1))
    paths[:, 0] = initial_bankroll
    np.multiply.accumulate(growth, axis=1, out=growth)
    paths[:, 1:] = initial_bankroll * growth

    # Absorbing ruin: once below threshold, bankroll stays at zero.
    ruined = np.maximum.accumulate(paths < ruin_threshold, axis=1)
    paths[ruined] = 0.0
    return paths
