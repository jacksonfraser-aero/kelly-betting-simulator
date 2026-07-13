"""Kelly betting under parameter uncertainty.

Day 1 assumed the bettor knows the true win probability p. In reality p
must be *estimated* -- from a finite sample of past outcomes, or from a
model that may be biased. This module studies what estimation error does
to Kelly betting, and why fractional Kelly is the practical answer.

Setup: before betting, the bettor observes `n_history` past outcomes of
the same bet, estimates p_hat (sample win rate), and then repeatedly bets

    f = multiplier * kelly_fraction(p_hat, b)

against the TRUE probability p. Because the log-growth curve g(f) is
asymmetric around its peak (over-betting is punished far more than
under-betting), noise in p_hat makes aggressive sizing costly: the
multiplier that maximises realised growth drops below 1.
"""

from __future__ import annotations

import numpy as np

from .kelly import kelly_fraction


def estimate_win_prob(
    true_p: float,
    n_history: int,
    n_paths: int,
    bias: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample-win-rate estimates of p from n_history observed bets.

    Parameters
    ----------
    true_p : float
        The true win probability generating the historical outcomes.
    n_history : int
        Number of past outcomes the bettor observes before sizing bets.
    n_paths : int
        Number of independent bettors (simulation paths).
    bias : float
        Optional additive bias applied to the estimate, modelling
        systematic overconfidence (positive) or conservatism (negative).

    Returns
    -------
    np.ndarray of shape (n_paths,)
        Estimated win probabilities, clipped to [0.001, 0.999].
    """
    if n_history <= 0:
        raise ValueError("n_history must be positive")
    rng = rng or np.random.default_rng()
    wins = rng.binomial(n_history, true_p, size=n_paths)
    p_hat = wins / n_history + bias
    return np.clip(p_hat, 0.001, 0.999)


def realised_growth_with_estimation(
    multiplier: float,
    true_p: float,
    b: float,
    n_history: int,
    n_bets: int = 1_000,
    n_paths: int = 10_000,
    bias: float = 0.0,
    max_fraction: float = 0.99,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Realised per-bet log-growth when Kelly is sized on an ESTIMATED edge.

    Each path: estimate p_hat from n_history outcomes, bet
    f = multiplier * kelly(p_hat, b) for n_bets rounds against true_p,
    and record the realised mean log-growth per bet.

    Returns
    -------
    np.ndarray of shape (n_paths,)
        Realised log-growth per bet for each path. Paths whose estimate
        implies no edge (f = 0) realise exactly 0 growth.
    """
    if multiplier < 0:
        raise ValueError("multiplier must be non-negative")
    rng = rng or np.random.default_rng()

    p_hat = estimate_win_prob(true_p, n_history, n_paths, bias=bias, rng=rng)
    f = multiplier * np.array([kelly_fraction(p, b) for p in p_hat])
    f = np.clip(f, 0.0, max_fraction)

    # Simulate n_bets outcomes at the TRUE p; growth is analytic given
    # the win count since bet size is a constant fraction per path.
    n_wins = rng.binomial(n_bets, true_p, size=n_paths)
    growth = (
        n_wins * np.log1p(b * f) + (n_bets - n_wins) * np.log1p(-f)
    ) / n_bets
    return growth


def growth_vs_multiplier(
    multipliers: np.ndarray,
    true_p: float,
    b: float,
    n_history: int,
    n_bets: int = 1_000,
    n_paths: int = 10_000,
    bias: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and 10th-percentile realised growth across a grid of multipliers.

    Returns (mean_growth, p10_growth), each shaped like `multipliers`.
    The 10th percentile captures downside risk, not just the average.
    """
    rng = rng or np.random.default_rng()
    means = np.empty(len(multipliers))
    p10s = np.empty(len(multipliers))
    for i, m in enumerate(multipliers):
        g = realised_growth_with_estimation(
            m, true_p, b, n_history,
            n_bets=n_bets, n_paths=n_paths, bias=bias, rng=rng,
        )
        means[i] = g.mean()
        p10s[i] = np.percentile(g, 10)
    return means, p10s
