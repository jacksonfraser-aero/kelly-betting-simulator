"""Unit tests for src.estimation -- run with `python -m pytest`."""

import numpy as np
import pytest

from src.estimation import (
    estimate_win_prob,
    growth_vs_multiplier,
    realised_growth_with_estimation,
)


def test_estimates_are_unbiased_and_consistent():
    rng = np.random.default_rng(0)
    p_hat = estimate_win_prob(0.55, n_history=1_000, n_paths=20_000, rng=rng)
    assert p_hat.mean() == pytest.approx(0.55, abs=2e-3)
    # More history -> tighter estimates
    p_hat_small = estimate_win_prob(0.55, n_history=50, n_paths=20_000, rng=rng)
    assert p_hat.std() < p_hat_small.std()


def test_bias_shifts_estimates():
    rng = np.random.default_rng(1)
    p_hat = estimate_win_prob(0.52, 500, 20_000, bias=0.03, rng=rng)
    assert p_hat.mean() == pytest.approx(0.55, abs=2e-3)


def test_zero_multiplier_means_zero_growth():
    g = realised_growth_with_estimation(0.0, 0.55, 1.0, n_history=100,
                                        n_bets=100, n_paths=500)
    assert np.all(g == 0.0)


def test_negative_multiplier_raises():
    with pytest.raises(ValueError):
        realised_growth_with_estimation(-0.5, 0.55, 1.0, n_history=100)
    with pytest.raises(ValueError):
        estimate_win_prob(0.55, n_history=0, n_paths=10)


def test_noisy_optimum_below_full_kelly():
    # With a short history, the growth-maximising multiplier is < 1.
    rng = np.random.default_rng(2)
    multipliers = np.linspace(0.0, 2.0, 21)
    means, _ = growth_vs_multiplier(
        multipliers, 0.55, 1.0, n_history=50,
        n_bets=500, n_paths=4_000, rng=rng,
    )
    assert multipliers[np.argmax(means)] < 1.0


def test_perfect_information_recovers_full_kelly():
    # With a very long history the optimum approaches multiplier = 1.
    rng = np.random.default_rng(3)
    multipliers = np.linspace(0.0, 2.0, 21)
    means, _ = growth_vs_multiplier(
        multipliers, 0.55, 1.0, n_history=100_000,
        n_bets=500, n_paths=4_000, rng=rng,
    )
    assert multipliers[np.argmax(means)] == pytest.approx(1.0, abs=0.15)
