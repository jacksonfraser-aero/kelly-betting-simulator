"""Unit tests for src.kelly -- run with `python -m pytest` from the repo root."""

import numpy as np
import pytest

from src.kelly import expected_log_growth, kelly_fraction, simulate_bankroll_paths


def test_kelly_even_money():
    # f* = 2p - 1 for even-money odds
    assert kelly_fraction(0.55, 1.0) == pytest.approx(0.10)
    assert kelly_fraction(0.60, 1.0) == pytest.approx(0.20)


def test_kelly_negative_edge_clipped_to_zero():
    assert kelly_fraction(0.45, 1.0) == 0.0


def test_kelly_fraction_maximises_growth():
    p, b = 0.55, 1.5
    f_star = kelly_fraction(p, b)
    f_grid = np.linspace(0.0, 0.6, 2_000)
    g = expected_log_growth(f_grid, p, b)
    assert f_grid[np.argmax(g)] == pytest.approx(f_star, abs=1e-3)


def test_growth_zero_at_zero_fraction():
    assert expected_log_growth(0.0, 0.55, 1.0) == pytest.approx(0.0)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        kelly_fraction(1.2, 1.0)
    with pytest.raises(ValueError):
        kelly_fraction(0.55, -1.0)
    with pytest.raises(ValueError):
        expected_log_growth(1.0, 0.55, 1.0)  # f = 1 not allowed


def test_simulated_growth_matches_theory():
    # Empirical mean log-growth per bet should match g(f) closely.
    p, b, f = 0.55, 1.0, 0.10
    rng = np.random.default_rng(42)
    paths = simulate_bankroll_paths(f, p, b, n_bets=2_000, n_paths=5_000, rng=rng)
    final = paths[:, -1]
    emp_growth = np.log(final[final > 0]).mean() / 2_000
    assert emp_growth == pytest.approx(expected_log_growth(f, p, b), abs=5e-4)


def test_paths_shape_and_start():
    paths = simulate_bankroll_paths(0.05, 0.55, 1.0, n_bets=10, n_paths=100)
    assert paths.shape == (100, 11)
    assert np.all(paths[:, 0] == 1.0)
