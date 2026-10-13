"""Unit tests for src.strategies -- run with `python -m pytest`."""

import numpy as np
import pytest

from src.market_game import NaiveMarketMaker, Trade, play_game, tournament
from src.strategies import (
    BayesianMarketMaker,
    InventoryAwareMarketMaker,
    _three_dice_pmf,
)


def test_prior_is_valid_and_symmetric():
    pmf = _three_dice_pmf()
    assert pmf.sum() == pytest.approx(1.0)
    assert pmf[0] == pytest.approx(1 / 216)      # P(sum = 3)
    assert np.allclose(pmf, pmf[::-1])           # symmetric around 10.5


def test_inventory_aware_skews_quotes():
    mm = InventoryAwareMarketMaker(1.0, skew=0.5)
    b0, a0 = mm.quote(0)
    mm.on_trade(Trade(0, -1, b0, informed=False))  # maker buys one
    assert mm.inventory == +1
    b1, a1 = mm.quote(1)
    assert b1 == b0 - 0.5 and a1 == a0 - 0.5       # long -> quotes drop


def test_bayesian_starts_at_prior_mean():
    mm = BayesianMarketMaker(1.0)
    assert mm.fair_value == pytest.approx(10.5)


def test_bayesian_updates_toward_flow():
    mm = BayesianMarketMaker(1.0)
    bid, ask = mm.quote(0)
    mm.on_trade(Trade(0, +1, ask, informed=False))  # a buy
    assert mm.fair_value > 10.5                      # estimate rises
    up = mm.fair_value
    mm.quote(1)
    mm.on_trade(Trade(1, -1, bid, informed=False))  # a sell
    assert mm.fair_value < up                        # estimate falls back


def _posterior_std(mm):
    values = np.arange(3, 19)
    mean = (values * mm.posterior).sum()
    return float(np.sqrt(((values - mean) ** 2 * mm.posterior).sum()))


def test_pass_concentrates_posterior_between_quotes():
    mm = BayesianMarketMaker(1.0)
    before = _posterior_std(mm)
    for t in range(10):
        mm.quote(t)
        mm.on_pass(t)
    # Repeated passes shrink uncertainty about V and keep the mean
    # near the mid: only informed traders pass, and only when V is
    # inside the spread.
    assert _posterior_std(mm) < before / 2
    assert abs(mm.fair_value - 10.5) < 1.0


def test_bayesian_beats_naive_under_informed_flow():
    rng = np.random.default_rng(10)
    naive = tournament(lambda: NaiveMarketMaker(1.0),
                       n_games=300, n_rounds=100, informed_share=0.5, rng=rng)
    rng = np.random.default_rng(10)
    bayes = tournament(lambda: BayesianMarketMaker(1.0),
                       n_games=300, n_rounds=100, informed_share=0.5, rng=rng)
    assert bayes.mean() > naive.mean() + 30
