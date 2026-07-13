"""Unit tests for src.market_game -- run with `python -m pytest`."""

import numpy as np
import pytest

from src.market_game import (
    PRIOR_MEAN,
    NaiveMarketMaker,
    Trade,
    draw_true_value,
    play_game,
    tournament,
)


def test_true_value_distribution():
    rng = np.random.default_rng(5)
    vals = [draw_true_value(rng) for _ in range(50_000)]
    assert min(vals) >= 3 and max(vals) <= 18
    assert np.mean(vals) == pytest.approx(PRIOR_MEAN, abs=0.05)


def test_maker_accounting():
    mm = NaiveMarketMaker(1.0)
    bid, ask = mm.quote(0)
    assert (bid, ask) == (9.5, 11.5)
    mm.on_trade(Trade(0, +1, ask, informed=False))  # trader buys at ask
    assert mm.inventory == -1
    assert mm.cash == 11.5
    assert mm.pnl(true_value=11.5) == 0.0   # sold exactly at value
    assert mm.pnl(true_value=10.0) == 1.5   # sold above value: profit


def test_noise_flow_is_profitable():
    rng = np.random.default_rng(6)
    pnls = tournament(
        lambda: NaiveMarketMaker(1.0),
        n_games=400, n_rounds=100, informed_share=0.0, rng=rng,
    )
    assert pnls.mean() > 50  # spread income with no adverse selection


def test_informed_flow_hurts():
    rng = np.random.default_rng(6)
    pnls_noise = tournament(
        lambda: NaiveMarketMaker(1.0),
        n_games=400, n_rounds=100, informed_share=0.0, rng=rng,
    )
    rng = np.random.default_rng(7)
    pnls_informed = tournament(
        lambda: NaiveMarketMaker(1.0),
        n_games=400, n_rounds=100, informed_share=0.6, rng=rng,
    )
    assert pnls_informed.mean() < pnls_noise.mean() - 50


def test_crossed_market_raises():
    class BadMaker(NaiveMarketMaker):
        def quote(self, round_idx):
            return (11.0, 10.0)

    with pytest.raises(ValueError):
        play_game(BadMaker(), n_rounds=5, rng=np.random.default_rng(8))


def test_game_result_paths_shape():
    result = play_game(
        NaiveMarketMaker(1.0), n_rounds=50, rng=np.random.default_rng(9)
    )
    assert result.inventory_path.shape == (51,)
    assert result.pnl_path.shape == (51,)
    assert result.pnl_path[0] == 0.0
