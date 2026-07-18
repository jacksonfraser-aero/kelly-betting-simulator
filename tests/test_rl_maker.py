"""Unit tests for src.rl_maker -- run with `python -m pytest`."""

import numpy as np
import pytest

from src.market_game import tournament
from src.qlearning import train
from src.rl_env import N_BINS
from src.rl_maker import RLMarketMaker


def test_zero_table_holds_still_or_moves_randomly_but_quotes_validly():
    q = np.zeros(N_BINS + (5,))
    mm = RLMarketMaker(q, rng=np.random.default_rng(0))
    for t in range(50):
        bid, ask = mm.quote(t)
        assert ask - bid == pytest.approx(2.0)


def test_trained_table_beats_naive_in_tournament():
    """Wrapped frozen policy must reproduce (approximately) its
    env-side evaluation when run through the tournament machinery."""
    agent, _ = train(n_episodes=4000, eval_every=4000, eval_games=100,
                     seed=0, verbose=False)
    rng = np.random.default_rng(99)
    pnls = tournament(lambda: RLMarketMaker(agent.q),
                      n_games=300, n_rounds=100,
                      informed_share=0.3, rng=rng)
    assert pnls.mean() > 35   # naive is ~28 here; trained agent well above
