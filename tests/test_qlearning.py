"""Unit tests for src.qlearning -- run with `python -m pytest`."""

import numpy as np
import pytest

from src.qlearning import (
    N_ACTIONS,
    QLearningAgent,
    epsilon_schedule,
    evaluate,
    train,
)


def test_td_update_matches_hand_calculation():
    agent = QLearningAgent(rng=np.random.default_rng(0))
    s = np.array([0, 0, 0, 0])
    s2 = np.array([1, 0, 0, 0])
    agent.q[tuple(s2)] = [1.0, 3.0, 2.0, 0.0, 0.0]   # max = 3.0

    # Q[s,a] starts at 0; r = 2, gamma = 0.5, alpha = 0.1
    # target = 2 + 0.5 * 3 = 3.5; error = 3.5; new Q = 0.35
    err = agent.update(s, action=1, reward=2.0, next_obs=s2,
                       terminated=False, alpha=0.1, gamma=0.5)
    assert err == pytest.approx(3.5)
    assert agent.q[tuple(s) + (1,)] == pytest.approx(0.35)


def test_terminal_update_ignores_future():
    agent = QLearningAgent(rng=np.random.default_rng(0))
    s = np.array([0, 0, 0, 0])
    s2 = np.array([1, 0, 0, 0])
    agent.q[tuple(s2)] = [100.0] * N_ACTIONS  # must be ignored

    err = agent.update(s, action=0, reward=5.0, next_obs=s2,
                       terminated=True, alpha=1.0, gamma=1.0)
    assert err == pytest.approx(5.0)
    assert agent.q[tuple(s) + (0,)] == pytest.approx(5.0)


def test_greedy_action_is_argmax():
    agent = QLearningAgent(rng=np.random.default_rng(0))
    s = np.array([2, 3, 1, 0])
    agent.q[tuple(s)] = [0.0, -1.0, 4.0, 2.0, 3.0]
    assert agent.greedy_action(s) == 2


def test_epsilon_greedy_explores_and_exploits():
    agent = QLearningAgent(rng=np.random.default_rng(0))
    s = np.array([0, 0, 0, 0])
    agent.q[tuple(s)] = [0.0, 0.0, 10.0, 0.0, 0.0]
    # epsilon = 0: always greedy
    assert all(agent.act(s, 0.0) == 2 for _ in range(20))
    # epsilon = 1: all actions appear
    seen = {agent.act(s, 1.0) for _ in range(300)}
    assert seen == set(range(N_ACTIONS))


def test_epsilon_schedule_decays_then_flattens():
    assert epsilon_schedule(0, 1000) == pytest.approx(1.0)
    assert epsilon_schedule(600, 1000) == pytest.approx(0.05)
    assert epsilon_schedule(999, 1000) == pytest.approx(0.05)
    mid = epsilon_schedule(300, 1000)
    assert 0.05 < mid < 1.0


def test_training_beats_hold_policy():
    """A short run must already beat an untrained agent clearly.

    Uses reduced sizes to keep the test fast; the full training run
    lives in scripts/train_rl_maker.py.
    """
    agent, hist = train(n_episodes=1500, eval_every=1500, eval_games=100,
                        seed=3, verbose=False, n_rounds=50)
    final = hist.eval_mean_pnl[-1]
    baseline = evaluate(QLearningAgent(rng=np.random.default_rng(1)),
                        n_games=100, n_rounds=50)
    # an untrained table is all zeros -> ties broken randomly, so the
    # baseline is a random policy; trained must beat it clearly and
    # must be profitable in absolute terms
    assert final > baseline + 10
    assert final > 0
