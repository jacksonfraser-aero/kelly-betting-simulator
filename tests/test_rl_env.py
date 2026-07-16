"""Unit tests for src.rl_env -- run with `python -m pytest`."""

import numpy as np
import pytest

from src.rl_env import CENTRE_MOVES, N_BINS, MarketMakingEnv


def test_rewards_telescope_to_final_pnl():
    """The dense per-step rewards must sum EXACTLY to the settled P&L.

    This is the single most important property of the environment: if
    it holds, the agent's optimisation target is the true objective.
    """
    env = MarketMakingEnv(n_rounds=100)
    env.reset(seed=123)
    rng = np.random.default_rng(0)
    total, done = 0.0, False
    while not done:
        _, r, done, _, info = env.step(int(rng.integers(len(CENTRE_MOVES))))
        total += r
    assert total == pytest.approx(info["final_pnl"], abs=1e-9)


def test_same_seed_same_trajectory():
    def rollout(seed, actions):
        env = MarketMakingEnv(n_rounds=50)
        obs, _ = env.reset(seed=seed)
        traj, rews = [tuple(obs)], []
        for a in actions:
            obs, r, *_ = env.step(a)
            traj.append(tuple(obs))
            rews.append(r)
        return traj, rews

    actions = list(np.random.default_rng(1).integers(0, 5, size=50))
    assert rollout(7, actions) == rollout(7, actions)


def test_observations_stay_in_declared_bins():
    env = MarketMakingEnv(n_rounds=200)
    env.reset(seed=5)
    rng = np.random.default_rng(2)
    done = False
    while not done:
        obs, _, done, _, _ = env.step(int(rng.integers(5)))
        assert all(0 <= o < b for o, b in zip(obs, N_BINS))


def test_centre_moves_shift_centre_bin():
    env = MarketMakingEnv(n_rounds=10)
    obs0, _ = env.reset(seed=9)
    obs = obs0
    for _ in range(4):
        obs, *_ = env.step(4)  # +1.0 per step
    assert obs[1] > obs0[1]


def test_episode_terminates_on_schedule():
    env = MarketMakingEnv(n_rounds=30)
    env.reset(seed=3)
    steps, done = 0, False
    while not done:
        _, _, done, truncated, _ = env.step(2)
        assert truncated is False
        steps += 1
    assert steps == 30


def test_guards():
    env = MarketMakingEnv()
    with pytest.raises(RuntimeError):
        env.step(0)          # step before reset
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(99)         # invalid action


def test_hold_policy_matches_naive_maker_economics():
    """Action 2 (never move) IS the naive maker: mean P&L at 30%%
    informed flow should land near the Day 3 figure (~28)."""
    env = MarketMakingEnv(n_rounds=100, informed_share=0.3)
    pnls = []
    for seed in range(300):
        env.reset(seed=seed)
        done = False
        while not done:
            _, _, done, _, info = env.step(2)
        pnls.append(info["final_pnl"])
    assert 10 < np.mean(pnls) < 45
