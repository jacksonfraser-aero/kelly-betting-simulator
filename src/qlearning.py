"""Tabular Q-learning for the market-making environment.

The agent maintains a table Q[s, a]: its running estimate of the total
future P&L from taking action a in (discretised) state s and acting
greedily ever after. Learning is the classic Q-learning update
(Watkins, 1989): after experiencing (s, a, r, s'),

    Q[s, a] <- Q[s, a] + alpha * (r + gamma * max_a' Q[s', a'] - Q[s, a])

The bracketed term is the temporal-difference error: the gap between
the "TD target" (what this step's evidence says the action was worth:
immediate reward plus the discounted value of the best continuation)
and the current estimate. alpha is the learning rate.

Exploration is epsilon-greedy: with probability epsilon act uniformly
at random, otherwise take the greedy action. Epsilon decays linearly
from eps_start to eps_end over the first `eps_decay_frac` of training,
then stays at eps_end. Q-learning is off-policy: the update uses the
max over next actions even when the behaviour was random, so it learns
the greedy policy's value while still exploring.

Evaluation is strictly separated from training: a frozen greedy policy
(epsilon = 0) is scored on a fixed block of evaluation seeds disjoint
from training seeds, so learning curves measure the policy, not the
exploration noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rl_env import CENTRE_MOVES, N_BINS, MarketMakingEnv

N_ACTIONS = len(CENTRE_MOVES)

# Training episodes use seeds >= TRAIN_SEED_BASE; evaluation uses
# 0..n_eval-1. Keeping the blocks disjoint means the agent is always
# scored on games it never trained on.
TRAIN_SEED_BASE = 1_000_000


class QLearningAgent:
    """A Q-table over the environment's discretised observation space."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.q = np.zeros(N_BINS + (N_ACTIONS,))
        self.visits = np.zeros(N_BINS + (N_ACTIONS,), dtype=np.int64)
        self.rng = rng or np.random.default_rng()

    # -- acting -----------------------------------------------------------
    def greedy_action(self, obs: np.ndarray) -> int:
        """Argmax over actions; ties broken at random for symmetry."""
        values = self.q[tuple(obs)]
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best))

    def act(self, obs: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(N_ACTIONS))
        return self.greedy_action(obs)

    # -- learning ---------------------------------------------------------
    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        alpha: float,
        gamma: float,
    ) -> float:
        """One Q-learning update; returns the TD error (for logging)."""
        idx = tuple(obs) + (action,)
        # A terminated state has no future: its continuation value is 0.
        bootstrap = 0.0 if terminated else float(self.q[tuple(next_obs)].max())
        td_error = reward + gamma * bootstrap - self.q[idx]
        self.q[idx] += alpha * td_error
        self.visits[idx] += 1
        return float(td_error)


@dataclass
class TrainingHistory:
    """Learning-curve data collected during training."""
    eval_episodes: list[int] = field(default_factory=list)
    eval_mean_pnl: list[float] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)


def epsilon_schedule(
    episode: int, n_episodes: int,
    eps_start: float = 1.0, eps_end: float = 0.05,
    eps_decay_frac: float = 0.6,
) -> float:
    """Linear decay from eps_start to eps_end, then flat."""
    horizon = max(1, int(n_episodes * eps_decay_frac))
    frac = min(1.0, episode / horizon)
    return eps_start + frac * (eps_end - eps_start)


def evaluate(
    agent: QLearningAgent,
    n_games: int = 200,
    seed_base: int = 0,
    **env_kwargs,
) -> float:
    """Mean final P&L of the frozen greedy policy on held-out seeds."""
    env = MarketMakingEnv(**env_kwargs)
    pnls = np.empty(n_games)
    for i in range(n_games):
        obs, _ = env.reset(seed=seed_base + i)
        done = False
        while not done:
            obs, _, done, _, info = env.step(agent.greedy_action(obs))
        pnls[i] = info["final_pnl"]
    return float(pnls.mean())


def train(
    n_episodes: int = 20_000,
    alpha: float = 0.1,
    gamma: float = 1.0,
    eval_every: int = 1_000,
    eval_games: int = 200,
    seed: int = 0,
    verbose: bool = True,
    **env_kwargs,
) -> tuple[QLearningAgent, TrainingHistory]:
    """Train an agent from scratch; returns (agent, learning history).

    gamma defaults to 1.0: the task is episodic and rewards are money,
    so a unit of P&L is worth the same whenever it arrives.
    """
    agent = QLearningAgent(rng=np.random.default_rng(seed))
    env = MarketMakingEnv(**env_kwargs)
    history = TrainingHistory()
    best_pnl, best_q = -np.inf, agent.q.copy()

    for episode in range(n_episodes):
        eps = epsilon_schedule(episode, n_episodes)
        obs, _ = env.reset(seed=TRAIN_SEED_BASE + seed * n_episodes + episode)
        done = False
        while not done:
            action = agent.act(obs, eps)
            next_obs, reward, done, _, _ = env.step(action)
            agent.update(obs, action, reward, next_obs, done, alpha, gamma)
            obs = next_obs

        if (episode + 1) % eval_every == 0 or episode == 0:
            mean_pnl = evaluate(agent, n_games=eval_games, **env_kwargs)
            history.eval_episodes.append(episode + 1)
            history.eval_mean_pnl.append(mean_pnl)
            history.epsilons.append(eps)
            if mean_pnl > best_pnl:
                best_pnl, best_q = mean_pnl, agent.q.copy()
            if verbose:
                print(f"episode {episode + 1:>6d}  eps {eps:.2f}  "
                      f"eval mean P&L {mean_pnl:7.1f}")

    # Keep the best checkpoint, not the last: tabular Q-learning with a
    # constant learning rate oscillates, and the final episode is an
    # arbitrary place to stop. Selection uses the eval seed block, so
    # any final reported number must come from a disjoint block.
    agent.q = best_q
    return agent, history
