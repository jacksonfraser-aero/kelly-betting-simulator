"""A Gymnasium-style RL environment wrapping the market-making game.

The agent is the market maker. Each step it adjusts its quote centre;
a trader arrives (same mechanics as market_game.play_game); the agent
receives the change in mark-to-settlement P&L as reward. The hidden
value V never appears in the observation -- the agent must infer it
from the flow, exactly as a human player does.

Observation (all discretised into small integer bins):
    [ flow imbalance bin, quote-centre offset bin, inventory bin,
      time-remaining bin ]

- flow imbalance = (number of buys - number of sells) so far. Buys
  hitting the ask suggest V is above the quotes, sells below. This is
  the agent's only channel of information about V, which is what makes
  the problem a POMDP: the feature is a hand-crafted summary of history
  standing in for the belief state.
- quote-centre offset = current centre - prior mean (10.5). Without
  this the agent cannot know whether it has already responded to the
  imbalance it sees.
- inventory = current position (positive = long).
- time remaining = coarse early/mid/late phase. Late-game, inventory
  risk dominates; early-game, information gathering pays.

Action (Discrete(5)): move the quote centre by
    {-1.0, -0.5, 0.0, +0.5, +1.0}
with a fixed half-spread of 1.0 around the centre.

Reward (dense): the per-step change in mark-to-settlement P&L.
    trader buys at ask  -> reward = ask - V   (maker sold one unit)
    trader sells at bid -> reward = V - bid   (maker bought one unit)
    pass                -> reward = 0
These telescope: summed over an episode they equal the final settled
P&L exactly (a property the tests pin down). V is used to COMPUTE the
reward during training but is never observable to the agent, which is
the standard and legitimate pattern (training signal, not input).

The class inherits gymnasium.Env when gymnasium is installed, and
otherwise falls back to a plain object with the same reset/step API,
so the module works without the dependency.
"""

from __future__ import annotations

import numpy as np

from .market_game import PRIOR_MEAN, draw_true_value

try:  # gymnasium is optional: same API either way
    import gymnasium as gym
    from gymnasium import spaces

    _BASE = gym.Env
    _HAS_GYM = True
except ImportError:  # pragma: no cover
    _BASE = object
    _HAS_GYM = False


# Action index -> change applied to the quote centre.
CENTRE_MOVES = np.array([-1.0, -0.5, 0.0, +0.5, +1.0])

# Discretisation bin edges (np.digitize convention: value < first edge
# -> bin 0, value >= last edge -> last bin).
IMBALANCE_EDGES = np.array([-4, -2, -1, 1, 2, 4])     # 7 bins
CENTRE_EDGES = np.array([-4, -2, -1, 1, 2, 4])        # 7 bins
INVENTORY_EDGES = np.array([-4, -1, 1, 4])            # 5 bins
TIME_EDGES = np.array([1 / 3, 2 / 3])                 # 3 bins

N_BINS = (
    len(IMBALANCE_EDGES) + 1,
    len(CENTRE_EDGES) + 1,
    len(INVENTORY_EDGES) + 1,
    len(TIME_EDGES) + 1,
)


class MarketMakingEnv(_BASE):
    """Market making as a sequential decision problem.

    Parameters
    ----------
    n_rounds : int
        Episode length (traders per game).
    informed_share : float
        Probability an arriving trader is informed.
    signal_noise : float
        Std dev of the informed trader's signal noise.
    half_spread : float
        Fixed half-spread quoted around the centre.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_rounds: int = 100,
        informed_share: float = 0.3,
        signal_noise: float = 1.0,
        half_spread: float = 1.0,
        dense_reward: bool = True,
    ) -> None:
        self.n_rounds = n_rounds
        self.phi = informed_share
        self.sigma = signal_noise
        self.half_spread = half_spread
        self.dense_reward = dense_reward

        if _HAS_GYM:
            self.observation_space = spaces.MultiDiscrete(list(N_BINS))
            self.action_space = spaces.Discrete(len(CENTRE_MOVES))

        self._rng = np.random.default_rng()
        self._v: int | None = None
        self._sparse_accum = 0.0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options=None):
        if _HAS_GYM:
            super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._v = draw_true_value(self._rng)
        self._t = 0
        self._centre = PRIOR_MEAN
        self._inventory = 0
        self._cash = 0.0
        self._imbalance = 0
        self._sparse_accum = 0.0
        return self._observation(), {}

    def step(self, action: int):
        if self._v is None:
            raise RuntimeError("call reset() before step()")
        if not 0 <= int(action) < len(CENTRE_MOVES):
            raise ValueError(f"invalid action {action}")

        # 1. Apply the quote adjustment.
        self._centre += CENTRE_MOVES[int(action)]
        bid = self._centre - self.half_spread
        ask = self._centre + self.half_spread

        # 2. A trader arrives (identical mechanics to play_game).
        informed = bool(self._rng.random() < self.phi)
        if informed:
            signal = self._v + self._rng.normal(0.0, self.sigma)
            side = 1 if signal > ask else (-1 if signal < bid else 0)
        else:
            side = 1 if self._rng.random() < 0.5 else -1

        # 3. Settle the fill and compute the dense reward.
        if side == 1:      # trader buys: maker sells at the ask
            self._inventory -= 1
            self._cash += ask
            reward = ask - self._v
        elif side == -1:   # trader sells: maker buys at the bid
            self._inventory += 1
            self._cash -= bid
            reward = self._v - bid
        else:
            reward = 0.0
        self._imbalance += side

        # 4. Advance time; terminate at settlement.
        self._t += 1
        terminated = self._t >= self.n_rounds
        info = {"side": side, "informed": informed}
        if terminated:
            info["true_value"] = self._v
            info["final_pnl"] = self._cash + self._inventory * self._v

        if self.dense_reward:
            returned_reward = reward
        else:
            # Sparse mode: withhold all reward until settlement, where
            # the agent receives the FULL episode P&L at once. Credit
            # assignment across n_rounds steps must then be learned
            # entirely through bootstrapping (gamma * max Q(s', .)),
            # rather than being handed a signal every round.
            self._sparse_accum += reward
            returned_reward = self._sparse_accum if terminated else 0.0

        return self._observation(), float(returned_reward), terminated, False, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _observation(self) -> np.ndarray:
        imb = int(np.digitize(self._imbalance, IMBALANCE_EDGES))
        cen = int(np.digitize(self._centre - PRIOR_MEAN, CENTRE_EDGES))
        inv = int(np.digitize(self._inventory, INVENTORY_EDGES))
        tim = int(np.digitize(self._t / self.n_rounds, TIME_EDGES))
        return np.array([imb, cen, inv, tim], dtype=np.int64)

    @property
    def pnl(self) -> float:
        """Mark-to-settlement P&L of the current position (uses V --
        for logging and tests, never for the agent's observation)."""
        return self._cash + self._inventory * self._v
