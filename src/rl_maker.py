"""A frozen Q-learning policy wrapped as a tournament MarketMaker.

The adapter mirrors the environment's feature extraction: it tracks
flow imbalance, its own quote centre, and the round clock, rebuilds
the discretised observation each round, and takes the greedy action
from a trained Q-table. No learning happens at tournament time.
"""

from __future__ import annotations

import numpy as np

from .market_game import PRIOR_MEAN, MarketMaker, Trade
from .rl_env import (
    CENTRE_EDGES,
    CENTRE_MOVES,
    IMBALANCE_EDGES,
    INVENTORY_EDGES,
    TIME_EDGES,
)


class RLMarketMaker(MarketMaker):
    """Quotes according to a trained (frozen) Q-table."""

    def __init__(self, q_table: np.ndarray, n_rounds: int = 100,
                 half_spread: float = 1.0,
                 rng: np.random.Generator | None = None) -> None:
        super().__init__()
        self.q = q_table
        self.n_rounds = n_rounds
        self.half_spread = half_spread
        self.rng = rng or np.random.default_rng(0)
        self._centre = PRIOR_MEAN
        self._imbalance = 0
        self._t = 0
        self.name = "RL (tabular Q)"

    def _observation(self) -> tuple[int, int, int, int]:
        imb = int(np.digitize(self._imbalance, IMBALANCE_EDGES))
        cen = int(np.digitize(self._centre - PRIOR_MEAN, CENTRE_EDGES))
        inv = int(np.digitize(self.inventory, INVENTORY_EDGES))
        tim = int(np.digitize(self._t / self.n_rounds, TIME_EDGES))
        return (imb, cen, inv, tim)

    def quote(self, round_idx: int) -> tuple[float, float]:
        values = self.q[self._observation()]
        best = np.flatnonzero(values == values.max())
        action = int(self.rng.choice(best))
        self._centre += CENTRE_MOVES[action]
        self._t += 1
        return self._centre - self.half_spread, self._centre + self.half_spread

    def on_trade(self, trade: Trade) -> None:
        super().on_trade(trade)
        self._imbalance += trade.side
