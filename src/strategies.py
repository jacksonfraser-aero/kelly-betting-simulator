"""Smarter market-making strategies.

Day 3 showed a fixed-spread maker bleeding to informed flow. The fix has
two independent parts, implemented here as two strategies:

- **InventoryAwareMarketMaker**: skews its quote centre against its
  inventory (long -> quote lower to attract buyers, short -> higher).
  This manages position risk but still never learns where V is.

- **BayesianMarketMaker**: maintains a posterior over the hidden value V
  and centres quotes on the posterior mean. Every event is evidence: a
  buy is more likely when V is above the ask, a sell when V is below the
  bid, and a pass -- which only an informed trader produces -- means V
  is probably between the quotes. This is the Glosten-Milgrom idea:
  quotes must reflect what the act of trading itself reveals.
"""

from __future__ import annotations

import math

import numpy as np

from .market_game import PRIOR_MEAN, MarketMaker, Trade

# Prior pmf of the sum of three fair dice, values 3..18.
_VALUES = np.arange(3, 19)


def _three_dice_pmf() -> np.ndarray:
    counts = np.zeros(19)
    for a in range(1, 7):
        for b in range(1, 7):
            for c in range(1, 7):
                counts[a + b + c] += 1
    return counts[3:19] / counts.sum()


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF via erf (avoids a scipy dependency)."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


class InventoryAwareMarketMaker(MarketMaker):
    """Fixed spread around a centre skewed against current inventory.

    centre = prior mean - skew * inventory. When long, quotes drop,
    making it easier to sell out and harder to buy more; vice versa
    when short. Position control without any learning about V.
    """

    def __init__(self, half_spread: float = 1.0, skew: float = 0.3) -> None:
        super().__init__()
        self.half_spread = half_spread
        self.skew = skew
        self.name = f"inventory-aware (skew = {skew})"

    def quote(self, round_idx: int) -> tuple[float, float]:
        centre = PRIOR_MEAN - self.skew * self.inventory
        return centre - self.half_spread, centre + self.half_spread


class BayesianMarketMaker(MarketMaker):
    """Quotes around the posterior mean of V, updated from all flow.

    The maker knows the game's structure: the share of informed traders
    (informed_share) and their signal noise (signal_noise). Each round's
    outcome updates a discrete posterior over V in {3, ..., 18}:

        P(buy  | V) = phi * P(V + eps > ask) + (1 - phi) / 2
        P(sell | V) = phi * P(V + eps < bid) + (1 - phi) / 2
        P(pass | V) = phi * P(bid <= V + eps <= ask)

    Noise traders never pass, which is why a pass concentrates the
    posterior between the quotes.
    """

    def __init__(
        self,
        half_spread: float = 1.0,
        informed_share: float = 0.3,
        signal_noise: float = 1.0,
        inventory_skew: float = 0.0,
    ) -> None:
        super().__init__()
        self.half_spread = half_spread
        self.phi = informed_share
        self.sigma = max(signal_noise, 1e-6)
        self.inventory_skew = inventory_skew
        self.posterior = _three_dice_pmf().copy()
        self._last_quote: tuple[float, float] | None = None
        label = "bayesian" if inventory_skew == 0.0 else "bayesian + skew"
        self.name = f"{label} (s = {half_spread})"

    @property
    def fair_value(self) -> float:
        return float((_VALUES * self.posterior).sum())

    def quote(self, round_idx: int) -> tuple[float, float]:
        centre = self.fair_value - self.inventory_skew * self.inventory
        quote = (centre - self.half_spread, centre + self.half_spread)
        self._last_quote = quote
        return quote

    def _update(self, likelihood: np.ndarray) -> None:
        post = self.posterior * likelihood
        total = post.sum()
        if total > 0:
            self.posterior = post / total
        # If evidence is impossible under the model, keep the old posterior.

    def _event_likelihoods(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bid, ask = self._last_quote
        p_above_ask = 1.0 - _norm_cdf((ask - _VALUES) / self.sigma)
        p_below_bid = _norm_cdf((bid - _VALUES) / self.sigma)
        buy = self.phi * p_above_ask + (1.0 - self.phi) * 0.5
        sell = self.phi * p_below_bid + (1.0 - self.phi) * 0.5
        pas = self.phi * (1.0 - p_above_ask - p_below_bid)
        return buy, sell, pas

    def on_trade(self, trade: Trade) -> None:
        super().on_trade(trade)
        if self._last_quote is None:
            return
        buy, sell, _ = self._event_likelihoods()
        self._update(buy if trade.side == 1 else sell)

    def on_pass(self, round_idx: int) -> None:
        if self._last_quote is None:
            return
        _, _, pas = self._event_likelihoods()
        self._update(pas)
