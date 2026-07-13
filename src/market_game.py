"""A market-making game with informed and uninformed order flow.

The game: a hidden true value V is drawn (the sum of three dice, so
V is in [3, 18] with mean 10.5). Each round a trader arrives and the
market maker must quote a bid and an ask around its estimate of V.

Traders come in two kinds:

- **Noise traders** (probability 1 - phi): buy or sell at random,
  regardless of price. They are the market maker's profit source.
- **Informed traders** (probability phi): observe V plus noise, buy if
  their signal is above the ask, sell if below the bid, else pass.
  They are the market maker's adverse-selection cost.

At the end of the game every position is settled at the true V. The
market maker's P&L therefore decomposes into spread income earned from
noise traders minus losses to informed traders who traded against
mispriced quotes.

Day 3 implements the engine plus a naive fixed-spread market maker.
Smarter strategies (inventory-aware, Bayesian) arrive on Day 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PRIOR_MEAN = 10.5  # mean of the sum of three fair dice


def draw_true_value(rng: np.random.Generator) -> int:
    """Hidden value: sum of three fair six-sided dice."""
    return int(rng.integers(1, 7, size=3).sum())


@dataclass
class Trade:
    """Record of a single fill against the market maker's quotes."""
    round_idx: int
    side: int          # +1 = trader bought (MM sold), -1 = trader sold
    price: float
    informed: bool


class MarketMaker:
    """Base class: subclasses implement quoting logic.

    State tracked here: inventory (positive = long), cash, and fills.
    Sign convention: when a trader BUYS at the ask, the MM's inventory
    falls by 1 and cash rises by the ask price.
    """

    name = "base"

    def __init__(self) -> None:
        self.inventory = 0
        self.cash = 0.0
        self.trades: list[Trade] = []

    def quote(self, round_idx: int) -> tuple[float, float]:
        raise NotImplementedError

    def on_trade(self, trade: Trade) -> None:
        self.trades.append(trade)
        self.inventory -= trade.side
        self.cash += trade.side * trade.price

    def pnl(self, true_value: float) -> float:
        """Mark-to-settlement P&L: cash plus inventory valued at V."""
        return self.cash + self.inventory * true_value


class NaiveMarketMaker(MarketMaker):
    """Quotes a fixed spread around the prior mean, ignoring all flow.

    This maker never learns: it keeps quoting around 10.5 even as
    informed traders repeatedly hit the same side. Its performance as
    informed flow rises is the adverse-selection benchmark.
    """

    def __init__(self, half_spread: float = 1.0) -> None:
        super().__init__()
        self.half_spread = half_spread
        self.name = f"naive (s = {half_spread})"

    def quote(self, round_idx: int) -> tuple[float, float]:
        return PRIOR_MEAN - self.half_spread, PRIOR_MEAN + self.half_spread


@dataclass
class GameResult:
    """Outcome of one full game for one market maker."""
    true_value: int
    final_pnl: float
    n_trades: int
    n_informed_trades: int
    inventory_path: np.ndarray = field(repr=False)
    pnl_path: np.ndarray = field(repr=False)


def play_game(
    maker: MarketMaker,
    n_rounds: int = 100,
    informed_share: float = 0.3,
    signal_noise: float = 1.0,
    rng: np.random.Generator | None = None,
) -> GameResult:
    """Run one game: n_rounds of arriving traders against the maker.

    Parameters
    ----------
    maker : MarketMaker
        A fresh maker instance (state is mutated in place).
    informed_share : float
        Probability phi that an arriving trader is informed.
    signal_noise : float
        Std dev of Gaussian noise on the informed trader's view of V.
    """
    rng = rng or np.random.default_rng()
    v = draw_true_value(rng)

    inventory_path = np.zeros(n_rounds + 1)
    pnl_path = np.zeros(n_rounds + 1)
    pnl_path[0] = maker.pnl(v)

    for t in range(n_rounds):
        bid, ask = maker.quote(t)
        if bid >= ask:
            raise ValueError("maker quoted a crossed market (bid >= ask)")

        informed = bool(rng.random() < informed_share)
        if informed:
            signal = v + rng.normal(0.0, signal_noise)
            side = 1 if signal > ask else (-1 if signal < bid else 0)
        else:
            side = 1 if rng.random() < 0.5 else -1

        if side != 0:
            price = ask if side == 1 else bid
            maker.on_trade(Trade(t, side, price, informed))

        inventory_path[t + 1] = maker.inventory
        pnl_path[t + 1] = maker.pnl(v)

    return GameResult(
        true_value=v,
        final_pnl=maker.pnl(v),
        n_trades=len(maker.trades),
        n_informed_trades=sum(tr.informed for tr in maker.trades),
        inventory_path=inventory_path,
        pnl_path=pnl_path,
    )


def tournament(
    maker_factory,
    n_games: int = 2_000,
    n_rounds: int = 100,
    informed_share: float = 0.3,
    signal_noise: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Final P&L across many independent games for a maker strategy.

    maker_factory: zero-argument callable returning a fresh maker.
    Returns an array of final P&Ls, one per game.
    """
    rng = rng or np.random.default_rng()
    pnls = np.empty(n_games)
    for i in range(n_games):
        result = play_game(
            maker_factory(), n_rounds=n_rounds,
            informed_share=informed_share, signal_noise=signal_noise,
            rng=rng,
        )
        pnls[i] = result.final_pnl
    return pnls
