"""Play the market-making game yourself.

You are the market maker. A hidden value V (sum of three dice, mean
10.5) has been drawn. Each round you quote a bid and an ask; a trader
arrives and may buy at your ask, sell at your bid, or pass. Some traders
are informed (they see V plus noise) -- but you never know which. Watch
the flow: if one side keeps getting hit, your quotes are probably wrong.

At the end V is revealed and your position settles. The naive bot then
plays the *same* trader sequence, so the comparison is fair.

Run from the repo root:
    python scripts/play_market_game.py
Options:
    python scripts/play_market_game.py --rounds 30 --informed 0.4
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.market_game import (
    PRIOR_MEAN,
    MarketMaker,
    NaiveMarketMaker,
    Trade,
    draw_true_value,
)


def pregenerate_flow(
    n_rounds: int, informed_share: float, signal_noise: float,
    rng: np.random.Generator,
) -> list[dict]:
    """Draw the trader sequence up front so human and bot face identical flow."""
    flow = []
    for _ in range(n_rounds):
        flow.append({
            "informed": bool(rng.random() < informed_share),
            "noise_side": 1 if rng.random() < 0.5 else -1,
            "signal_eps": float(rng.normal(0.0, signal_noise)),
        })
    return flow


def trader_action(event: dict, true_value: int, bid: float, ask: float) -> int:
    """What this trader does against these quotes: +1 buy, -1 sell, 0 pass."""
    if event["informed"]:
        signal = true_value + event["signal_eps"]
        return 1 if signal > ask else (-1 if signal < bid else 0)
    return event["noise_side"]


def replay(maker: MarketMaker, flow: list[dict], true_value: int) -> float:
    """Run a maker through a pregenerated flow; return final P&L."""
    for t, event in enumerate(flow):
        bid, ask = maker.quote(t)
        side = trader_action(event, true_value, bid, ask)
        if side != 0:
            price = ask if side == 1 else bid
            maker.on_trade(Trade(t, side, price, event["informed"]))
    return maker.pnl(true_value)


def ask_quote(prev_bid: float, prev_ask: float) -> tuple[float, float]:
    """Prompt for a bid/ask pair; enter keeps the previous quotes."""
    while True:
        raw = input(
            f"  your quote [bid ask] (enter = keep {prev_bid:.1f}/{prev_ask:.1f}): "
        ).strip()
        if raw == "":
            return prev_bid, prev_ask
        parts = raw.replace(",", " ").split()
        try:
            bid, ask = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            print("  -> enter two numbers, e.g.: 9.5 11.5")
            continue
        if bid >= ask:
            print("  -> bid must be below ask")
            continue
        return bid, ask


def main() -> None:
    parser = argparse.ArgumentParser(description="Play as the market maker.")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--informed", type=float, default=0.3,
                        help="share of informed traders (default 0.3)")
    parser.add_argument("--noise", type=float, default=1.0,
                        help="informed signal noise std (default 1.0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="set for a reproducible game")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    true_value = draw_true_value(rng)
    flow = pregenerate_flow(args.rounds, args.informed, args.noise, rng)

    print("=" * 60)
    print("MARKET MAKING GAME")
    print(f"Hidden value V = sum of three dice (3-18, mean {PRIOR_MEAN}).")
    print(f"{args.rounds} rounds. About {args.informed:.0%} of traders are")
    print("informed, but you never know which. Watch the flow.")
    print("=" * 60)

    inventory = 0
    cash = 0.0
    bid, ask = PRIOR_MEAN - 1.0, PRIOR_MEAN + 1.0

    for t, event in enumerate(flow):
        print(f"\nRound {t + 1}/{args.rounds}   "
              f"inventory: {inventory:+d}   cash: {cash:+.1f}")
        bid, ask = ask_quote(bid, ask)
        side = trader_action(event, true_value, bid, ask)
        if side == 1:
            inventory -= 1
            cash += ask
            print(f"  trader BUYS at your ask {ask:.1f}")
        elif side == -1:
            inventory += 1
            cash -= bid
            print(f"  trader SELLS at your bid {bid:.1f}")
        else:
            print("  trader passes")

    human_pnl = cash + inventory * true_value

    bot = NaiveMarketMaker(half_spread=1.0)
    bot_pnl = replay(bot, flow, true_value)

    n_informed = sum(e["informed"] for e in flow)
    print("\n" + "=" * 60)
    print(f"THE VALUE WAS  V = {true_value}")
    print(f"({n_informed} of {args.rounds} traders were informed)")
    print("-" * 60)
    print(f"you:        inventory {inventory:+d} x {true_value} "
          f"+ cash {cash:+.1f}  =  P&L {human_pnl:+.1f}")
    print(f"naive bot:  P&L {bot_pnl:+.1f}  (fixed 9.5/11.5 quotes)")
    print("-" * 60)
    if human_pnl > bot_pnl:
        print("You beat the bot -- you read the flow.")
    elif human_pnl == bot_pnl:
        print("Dead heat with the bot.")
    else:
        print("The bot wins this one. Were you slow to move your quotes?")
    print("=" * 60)


if __name__ == "__main__":
    main()
