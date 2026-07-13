# Kelly Criterion Betting Simulator

A research project on optimal bet sizing and market making under
uncertainty: how much to bet when you have an edge, what happens when
that edge is only estimated, and how quoting prices exposes you to
traders who know more than you do.

## Day 1 -- Kelly with a known edge

Even-money bet, true win probability p = 0.55 (Kelly optimal f* = 0.10),
1,000 sequential bets, 20,000 Monte Carlo paths:

| Strategy | Median final bankroll | Mean final bankroll | P(losing >50%) |
|---|---:|---:|---:|
| Fixed 2% | 6.05 | 7.41 | 0.000 |
| Half Kelly (f = 0.05) | 42.6 | 142 | 0.003 |
| Full Kelly (f = 0.10) | 150 | 15,006 | 0.037 |
| Double Kelly (f = 0.20) | **0.87** | 23.9M | **0.457** |

Key observations:

- **Full Kelly maximises median (typical) growth**, consistent with theory.
- **Over-betting is catastrophic in the typical case**: double Kelly has a
  median *below the starting bankroll* and a ~46% chance of losing half the
  bankroll -- despite an enormous *mean*, driven entirely by a tiny number of
  extreme lucky paths. Mean outcomes are deeply misleading for multiplicative
  wealth processes.
- **Half Kelly gives up surprisingly little growth for a large cut in risk**,
  which is why practitioners rarely bet full Kelly.

![Log-growth vs bet fraction](figures/growth_curve.png)
![Bankroll paths](figures/bankroll_paths.png)

## Day 2 -- Kelly with an estimated edge

Real bettors never know p; they estimate it. Here the bettor observes a
finite history of outcomes, estimates the win rate, and bets
m x Kelly(estimate) against the true probability.

**Noise study.** The growth-maximising Kelly multiplier m falls as the
estimate gets noisier:

| Estimate quality | Best multiplier | Growth at best |
|---|---:|---:|
| Known edge (theory) | 1.00 | 0.00501 |
| Estimated from 1,000 bets | 0.90 | 0.00459 |
| Estimated from 200 bets | 0.65 | 0.00360 |
| Estimated from 50 bets | 0.45 | 0.00263 |

**Bias study.** A bettor who *believes* p = 0.55 when the truth is 0.52
(3 points of overconfidence) realises **negative** growth betting full
Kelly on the believed edge (-0.0035 per bet), while a multiplier around
0.3 remains profitable.

Why: the log-growth curve is asymmetric around its peak -- over-betting is
punished far more than under-betting -- so estimation error pushes optimal
sizing below full Kelly. This is the quantitative case for fractional
Kelly, and for conservatism whenever your edge is self-estimated.

![Noisy edge](figures/noisy_edge_multiplier.png)
![Biased edge](figures/biased_edge_multiplier.png)

## Day 3 -- Market making against informed flow

A market-making game: a hidden value V (sum of three dice, mean 10.5)
and a maker quoting bid/ask each round. Noise traders (buy/sell at
random) pay the spread; informed traders (see V plus noise) only trade
when the maker's quotes are wrong. Positions settle at the true V.

A naive maker quoting a fixed +-1.0 spread around the prior, across
2,000 games of 100 rounds:

| Informed share | Mean P&L | Median P&L | P(loss) |
|---:|---:|---:|---:|
| 0% | 98.4 | 100.0 | 0.005 |
| 15% | 63.6 | 77.2 | 0.070 |
| 30% | 27.7 | 45.5 | 0.255 |
| 45% | **-9.0** | 19.0 | 0.445 |
| 60% | **-48.4** | -24.0 | 0.533 |

This is **adverse selection**: spread income from noise traders is a
roughly constant revenue stream, while losses to informed traders scale
with how often the maker's quotes are far from the truth. In the example
game below, V = 3 -- far below the prior of 10.5 -- and informed traders
sell to the maker's bid all game long while it never updates its view.

![Example game](figures/example_game.png)
![Informed share sweep](figures/informed_share_sweep.png)

The obvious fix -- update your fair-value estimate from the flow hitting
you, and skew quotes with inventory -- is Day 4.

## Usage

Install dependencies and run the experiments:

    pip install -r requirements.txt
    python scripts/run_bankroll_sim.py       # Day 1: fraction comparison
    python scripts/run_estimation_study.py   # Day 2: estimated-edge studies
    python scripts/run_market_game.py        # Day 3: adverse selection

Run the test suite:

    python -m pytest

## Roadmap

- [x] Core Kelly criterion + expected log-growth
- [x] Vectorised bankroll path simulation with absorbing ruin
- [x] Fraction comparison study (fixed / half / full / double Kelly)
- [x] Kelly under parameter uncertainty (noisy and biased edge estimates)
- [x] Fractional Kelly as a robustness result
- [x] Market-making game engine with informed and noise flow
- [x] Adverse selection study for a fixed-spread maker
- [ ] Inventory-aware and Bayesian market-making strategies
- [ ] Strategy tournament: P&L vs informed share by strategy
- [ ] Human-playable CLI mode
- [ ] Simultaneous correlated bets (portfolio Kelly)

## References

- Kelly, J. L. (1956). *A New Interpretation of Information Rate.*
- Thorp, E. O. (2006). *The Kelly Criterion in Blackjack, Sports Betting, and
  the Stock Market.*
- MacLean, Thorp & Ziemba (2011). *The Kelly Capital Growth Investment
  Criterion.*
- Glosten, L. & Milgrom, P. (1985). *Bid, Ask and Transaction Prices in a
  Specialist Market with Heterogeneously Informed Traders.*
