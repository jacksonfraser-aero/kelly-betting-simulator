# Kelly Criterion Betting Simulator

A research project on optimal bet sizing under uncertainty, built as the first
stage of a betting / market-making strategy engine.

**Motivation.** The Kelly criterion (Kelly, 1956) maximises the long-run
exponential growth rate of wealth for a repeated favourable bet. It is central
to both professional betting and position sizing in quantitative trading. This
project implements Kelly sizing from first principles, verifies the theory by
simulation, and (in later stages) studies what happens when the bettor's edge
is *estimated* rather than known -- the realistic case.

## Results so far (Day 1)

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
  median *below the starting bankroll* and a large chance of losing half the
  bankroll -- despite an enormous *mean*, which is driven entirely by a tiny
  number of extreme lucky paths. Mean outcomes are deeply misleading for
  multiplicative wealth processes.
- **Half Kelly gives up surprisingly little growth for a large cut in risk**,
  which is why practitioners rarely bet full Kelly.

![Log-growth vs bet fraction](figures/growth_curve.png)
![Bankroll paths](figures/bankroll_paths.png)

## Usage

Install dependencies and run the Day 1 experiment:

    pip install -r requirements.txt
    python scripts/run_bankroll_sim.py

Run the test suite:

    python -m pytest

## Roadmap

- [x] Core Kelly criterion + expected log-growth
- [x] Vectorised bankroll path simulation with absorbing ruin
- [x] Fraction comparison study (fixed / half / full / double Kelly)
- [ ] Kelly under parameter uncertainty (noisy and biased edge estimates)
- [ ] Fractional Kelly as a robustness result
- [ ] Simultaneous correlated bets (portfolio Kelly)
- [ ] Market-making game with informed flow (adverse selection)

## References

- Kelly, J. L. (1956). *A New Interpretation of Information Rate.*
- Thorp, E. O. (2006). *The Kelly Criterion in Blackjack, Sports Betting, and
  the Stock Market.*
- MacLean, Thorp & Ziemba (2011). *The Kelly Capital Growth Investment
  Criterion.*
