"""USD/CHF VOLA-Filter mit verschiedenen Hebel-Stufen."""
import warnings
import numpy as np
import pandas as pd

from scripts.usdchf_voting_filters import (
    prepare_data, build_signals, STARTING_CAPITAL_CHF, FEE_RATE,
    TRAILING_MULT,
)
from src.backtest.trend import trend_following_backtest

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    df = prepare_data()
    n_years = (df.index[-1] - df.index[0]).days / 365.25
    long_sig, short_sig = build_signals(df, "VOLA")
    print(f"VOLA-Filter: {int(long_sig.sum())} Long + {int(short_sig.sum())} Short = "
          f"{int(long_sig.sum() + short_sig.sum())} Signale über {n_years:.1f} Jahre\n")
    print(f"  {'Pos':>4s} {'Hebel':>5s} {'Endkap.':>10s} {'Return':>9s} {'CAGR':>8s} {'MaxDD':>8s} {'WR':>5s}")

    for pos in [0.40, 0.60, 0.80]:
        for lev in [2.0, 3.0, 5.0, 7.0, 10.0, 20.0]:
            res = trend_following_backtest(
                df=df, long_entry=long_sig, short_entry=short_sig,
                atr=df["atr"], starting_capital=STARTING_CAPITAL_CHF,
                position_fraction=pos, fee_rate=FEE_RATE,
                leverage=lev, atr_mult=TRAILING_MULT, sma_exit=None,
            )
            eq = res["equity"]
            max_dd = (eq / eq.cummax() - 1).min() if len(eq) else 0
            trades = res["trades"]
            wr = (trades["trade_net_return"] > 0).mean() if len(trades) else 0
            cagr = (res["final_capital"] / STARTING_CAPITAL_CHF) ** (1 / n_years) - 1
            liq = (trades["reason"] == "LIQUIDATED").sum() if len(trades) else 0
            liq_marker = f" LIQ={liq}" if liq > 0 else ""
            print(f"  {pos*100:>3.0f}% {lev:>4.1f}x {res['final_capital']:>10.2f} "
                  f"{res['total_return']*100:>+8.2f}% {cagr*100:>+7.2f}% "
                  f"{max_dd*100:>+7.2f}% {wr*100:>4.1f}%{liq_marker}")
        print()


if __name__ == "__main__":
    main()
