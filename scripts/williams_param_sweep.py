"""Williams Parameter-Sweep auf USD/CHF.

Testet 4×4×4×3×4 = 768 Parameter-Kombinationen:
- WR_OVERSOLD: -95, -90, -85, -80
- WR_OVERBOUGHT: -20, -15, -10, -5
- HOLD_DAYS: 1, 2, 3, 5
- POSITION: 0.20, 0.30, 0.50
- LEVERAGE: 2, 3, 5, 7

Liefert Top-10 Setups + Stabilitäts-Analyse (sind alle Top-Setups ähnlich?).
"""
import warnings
import itertools
import numpy as np
import pandas as pd

from src.data.storage import load_ohlcv
from src.indicators.technical import williams_r

warnings.filterwarnings("ignore", category=UserWarning)

STARTING_CAPITAL_CHF = 1000.0
FEE_RATE = 0.0002

WR_OVERSOLD_VALS = [-95, -90, -85, -80]
WR_OVERBOUGHT_VALS = [-20, -15, -10, -5]
HOLD_VALS = [1, 2, 3, 5]
POS_VALS = [0.20, 0.30, 0.50]
LEV_VALS = [2.0, 3.0, 5.0, 7.0]


def backtest(df, wr_o, wr_u, hold, pos, lev):
    long_sig = (df["wr"].shift(1) >= wr_o) & (df["wr"] < wr_o)
    short_sig = (df["wr"].shift(1) <= wr_u) & (df["wr"] > wr_u)

    capital = STARTING_CAPITAL_CHF
    in_pos = 0
    entry_idx = None
    entry_price = 0.0
    n_trades = 0
    n_wins = 0
    max_cap = capital
    min_cap_after_peak = capital
    close = df["close"].values

    for i in range(len(df) - 1):
        if in_pos != 0 and (i - entry_idx) >= hold:
            price_ret = (close[i] / entry_price - 1.0) * in_pos
            trade_net = (1 + price_ret) * (1 - FEE_RATE) ** 2 - 1.0
            port_ret = pos * trade_net * lev
            if port_ret <= -0.99:
                port_ret = -0.99
            capital *= (1 + port_ret)
            n_trades += 1
            if port_ret > 0:
                n_wins += 1
            if capital > max_cap:
                max_cap = capital
                min_cap_after_peak = capital
            elif capital < min_cap_after_peak:
                min_cap_after_peak = capital
            in_pos = 0
        if in_pos == 0:
            if long_sig.iloc[i]:
                in_pos = 1; entry_idx = i; entry_price = close[i]
            elif short_sig.iloc[i]:
                in_pos = -1; entry_idx = i; entry_price = close[i]

    max_dd = (min_cap_after_peak / max_cap - 1) if max_cap > 0 else 0
    wr_rate = n_wins / n_trades if n_trades else 0
    return capital, n_trades, wr_rate, max_dd


def main():
    df = load_ohlcv("forex", "USD/CHF", "1d")
    df["wr"] = williams_r(df["high"], df["low"], df["close"], 14)
    df = df.dropna()
    n_years = (df.index[-1] - df.index[0]).days / 365.25

    print(f"USD/CHF Williams Parameter-Sweep ({n_years:.1f} Jahre, {len(df)} Bars)")
    print(f"  {len(WR_OVERSOLD_VALS) * len(WR_OVERBOUGHT_VALS) * len(HOLD_VALS) * len(POS_VALS) * len(LEV_VALS)} "
          f"Kombinationen werden getestet …\n")

    results = []
    for wr_o, wr_u, hold, pos, lev in itertools.product(
        WR_OVERSOLD_VALS, WR_OVERBOUGHT_VALS, HOLD_VALS, POS_VALS, LEV_VALS
    ):
        capital, n_trades, wr_rate, max_dd = backtest(df, wr_o, wr_u, hold, pos, lev)
        if n_trades < 50:
            continue
        cagr = (capital / STARTING_CAPITAL_CHF) ** (1 / n_years) - 1
        results.append({
            "wr_o": wr_o, "wr_u": wr_u, "hold": hold, "pos": pos, "lev": lev,
            "capital": capital, "cagr": cagr, "trades": n_trades,
            "wr": wr_rate, "max_dd": max_dd,
            "risk_adj": cagr / abs(max_dd) if max_dd != 0 else 0,
        })

    df_res = pd.DataFrame(results)
    print(f"=== TOP-10 NACH CAGR ===")
    print(f"  {'WR_O':>5s} {'WR_U':>5s} {'Hold':>4s} {'Pos':>5s} {'Lev':>4s} "
          f"{'Trades':>6s} {'WR':>5s} {'CAGR':>8s} {'MaxDD':>8s} {'RiskAdj':>8s}")
    for _, r in df_res.nlargest(10, "cagr").iterrows():
        print(f"  {r['wr_o']:>5.0f} {r['wr_u']:>5.0f} {r['hold']:>4.0f} {r['pos']*100:>4.0f}% "
              f"{r['lev']:>3.0f}x {r['trades']:>6.0f} {r['wr']*100:>4.1f}% "
              f"{r['cagr']*100:>+7.2f}% {r['max_dd']*100:>+7.2f}% {r['risk_adj']:>7.2f}")

    print(f"\n=== TOP-10 NACH RISIKO-ADJUSTIERTEM RETURN (CAGR / |MaxDD|) ===")
    print(f"  {'WR_O':>5s} {'WR_U':>5s} {'Hold':>4s} {'Pos':>5s} {'Lev':>4s} "
          f"{'Trades':>6s} {'WR':>5s} {'CAGR':>8s} {'MaxDD':>8s} {'RiskAdj':>8s}")
    for _, r in df_res.nlargest(10, "risk_adj").iterrows():
        print(f"  {r['wr_o']:>5.0f} {r['wr_u']:>5.0f} {r['hold']:>4.0f} {r['pos']*100:>4.0f}% "
              f"{r['lev']:>3.0f}x {r['trades']:>6.0f} {r['wr']*100:>4.1f}% "
              f"{r['cagr']*100:>+7.2f}% {r['max_dd']*100:>+7.2f}% {r['risk_adj']:>7.2f}")

    # Stabilitäts-Analyse: schwankt CAGR stark bei Parameter-Änderung?
    print(f"\n=== STABILITÄTS-ANALYSE ===")
    print(f"  Anzahl getestet:           {len(df_res)}")
    print(f"  Anzahl profitabel (>0%):   {(df_res['cagr'] > 0).sum()} ({(df_res['cagr'] > 0).mean()*100:.1f}%)")
    print(f"  Anzahl >5% p. a.:          {(df_res['cagr'] > 0.05).sum()}")
    print(f"  Anzahl >10% p. a.:         {(df_res['cagr'] > 0.10).sum()}")
    print(f"  Avg CAGR:                  {df_res['cagr'].mean()*100:+.2f}%")
    print(f"  Median CAGR:               {df_res['cagr'].median()*100:+.2f}%")
    print(f"  CAGR Std:                  {df_res['cagr'].std()*100:.2f}%")

    # Welche Parameter sind robust? Marginalize über andere
    print(f"\n=== BESTE EINZEL-PARAMETER (Marginalized) ===")
    for col in ["wr_o", "wr_u", "hold", "pos", "lev"]:
        marg = df_res.groupby(col)["cagr"].mean().sort_values(ascending=False)
        print(f"  {col}: " + ", ".join(f"{k}={v*100:+.2f}%" for k, v in marg.items()))


if __name__ == "__main__":
    main()
