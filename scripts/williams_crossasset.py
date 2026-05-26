"""Williams %R Mean-Reversion auf 5 Forex Major Pairs.

Wenn der Edge auf USD/CHF ECHT struktureller Effekt ist (nicht USD/CHF-spezifisches
Zufall), sollte er auf anderen Major Pairs auch funktionieren.
"""
import warnings
import numpy as np
import pandas as pd

from src.data.storage import load_ohlcv
from src.indicators.technical import williams_r

warnings.filterwarnings("ignore", category=UserWarning)

STARTING_CAPITAL_CHF = 1000.0
FEE_RATE = 0.0002
POSITION_FRAC = 0.30
LEVERAGE = 5.0
HOLD_DAYS = 1
WR_OVERSOLD = -90
WR_OVERBOUGHT = -10

PAIRS = ["USD/CHF", "EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD"]


def backtest_pair(symbol):
    df = load_ohlcv("forex", symbol, "1d")
    df["wr"] = williams_r(df["high"], df["low"], df["close"], 14)
    df = df.dropna()

    long_sig = (df["wr"].shift(1) >= WR_OVERSOLD) & (df["wr"] < WR_OVERSOLD)
    short_sig = (df["wr"].shift(1) <= WR_OVERBOUGHT) & (df["wr"] > WR_OVERBOUGHT)

    capital = STARTING_CAPITAL_CHF
    in_pos = 0
    entry_idx = None
    entry_price = 0.0
    trades = []
    close = df["close"].values

    for i in range(len(df) - 1):
        if in_pos != 0 and (i - entry_idx) >= HOLD_DAYS:
            price_ret = (close[i] / entry_price - 1.0) * in_pos
            trade_net = (1 + price_ret) * (1 - FEE_RATE) ** 2 - 1.0
            port_ret = POSITION_FRAC * trade_net * LEVERAGE
            if port_ret <= -0.99:
                port_ret = -0.99
            capital *= (1 + port_ret)
            trades.append(port_ret)
            in_pos = 0
        if in_pos == 0:
            if long_sig.iloc[i]:
                in_pos = 1; entry_idx = i; entry_price = close[i]
            elif short_sig.iloc[i]:
                in_pos = -1; entry_idx = i; entry_price = close[i]

    n_years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (capital / STARTING_CAPITAL_CHF) ** (1 / n_years) - 1
    wr_rate = sum(1 for r in trades if r > 0) / len(trades) if trades else 0
    bh = STARTING_CAPITAL_CHF * (df["close"].iloc[-1] / df["close"].iloc[0])
    return {
        "symbol": symbol, "years": n_years, "trades": len(trades),
        "final": capital, "cagr": cagr, "wr": wr_rate,
        "n_long": int(long_sig.sum()), "n_short": int(short_sig.sum()),
        "bh_final": bh,
    }


def main():
    print(f"Williams %R Cross-Asset Test (WR<{WR_OVERSOLD}, WR>{WR_OVERBOUGHT}, hold {HOLD_DAYS}d, "
          f"5x Hebel, 30% Position)\n")
    print(f"  {'Pair':>8s} {'Jahre':>5s} {'Bars':>5s} {'Long':>5s} {'Short':>5s} "
          f"{'Trades':>6s} {'WR':>5s} {'Endkap.':>10s} {'CAGR':>8s} {'B&H Vergl.':>12s}")
    results = []
    for sym in PAIRS:
        r = backtest_pair(sym)
        results.append(r)
        bh_cagr = (r["bh_final"] / STARTING_CAPITAL_CHF) ** (1 / r["years"]) - 1
        print(f"  {sym:>8s} {r['years']:>5.1f} {'-':>5s} {r['n_long']:>5d} {r['n_short']:>5d} "
              f"{r['trades']:>6d} {r['wr']*100:>4.1f}% {r['final']:>10.2f} "
              f"{r['cagr']*100:>+7.2f}% {bh_cagr*100:>+8.2f}%/yr")

    print(f"\n=== KONSISTENZ-CHECK ===")
    cagrs = [r["cagr"] for r in results]
    profitable = sum(1 for c in cagrs if c > 0)
    print(f"  Profitable Pairs: {profitable} von {len(results)}")
    print(f"  Avg CAGR: {np.mean(cagrs)*100:+.2f}% p. a.")
    print(f"  Min CAGR: {min(cagrs)*100:+.2f}% / Max CAGR: {max(cagrs)*100:+.2f}%")
    if profitable >= len(results) * 0.8:
        print(f"  → EDGE IST CROSS-ASSET KONSISTENT — strukturell robust")
    elif profitable >= len(results) // 2:
        print(f"  → Edge partiell konsistent — Vorsicht")
    else:
        print(f"  → Edge ist NICHT cross-asset robust — vermutlich USD/CHF-spezifisches Glück")


if __name__ == "__main__":
    main()
