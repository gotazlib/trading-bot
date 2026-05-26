"""Walk-Forward Variante A auf ALLE 5 Forex-Pairs.

Bestätigt, ob +22% p.a. OOS auch auf EUR/USD, USD/JPY etc. funktioniert
oder USD/CHF-spezifisches Glück war.
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
POSITION_FRAC = 0.50
LEVERAGE = 7.0
HOLD_DAYS = 1

TRAIN_YEARS = 5
TEST_YEARS = 1
WR_O_VALS = [-95, -90, -85, -80]
WR_U_VALS = [-20, -15, -10, -5]

PAIRS = ["USD/CHF", "EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD"]


def backtest(df, wr_o, wr_u):
    long_sig = (df["wr"].shift(1) >= wr_o) & (df["wr"] < wr_o)
    short_sig = (df["wr"].shift(1) <= wr_u) & (df["wr"] > wr_u)
    capital = STARTING_CAPITAL_CHF
    in_pos = 0
    entry_idx = None
    entry_price = 0.0
    trades = []
    max_cap = capital
    min_cap = capital
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
            if capital > max_cap:
                max_cap = capital; min_cap = capital
            elif capital < min_cap:
                min_cap = capital
            in_pos = 0
        if in_pos == 0:
            if long_sig.iloc[i]:
                in_pos = 1; entry_idx = i; entry_price = close[i]
            elif short_sig.iloc[i]:
                in_pos = -1; entry_idx = i; entry_price = close[i]
    max_dd = (min_cap / max_cap - 1) if max_cap > 0 else 0
    return capital, trades, max_dd


def find_best(df_train):
    best_cap = -1
    best_p = None
    for wr_o, wr_u in itertools.product(WR_O_VALS, WR_U_VALS):
        cap, _, _ = backtest(df_train, wr_o, wr_u)
        if cap > best_cap:
            best_cap = cap; best_p = (wr_o, wr_u)
    return best_p


def run_pair(symbol):
    df = load_ohlcv("forex", symbol, "1d")
    df["wr"] = williams_r(df["high"], df["low"], df["close"], 14)
    df = df.dropna()
    train_days = TRAIN_YEARS * 252
    test_days = TEST_YEARS * 252
    n = len(df)
    total_cap = STARTING_CAPITAL_CHF
    yearly = []
    yearly_dds = []
    all_trades = []
    fold = 0
    fold_idx = 0
    while True:
        train_start = fold_idx * test_days
        train_end = train_start + train_days
        test_start = train_end
        test_end = test_start + test_days
        if test_end > n:
            break
        df_train = df.iloc[train_start:train_end]
        df_test = df.iloc[test_start:test_end]
        wr_o, wr_u = find_best(df_train)
        cap, trades, dd = backtest(df_test, wr_o, wr_u)
        ret = (cap / STARTING_CAPITAL_CHF) - 1
        total_cap *= (1 + ret)
        yearly.append(ret)
        yearly_dds.append(dd)
        all_trades.extend(trades)
        fold += 1
        fold_idx += 1
    cagr = (total_cap / STARTING_CAPITAL_CHF) ** (1 / fold) - 1 if fold > 0 else 0
    win_y = sum(1 for r in yearly if r > 0)
    wr = sum(1 for t in all_trades if t > 0) / len(all_trades) if all_trades else 0
    return {
        "symbol": symbol, "folds": fold, "final": total_cap, "cagr": cagr,
        "win_years": win_y, "win_rate": wr,
        "best_year": max(yearly), "worst_year": min(yearly),
        "worst_dd": min(yearly_dds), "trades": len(all_trades),
    }


def main():
    print(f"Walk-Forward Cross-Asset VARIANTE A (50% Pos × 7x Hebel)")
    print(f"  Train 5y, Test 1y, alle 5 Major Pairs\n")
    print(f"  {'Pair':>8s} {'Folds':>5s} {'Trades':>6s} {'WR':>5s} {'CAGR':>8s} "
          f"{'WinYrs':>7s} {'Best':>7s} {'Worst':>7s} {'WorstDD':>8s} {'Endkap.':>10s}")
    results = []
    for sym in PAIRS:
        r = run_pair(sym)
        results.append(r)
        print(f"  {sym:>8s} {r['folds']:>5d} {r['trades']:>6d} {r['win_rate']*100:>4.1f}% "
              f"{r['cagr']*100:>+7.2f}% {r['win_years']:>3d}/{r['folds']:<3d} "
              f"{r['best_year']*100:>+6.2f}% {r['worst_year']*100:>+6.2f}% "
              f"{r['worst_dd']*100:>+7.2f}% {r['final']:>10.2f}")

    print(f"\n=== ZUSAMMENFASSUNG ===")
    cagrs = [r["cagr"] for r in results]
    print(f"  Profitable: {sum(1 for c in cagrs if c > 0)}/{len(results)}")
    print(f"  Avg CAGR:   {np.mean(cagrs)*100:+.2f}% p. a.")
    print(f"  Min/Max:    {min(cagrs)*100:+.2f}% / {max(cagrs)*100:+.2f}%")
    print(f"  Std:        {np.std(cagrs)*100:.2f}pp")
    if all(c > 0.15 for c in cagrs):
        print(f"  → STRUKTURELL ROBUST: ALLE Pairs > +15% p.a. OOS")
    elif all(c > 0.05 for c in cagrs):
        print(f"  → ROBUST: alle Pairs > +5% p.a.")
    elif sum(1 for c in cagrs if c > 0) == len(results):
        print(f"  → POSITIV aber variabel")
    else:
        print(f"  → NICHT cross-asset robust")


if __name__ == "__main__":
    main()
