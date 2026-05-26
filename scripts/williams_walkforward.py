"""Walk-Forward Validation der Williams-Strategie auf USD/CHF.

Methodisch sauber:
- 5 Jahre Training → finde beste WR-Schwelle in [-95,-85] und [-15,-5]
- 1 Jahr Test mit gefundener Schwelle
- Rolle vorwärts
- Aggregiere alle Test-Jahre = ehrliches OOS

Wenn dies +10 % p. a. liefert, ist der Edge real.
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
POSITION_FRAC = 0.30
LEVERAGE = 5.0
HOLD_DAYS = 1

TRAIN_YEARS = 5
TEST_YEARS = 1
WR_OVERSOLD_CANDIDATES = [-95, -90, -85, -80]
WR_OVERBOUGHT_CANDIDATES = [-20, -15, -10, -5]


def backtest(df, wr_over, wr_under, hold=HOLD_DAYS):
    """Single-pair Williams backtest. Liefert Endkapital."""
    long_sig = (df["wr"].shift(1) >= wr_over) & (df["wr"] < wr_over)
    short_sig = (df["wr"].shift(1) <= wr_under) & (df["wr"] > wr_under)

    capital = STARTING_CAPITAL_CHF
    in_pos = 0
    entry_idx = None
    entry_price = 0.0
    trades = []
    close = df["close"].values

    for i in range(len(df) - 1):
        if in_pos != 0 and (i - entry_idx) >= hold:
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
                in_pos = 1
                entry_idx = i
                entry_price = close[i]
            elif short_sig.iloc[i]:
                in_pos = -1
                entry_idx = i
                entry_price = close[i]
    return capital, trades


def find_best_params(df_train):
    """Grid-Search auf Trainingsdaten."""
    best_cap = -1
    best_params = None
    for wr_o in WR_OVERSOLD_CANDIDATES:
        for wr_u in WR_OVERBOUGHT_CANDIDATES:
            cap, _ = backtest(df_train, wr_o, wr_u)
            if cap > best_cap:
                best_cap = cap
                best_params = (wr_o, wr_u)
    return best_params, best_cap


def main():
    df = load_ohlcv("forex", "USD/CHF", "1d")
    df["wr"] = williams_r(df["high"], df["low"], df["close"], 14)
    df = df.dropna()

    train_days = TRAIN_YEARS * 252
    test_days = TEST_YEARS * 252
    n = len(df)

    print(f"USD/CHF Walk-Forward Williams")
    print(f"  Daten: {n} Bars von {df.index[0].date()} bis {df.index[-1].date()}")
    print(f"  Train: 5 Jahre, Test: 1 Jahr, rolle vorwärts\n")

    all_returns = []
    fold = 0
    total_cap = STARTING_CAPITAL_CHF
    yearly_results = []
    fold_idx = 0
    print(f"  {'Fold':>4s} {'Train':<22s} {'Test':<22s} {'WR_o':>5s} {'WR_u':>5s} {'Train End':>10s} {'Test Ret':>9s}")
    while True:
        train_start = fold_idx * test_days
        train_end = train_start + train_days
        test_start = train_end
        test_end = test_start + test_days
        if test_end > n:
            break
        df_train = df.iloc[train_start:train_end]
        df_test = df.iloc[test_start:test_end]

        best_params, best_train_cap = find_best_params(df_train)
        wr_o, wr_u = best_params

        # Test mit gefundenen Parametern, neues Kapital startet bei total_cap
        capital = STARTING_CAPITAL_CHF
        in_pos = 0
        entry_idx = None
        entry_price = 0.0
        close = df_test["close"].values
        long_sig = (df_test["wr"].shift(1) >= wr_o) & (df_test["wr"] < wr_o)
        short_sig = (df_test["wr"].shift(1) <= wr_u) & (df_test["wr"] > wr_u)
        for i in range(len(df_test) - 1):
            if in_pos != 0 and (i - entry_idx) >= HOLD_DAYS:
                price_ret = (close[i] / entry_price - 1.0) * in_pos
                trade_net = (1 + price_ret) * (1 - FEE_RATE) ** 2 - 1.0
                port_ret = POSITION_FRAC * trade_net * LEVERAGE
                if port_ret <= -0.99:
                    port_ret = -0.99
                capital *= (1 + port_ret)
                all_returns.append(port_ret)
                in_pos = 0
            if in_pos == 0:
                if long_sig.iloc[i]:
                    in_pos = 1
                    entry_idx = i
                    entry_price = close[i]
                elif short_sig.iloc[i]:
                    in_pos = -1
                    entry_idx = i
                    entry_price = close[i]
        test_ret = (capital / STARTING_CAPITAL_CHF) - 1
        total_cap *= (1 + test_ret)
        yearly_results.append(test_ret)
        print(f"  {fold+1:>4d} {str(df_train.index[0].date())+'→'+str(df_train.index[-1].date()):<22s} "
              f"{str(df_test.index[0].date())+'→'+str(df_test.index[-1].date()):<22s} "
              f"{wr_o:>5d} {wr_u:>5d} {best_train_cap:>10.2f} {test_ret*100:>+8.2f}%")
        fold += 1
        fold_idx += 1

    print(f"\n=== Walk-Forward-Ergebnis ===")
    print(f"  Anzahl OOS-Test-Jahre:   {fold}")
    print(f"  Endkapital (compound):   {total_cap:.2f} CHF")
    print(f"  Total Return:            {(total_cap/STARTING_CAPITAL_CHF-1)*100:+.2f}%")
    cagr = (total_cap / STARTING_CAPITAL_CHF) ** (1 / fold) - 1 if fold > 0 else 0
    print(f"  CAGR:                    {cagr*100:+.2f}% p. a.")
    if all_returns:
        wr = sum(1 for r in all_returns if r > 0) / len(all_returns)
        print(f"  Trades gesamt OOS:       {len(all_returns)}")
        print(f"  Win-Rate OOS:            {wr*100:.1f}%")
    win_years = sum(1 for r in yearly_results if r > 0)
    print(f"  Gewinn-Jahre OOS:        {win_years}/{fold}")
    if yearly_results:
        print(f"  Bestes Jahr:             {max(yearly_results)*100:+.2f}%")
        print(f"  Schlimmstes Jahr:        {min(yearly_results)*100:+.2f}%")


if __name__ == "__main__":
    main()
