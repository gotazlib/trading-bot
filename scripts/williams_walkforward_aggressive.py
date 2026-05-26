"""Walk-Forward Variante A: 50% Position × 7x Hebel auf USD/CHF.

Letzter Check vor Live: bleibt die +24% p.a. auch OOS bestehen?
"""
import warnings
import numpy as np
import pandas as pd

from src.data.storage import load_ohlcv
from src.indicators.technical import williams_r

warnings.filterwarnings("ignore", category=UserWarning)

STARTING_CAPITAL_CHF = 1000.0
FEE_RATE = 0.0002

# Variante A — aggressive Einstellungen
POSITION_FRAC = 0.50
LEVERAGE = 7.0
HOLD_DAYS = 1

TRAIN_YEARS = 5
TEST_YEARS = 1
WR_OVERSOLD_CANDIDATES = [-95, -90, -85, -80]
WR_OVERBOUGHT_CANDIDATES = [-20, -15, -10, -5]


def backtest(df, wr_o, wr_u, hold=HOLD_DAYS, pos=POSITION_FRAC, lev=LEVERAGE):
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
        if in_pos != 0 and (i - entry_idx) >= hold:
            price_ret = (close[i] / entry_price - 1.0) * in_pos
            trade_net = (1 + price_ret) * (1 - FEE_RATE) ** 2 - 1.0
            port_ret = pos * trade_net * lev
            if port_ret <= -0.99:
                port_ret = -0.99
            capital *= (1 + port_ret)
            trades.append(port_ret)
            if capital > max_cap:
                max_cap = capital
                min_cap = capital
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


def find_best_params(df_train):
    best_cap = -1
    best_params = None
    for wr_o in WR_OVERSOLD_CANDIDATES:
        for wr_u in WR_OVERBOUGHT_CANDIDATES:
            cap, _, _ = backtest(df_train, wr_o, wr_u)
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

    print(f"USD/CHF Walk-Forward Williams VARIANTE A (50% Pos × 7x Hebel)")
    print(f"  Daten: {n} Bars von {df.index[0].date()} bis {df.index[-1].date()}\n")
    print(f"  {'Fold':>4s} {'Train':<22s} {'Test':<22s} "
          f"{'WR_o':>5s} {'WR_u':>5s} {'Test Ret':>9s} {'MaxDD':>8s}")

    all_returns = []
    yearly_results = []
    yearly_dds = []
    total_cap = STARTING_CAPITAL_CHF
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
        best_params, _ = find_best_params(df_train)
        wr_o, wr_u = best_params

        test_cap, test_trades, test_dd = backtest(df_test, wr_o, wr_u)
        test_ret = (test_cap / STARTING_CAPITAL_CHF) - 1
        total_cap *= (1 + test_ret)
        yearly_results.append(test_ret)
        yearly_dds.append(test_dd)
        all_returns.extend(test_trades)
        print(f"  {fold+1:>4d} {str(df_train.index[0].date())+'→'+str(df_train.index[-1].date()):<22s} "
              f"{str(df_test.index[0].date())+'→'+str(df_test.index[-1].date()):<22s} "
              f"{wr_o:>5d} {wr_u:>5d} {test_ret*100:>+8.2f}% {test_dd*100:>+7.2f}%")
        fold += 1
        fold_idx += 1

    print(f"\n=== Walk-Forward-Ergebnis VARIANTE A ===")
    print(f"  OOS-Jahre:               {fold}")
    print(f"  Endkapital (compound):   {total_cap:.2f} CHF")
    print(f"  Total Return:            {(total_cap/STARTING_CAPITAL_CHF-1)*100:+.2f}%")
    cagr = (total_cap / STARTING_CAPITAL_CHF) ** (1 / fold) - 1 if fold > 0 else 0
    print(f"  CAGR:                    {cagr*100:+.2f}% p. a.")
    wr_overall = sum(1 for r in all_returns if r > 0) / len(all_returns) if all_returns else 0
    print(f"  Trades OOS gesamt:       {len(all_returns)}")
    print(f"  Win-Rate OOS:            {wr_overall*100:.1f}%")
    win_y = sum(1 for r in yearly_results if r > 0)
    print(f"  Gewinn-Jahre:            {win_y}/{fold}")
    if yearly_results:
        print(f"  Bestes Jahr:             {max(yearly_results)*100:+.2f}%")
        print(f"  Schlimmstes Jahr:        {min(yearly_results)*100:+.2f}%")
        print(f"  Schlimmster Jahres-DD:   {min(yearly_dds)*100:+.2f}%")

    # Vergleich mit IN-SAMPLE Variante A (+24.52%)
    print(f"\n=== VERGLEICH ===")
    print(f"  In-Sample CAGR (Sweep):   +24.52 % p. a.")
    print(f"  OOS Walk-Forward CAGR:    {cagr*100:+.2f} % p. a.")
    diff = cagr - 0.2452
    print(f"  Differenz:                {diff*100:+.2f} pp")
    if cagr >= 0.20:
        print(f"  → EXZELLENT: Variante A haelt OOS")
    elif cagr >= 0.15:
        print(f"  → SEHR GUT: leicht reduziert aber stabil")
    elif cagr >= 0.10:
        print(f"  → GUT: ähnlich Variante B")
    elif cagr >= 0.05:
        print(f"  → AKZEPTABEL: Edge da, aber kleiner als in-sample")
    else:
        print(f"  → WARNUNG: deutliche Abnahme, Hebel/Position evtl. zu aggressiv")


if __name__ == "__main__":
    main()
