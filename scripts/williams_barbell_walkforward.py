"""Walk-Forward des Barbell-Mix (60% @ 2x + 40% @ 15x).

Pro Fold: 5 Jahre Train (WR-Schwellen finden), 1 Jahr Test mit Barbell-Allokation.
Validiert, ob die +58% Backtest-CAGR OOS hält.
"""
import warnings
import itertools
import numpy as np
import pandas as pd

from src.data.storage import load_ohlcv
from src.indicators.technical import williams_r

warnings.filterwarnings("ignore", category=UserWarning)

TOTAL_CAPITAL_CHF = 5000.0
FEE_RATE = 0.0002
HOLD_DAYS = 1
POSITION_PER_PAIR = 0.10
MAX_TOTAL_EXPOSURE = 0.60

BARBELL = [(2.0, 0.60), (15.0, 0.40)]   # (leverage, capital_share)

TRAIN_YEARS = 5
TEST_YEARS = 1
WR_O_VALS = [-95, -90, -85, -80]
WR_U_VALS = [-20, -15, -10, -5]

PAIRS = ["USD/CHF", "EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD"]


def run_tranche(coins, ts_range, leverage, start_capital, wr_o, wr_u):
    capital = start_capital
    open_pos = {}
    equity_curve = []
    for ts in ts_range:
        # Exits
        for symbol in list(open_pos.keys()):
            pos = open_pos[symbol]
            df = coins[symbol]
            if ts not in df.index:
                continue
            try:
                entry_idx = df.index.get_loc(pos["entry_ts"])
                cur_idx = df.index.get_loc(ts)
            except KeyError:
                continue
            if (cur_idx - entry_idx) >= HOLD_DAYS:
                exit_price = df.loc[ts, "close"]
                price_ret = (exit_price / pos["entry_price"] - 1.0) * pos["direction"]
                trade_net = (1 + price_ret) * (1 - FEE_RATE) ** 2 - 1.0
                port_ret = pos["position_frac"] * trade_net * leverage
                if port_ret <= -0.99:
                    port_ret = -0.99
                capital *= (1 + port_ret)
                del open_pos[symbol]

        # Mark-to-market
        unreal = 0.0
        for symbol, pos in open_pos.items():
            df = coins[symbol]
            if ts not in df.index:
                continue
            cp = df.loc[ts, "close"]
            pr = (cp / pos["entry_price"] - 1.0) * pos["direction"]
            ur = pos["position_frac"] * pr * leverage
            if ur <= -0.99:
                ur = -0.99
            unreal += capital * ur
        equity_curve.append(capital + unreal)

        # Entries
        cur_exp = sum(p["position_frac"] for p in open_pos.values())
        for symbol in PAIRS:
            if symbol in open_pos:
                continue
            df = coins[symbol]
            if ts not in df.index:
                continue
            row = df.loc[ts]
            avail = MAX_TOTAL_EXPOSURE - cur_exp
            if avail < POSITION_PER_PAIR * 0.5:
                break
            pos_frac = min(POSITION_PER_PAIR, avail)
            # Signale on the fly
            wr_prev = df["wr"].shift(1).loc[ts]
            wr_now = df["wr"].loc[ts]
            if (wr_prev >= wr_o) and (wr_now < wr_o):
                open_pos[symbol] = {"entry_ts": ts, "direction": 1,
                                    "entry_price": row["close"], "position_frac": pos_frac}
                cur_exp += pos_frac
            elif (wr_prev <= wr_u) and (wr_now > wr_u):
                open_pos[symbol] = {"entry_ts": ts, "direction": -1,
                                    "entry_price": row["close"], "position_frac": pos_frac}
                cur_exp += pos_frac

    return capital, pd.Series(equity_curve)


def find_best_params(coins, train_ts, leverage):
    best_cap = -1
    best_p = None
    for wr_o, wr_u in itertools.product(WR_O_VALS, WR_U_VALS):
        cap, _ = run_tranche(coins, train_ts, leverage, 1000.0, wr_o, wr_u)
        if cap > best_cap:
            best_cap = cap; best_p = (wr_o, wr_u)
    return best_p


def main():
    print("Lade Daten …")
    coins = {}
    for sym in PAIRS:
        df = load_ohlcv("forex", sym, "1d")
        df["wr"] = williams_r(df["high"], df["low"], df["close"], 14)
        df = df.dropna()
        coins[sym] = df
    common_start = max(df.index.min() for df in coins.values())
    common_end = min(df.index.max() for df in coins.values())
    for sym in PAIRS:
        coins[sym] = coins[sym].loc[common_start:common_end]
    all_ts = sorted(set().union(*[set(df.index) for df in coins.values()]))
    n_years = (common_end - common_start).days / 365.25

    print(f"Zeitraum: {common_start.date()} → {common_end.date()} ({n_years:.1f} Jahre)\n")
    print(f"Walk-Forward Barbell: 60% @ 2x + 40% @ 15x")
    print(f"Train 5y, Test 1y, rollende WR-Optimierung pro Fold\n")

    train_days = TRAIN_YEARS * 252
    test_days = TEST_YEARS * 252
    n = len(all_ts)

    total_capital = TOTAL_CAPITAL_CHF
    yearly_returns = []
    yearly_dds = []
    fold = 0
    fold_idx = 0
    print(f"  {'Fold':>4s} {'Test':<24s} {'WR_o':>5s} {'WR_u':>5s} "
          f"{'Tr1 (2x)':>10s} {'Tr2 (15x)':>11s} {'Mix Ret':>9s} {'MaxDD':>8s}")
    while True:
        train_start_idx = fold_idx * test_days
        train_end_idx = train_start_idx + train_days
        test_start_idx = train_end_idx
        test_end_idx = test_start_idx + test_days
        if test_end_idx > n:
            break
        train_ts = all_ts[train_start_idx:train_end_idx]
        test_ts = all_ts[test_start_idx:test_end_idx]
        # Optimierung auf Train: nutze 7x als "Referenz" (Top Hebel des Mix)
        best_params = find_best_params(coins, train_ts, 7.0)
        wr_o, wr_u = best_params

        # Test: beide Tranchen separat
        tr1_cap, tr1_eq = run_tranche(coins, test_ts, 2.0, TOTAL_CAPITAL_CHF * 0.60, wr_o, wr_u)
        tr2_cap, tr2_eq = run_tranche(coins, test_ts, 15.0, TOTAL_CAPITAL_CHF * 0.40, wr_o, wr_u)
        # Aggregiere
        combined_eq = tr1_eq.values + tr2_eq.values if len(tr1_eq) == len(tr2_eq) else None
        if combined_eq is not None:
            ceq = pd.Series(combined_eq)
            mix_dd = (ceq / ceq.cummax() - 1).min()
        else:
            mix_dd = 0
        mix_final = tr1_cap + tr2_cap
        mix_ret = mix_final / TOTAL_CAPITAL_CHF - 1
        total_capital *= (1 + mix_ret)
        yearly_returns.append(mix_ret)
        yearly_dds.append(mix_dd)
        print(f"  {fold+1:>4d} {str(test_ts[0].date())+'→'+str(test_ts[-1].date()):<24s} "
              f"{wr_o:>5d} {wr_u:>5d} {tr1_cap:>10.0f} {tr2_cap:>11.0f} "
              f"{mix_ret*100:>+8.2f}% {mix_dd*100:>+7.2f}%")
        fold += 1
        fold_idx += 1

    cagr = (total_capital / TOTAL_CAPITAL_CHF) ** (1 / fold) - 1 if fold > 0 else 0
    win_y = sum(1 for r in yearly_returns if r > 0)
    print(f"\n=== Barbell Walk-Forward Ergebnis ===")
    print(f"  OOS-Jahre:             {fold}")
    print(f"  Endkapital:            {total_capital:,.2f} CHF")
    print(f"  Total Return:          {(total_capital/TOTAL_CAPITAL_CHF-1)*100:+.2f}%")
    print(f"  CAGR:                  {cagr*100:+.2f}% p. a.")
    print(f"  Gewinn-Jahre:          {win_y}/{fold}")
    if yearly_returns:
        print(f"  Bestes Jahr:           {max(yearly_returns)*100:+.2f}%")
        print(f"  Schlimmstes Jahr:      {min(yearly_returns)*100:+.2f}%")
        print(f"  Worst Jahres-DD:       {min(yearly_dds)*100:+.2f}%")
    print(f"\n  In-Sample Vergleich:   +58.34% p. a.")
    print(f"  OOS Differenz:         {(cagr - 0.5834)*100:+.2f} pp")


if __name__ == "__main__":
    main()
