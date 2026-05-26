"""20-Jahre Forex Walk-Forward Brute-Force-Analyse.

Methodisch sauber:
- 5 Jahre Training → 1 Jahr Test → rolle vorwärts
- 16 Iterationen → 16 Jahre OOS-Trade-Liste
- Pro Iteration: Brute-Force Top-5 Paare auf Train, teste auf Holdout
- Hebel-Vergleich am Ende
"""
import itertools
import warnings

import numpy as np
import pandas as pd

from src.data.storage import load_ohlcv
from src.backtest.engine import triple_barrier_backtest_ls
from scripts.run_forex_brute_force import (
    BULL_RULES, BEAR_RULES, compute_indicators, evaluate_pair_on_pair,
    TP_PCT, SL_PCT, MAX_HOLD_BARS, FEE_RATE, COOLDOWN, MIN_VOTES_PCT,
)

warnings.filterwarnings("ignore", category=UserWarning)

EXCHANGE = "forex"
TIMEFRAME = "1d"
PAIRS = ["USD/CHF", "EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD"]

STARTING_CAPITAL_CHF = 1000.0
TRAIN_YEARS = 5
TEST_YEARS = 1
TOP_N_PAIRS = 5
MIN_TRADES_PER_PAIR = 5
DAILY_LOSS_LIMIT = -0.01

LEVERAGES_TO_TEST = [2.0, 5.0, 10.0, 20.0]


def main():
    print(f"Lade {len(PAIRS)} Forex Pairs für 20-Jahre Walk-Forward …")
    coins = {}
    for sym in PAIRS:
        df = load_ohlcv(EXCHANGE, sym, TIMEFRAME)
        ind = compute_indicators(df)
        df = pd.concat([df, ind], axis=1).dropna()
        coins[sym] = df
        print(f"  {sym}: {len(df)} Bars von {df.index.min().date()} bis {df.index.max().date()}")

    common_start = max(df.index.min() for df in coins.values())
    common_end = min(df.index.max() for df in coins.values())
    for sym in coins:
        coins[sym] = coins[sym].loc[common_start:common_end]
    print(f"\nGemeinsamer Zeitraum: {common_start.date()} → {common_end.date()}")
    total_years = (common_end - common_start).days / 365
    print(f"Total: {total_years:.1f} Jahre")

    # Walk-Forward Folds
    train_days = TRAIN_YEARS * 252
    test_days = TEST_YEARS * 252
    n = len(next(iter(coins.values())))
    folds = []
    fold_idx = 0
    while True:
        train_start = fold_idx * test_days
        train_end = train_start + train_days
        test_start = train_end
        test_end = test_start + test_days
        if test_end > n:
            break
        folds.append((train_start, train_end, test_start, test_end))
        fold_idx += 1
    print(f"\nWalk-Forward: {len(folds)} Folds × 5 Pairs × 210 Paare = {len(folds)*5*210:,} Backtests")

    indicators = list(BULL_RULES.keys())
    all_pairs = list(itertools.combinations(indicators, 2))

    # Pro Fold: finde Top-5 Paare auf Train, sammle Holdout-Signale fuer alle Pairs
    holdout_signals = {sym: pd.DataFrame() for sym in PAIRS}
    print(f"\n{'Fold':>4s} {'Train':<22s} {'Test':<22s} {'Top Pair':<32s} {'AvgRet':>8s}")
    for fi, (ts, te, hs, he) in enumerate(folds):
        train_data = {sym: coins[sym].iloc[ts:te] for sym in PAIRS}
        test_data = {sym: coins[sym].iloc[hs:he] for sym in PAIRS}

        # Brute-Force auf Train
        pair_scores = []
        for ind_a, ind_b in all_pairs:
            per_fx = {}
            valid = True
            for sym in PAIRS:
                r = evaluate_pair_on_pair(train_data[sym]["close"], train_data[sym], ind_a, ind_b)
                per_fx[sym] = r
                if r["n_trades"] < MIN_TRADES_PER_PAIR:
                    valid = False
                    break
            if not valid:
                continue
            returns = [per_fx[s]["total_return"] for s in PAIRS]
            min_ret = min(returns)
            avg_ret = float(np.mean(returns))
            pair_scores.append({"ind_a": ind_a, "ind_b": ind_b,
                                "avg_return": avg_ret, "min_return": min_ret})

        pair_df = pd.DataFrame(pair_scores)
        robust = pair_df[pair_df["min_return"] > 0].sort_values("avg_return", ascending=False)
        if len(robust) == 0:
            top_pairs = pd.DataFrame()
            top_pair_str = "(keine robusten Paare)"
            avg_ret_str = "—"
        else:
            top_pairs = robust.head(TOP_N_PAIRS)
            top_pair_str = f"{top_pairs.iloc[0]['ind_a']}+{top_pairs.iloc[0]['ind_b']}"
            avg_ret_str = f"{top_pairs.iloc[0]['avg_return']*100:>+6.2f}%"

        train_start_d = train_data[PAIRS[0]].index[0].date()
        train_end_d = train_data[PAIRS[0]].index[-1].date()
        test_start_d = test_data[PAIRS[0]].index[0].date()
        test_end_d = test_data[PAIRS[0]].index[-1].date()
        print(f"{fi+1:>4d} {str(train_start_d)+'-'+str(train_end_d):<22s} "
              f"{str(test_start_d)+'-'+str(test_end_d):<22s} "
              f"{top_pair_str:<32s} {avg_ret_str:>8s}")

        # Wende Top-Paare auf Holdout an
        for sym in PAIRS:
            df_h = test_data[sym]
            if df_h.empty or len(top_pairs) == 0:
                continue
            long_votes = pd.Series(0, index=df_h.index)
            short_votes = pd.Series(0, index=df_h.index)
            for _, r in top_pairs.iterrows():
                a, b = r["ind_a"], r["ind_b"]
                long_votes += (BULL_RULES[a](df_h[a]).fillna(False)
                               & BULL_RULES[b](df_h[b]).fillna(False)).astype(int)
                short_votes += (BEAR_RULES[a](df_h[a]).fillna(False)
                                & BEAR_RULES[b](df_h[b]).fillna(False)).astype(int)
            df_h = df_h.copy()
            df_h["p_long"] = long_votes / len(top_pairs)
            df_h["p_short"] = short_votes / len(top_pairs)
            holdout_signals[sym] = pd.concat([holdout_signals[sym], df_h])

    # Holdout-Backtest mit verschiedenen Hebeln
    print(f"\n{'='*90}")
    print(f"=== HOLDOUT-BACKTEST ({len(folds)} Test-Jahre) — Hebel-Vergleich ===")
    print(f"{'='*90}")
    print(f"  {'Hebel':>5s} | {'Endkap.':>10s} {'Return':>9s} {'Trades':>6s} {'WR':>5s} {'MaxDD':>8s}")

    bh_total = 0
    for sym in PAIRS:
        df_h = holdout_signals[sym]
        if df_h.empty:
            continue
        cap0 = STARTING_CAPITAL_CHF / len(PAIRS)
        bh_total += cap0 * (df_h["close"].iloc[-1] / df_h["close"].iloc[0])

    for lev in LEVERAGES_TO_TEST:
        total_final = 0
        all_trades = []
        all_eq = []
        for sym in PAIRS:
            df_h = holdout_signals[sym]
            if df_h.empty:
                continue
            pos_frac = np.where(
                (df_h["p_long"] > MIN_VOTES_PCT) | (df_h["p_short"] > MIN_VOTES_PCT),
                np.maximum(df_h["p_long"], df_h["p_short"]).values * 0.30, 0.0
            )
            cap0 = STARTING_CAPITAL_CHF / len(PAIRS)
            res = triple_barrier_backtest_ls(
                close=df_h["close"], p_tp_long=df_h["p_long"], p_tp_short=df_h["p_short"],
                entry_threshold=MIN_VOTES_PCT,
                tp_pct=TP_PCT, sl_pct=SL_PCT, max_hold=MAX_HOLD_BARS,
                fee_rate=FEE_RATE, starting_capital=cap0,
                cooldown=COOLDOWN, position_fraction=pos_frac,
                leverage=lev, daily_loss_limit=DAILY_LOSS_LIMIT,
            )
            total_final += res["final_capital"]
            all_trades.append(res["trades"])
            all_eq.append(res["equity"])

        combined_eq = pd.concat(all_eq, axis=1).ffill().sum(axis=1) if all_eq else pd.Series([STARTING_CAPITAL_CHF])
        max_dd = (combined_eq / combined_eq.cummax() - 1).min() if len(combined_eq) > 1 else 0
        all_t = pd.concat(all_trades) if any(len(t) for t in all_trades) else pd.DataFrame()
        wr = (all_t["trade_net_return"] > 0).mean() if len(all_t) else 0
        print(f"  {lev:>4.1f}x | {total_final:>10.2f} "
              f"{(total_final/STARTING_CAPITAL_CHF-1)*100:>+8.2f}% "
              f"{len(all_t):>6d} {wr*100:>4.1f}% {max_dd*100:>+7.2f}%")

    print(f"\n  B&H Portfolio: {bh_total:.2f} CHF ({(bh_total/STARTING_CAPITAL_CHF-1)*100:+.2f}%)")


if __name__ == "__main__":
    main()
