"""Hebel-Test auf dem Brute-Force-Pair-System (Top-3 Coins).

Selektiert die Top-10 Paare einmal auf Optimization-Set,
testet dann das Voting-Ensemble auf Holdout mit verschiedenen Hebeln.
"""
import itertools
import warnings

import numpy as np
import pandas as pd

from scripts.run_brute_force_pairs import (
    BULL_RULES, BEAR_RULES, COINS, RESAMPLE_TO, INPUT_TIMEFRAME,
    EXCHANGE, TP_PCT, SL_PCT, MAX_HOLD_BARS, COOLDOWN, FEE_RATE,
    OPT_FRAC, TOP_N_PAIRS, MIN_VOTES_PCT, MIN_TRADES_PER_COIN,
    STARTING_CAPITAL_CHF, DAILY_LOSS_LIMIT,
    aggregate_to, compute_indicators, evaluate_pair_on_coin,
)
from src.data.storage import load_ohlcv
from src.backtest.engine import triple_barrier_backtest_ls

warnings.filterwarnings("ignore", category=UserWarning)

LEVERAGES = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]


def main():
    print("Lade Coins, finde Top-Paare (einmal) …")
    coins = {}
    for sym in COINS:
        df_1h = load_ohlcv(EXCHANGE, sym, INPUT_TIMEFRAME)
        df = aggregate_to(df_1h, RESAMPLE_TO)
        ind = compute_indicators(df)
        df = pd.concat([df, ind], axis=1).dropna()
        coins[sym] = df

    common_start = max(df.index.min() for df in coins.values())
    common_end = min(df.index.max() for df in coins.values())
    for sym in COINS:
        coins[sym] = coins[sym].loc[common_start:common_end]
    n = len(next(iter(coins.values())))
    opt_end = int(n * OPT_FRAC)

    indicators = list(BULL_RULES.keys())
    pairs = list(itertools.combinations(indicators, 2))
    pair_results = []
    for ind_a, ind_b in pairs:
        per_coin = {}
        valid = True
        for sym in COINS:
            df_opt = coins[sym].iloc[:opt_end]
            res = evaluate_pair_on_coin(df_opt["close"], df_opt, ind_a, ind_b)
            per_coin[sym] = res
            if res["n_trades"] < MIN_TRADES_PER_COIN:
                valid = False
                break
        if not valid:
            continue
        returns = [per_coin[s]["total_return"] for s in COINS]
        pair_results.append({
            "ind_a": ind_a, "ind_b": ind_b,
            "avg_return": float(np.mean(returns)),
            "min_return": float(min(returns)),
        })

    pair_df = pd.DataFrame(pair_results)
    robust = pair_df[pair_df["min_return"] > 0].sort_values("avg_return", ascending=False)
    top_pairs = robust.head(TOP_N_PAIRS)
    print(f"Top-{len(top_pairs)} robuste Paare gefunden:")
    for _, r in top_pairs.iterrows():
        print(f"  {r['ind_a']}+{r['ind_b']}  avg {r['avg_return']*100:+.2f}%")

    # Backtest pro Hebel
    print(f"\n{'='*100}")
    print("HEBEL-TEST auf Holdout mit Top-Paare-Ensemble:")
    print("=" * 100)
    print(f"  {'Hebel':>5s} | {'Endkap.':>10s} {'Return':>9s} {'Best Day':>9s} {'Worst Day':>10s} "
          f"{'MaxDD':>8s} {'Trades':>6s} {'WR':>5s} {'Liq':>4s}")

    summary = []
    for lev in LEVERAGES:
        total_final = 0
        all_trades = []
        all_eq = []
        for sym in COINS:
            df_h = coins[sym].iloc[opt_end:]
            long_votes = pd.Series(0, index=df_h.index)
            short_votes = pd.Series(0, index=df_h.index)
            for _, r in top_pairs.iterrows():
                a, b = r["ind_a"], r["ind_b"]
                long_votes += (BULL_RULES[a](df_h[a]).fillna(False)
                               & BULL_RULES[b](df_h[b]).fillna(False)).astype(int)
                short_votes += (BEAR_RULES[a](df_h[a]).fillna(False)
                                & BEAR_RULES[b](df_h[b]).fillna(False)).astype(int)
            p_long = long_votes / len(top_pairs)
            p_short = short_votes / len(top_pairs)
            pos_frac = np.where((p_long > MIN_VOTES_PCT) | (p_short > MIN_VOTES_PCT),
                                np.maximum(p_long, p_short).values * 0.30, 0.0)
            res = triple_barrier_backtest_ls(
                close=df_h["close"], p_tp_long=p_long, p_tp_short=p_short,
                entry_threshold=MIN_VOTES_PCT,
                tp_pct=TP_PCT, sl_pct=SL_PCT, max_hold=MAX_HOLD_BARS,
                fee_rate=FEE_RATE, starting_capital=STARTING_CAPITAL_CHF / len(COINS),
                cooldown=COOLDOWN, position_fraction=pos_frac,
                leverage=lev, daily_loss_limit=DAILY_LOSS_LIMIT,
            )
            total_final += res["final_capital"]
            all_trades.append(res["trades"])
            all_eq.append(res["equity"])

        # Portfolio-Equity-Kurve approximieren via Sum der einzelnen Equities
        combined_eq = pd.concat(all_eq, axis=1).ffill().sum(axis=1)
        max_dd = (combined_eq / combined_eq.cummax() - 1).min()
        all_t = pd.concat(all_trades) if any(len(t) for t in all_trades) else pd.DataFrame()
        wr = (all_t["trade_net_return"] > 0).mean() if len(all_t) else 0
        liq = (all_t["reason"] == "LIQUIDATED").sum() if len(all_t) else 0
        # Tages-Returns aus combined_eq
        daily = combined_eq.pct_change().dropna()
        best_d = daily.max() if len(daily) else 0
        worst_d = daily.min() if len(daily) else 0

        summary.append({
            "lev": lev, "final": total_final,
            "ret": total_final / STARTING_CAPITAL_CHF - 1,
            "best_d": best_d, "worst_d": worst_d,
            "max_dd": max_dd, "trades": len(all_t),
            "wr": wr, "liq": liq,
        })

        print(f"  {lev:>4.1f}x | {total_final:>10.2f} {(total_final/STARTING_CAPITAL_CHF-1)*100:>+8.2f}% "
              f"{best_d*100:>+8.2f}% {worst_d*100:>+9.2f}% "
              f"{max_dd*100:>+7.2f}% {len(all_t):>6d} {wr*100:>4.1f}% {liq:>4d}")

    # B&H Vergleich
    bh_total = 0
    for sym in COINS:
        df_h = coins[sym].iloc[opt_end:]
        bh_total += (STARTING_CAPITAL_CHF / len(COINS)) * (df_h["close"].iloc[-1] / df_h["close"].iloc[0])
    print(f"\n  B&H Portfolio (1x): {bh_total:.2f} CHF "
          f"({(bh_total/STARTING_CAPITAL_CHF-1)*100:+.2f} %)")

    # Sweet-Spot-Analyse
    best = max(summary, key=lambda x: x["ret"])
    print(f"\n--- SWEET SPOT ---")
    print(f"  Bester Hebel: {best['lev']:.1f}x  →  {best['ret']*100:+.2f} %  "
          f"(MaxDD {best['max_dd']*100:.1f} %)")
    # Risiko-adjusted: Return / |MaxDD|
    for r in summary:
        if r["max_dd"] != 0:
            r["risk_adj"] = r["ret"] / abs(r["max_dd"])
    best_risk = max(summary, key=lambda x: x.get("risk_adj", 0))
    print(f"  Bester Risk-Adjusted: {best_risk['lev']:.1f}x  →  "
          f"Return {best_risk['ret']*100:+.2f} % / |DD| = {best_risk.get('risk_adj', 0):.2f}")


if __name__ == "__main__":
    main()
