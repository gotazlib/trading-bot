"""Mean-Reversion-Variante: Brute-Force-Pair-Analyse auf 5 Forex-Major-Pairs (2010-2026).

Anpassungen gegenueber Crypto:
- Forex hat keine echten Volumen → CMF/OBV/vol_change entfernt
- Kleinere TP/SL: Forex bewegt sich ~5x weniger als Crypto
- Niedrigere Fee: 0.02 % Spread statt 0.1 % Spot-Fee
- 16 Jahre Daten = viel mehr Trainingsdaten
- Mean-Reversion-Parameter: kleine TP/SL, kurze Haltedauer, mehr Hebel
"""
import itertools
import warnings

import numpy as np
import pandas as pd

from src.data.storage import load_ohlcv
from src.indicators.technical import (
    sma, ema, rsi, macd, bollinger_pct,
    atr, stochastic, roc, cci, williams_r, adx,
    keltner_pct, aroon, trend_strength,
)
from src.backtest.engine import triple_barrier_backtest_ls

warnings.filterwarnings("ignore", category=UserWarning)

EXCHANGE = "forex"
TIMEFRAME = "1d"
PAIRS = ["USD/CHF", "EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD"]

STARTING_CAPITAL_CHF = 1000.0
FEE_RATE = 0.0002          # 0.02 % Forex-Spread (Major Pairs)
TP_PCT = 0.005             # 0.5 % TP (Mean-Reversion: kleine Bewegungen)
SL_PCT = 0.003             # 0.3 % SL (enger Stop)
MAX_HOLD_BARS = 5          # kurz halten — Mean-Reversion will schnell raus
LEVERAGE = 5.0             # mit kleinen Bewegungen brauchen wir mehr Hebel
DAILY_LOSS_LIMIT = -0.01   # -1 % (Forex weniger volatil als Crypto)
COOLDOWN = 1

OPT_FRAC = 0.70            # 70 % Optimization, 30 % Holdout
TOP_N_PAIRS = 10
MIN_VOTES_PCT = 0.30
MIN_TRADES_PER_PAIR = 30   # mehr Trades verlangen


# Bull/Bear-Regeln (ohne Volumen-Indikatoren)
BULL_RULES = {
    "rsi_14":         lambda x: x < 35,
    "bb_pct":         lambda x: x < 0.2,
    "stoch_k":        lambda x: x < 25,
    "williams_r_14":  lambda x: x < -80,
    "cci_20":         lambda x: x < -100,
    "macd_hist":      lambda x: x > 0,
    "ema_ratio":      lambda x: x > 0,
    "sma_ratio":      lambda x: x > 0,
    "roc_10":         lambda x: x > 0,
    "roc_30":         lambda x: x > 0,
    "ret_1":          lambda x: x > 0,
    "ret_3":          lambda x: x > 0,
    "ret_6":          lambda x: x > 0,
    "keltner_pct":    lambda x: x < 0.2,
    "aroon_up":       lambda x: x > 70,
    "aroon_osc":      lambda x: x > 50,
    "trend_strength": lambda x: x > 0,
    "trend_strength_long": lambda x: x > 0,
    "adx_14":         lambda x: x > 25,
    "dist_to_20d_high": lambda x: x < 0.01,
    "dist_to_20d_low":  lambda x: x > 0.05,
}

BEAR_RULES = {
    "rsi_14":         lambda x: x > 65,
    "bb_pct":         lambda x: x > 0.8,
    "stoch_k":        lambda x: x > 75,
    "williams_r_14":  lambda x: x > -20,
    "cci_20":         lambda x: x > 100,
    "macd_hist":      lambda x: x < 0,
    "ema_ratio":      lambda x: x < 0,
    "sma_ratio":      lambda x: x < 0,
    "roc_10":         lambda x: x < 0,
    "roc_30":         lambda x: x < 0,
    "ret_1":          lambda x: x < 0,
    "ret_3":          lambda x: x < 0,
    "ret_6":          lambda x: x < 0,
    "keltner_pct":    lambda x: x > 0.8,
    "aroon_up":       lambda x: x < 30,
    "aroon_osc":      lambda x: x < -50,
    "trend_strength": lambda x: x < 0,
    "trend_strength_long": lambda x: x < 0,
    "adx_14":         lambda x: x > 25,
    "dist_to_20d_high": lambda x: x > 0.05,
    "dist_to_20d_low":  lambda x: x < 0.01,
}


def compute_indicators(df):
    out = pd.DataFrame(index=df.index)
    close, high, low = df["close"], df["high"], df["low"]
    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_6"] = close.pct_change(6)
    out["roc_10"] = roc(close, 10)
    out["roc_30"] = roc(close, 30)
    out["atr_14"] = atr(high, low, close, 14)
    out["rsi_14"] = rsi(close, 14)
    sk, _ = stochastic(high, low, close, 14, 3)
    out["stoch_k"] = sk
    out["williams_r_14"] = williams_r(high, low, close, 14)
    out["cci_20"] = cci(high, low, close, 20)
    out["macd_hist"] = macd(close)[2]
    out["bb_pct"] = bollinger_pct(close, 20)
    out["sma_ratio"] = sma(close, 10) / sma(close, 50) - 1
    out["ema_ratio"] = ema(close, 12) / ema(close, 26) - 1
    out["adx_14"] = adx(high, low, close, 14)
    out["keltner_pct"] = keltner_pct(high, low, close, 20)
    aup, _, aosc = aroon(high, low, 25)
    out["aroon_up"] = aup
    out["aroon_osc"] = aosc
    out["trend_strength"] = trend_strength(close, 10, 50)
    out["trend_strength_long"] = trend_strength(close, 20, 200)
    out["dist_to_20d_high"] = (high.rolling(20).max() - close) / close
    out["dist_to_20d_low"] = (close - low.rolling(20).min()) / close
    return out


def evaluate_pair_on_pair(close, ind_data, ind_a, ind_b):
    bull_a = BULL_RULES[ind_a](ind_data[ind_a]).fillna(False)
    bull_b = BULL_RULES[ind_b](ind_data[ind_b]).fillna(False)
    bear_a = BEAR_RULES[ind_a](ind_data[ind_a]).fillna(False)
    bear_b = BEAR_RULES[ind_b](ind_data[ind_b]).fillna(False)
    p_long = (bull_a & bull_b).astype(float)
    p_short = (bear_a & bear_b).astype(float)
    pos_frac = np.where((p_long > 0.5) | (p_short > 0.5), 0.20, 0.0)
    res = triple_barrier_backtest_ls(
        close=close, p_tp_long=p_long, p_tp_short=p_short,
        entry_threshold=0.5,
        tp_pct=TP_PCT, sl_pct=SL_PCT, max_hold=MAX_HOLD_BARS,
        fee_rate=FEE_RATE, starting_capital=STARTING_CAPITAL_CHF,
        cooldown=COOLDOWN, position_fraction=pos_frac, leverage=1.0,
    )
    n = len(res["trades"])
    return {
        "total_return": res["total_return"],
        "n_trades": n,
        "win_rate": (res["trades"]["trade_net_return"] > 0).mean() if n else 0,
    }


def main():
    print(f"Lade {len(PAIRS)} Forex-Pairs (2010-heute) …")
    pairs_data = {}
    for sym in PAIRS:
        df = load_ohlcv(EXCHANGE, sym, TIMEFRAME)
        if df.empty:
            print(f"  {sym}: leer, ueberspringe")
            continue
        ind = compute_indicators(df)
        df = pd.concat([df, ind], axis=1).dropna()
        pairs_data[sym] = df
        print(f"  {sym}: {len(df)} Bars von {df.index.min().date()} bis {df.index.max().date()}")

    common_start = max(df.index.min() for df in pairs_data.values())
    common_end = min(df.index.max() for df in pairs_data.values())
    for sym in pairs_data:
        pairs_data[sym] = pairs_data[sym].loc[common_start:common_end]
    n = len(next(iter(pairs_data.values())))
    opt_end = int(n * OPT_FRAC)
    opt_date = next(iter(pairs_data.values())).index[opt_end - 1]
    print(f"\nOpt-Set:  {common_start.date()} → {opt_date.date()}  ({opt_end} Bars)")
    print(f"Holdout:  {opt_date.date()} → {common_end.date()}  ({n - opt_end} Bars)")

    indicators = list(BULL_RULES.keys())
    all_pairs = list(itertools.combinations(indicators, 2))
    print(f"\nBrute-Force: {len(all_pairs)} Paare × {len(pairs_data)} Forex-Pairs = "
          f"{len(all_pairs) * len(pairs_data)} Backtests …")

    pair_results = []
    for ind_a, ind_b in all_pairs:
        per_fx = {}
        valid = True
        for sym in pairs_data:
            df_opt = pairs_data[sym].iloc[:opt_end]
            res = evaluate_pair_on_pair(df_opt["close"], df_opt, ind_a, ind_b)
            per_fx[sym] = res
            if res["n_trades"] < MIN_TRADES_PER_PAIR:
                valid = False
                break
        if not valid:
            continue
        returns = [per_fx[s]["total_return"] for s in pairs_data]
        pair_results.append({
            "pair": f"{ind_a}+{ind_b}", "ind_a": ind_a, "ind_b": ind_b,
            "avg_return": float(np.mean(returns)),
            "min_return": float(min(returns)),
            "total_trades": sum(per_fx[s]["n_trades"] for s in pairs_data),
            "avg_wr": float(np.mean([per_fx[s]["win_rate"] for s in pairs_data])),
            **{f"ret_{s.replace('/','')}": per_fx[s]["total_return"] for s in pairs_data},
        })

    pair_df = pd.DataFrame(pair_results)
    print(f"\n{len(pair_df)} Paare mit ≥{MIN_TRADES_PER_PAIR} Trades je Forex-Pair.")
    robust = pair_df[pair_df["min_return"] > 0].sort_values("avg_return", ascending=False)
    print(f"Davon {len(robust)} Paare auf ALLEN 5 Forex-Pairs profitabel.")

    if len(robust) == 0:
        print("\nKein robustes Paar gefunden. Top-10 nach Average (zur Info):")
        for _, r in pair_df.sort_values("avg_return", ascending=False).head(10).iterrows():
            print(f"  {r['pair']:<32s} avg {r['avg_return']*100:>+6.2f}%  min {r['min_return']*100:>+6.2f}%")
        return

    print(f"\n=== TOP-{min(TOP_N_PAIRS, len(robust))} ROBUSTE PAARE (alle 5 Forex-Pairs positiv) ===")
    print(f"  {'Pair':<32s} {'AvgRet':>7s} {'MinRet':>7s} {'WR':>5s} {'Trades':>6s}")
    top_pairs = robust.head(TOP_N_PAIRS)
    for _, r in top_pairs.iterrows():
        print(f"  {r['pair']:<32s} {r['avg_return']*100:>+6.2f}% {r['min_return']*100:>+6.2f}% "
              f"{r['avg_wr']*100:>4.1f}% {int(r['total_trades']):>6d}")

    # --- HOLDOUT Ensemble mit Hebel ---
    print(f"\n--- HOLDOUT Ensemble (Threshold {MIN_VOTES_PCT*100:.0f}% Votes, Hebel {LEVERAGE}x) ---")
    print(f"  {'Pair':>8s}  {'Endkap.':>10s}  {'Return':>9s}  {'Trades':>6s}  {'WR':>5s}  {'MaxDD':>7s}")
    total_final = 0
    total_bh = 0
    for sym in pairs_data:
        df_h = pairs_data[sym].iloc[opt_end:]
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
        pos_frac = np.where(
            (p_long > MIN_VOTES_PCT) | (p_short > MIN_VOTES_PCT),
            np.maximum(p_long, p_short).values * 0.30, 0.0
        )
        cap0 = STARTING_CAPITAL_CHF / len(pairs_data)
        res = triple_barrier_backtest_ls(
            close=df_h["close"], p_tp_long=p_long, p_tp_short=p_short,
            entry_threshold=MIN_VOTES_PCT,
            tp_pct=TP_PCT, sl_pct=SL_PCT, max_hold=MAX_HOLD_BARS,
            fee_rate=FEE_RATE, starting_capital=cap0,
            cooldown=COOLDOWN, position_fraction=pos_frac,
            leverage=LEVERAGE, daily_loss_limit=DAILY_LOSS_LIMIT,
        )
        trades = res["trades"]
        eq = res["equity"]
        wr = (trades["trade_net_return"] > 0).mean() if len(trades) else 0
        max_dd = (eq / eq.cummax() - 1).min() if len(eq) else 0
        bh = cap0 * (df_h["close"].iloc[-1] / df_h["close"].iloc[0])
        total_final += res["final_capital"]
        total_bh += bh
        print(f"  {sym:>8s}  {res['final_capital']:>10.2f}  {res['total_return']*100:>+8.2f}%  "
              f"{len(trades):>6d}  {wr*100:>4.1f}%  {max_dd*100:>+6.2f}%")

    print(f"\n  {'GESAMT':>8s}  {total_final:>10.2f}  "
          f"{(total_final/STARTING_CAPITAL_CHF-1)*100:>+8.2f}%  "
          f"(B&H Portfolio {total_bh:.2f} = {(total_bh/STARTING_CAPITAL_CHF-1)*100:+.2f}%)")


if __name__ == "__main__":
    main()
