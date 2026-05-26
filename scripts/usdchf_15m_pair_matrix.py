"""USD/CHF 15m Brute-Force-Pair-Matrix.

Berechnet pro Paar (17 Indikatoren = 136 Paare) die Performance auf USD/CHF 15m
und speichert eine komplette Pair-Matrix als CSV fuer Agenten-Analyse.
"""
import itertools
import json
import os
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
SYMBOL = "USD/CHF"
TIMEFRAME = "15m"

STARTING_CAPITAL_CHF = 1000.0
FEE_RATE = 0.0002        # 0.02 % Spread fuer Major Pair
TP_PCT = 0.003           # 0.3 % TP (15m typische Bewegung)
SL_PCT = 0.0015          # 0.15 % SL (2:1 Reward)
MAX_HOLD_BARS = 24       # 6 Stunden
COOLDOWN = 1
LEVERAGE = 1.0           # ungehebelt fuer Pair-Ranking

OPT_FRAC = 0.70

# 17 Indikatoren mit Bull/Bear-Regeln
BULL_RULES = {
    "rsi_14":              lambda x: x < 35,
    "bb_pct":              lambda x: x < 0.2,
    "stoch_k":             lambda x: x < 25,
    "williams_r_14":       lambda x: x < -80,
    "cci_20":              lambda x: x < -100,
    "macd_hist":           lambda x: x > 0,
    "ema_ratio":           lambda x: x > 0,
    "sma_ratio":           lambda x: x > 0,
    "roc_10":              lambda x: x > 0,
    "roc_30":              lambda x: x > 0,
    "ret_1":               lambda x: x > 0,
    "ret_3":               lambda x: x > 0,
    "ret_6":               lambda x: x > 0,
    "keltner_pct":         lambda x: x < 0.2,
    "aroon_up":            lambda x: x > 70,
    "aroon_osc":           lambda x: x > 50,
    "adx_14":              lambda x: x > 25,
    "trend_strength":      lambda x: x > 0,
}

BEAR_RULES = {
    "rsi_14":              lambda x: x > 65,
    "bb_pct":              lambda x: x > 0.8,
    "stoch_k":             lambda x: x > 75,
    "williams_r_14":       lambda x: x > -20,
    "cci_20":              lambda x: x > 100,
    "macd_hist":           lambda x: x < 0,
    "ema_ratio":           lambda x: x < 0,
    "sma_ratio":           lambda x: x < 0,
    "roc_10":              lambda x: x < 0,
    "roc_30":              lambda x: x < 0,
    "ret_1":               lambda x: x < 0,
    "ret_3":               lambda x: x < 0,
    "ret_6":               lambda x: x < 0,
    "keltner_pct":         lambda x: x > 0.8,
    "aroon_up":            lambda x: x < 30,
    "aroon_osc":           lambda x: x < -50,
    "adx_14":              lambda x: x > 25,
    "trend_strength":      lambda x: x < 0,
}


def compute_indicators(df):
    out = pd.DataFrame(index=df.index)
    close, high, low = df["close"], df["high"], df["low"]
    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_6"] = close.pct_change(6)
    out["roc_10"] = roc(close, 10)
    out["roc_30"] = roc(close, 30)
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
    return out


def evaluate_pair(close, ind_data, ind_a, ind_b):
    bull_a = BULL_RULES[ind_a](ind_data[ind_a]).fillna(False)
    bull_b = BULL_RULES[ind_b](ind_data[ind_b]).fillna(False)
    bear_a = BEAR_RULES[ind_a](ind_data[ind_a]).fillna(False)
    bear_b = BEAR_RULES[ind_b](ind_data[ind_b]).fillna(False)
    p_long = (bull_a & bull_b).astype(float)
    p_short = (bear_a & bear_b).astype(float)
    pos_frac = np.where((p_long > 0.5) | (p_short > 0.5), 0.25, 0.0)
    res = triple_barrier_backtest_ls(
        close=close, p_tp_long=p_long, p_tp_short=p_short,
        entry_threshold=0.5,
        tp_pct=TP_PCT, sl_pct=SL_PCT, max_hold=MAX_HOLD_BARS,
        fee_rate=FEE_RATE, starting_capital=STARTING_CAPITAL_CHF,
        cooldown=COOLDOWN, position_fraction=pos_frac, leverage=LEVERAGE,
    )
    n = len(res["trades"])
    return {
        "total_return": res["total_return"],
        "n_trades": n,
        "win_rate": (res["trades"]["trade_net_return"] > 0).mean() if n else 0,
    }


def main():
    print(f"Lade {SYMBOL} {TIMEFRAME} …")
    df = load_ohlcv(EXCHANGE, SYMBOL, TIMEFRAME)
    print(f"  {len(df)} Bars von {df.index.min()} bis {df.index.max()}")

    ind = compute_indicators(df)
    df = pd.concat([df, ind], axis=1).dropna()
    n = len(df)
    opt_end = int(n * OPT_FRAC)
    df_opt = df.iloc[:opt_end]
    print(f"\nOpt-Set:  {df_opt.index[0]} → {df_opt.index[-1]}  ({len(df_opt)} Bars)")
    print(f"Holdout:  {df.iloc[opt_end:].index[0]} → {df.iloc[-1].name}  ({n-opt_end} Bars)")

    indicators = list(BULL_RULES.keys())
    pairs = list(itertools.combinations(indicators, 2))
    print(f"\nBrute-Force: {len(pairs)} Paare auf {SYMBOL} 15m …")

    results = []
    for ind_a, ind_b in pairs:
        res = evaluate_pair(df_opt["close"], df_opt, ind_a, ind_b)
        results.append({
            "ind_a": ind_a, "ind_b": ind_b, "pair": f"{ind_a}+{ind_b}",
            **res,
        })

    pair_df = pd.DataFrame(results).sort_values("total_return", ascending=False)
    os.makedirs("results", exist_ok=True)
    out_csv = "results/usdchf_15m_pair_matrix.csv"
    pair_df.to_csv(out_csv, index=False)
    print(f"\nCSV gespeichert: {out_csv}")

    # Per-Indikator Top-Paare als JSON fuer Agenten
    per_indicator = {}
    for ind in indicators:
        sub = pair_df[(pair_df["ind_a"] == ind) | (pair_df["ind_b"] == ind)].copy()
        sub["partner"] = sub.apply(
            lambda r: r["ind_b"] if r["ind_a"] == ind else r["ind_a"], axis=1
        )
        sub = sub[["partner", "total_return", "n_trades", "win_rate"]].sort_values(
            "total_return", ascending=False
        )
        per_indicator[ind] = {
            "n_partners": len(sub),
            "best_partner": sub.iloc[0]["partner"] if len(sub) else None,
            "best_return": float(sub.iloc[0]["total_return"]) if len(sub) else 0,
            "n_profitable": int((sub["total_return"] > 0).sum()),
            "avg_return": float(sub["total_return"].mean()),
            "all_partners": sub.to_dict(orient="records"),
        }
    out_json = "results/usdchf_15m_per_indicator.json"
    with open(out_json, "w") as f:
        json.dump(per_indicator, f, indent=2, default=float)
    print(f"JSON gespeichert: {out_json}")

    # Quick-Summary
    print(f"\n--- QUICK SUMMARY ---")
    print(f"  Gesamt-Paare:       {len(pair_df)}")
    print(f"  Mit Trades:         {(pair_df['n_trades'] > 0).sum()}")
    print(f"  Profitabel:         {(pair_df['total_return'] > 0).sum()}")
    print(f"  Bestes Paar:        {pair_df.iloc[0]['pair']}  "
          f"({pair_df.iloc[0]['total_return']*100:+.2f} %)")
    print(f"  Schlechtestes:      {pair_df.iloc[-1]['pair']}  "
          f"({pair_df.iloc[-1]['total_return']*100:+.2f} %)")
    print(f"  Avg Return:         {pair_df['total_return'].mean()*100:+.2f} %")
    print(f"  Median Win-Rate:    {pair_df['win_rate'].median()*100:.1f} %")


if __name__ == "__main__":
    main()
