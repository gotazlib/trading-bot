"""USD/CHF Signal-to-Outcome-Analyse.

Pro Indikator-Signal-Trigger:
- Identifiziere alle Bars wo das Signal ausgelöst hat
- Messe was in N Bars/Tagen danach mit dem Preis passiert ist
- Hit-Rate, Avg Return, Sharpe-aequivalentes Maß (IR)

Liefert ehrliche Edge-Messung pro Indikator OHNE Backtest-Mechanik
(keine Stops, keine Position-Sizing — pure Vorhersage-Statistik).
"""
import warnings

import numpy as np
import pandas as pd

from src.data.storage import load_ohlcv
from scripts.run_forex_brute_force import compute_indicators

warnings.filterwarnings("ignore", category=UserWarning)


# Signal-Trigger: Wann LÖST der Indikator ein Bull-/Bear-Signal aus?
# Wir nehmen Cross-Events (statt "ist Zustand X") für saubere Trigger
SIGNAL_TRIGGERS = {
    "RSI<30 Bull (oversold)":        ("rsi_14",      lambda s: (s.shift(1) >= 30) & (s < 30),  "bull"),
    "RSI>70 Bear (overbought)":      ("rsi_14",      lambda s: (s.shift(1) <= 70) & (s > 70),  "bear"),
    "BB<0.05 Bull (extreme oversold)": ("bb_pct",    lambda s: (s.shift(1) >= 0.05) & (s < 0.05), "bull"),
    "BB>0.95 Bear (extreme overbought)": ("bb_pct",  lambda s: (s.shift(1) <= 0.95) & (s > 0.95), "bear"),
    "MACD Cross Up Bull":            ("macd_hist",   lambda s: (s.shift(1) <= 0) & (s > 0),    "bull"),
    "MACD Cross Down Bear":          ("macd_hist",   lambda s: (s.shift(1) >= 0) & (s < 0),    "bear"),
    "EMA Cross Up Bull":             ("ema_ratio",   lambda s: (s.shift(1) <= 0) & (s > 0),    "bull"),
    "EMA Cross Down Bear":           ("ema_ratio",   lambda s: (s.shift(1) >= 0) & (s < 0),    "bear"),
    "ROC30>0 Bull (momentum)":       ("roc_30",      lambda s: (s.shift(1) <= 0) & (s > 0),    "bull"),
    "ROC30<0 Bear (momentum)":       ("roc_30",      lambda s: (s.shift(1) >= 0) & (s < 0),    "bear"),
    "Stoch<20 Bull (oversold)":      ("stoch_k",     lambda s: (s.shift(1) >= 20) & (s < 20),  "bull"),
    "Stoch>80 Bear (overbought)":    ("stoch_k",     lambda s: (s.shift(1) <= 80) & (s > 80),  "bear"),
    "Williams<-90 Bull":             ("williams_r_14", lambda s: (s.shift(1) >= -90) & (s < -90), "bull"),
    "Williams>-10 Bear":             ("williams_r_14", lambda s: (s.shift(1) <= -10) & (s > -10), "bear"),
    "CCI<-150 Bull":                 ("cci_20",      lambda s: (s.shift(1) >= -150) & (s < -150), "bull"),
    "CCI>150 Bear":                  ("cci_20",      lambda s: (s.shift(1) <= 150) & (s > 150),  "bear"),
}


def analyze_signals(df, horizons, label):
    """Pro Signal-Trigger: messe Outcome auf verschiedenen Horizonten."""
    print(f"\n{'='*108}")
    print(f"=== {label} ===")
    print(f"{'='*108}")
    close = df["close"]
    rows = []
    for name, (indicator, trigger_fn, direction) in SIGNAL_TRIGGERS.items():
        if indicator not in df.columns:
            continue
        triggers = trigger_fn(df[indicator]).fillna(False)
        n_signals = int(triggers.sum())
        if n_signals < 5:
            continue
        result = {"signal": name, "direction": direction, "n_signals": n_signals}
        for h_label, h_bars in horizons:
            future_ret = (close.shift(-h_bars) / close - 1)
            # Für Bull-Signal erwarten wir positive Returns, für Bear negative
            ret_at_signal = future_ret[triggers]
            # Direction-adjusted Returns: für Bear-Signale invertieren
            if direction == "bear":
                directional = -ret_at_signal
            else:
                directional = ret_at_signal
            hit_rate = (directional > 0).mean() * 100
            avg_ret = directional.mean() * 100
            std_ret = directional.std() * 100
            ir = (avg_ret / std_ret) if std_ret > 0 else 0
            result[f"hit_{h_label}"] = hit_rate
            result[f"avg_{h_label}"] = avg_ret
            result[f"ir_{h_label}"] = ir
        rows.append(result)

    df_res = pd.DataFrame(rows)
    # Print table
    h_labels = [h[0] for h in horizons]
    print(f"  {'Signal':<38s} {'N':>5s}", end="")
    for h in h_labels:
        print(f" | {h+' Hit':>8s} {h+' Avg':>8s} {h+' IR':>7s}", end="")
    print()
    print("  " + "-" * (38 + 7 + (len(h_labels) * 27)))
    for _, r in df_res.iterrows():
        print(f"  {r['signal']:<38s} {r['n_signals']:>5d}", end="")
        for h in h_labels:
            print(f" | {r[f'hit_{h}']:>7.1f}% {r[f'avg_{h}']:>+7.3f}% {r[f'ir_{h}']:>+6.2f}", end="")
        print()
    return df_res


def main():
    # ============== DAILY ==============
    print("Lade USD/CHF Daily seit 2005 …")
    df_d = load_ohlcv("forex", "USD/CHF", "1d")
    ind_d = compute_indicators(df_d)
    df_d = pd.concat([df_d, ind_d], axis=1).dropna()
    print(f"  {len(df_d)} Bars von {df_d.index.min().date()} bis {df_d.index.max().date()}")

    daily_horizons = [("T+1", 1), ("T+3", 3), ("T+5", 5), ("T+10", 10)]
    res_d = analyze_signals(df_d, daily_horizons, "DAILY (5.500 Bars, 20 Jahre)")

    # Edge-Ranking: bestes IR insgesamt
    best_signals_d = res_d.copy()
    best_signals_d["max_ir"] = best_signals_d[[f"ir_{h[0]}" for h in daily_horizons]].max(axis=1)
    best_signals_d["best_horizon"] = best_signals_d[[f"ir_{h[0]}" for h in daily_horizons]].idxmax(axis=1)
    best_signals_d = best_signals_d.sort_values("max_ir", ascending=False)
    print(f"\n--- TOP-5 Signale Daily (nach bestem IR) ---")
    for _, r in best_signals_d.head(5).iterrows():
        h = r["best_horizon"].replace("ir_", "")
        print(f"  {r['signal']:<38s}  Best @ {h}: IR {r[r['best_horizon']]:+.2f}, "
              f"Hit {r[f'hit_{h}']:.1f}%, AvgRet {r[f'avg_{h}']:+.3f}%, N={r['n_signals']}")

    # ============== 15m ==============
    print("\n\nLade USD/CHF 15m (letzte 60 Tage) …")
    df_15 = load_ohlcv("forex", "USD/CHF", "15m")
    ind_15 = compute_indicators(df_15)
    df_15 = pd.concat([df_15, ind_15], axis=1).dropna()
    print(f"  {len(df_15)} Bars von {df_15.index.min()} bis {df_15.index.max()}")

    # Horizonte: 1, 4, 16, 96 Bars = 15min, 1h, 4h, 24h
    intraday_horizons = [("T+1 (15m)", 1), ("T+4 (1h)", 4), ("T+16 (4h)", 16), ("T+96 (1d)", 96)]
    res_15 = analyze_signals(df_15, intraday_horizons, "15-MINUTE (5.600 Bars, 60 Tage)")

    best_signals_15 = res_15.copy()
    best_signals_15["max_ir"] = best_signals_15[[f"ir_{h[0]}" for h in intraday_horizons]].max(axis=1)
    best_signals_15["best_horizon"] = best_signals_15[[f"ir_{h[0]}" for h in intraday_horizons]].idxmax(axis=1)
    best_signals_15 = best_signals_15.sort_values("max_ir", ascending=False)
    print(f"\n--- TOP-5 Signale 15m (nach bestem IR) ---")
    for _, r in best_signals_15.head(5).iterrows():
        h = r["best_horizon"].replace("ir_", "")
        print(f"  {r['signal']:<38s}  Best @ {h}: IR {r[r['best_horizon']]:+.2f}, "
              f"Hit {r[f'hit_{h}']:.1f}%, AvgRet {r[f'avg_{h}']:+.3f}%, N={r['n_signals']}")

    # === Statistische Bewertung ===
    print(f"\n\n{'='*108}")
    print("=== EDGE-BEWERTUNG ===")
    print(f"{'='*108}")
    print(f"Null-Hypothese: Hit-Rate = 50 % (kein Edge)")
    print(f"Praktisch profitabel: Hit-Rate > ~52-55 % UND Avg-Return > 0.05 % (nach Spread)")
    print(f"\nDaily-Signale ueber alle Horizonte: Hit-Rate-Mittel = "
          f"{res_d[[f'hit_{h[0]}' for h in daily_horizons]].values.mean():.2f}%")
    print(f"15m-Signale ueber alle Horizonte:   Hit-Rate-Mittel = "
          f"{res_15[[f'hit_{h[0]}' for h in intraday_horizons]].values.mean():.2f}%")


if __name__ == "__main__":
    main()
