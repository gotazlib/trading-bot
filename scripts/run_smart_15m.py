"""Triple-Barrier-Strategie auf 15m-BTC mit Marktregime-Filter.

Verbesserungen vs. 1m-Variante:
- Laengerer Horizont (15m): besseres Signal/Rausch-Verhaeltnis, Gebuehren werden
  zur Nebensache statt zum Hauptproblem
- Marktregime-Filter: trade NUR, wenn 24h-Volatilitaet im 25.-75. Perzentil
  liegt. Extreme Phasen (Crash, Pump) werden ausgeschlossen — dort sind
  trainierte Muster sowieso nicht verlaesslich.
- Probability-Kalibrierung beibehalten (isotonic regression)
- Asymmetrisches TP:SL — Break-Even Win-Rate ~32 %
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score

from src.data.storage import load_ohlcv
from src.ml.features import (
    build_features_with_interactions,
    build_label_triple_barrier,
)
from src.backtest.engine import triple_barrier_backtest

warnings.filterwarnings("ignore", category=UserWarning)

# --- Konfiguration ---
EXCHANGE, SYMBOL, TIMEFRAME = "binance", "BTC/USDT", "15m"
STARTING_CAPITAL_CHF = 1000.0
FEE_RATE = 0.001

# Triple-Barrier — 6h max Halten (24 Kerzen × 15min):
# Net_TP = +1.30 %, Net_SL = -0.60 %, Break-Even Win-Rate ≈ 31.6 %
TP_PCT = 0.015
SL_PCT = 0.004
MAX_HOLD_BARS = 24             # 24 × 15min = 6 Stunden

ENTRY_PROB_THRESHOLD = 0.40

# Marktregime-Filter:
# 24h-Vola wird gemessen als rolling std der 15m-Renditen ueber 96 Kerzen.
# Trade nur, wenn die aktuelle Vola im 25.-75. Perzentil der Trainingsverteilung liegt.
REGIME_LOW_PCT = 25
REGIME_HIGH_PCT = 75
VOL_WINDOW_BARS = 96           # 24h in 15m-Kerzen

TRAIN_FRAC = 0.60
CAL_FRAC = 0.10

COOLDOWN_BARS = 2              # 30 Minuten Pause nach jedem Trade


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=400,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=100,
        l2_regularization=1.0,
        random_state=42,
    )


def main():
    print("Lade 15m-Daten …")
    df = load_ohlcv(EXCHANGE, SYMBOL, TIMEFRAME)
    print(f"  {len(df)} Kerzen von {df.index.min()} bis {df.index.max()}")

    print("\nBaue Features (Basis + paarweise Produkte) …")
    X = build_features_with_interactions(df)

    # Regime-Indikator: 24h rolling std der 15m-Renditen
    rets = df["close"].pct_change()
    regime_vol = rets.rolling(VOL_WINDOW_BARS).std()

    print(f"\nBaue Triple-Barrier-Label (TP=+{TP_PCT*100:.2f} %, "
          f"SL=-{SL_PCT*100:.2f} %, max_hold={MAX_HOLD_BARS} Kerzen = "
          f"{MAX_HOLD_BARS*15/60:.0f}h) …")
    y = build_label_triple_barrier(df, tp_pct=TP_PCT, sl_pct=SL_PCT, max_hold=MAX_HOLD_BARS)

    data = X.copy()
    data["label"] = y
    data["close"] = df["close"]
    data["regime_vol"] = regime_vol
    data = data.dropna()
    data["label"] = data["label"].astype(int)
    features = list(X.columns)

    counts = data["label"].value_counts().to_dict()
    print("\nLabel-Verteilung (historische Outcomes):")
    total = len(data)
    for cls, name in [(1, "TP zuerst (Gewinn)"), (0, "Time-Stop (neutral)"), (-1, "SL zuerst (Verlust)")]:
        n = counts.get(cls, 0)
        print(f"  {name:25s} {n:>6d}  ({n/total*100:4.1f} %)")

    # --- Chronologischer Split ---
    n = len(data)
    train_end = int(n * TRAIN_FRAC)
    cal_end = train_end + int(n * CAL_FRAC)
    train = data.iloc[:train_end]
    cal = data.iloc[train_end:cal_end]
    test = data.iloc[cal_end:].copy()
    print(f"\nTraining:     {train.index.min()} -> {train.index.max()}  ({len(train)})")
    print(f"Kalibrierung: {cal.index.min()} -> {cal.index.max()}  ({len(cal)})")
    print(f"Test:         {test.index.min()} -> {test.index.max()}  ({len(test)})")

    # --- Regime-Grenzen aus Trainingsdaten lernen ---
    vol_lo = float(train["regime_vol"].quantile(REGIME_LOW_PCT / 100))
    vol_hi = float(train["regime_vol"].quantile(REGIME_HIGH_PCT / 100))
    print(f"\nMarktregime aus Trainingsdaten gelernt:"
          f"\n  Vola {REGIME_LOW_PCT}. Perzentil = {vol_lo:.5f}"
          f"\n  Vola {REGIME_HIGH_PCT}. Perzentil = {vol_hi:.5f}"
          f"\n  → Trades nur wenn Vola dazwischen liegt (normale Marktphase)")

    in_regime_test = (test["regime_vol"] >= vol_lo) & (test["regime_vol"] <= vol_hi)
    print(f"  Test-Kerzen im normalen Regime: {in_regime_test.sum()} / {len(test)} "
          f"({in_regime_test.mean()*100:.1f} %)")

    # Gebuehrenmathematik
    net_tp = (1 + TP_PCT) * (1 - FEE_RATE) ** 2 - 1
    net_sl = (1 - SL_PCT) * (1 - FEE_RATE) ** 2 - 1
    be_win = abs(net_sl) / (net_tp + abs(net_sl))
    print(f"\nGebuehren-Mathematik:"
          f"\n  Netto TP:  {net_tp*100:+.3f} %    Netto SL: {net_sl*100:+.3f} %"
          f"\n  Break-Even Win-Rate: {be_win*100:.1f} %")

    print("\nTrainiere Basis-Modell …")
    base = make_model()
    base.fit(train[features], train["label"])

    print("Kalibriere via Isotonic Regression …")
    model = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    model.fit(cal[features], cal["label"])

    classes = list(model.classes_)
    proba_raw = base.predict_proba(test[features])
    proba_cal = model.predict_proba(test[features])
    test["pred"] = model.predict(test[features])
    test["p_tp"] = proba_cal[:, classes.index(1)] if 1 in classes else 0.0
    test["p_tp_raw"] = proba_raw[:, list(base.classes_).index(1)] if 1 in base.classes_ else 0.0

    acc = accuracy_score(test["label"], test["pred"])
    print(f"  Out-of-Sample Trefferquote (3 Klassen): {acc*100:.1f} %")

    # --- Regime-Filter aufs Signal anwenden ---
    test["p_tp_filtered"] = np.where(in_regime_test, test["p_tp"], 0.0)
    n_above = int((test["p_tp"] > ENTRY_PROB_THRESHOLD).sum())
    n_above_in_regime = int((test["p_tp_filtered"] > ENTRY_PROB_THRESHOLD).sum())
    print(f"\nSignale mit kalibriertem P(TP) > {ENTRY_PROB_THRESHOLD}:")
    print(f"  ohne Regime-Filter: {n_above}")
    print(f"  mit  Regime-Filter: {n_above_in_regime}")

    print("\nKalibrierungs-Diagnose (war P=x wirklich x?):")
    print(f"  {'Bucket':<12s} {'roh: n':>7s} {'roh: TP%':>9s}  {'kal: n':>7s} {'kal: TP%':>9s}")
    for lo, hi in [(0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.01)]:
        m_raw = (test["p_tp_raw"] >= lo) & (test["p_tp_raw"] < hi)
        m_cal = (test["p_tp"] >= lo) & (test["p_tp"] < hi)
        n_raw = int(m_raw.sum())
        n_cal = int(m_cal.sum())
        a_raw = (test.loc[m_raw, "label"] == 1).mean() * 100 if n_raw else 0.0
        a_cal = (test.loc[m_cal, "label"] == 1).mean() * 100 if n_cal else 0.0
        print(f"  [{lo:.2f},{hi:.2f})  {n_raw:>7d} {a_raw:>8.1f}%  {n_cal:>7d} {a_cal:>8.1f}%")

    threshold = ENTRY_PROB_THRESHOLD
    if test["p_tp_filtered"].max() < threshold:
        demo_t = max(0.05, float(test["p_tp_filtered"][test["p_tp_filtered"] > 0].quantile(0.99)))
        print(f"\n[Hinweis] Max kalibriert+Regime = {test['p_tp_filtered'].max():.3f} < "
              f"Threshold {threshold:.2f}. → Demo mit {demo_t:.3f}.")
        threshold = demo_t

    print(f"\nLaufe Triple-Barrier-Backtest "
          f"({STARTING_CAPITAL_CHF:.0f} CHF, {FEE_RATE*100:.2f} % Gebuehr pro Seite) …")
    result = triple_barrier_backtest(
        close=test["close"],
        p_tp=test["p_tp_filtered"],
        entry_threshold=threshold,
        tp_pct=TP_PCT,
        sl_pct=SL_PCT,
        max_hold=MAX_HOLD_BARS,
        fee_rate=FEE_RATE,
        starting_capital=STARTING_CAPITAL_CHF,
        cooldown=COOLDOWN_BARS,
    )

    trades = result["trades"]
    equity = result["equity"]
    final = result["final_capital"]
    bh_final = STARTING_CAPITAL_CHF * (test["close"].iloc[-1] / test["close"].iloc[0])

    print("\n" + "=" * 72)
    print(f"=== ERGEBNIS {test.index.min().date()} → {test.index.max().date()} ===")
    print("=" * 72)
    print(f"  Strategie:        {final:>10.2f} CHF   ({result['total_return']*100:+.1f} %)")
    print(f"  Buy & Hold:       {bh_final:>10.2f} CHF   ({(bh_final/STARTING_CAPITAL_CHF-1)*100:+.1f} %)")
    print(f"  Anzahl Trades:    {len(trades):>10d}")

    if len(trades) > 0:
        win_rate = (trades["net_return"] > 0).mean()
        avg_hold_bars = trades["hold_bars"].mean()
        reasons = trades["reason"].value_counts().to_dict()
        total_fees = sum(abs(t["capital_before"]) * FEE_RATE * 2 for _, t in trades.iterrows())
        max_dd = (equity / equity.cummax() - 1).min()
        avg_pnl = trades["pnl_chf"].mean()

        print(f"  Win-Rate (Netto): {win_rate*100:>9.1f} %  (Break-Even: {be_win*100:.1f} %)")
        print(f"  Avg. Haltedauer:  {avg_hold_bars*15:>9.0f} Minuten")
        print(f"  Avg. PnL/Trade:   {avg_pnl:>9.2f} CHF")
        print(f"  Gebuehren total:  {total_fees:>9.2f} CHF")
        print(f"  Max Drawdown:     {max_dd*100:>9.1f} %")
        print(f"  Exits:            TP={reasons.get('TP',0)}  "
              f"SL={reasons.get('SL',0)}  TIME={reasons.get('TIME',0)}")

        print(f"\n--- ALLE {len(trades)} TRADES ---")
        print(f"  {'Eintritt':<19s}  {'Dauer':>6s}  {'P(TP)':>5s}  "
              f"{'Reason':>6s}  {'Brutto':>7s}  {'Netto':>7s}  {'Kapital':>10s}")
        for _, t in trades.iterrows():
            print(
                f"  {str(t['entry_time'])[:19]:<19s}  "
                f"{t['hold_bars']*15:>4.0f}m  "
                f"{t['p_tp']:>5.2f}  "
                f"{t['reason']:>6s}  "
                f"{t['gross_return']*100:>+6.2f} %  "
                f"{t['net_return']*100:>+6.2f} %  "
                f"{t['capital_after']:>9.2f} CHF"
            )


if __name__ == "__main__":
    main()
