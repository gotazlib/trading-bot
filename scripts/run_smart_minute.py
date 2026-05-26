"""Konservative Triple-Barrier-Strategie auf 1m-BTC.

Idee:
- Modell lernt direkt das Backtest-Ziel: 'Wird in den naechsten max_hold Minuten
  zuerst der Take-Profit oder der Stop-Loss getroffen?'
- Einstieg nur, wenn das Modell sich SEHR sicher ist (P(TP) > Threshold).
  Das bedeutet implizit: 'aehnliche historische Setups haben oft funktioniert'.
- Pro Trade variable Haltedauer (bis TP, SL oder Zeit-Stop).
- Realistischer Backtest in CHF mit Startguthaben.
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
EXCHANGE, SYMBOL, TIMEFRAME = "binance", "BTC/USDT", "1m"
STARTING_CAPITAL_CHF = 1000.0
FEE_RATE = 0.001               # 0.1 % pro Seite (Entry + Exit)

# Triple-Barrier mit asymmetrischem Reward (Reward 5x Risk):
# - TP 1.5 % minus 0.2 % Gebuehr = +1.30 % netto
# - SL 0.3 % plus 0.2 % Gebuehr = -0.50 % netto
# - Break-Even Win-Rate ≈ 0.50 / (1.30 + 0.50) ≈ 28 %
# Damit reicht eine *kalibrierte* Wahrscheinlichkeit > ~30 %, um statistisch
# profitabel zu sein — und auf einer Base-Rate von ~5 % ist das ein echter Edge.
TP_PCT = 0.015                 # +1.5 % Take-Profit
SL_PCT = 0.003                 # -0.3 % Stop-Loss
MAX_HOLD_MIN = 180             # 3 Stunden — genug fuer groessere Bewegungen

# Schwelle auf der kalibrierten Skala. Mit niedriger Base-Rate (<10 % TP-Anteil)
# sind 'kalibriert ≥0.30' bereits 3-6x ueber dem Marktdurchschnitt = echtes Signal.
ENTRY_PROB_THRESHOLD = 0.30

# Train/Calibrate/Test-Aufteilung:
#   60 % Modell-Training
#   10 % Kalibrierung (held-out fuer Isotonic Regression)
#   30 % Test (out-of-sample Backtest)
TRAIN_FRAC = 0.60
CAL_FRAC = 0.10

COOLDOWN_BARS = 5


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=400,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=200,
        l2_regularization=1.0,
        random_state=42,
    )


def main():
    print("Lade 1m-Daten …")
    df = load_ohlcv(EXCHANGE, SYMBOL, TIMEFRAME)
    print(f"  {len(df)} Kerzen von {df.index.min()} bis {df.index.max()}")

    print("\nBaue Features (Basis + paarweise Produkte) …")
    X = build_features_with_interactions(df)
    print(f"  {X.shape[1]} Features")

    print(f"\nBaue Triple-Barrier-Label (TP=+{TP_PCT*100:.2f} %, "
          f"SL=-{SL_PCT*100:.2f} %, max_hold={MAX_HOLD_MIN}m) …")
    y = build_label_triple_barrier(df, tp_pct=TP_PCT, sl_pct=SL_PCT, max_hold=MAX_HOLD_MIN)

    data = X.copy()
    data["label"] = y
    data["close"] = df["close"]
    data = data.dropna()
    data["label"] = data["label"].astype(int)
    features = list(X.columns)

    counts = data["label"].value_counts().to_dict()
    print(f"\nLabel-Verteilung (was waere historisch passiert?):")
    total = len(data)
    for cls, name in [(1, "TP zuerst (Gewinn)"), (0, "Time-Stop (neutral)"), (-1, "SL zuerst (Verlust)")]:
        n = counts.get(cls, 0)
        print(f"  {name:25s} {n:>7d}  ({n/total*100:4.1f} %)")

    # --- Chronologischer Split: Train / Calibration / Test ---
    n = len(data)
    train_end = int(n * TRAIN_FRAC)
    cal_end = train_end + int(n * CAL_FRAC)
    train = data.iloc[:train_end]
    cal = data.iloc[train_end:cal_end]
    test = data.iloc[cal_end:].copy()
    print(f"\nTraining:     {train.index.min()} -> {train.index.max()}  ({len(train)})")
    print(f"Kalibrierung: {cal.index.min()} -> {cal.index.max()}  ({len(cal)})")
    print(f"Test:         {test.index.min()} -> {test.index.max()}  ({len(test)})")

    # Break-Even-Mathematik vor dem Run
    net_tp = (1 + TP_PCT) * (1 - FEE_RATE) ** 2 - 1
    net_sl = (1 - SL_PCT) * (1 - FEE_RATE) ** 2 - 1
    be_win = abs(net_sl) / (net_tp + abs(net_sl))
    print(
        f"\nGebuehren-Mathematik:"
        f"\n  Netto TP nach 2x Gebuehr:  {net_tp*100:+.3f} %"
        f"\n  Netto SL nach 2x Gebuehr:  {net_sl*100:+.3f} %"
        f"\n  Break-Even Win-Rate:       {be_win*100:.1f} %  "
        f"(jede hoehere Win-Rate ergibt Gewinn)"
    )

    print("\nTrainiere Basis-Modell …")
    base = make_model()
    base.fit(train[features], train["label"])

    # --- Probability-Kalibrierung (isotonic regression) ---
    print("Kalibriere Wahrscheinlichkeiten via Isotonic Regression …")
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

    # --- Kalibrierungs-Diagnose ---
    print("\nKalibrierungs-Diagnose (war P=x wirklich x?) — Buckets aus Testdaten:")
    print(f"  {'Bucket':<12s} {'roh: n':>7s} {'roh: TP%':>9s}  {'kal: n':>7s} {'kal: TP%':>9s}")
    for lo, hi in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 1.01)]:
        mask_raw = (test["p_tp_raw"] >= lo) & (test["p_tp_raw"] < hi)
        mask_cal = (test["p_tp"]     >= lo) & (test["p_tp"]     < hi)
        n_raw = int(mask_raw.sum())
        n_cal = int(mask_cal.sum())
        actual_raw = (test.loc[mask_raw, "label"] == 1).mean() * 100 if n_raw else 0.0
        actual_cal = (test.loc[mask_cal, "label"] == 1).mean() * 100 if n_cal else 0.0
        print(f"  [{lo:.2f},{hi:.2f})  {n_raw:>7d} {actual_raw:>8.1f}%  {n_cal:>7d} {actual_cal:>8.1f}%")

    # Wie oft signalisiert das Modell ueberhaupt eine Einstiegschance?
    n_signals = int((test["p_tp"] > ENTRY_PROB_THRESHOLD).sum())
    print(f"  Signale mit P(TP) > {ENTRY_PROB_THRESHOLD:.2f}: "
          f"{n_signals} von {len(test)} Kerzen ({n_signals/len(test)*100:.2f} %)")

    # Wenn kalibrierter Threshold nie erreicht wird, automatisch auf 90. Perzentil
    # absenken — als Demo, was passieren WUERDE, wenn man die schwachen Signale
    # trotzdem handelt. Mathematisch: negative Erwartung, aber lehrreich.
    threshold = ENTRY_PROB_THRESHOLD
    if test["p_tp"].max() < threshold:
        demo_threshold = max(0.05, float(test["p_tp"].quantile(0.995)))
        print(
            f"\n[Hinweis] Kalibrierter Max-P(TP)={test['p_tp'].max():.3f} < "
            f"Threshold {threshold:.2f} → 0 Trades waeren die korrekte Antwort.\n"
            f"          Zur Demo: laufe mit Notschwelle {demo_threshold:.3f} "
            f"(99.5. Perzentil). Mathematik sagt: negative Erwartung."
        )
        threshold = demo_threshold

    print(f"\nLaufe Triple-Barrier-Backtest "
          f"({STARTING_CAPITAL_CHF:.0f} CHF Startkapital, {FEE_RATE*100:.2f} % Gebuehr pro Seite) …")
    result = triple_barrier_backtest(
        close=test["close"],
        p_tp=test["p_tp"],
        entry_threshold=threshold,
        tp_pct=TP_PCT,
        sl_pct=SL_PCT,
        max_hold=MAX_HOLD_MIN,
        fee_rate=FEE_RATE,
        starting_capital=STARTING_CAPITAL_CHF,
        cooldown=COOLDOWN_BARS,
    )

    trades = result["trades"]
    equity = result["equity"]
    final = result["final_capital"]

    bh_final = STARTING_CAPITAL_CHF * (test["close"].iloc[-1] / test["close"].iloc[0])

    print("\n" + "=" * 72)
    print(f"=== ERGEBNIS ueber Test-Zeitraum {test.index.min().date()} → {test.index.max().date()} ===")
    print("=" * 72)
    print(f"  Strategie-Endkapital:     {final:>10.2f} CHF   ({result['total_return']*100:+.1f} %)")
    print(f"  Buy & Hold-Endkapital:    {bh_final:>10.2f} CHF   ({(bh_final/STARTING_CAPITAL_CHF-1)*100:+.1f} %)")
    print(f"  Anzahl Trades:            {len(trades):>10d}")

    if len(trades) > 0:
        win_rate = (trades["net_return"] > 0).mean()
        avg_hold = trades["hold_bars"].mean()
        reasons = trades["reason"].value_counts().to_dict()
        total_fees = sum(abs(t["capital_before"]) * FEE_RATE * 2 for _, t in trades.iterrows())
        max_dd = (equity / equity.cummax() - 1).min()

        print(f"  Trefferquote (Net):       {win_rate*100:>9.1f} %")
        print(f"  Durchschn. Haltedauer:    {avg_hold:>9.1f} Minuten")
        print(f"  Gebuehren gezahlt:        {total_fees:>10.2f} CHF")
        print(f"  Max Drawdown:             {max_dd*100:>9.1f} %")
        print(f"  Exit-Gruende:             TP={reasons.get('TP',0)}  "
              f"SL={reasons.get('SL',0)}  TIME={reasons.get('TIME',0)}")

        # Detailliertes Trade-Log
        print(f"\n--- ALLE {len(trades)} TRADES ---")
        print(f"  {'Eintritt':<19s}  {'Dauer':>5s}  {'P(TP)':>5s}  "
              f"{'Reason':>6s}  {'Brutto':>7s}  {'Netto':>7s}  {'Kapital':>10s}")
        for _, t in trades.iterrows():
            print(
                f"  {str(t['entry_time'])[:19]:<19s}  "
                f"{t['hold_bars']:>3d}m  "
                f"{t['p_tp']:>5.2f}  "
                f"{t['reason']:>6s}  "
                f"{t['gross_return']*100:>+6.2f} %  "
                f"{t['net_return']*100:>+6.2f} %  "
                f"{t['capital_after']:>9.2f} CHF"
            )
    else:
        print("\n  Modell hat KEINE Einstiegs-Signale ueber dem Threshold gegeben.")
        print(f"  → Threshold {ENTRY_PROB_THRESHOLD:.2f} ist zu streng oder das Setup ist selten.")
        max_p = test["p_tp"].max()
        print(f"  → Maximales P(TP) im Test-Set war: {max_p:.3f}")


if __name__ == "__main__":
    main()
