"""Walk-Forward ML-Backtest mit Trade-Filter, Vol-skaliertem Label
und mehreren Vergleichsstrategien.

Vergleich (alle mit denselben Gebuehren):
  - Cash          : nichts tun
  - Buy & Hold    : durchgehend long
  - SMA-Crossover : long wenn SMA(10) > SMA(50), sonst cash
  - ML naiv       : Position = argmax(Modell)
  - ML gefiltert  : nur long, wenn P(up) > Schwelle  (selten handeln statt oft falsch)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.data.storage import load_ohlcv
from src.ml.features import build_features, build_label_vol_scaled
from src.backtest.engine import evaluate, print_row
from src.indicators.technical import sma

EXCHANGE, SYMBOL, TIMEFRAME = "binance", "BTC/USDT", "1h"
HORIZON = 6           # Vorhersage 6h voraus
VOL_WINDOW = 24       # 24h-Vola als Skala
K_VOL = 0.5           # Schwelle in Einheiten der Vola
FEE_RATE = 0.001      # 0.1 % pro Positionswechsel
PROB_THRESHOLD = 0.55 # ML-Filter: nur long, wenn Modell sich relativ sicher ist
N_FOLDS = 5           # Walk-Forward-Fenster
INITIAL_TRAIN_FRAC = 0.5  # erstes Trainingsfenster = 50 % der Daten


def walk_forward_predict(data: pd.DataFrame, features: list, n_folds: int, initial_train_frac: float):
    """Trainiert N Modelle hintereinander, jedes auf wachsendem Trainingsfenster.
    Liefert konkateniertes Test-DataFrame mit Spalten pred (+ p_up, p_down) sowie
    die durchschnittliche Feature-Importance ueber alle Folds.
    """
    n = len(data)
    initial_train = int(n * initial_train_frac)
    test_size = (n - initial_train) // n_folds
    importances = []
    test_chunks = []

    for fold in range(n_folds):
        train_end = initial_train + fold * test_size
        test_end = train_end + test_size if fold < n_folds - 1 else n
        train = data.iloc[:train_end]
        test = data.iloc[train_end:test_end].copy()

        model = RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=50,
            n_jobs=-1, random_state=42, class_weight="balanced",
        )
        model.fit(train[features], train["label"])

        classes = list(model.classes_)
        proba = model.predict_proba(test[features])
        test["pred"] = model.predict(test[features])
        test["p_up"] = proba[:, classes.index(1)] if 1 in classes else 0.0
        test["p_down"] = proba[:, classes.index(-1)] if -1 in classes else 0.0
        test["fold"] = fold

        importances.append(pd.Series(model.feature_importances_, index=features))
        test_chunks.append(test)

        print(
            f"  Fold {fold+1}/{n_folds}: train {train.index.min().date()}–{train.index.max().date()} "
            f"({len(train)}) | test {test.index.min().date()}–{test.index.max().date()} ({len(test)})"
        )

    return pd.concat(test_chunks), pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)


def main():
    print("Lade Daten …")
    df = load_ohlcv(EXCHANGE, SYMBOL, TIMEFRAME)
    print(f"  {len(df)} Kerzen von {df.index.min().date()} bis {df.index.max().date()}")

    X = build_features(df)
    y = build_label_vol_scaled(df, horizon=HORIZON, vol_window=VOL_WINDOW, k=K_VOL)

    data = X.copy()
    data["label"] = y
    data["close"] = df["close"]
    data = data.dropna()
    data["label"] = data["label"].astype(int)
    features = list(X.columns)

    label_counts = data["label"].value_counts().to_dict()
    print(f"\nLabel-Verteilung (vol-skaliert, k={K_VOL}, horizon={HORIZON}h):")
    for cls in sorted(label_counts):
        share = label_counts[cls] / len(data) * 100
        print(f"  {cls:+d}: {label_counts[cls]:>6d}  ({share:4.1f} %)")

    print(f"\nWalk-Forward ({N_FOLDS} Folds, Start-Train {INITIAL_TRAIN_FRAC*100:.0f} %):")
    test, importance = walk_forward_predict(data, features, N_FOLDS, INITIAL_TRAIN_FRAC)

    acc = accuracy_score(test["label"], test["pred"])
    print(f"\nOut-of-Sample Trefferquote (3 Klassen): {acc*100:.1f} %   "
          f"(Random-Baseline ≈ 33 %, groesste Klasse-Baseline = {test['label'].value_counts(normalize=True).max()*100:.1f} %)")

    # --- Marktrenditen aufs Test-Set ---
    market_ret = test["close"].pct_change().fillna(0)

    # --- Strategien ---
    cash_pos       = pd.Series(0,  index=test.index)
    bh_pos         = pd.Series(1,  index=test.index)

    # SMA-Crossover auf ganzen df berechnen, dann auf Test-Indizes zuschneiden
    sma_fast = sma(df["close"], 10)
    sma_slow = sma(df["close"], 50)
    sma_signal = (sma_fast > sma_slow).astype(int).reindex(test.index).fillna(0)

    ml_naive_pos = test["pred"].clip(lower=0)  # -1 → 0 (kein Short), 1 bleibt 1
    ml_filtered_pos = ((test["pred"] == 1) & (test["p_up"] > PROB_THRESHOLD)).astype(int)

    results = {
        "Cash":            evaluate(cash_pos,       market_ret, FEE_RATE),
        "Buy & Hold":      evaluate(bh_pos,         market_ret, FEE_RATE),
        "SMA(10/50)":      evaluate(sma_signal,     market_ret, FEE_RATE),
        "ML naiv (long-only)":   evaluate(ml_naive_pos,    market_ret, FEE_RATE),
        f"ML gefiltert P>{PROB_THRESHOLD:.2f}": evaluate(ml_filtered_pos, market_ret, FEE_RATE),
    }

    print(f"\n=== Ergebnisse (out-of-sample, {FEE_RATE*100:.1f} % Gebuehr pro Wechsel) ===")
    for name, res in results.items():
        print_row(name, res)

    print("\nDurchschnittliche Feature-Wichtigkeit ueber alle Folds:")
    for name, val in importance.items():
        print(f"  {name:12s} {val*100:5.1f} %")


if __name__ == "__main__":
    main()
