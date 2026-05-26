"""Walk-Forward ML-Backtest auf 1m-Kerzen mit Indikator-Interaktionen.

Idee:
- Basis-Indikatoren PLUS alle paarweisen Produkte als Features
- HistGradientBoostingClassifier (besser als RandomForest fuer Interaktionen)
- Permutation Importance auf dem letzten Fold zeigt, welche Indikator(-Paare)
  out-of-sample wirklich praediktiv sind
- Trade-Filter via predict_proba schuetzt vor Trade-Spam
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

from src.data.storage import load_ohlcv
from src.ml.features import build_features_with_interactions, build_label_vol_scaled
from src.backtest.engine import evaluate, print_row
from src.indicators.technical import sma

warnings.filterwarnings("ignore", category=UserWarning)

EXCHANGE, SYMBOL, TIMEFRAME = "binance", "BTC/USDT", "1m"
HORIZON = 15          # 15 Minuten voraus
VOL_WINDOW = 60       # 60-Minuten-Vola als Skala
K_VOL = 0.5
FEE_RATE = 0.001      # 0.1 % pro Positionswechsel
PROB_THRESHOLD = 0.50 # niedriger als bei 1h, weil 3 Klassen seltener > 0.5 erreichen
N_FOLDS = 3
INITIAL_TRAIN_FRAC = 0.5
PERM_REPEATS = 5      # Permutation-Importance: Repeats fuer Stabilitaet
TOP_N = 20            # Top-Features in der Auswertung


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=200,
        l2_regularization=1.0,
        random_state=42,
    )


def walk_forward_predict(data, features, n_folds, initial_train_frac):
    n = len(data)
    initial_train = int(n * initial_train_frac)
    test_size = (n - initial_train) // n_folds
    test_chunks = []
    last_model, last_test = None, None

    for fold in range(n_folds):
        train_end = initial_train + fold * test_size
        test_end = train_end + test_size if fold < n_folds - 1 else n
        train = data.iloc[:train_end]
        test = data.iloc[train_end:test_end].copy()

        model = make_model()
        model.fit(train[features], train["label"])

        classes = list(model.classes_)
        proba = model.predict_proba(test[features])
        test["pred"] = model.predict(test[features])
        test["p_up"]   = proba[:, classes.index(1)]  if  1 in classes else 0.0
        test["p_down"] = proba[:, classes.index(-1)] if -1 in classes else 0.0
        test["fold"] = fold

        test_chunks.append(test)
        last_model, last_test = model, test

        print(
            f"  Fold {fold+1}/{n_folds}: "
            f"train {train.index.min().date()}–{train.index.max().date()} ({len(train)}) | "
            f"test  {test.index.min().date()}–{test.index.max().date()} ({len(test)})"
        )

    return pd.concat(test_chunks), last_model, last_test


def main():
    print("Lade 1m-Daten …")
    df = load_ohlcv(EXCHANGE, SYMBOL, TIMEFRAME)
    print(f"  {len(df)} 1m-Kerzen von {df.index.min()} bis {df.index.max()}")

    print("Baue Features (Basis + paarweise Produkte) …")
    X = build_features_with_interactions(df)
    y = build_label_vol_scaled(df, horizon=HORIZON, vol_window=VOL_WINDOW, k=K_VOL)
    print(f"  Feature-Anzahl: {X.shape[1]}  (10 Basis + {X.shape[1]-10} Paare)")

    data = X.copy()
    data["label"] = y
    data["close"] = df["close"]
    data = data.dropna()
    data["label"] = data["label"].astype(int)
    features = list(X.columns)

    counts = data["label"].value_counts().to_dict()
    print(f"\nLabel-Verteilung (vol-skaliert, k={K_VOL}, horizon={HORIZON}m):")
    for cls in sorted(counts):
        share = counts[cls] / len(data) * 100
        print(f"  {cls:+d}: {counts[cls]:>7d}  ({share:4.1f} %)")

    print(f"\nWalk-Forward ({N_FOLDS} Folds, Start-Train {INITIAL_TRAIN_FRAC*100:.0f} %):")
    test, last_model, last_test = walk_forward_predict(
        data, features, N_FOLDS, INITIAL_TRAIN_FRAC
    )

    acc = accuracy_score(test["label"], test["pred"])
    majority = test["label"].value_counts(normalize=True).max()
    print(
        f"\nOut-of-Sample Trefferquote (3 Klassen): {acc*100:.1f} %   "
        f"(Random ≈ 33 %, gr. Klasse = {majority*100:.1f} %)"
    )

    # --- Backtest-Vergleich ---
    market_ret = test["close"].pct_change().fillna(0)

    cash_pos = pd.Series(0, index=test.index)
    bh_pos = pd.Series(1, index=test.index)

    sma_fast = sma(df["close"], 10)
    sma_slow = sma(df["close"], 50)
    sma_signal = (sma_fast > sma_slow).astype(int).reindex(test.index).fillna(0)

    ml_naive_pos = test["pred"].clip(lower=0)
    ml_filtered_pos = ((test["pred"] == 1) & (test["p_up"] > PROB_THRESHOLD)).astype(int)

    results = {
        "Cash":                            evaluate(cash_pos,        market_ret, FEE_RATE),
        "Buy & Hold":                      evaluate(bh_pos,          market_ret, FEE_RATE),
        "SMA(10/50)":                      evaluate(sma_signal,      market_ret, FEE_RATE),
        "ML naiv (long-only)":             evaluate(ml_naive_pos,    market_ret, FEE_RATE),
        f"ML gefiltert P_up>{PROB_THRESHOLD:.2f}": evaluate(ml_filtered_pos, market_ret, FEE_RATE),
    }

    print(f"\n=== Ergebnisse (out-of-sample, {FEE_RATE*100:.1f} % Gebuehr pro Wechsel) ===")
    for name, res in results.items():
        print_row(name, res)

    # --- Permutation Importance auf dem letzten Fold ---
    print(f"\nBerechne Permutation Importance auf Fold {N_FOLDS} "
          f"(n_repeats={PERM_REPEATS}, dauert je nach Datenmenge etwas) …")
    perm = permutation_importance(
        last_model, last_test[features], last_test["label"],
        n_repeats=PERM_REPEATS, random_state=42, n_jobs=-1,
    )
    imp = pd.DataFrame({
        "feature": features,
        "importance": perm.importances_mean,
        "std": perm.importances_std,
    }).sort_values("importance", ascending=False)

    is_pair = imp["feature"].str.contains(" X ")
    print(f"\nTop {TOP_N} Features insgesamt (out-of-sample Permutation Importance):")
    print(f"  {'Feature':<32s} {'Importance':>12s}  ±std")
    for _, row in imp.head(TOP_N).iterrows():
        print(f"  {row['feature']:<32s} {row['importance']*1000:>9.2f} e-3  ±{row['std']*1000:.2f}")

    print(f"\nTop 10 EINZEL-Indikatoren:")
    for _, row in imp[~is_pair].head(10).iterrows():
        print(f"  {row['feature']:<32s} {row['importance']*1000:>9.2f} e-3")

    print(f"\nTop 15 INDIKATOR-PAARE (das war die eigentliche Frage):")
    for _, row in imp[is_pair].head(15).iterrows():
        print(f"  {row['feature']:<32s} {row['importance']*1000:>9.2f} e-3")

    # Negative Importance = Feature SCHADET dem Modell out-of-sample
    harmful = imp[imp["importance"] < -1e-5]
    if len(harmful):
        print(f"\nFeatures mit negativer Importance ({len(harmful)} Stueck) — pures Rauschen / Overfitting-Quelle:")
        for _, row in harmful.head(10).iterrows():
            print(f"  {row['feature']:<32s} {row['importance']*1000:>9.2f} e-3")


if __name__ == "__main__":
    main()
