"""KNN-Pattern-Matching auf 15m-BTC.

Genau das, was der User beschrieben hat:
- Aktueller Bar wird mit allen historischen Bars verglichen
- Die K aehnlichsten historischen Setups werden gefunden
- Falls die Mehrheit von ihnen einen Gewinn-Outcome hatte → kaufen
- Pro Trade wird das Logbuch zeigen, welche historischen Patterns als
  Vorbild dienten (Interpretierbarkeit)
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.data.storage import load_ohlcv
from src.ml.features import (
    build_features_with_time,
    build_label_triple_barrier,
)
from src.backtest.engine import triple_barrier_backtest

warnings.filterwarnings("ignore", category=UserWarning)

EXCHANGE, SYMBOL, TIMEFRAME = "binance", "BTC/USDT", "15m"
STARTING_CAPITAL_CHF = 1000.0
FEE_RATE = 0.001

TP_PCT = 0.015
SL_PCT = 0.005
MAX_HOLD_BARS = 24

K_NEIGHBORS = 100               # Wie viele aehnliche historische Patterns?
ENTRY_PROB_THRESHOLD = 0.20     # Mind. 20 % der K Patterns muessen TP getroffen haben

TRAIN_FRAC = 0.70               # 70 % Train, 30 % Test (chronologisch)

REGIME_LOW_PCT = 25
REGIME_HIGH_PCT = 75
VOL_WINDOW_BARS = 96

DAILY_PROFIT_TARGET = 0.01
DAILY_LOSS_LIMIT = -0.01
COOLDOWN_BARS = 1

FIXED_POSITION = 0.15           # 15 % pro Trade
EXAMPLE_TRADES_TO_SHOW = 5      # Wie viele Trades sollen detailliert erklaert werden?


def main():
    print("Lade 15m-Daten …")
    df = load_ohlcv(EXCHANGE, SYMBOL, TIMEFRAME)
    print(f"  {len(df)} Kerzen von {df.index.min()} bis {df.index.max()}")

    print("\nBaue Features mit Zeit-Encoding …")
    X = build_features_with_time(df)
    print(f"  {X.shape[1]} Features: {list(X.columns)}")

    rets = df["close"].pct_change()
    regime_vol = rets.rolling(VOL_WINDOW_BARS).std()
    y = build_label_triple_barrier(df, tp_pct=TP_PCT, sl_pct=SL_PCT, max_hold=MAX_HOLD_BARS)

    data = X.copy()
    data["label"] = y
    data["close"] = df["close"]
    data["regime_vol"] = regime_vol
    data = data.dropna()
    data["label"] = data["label"].astype(int)
    features = list(X.columns)

    # Wir wollen nur "klare" Patterns trainieren (TP vs nicht-TP)
    # Label vereinfachen: +1 wenn TP, 0 sonst (binaer fuer KNN klarer)
    data["is_tp"] = (data["label"] == 1).astype(int)

    print(f"\nLabel-Anteile (Long-Sicht):")
    print(f"  TP-Anteil: {data['is_tp'].mean()*100:.1f} %  (Base-Rate)")

    # Chronologischer Split
    split = int(len(data) * TRAIN_FRAC)
    train = data.iloc[:split]
    test = data.iloc[split:].copy()
    print(f"\nTrain:  {train.index.min()} → {train.index.max()}  ({len(train)})")
    print(f"Test:   {test.index.min()} → {test.index.max()}  ({len(test)})")

    # StandardScaler — wichtig fuer KNN, weil unterschiedliche Feature-Skalen
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train[features])
    X_test_scaled = scaler.transform(test[features])

    print(f"\nTrainiere KNN (K={K_NEIGHBORS}) …")
    knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS, n_jobs=-1, weights="distance")
    knn.fit(X_train_scaled, train["is_tp"])

    print("Berechne P(TP) fuer alle Test-Kerzen …")
    proba = knn.predict_proba(X_test_scaled)
    classes = list(knn.classes_)
    test["p_tp"] = proba[:, classes.index(1)] if 1 in classes else 0.0

    # Regime-Filter
    vol_lo = float(train["regime_vol"].quantile(REGIME_LOW_PCT / 100))
    vol_hi = float(train["regime_vol"].quantile(REGIME_HIGH_PCT / 100))
    in_regime = (test["regime_vol"] >= vol_lo) & (test["regime_vol"] <= vol_hi)
    test["p_tp_filtered"] = np.where(in_regime, test["p_tp"], 0.0)

    n_above = int((test["p_tp_filtered"] > ENTRY_PROB_THRESHOLD).sum())
    print(f"  Signale mit P(TP) > {ENTRY_PROB_THRESHOLD}: {n_above} von {len(test)}")
    print(f"  Max P(TP) im Test: {test['p_tp_filtered'].max():.3f}")

    print(f"\nBacktest (Fixed {FIXED_POSITION*100:.0f} % Position, Daily-Limits aktiv) …")
    test["pos_frac"] = np.where(test["p_tp_filtered"] > ENTRY_PROB_THRESHOLD, FIXED_POSITION, 0.0)
    result = triple_barrier_backtest(
        close=test["close"],
        p_tp=test["p_tp_filtered"],
        entry_threshold=ENTRY_PROB_THRESHOLD,
        tp_pct=TP_PCT, sl_pct=SL_PCT, max_hold=MAX_HOLD_BARS,
        fee_rate=FEE_RATE, starting_capital=STARTING_CAPITAL_CHF,
        cooldown=COOLDOWN_BARS, position_fraction=test["pos_frac"].values,
        daily_profit_target=DAILY_PROFIT_TARGET, daily_loss_limit=DAILY_LOSS_LIMIT,
    )

    trades = result["trades"]
    equity = result["equity"]
    final = result["final_capital"]
    bh_final = STARTING_CAPITAL_CHF * (test["close"].iloc[-1] / test["close"].iloc[0])

    print("\n" + "=" * 78)
    print(f"=== KNN-PATTERN-MATCHING ERGEBNIS {test.index.min().date()} → {test.index.max().date()} ===")
    print("=" * 78)
    print(f"  Strategie:        {final:>10.2f} CHF   ({result['total_return']*100:+.2f} %)")
    print(f"  Buy & Hold:       {bh_final:>10.2f} CHF   ({(bh_final/STARTING_CAPITAL_CHF-1)*100:+.2f} %)")
    print(f"  Anzahl Trades:    {len(trades):>10d}")

    if len(trades) > 0:
        win_rate = (trades["trade_net_return"] > 0).mean()
        max_dd = (equity / equity.cummax() - 1).min()
        reasons = trades["reason"].value_counts().to_dict()
        net_tp = (1 + TP_PCT) * (1 - FEE_RATE) ** 2 - 1
        net_sl = (1 - SL_PCT) * (1 - FEE_RATE) ** 2 - 1
        be_win = abs(net_sl) / (net_tp + abs(net_sl))
        print(f"  Win-Rate:         {win_rate*100:>9.1f} %  (Break-Even: {be_win*100:.1f} %)")
        print(f"  Max Drawdown:     {max_dd*100:>9.2f} %")
        print(f"  Exits:            TP={reasons.get('TP',0)}  "
              f"SL={reasons.get('SL',0)}  TIME={reasons.get('TIME',0)}")

    # ----------- INTERPRETIERBARKEIT: Pattern-Beispiele -----------
    if len(trades) > 0:
        print("\n" + "=" * 78)
        print(f"=== INTERPRETIERBARKEIT: 'Warum hat der Bot gekauft?' ===")
        print("=" * 78)
        print(f"Fuer {min(EXAMPLE_TRADES_TO_SHOW, len(trades))} Beispiel-Trades zeige ich die "
              f"5 aehnlichsten historischen Patterns:")

        nn = NearestNeighbors(n_neighbors=5, n_jobs=-1)
        nn.fit(X_train_scaled)

        # Waehle gleichmaessig verteilte Beispiel-Trades
        sample_idx = np.linspace(0, len(trades) - 1, EXAMPLE_TRADES_TO_SHOW, dtype=int)
        for ti in sample_idx:
            trade = trades.iloc[ti]
            entry_ts = trade["entry_time"]
            # Finde Test-Index zur entry_time
            test_pos = test.index.get_indexer([entry_ts])[0]
            if test_pos < 0:
                continue
            query_vec = X_test_scaled[test_pos].reshape(1, -1)
            distances, indices = nn.kneighbors(query_vec)

            print(f"\n--- TRADE am {entry_ts}  (P_TP={trade['p_tp']:.3f}, "
                  f"Outcome: {trade['reason']}, "
                  f"Brutto {trade['gross_return']*100:+.2f} %) ---")
            print(f"  Aktuelle Indikatoren:")
            current = test.iloc[test_pos]
            for f in ["hour_sin", "hour_cos", "rsi_14", "macd_hist", "bb_pct",
                      "ema_ratio", "atr_14", "stoch_k", "adx_14", "obv_slope"]:
                if f in current.index:
                    print(f"    {f:<14s} = {current[f]:>+8.4f}")
            print(f"\n  5 aehnlichste historische Patterns:")
            for j, (dist, hist_pos) in enumerate(zip(distances[0], indices[0])):
                hist_ts = train.index[hist_pos]
                hist_label = train["label"].iloc[hist_pos]
                outcome_text = {1: "TP gewonnen", -1: "SL verloren", 0: "Time-Stop"}.get(hist_label, "?")
                print(f"    {j+1}. {hist_ts}  (Distanz {dist:.2f})  → {outcome_text}")

    # Daily-Statistik
    day_log = result["day_log"]
    if len(day_log) > 0:
        n_days = len(day_log)
        win_days = (day_log["day_return"] > 0).sum()
        loss_days = (day_log["day_return"] < 0).sum()
        active = (day_log["day_return"] != 0).sum()
        print(f"\n--- TAGES-STATISTIK ---")
        print(f"  Trading-Tage:          {n_days}")
        print(f"  Aktive Tage:           {active}  ({active/n_days*100:.1f} %)")
        print(f"  Gewinn-Tage:           {win_days}  ({win_days/n_days*100:.1f} %)")
        print(f"  Verlust-Tage:          {loss_days}  ({loss_days/n_days*100:.1f} %)")
        print(f"  Bester Tag:           {day_log['day_return'].max()*100:>+6.2f} %")
        print(f"  Schlechtester Tag:    {day_log['day_return'].min()*100:>+6.2f} %")


if __name__ == "__main__":
    main()
