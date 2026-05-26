"""Trend-Following + ML-Filter auf 1d-BTC.

Kombiniert klassische TA (Donchian-Breakout, ATR-Trailing-Stop) mit
ML-Vorhersage: Modell beantwortet 'ist das ein echter Trend oder Whipsaw?'.

Vergleich auf gleichem OOS-Zeitraum:
  A) Trend-Following PUR (ohne ML)
  B) Trend-Following + ML-Filter

So sehen wir EHRLICH, ob ML hier Mehrwert bringt.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance

from src.data.storage import load_ohlcv
from src.indicators.technical import sma, atr, adx
from src.ml.features import build_extended_features
from src.backtest.trend import trend_following_backtest

warnings.filterwarnings("ignore", category=UserWarning)

EXCHANGE, SYMBOL = "binance", "BTC/USDT"
INPUT_TIMEFRAME = "1h"
RESAMPLE_TO = "1D"
STARTING_CAPITAL_CHF = 1000.0
FEE_RATE = 0.001

DONCHIAN_PERIOD = 55
SMA_FAST = 20
SMA_SLOW = 50
ATR_PERIOD = 20
ATR_TRAILING_MULT = 5.0

POSITION_FRAC = 0.50
LEVERAGE = 2.0

# ML-spezifisch
ML_HORIZON_DAYS = 30                # "Trend laeuft 30 Tage weiter"
ML_TREND_THRESHOLD = 0.05           # +5 % in 30 Tagen = echter Trend
ML_PROB_THRESHOLD = 0.55            # ML muss ≥55 % Vertrauen haben
TRAIN_END_FRAC = 0.50               # erste 50 % = Training, danach OOS
WALK_FORWARD_FOLDS = 3              # 3 Folds in der OOS-Periode


def aggregate_to(df: pd.DataFrame, target: str) -> pd.DataFrame:
    return df.resample(target).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=300, max_depth=5, learning_rate=0.05,
        min_samples_leaf=30, l2_regularization=1.0, random_state=42,
    )


def main():
    print("Lade 1h-Daten …")
    df_1h = load_ohlcv(EXCHANGE, SYMBOL, INPUT_TIMEFRAME)
    df = aggregate_to(df_1h, RESAMPLE_TO)
    print(f"  {len(df)} 1d-Kerzen von {df.index.min().date()} bis {df.index.max().date()}")

    print("Berechne TA-Indikatoren …")
    df["sma_fast"] = sma(df["close"], SMA_FAST)
    df["sma_slow"] = sma(df["close"], SMA_SLOW)
    df["sma_200"] = sma(df["close"], 200)
    df["adx"] = adx(df["high"], df["low"], df["close"], ATR_PERIOD)
    df["atr"] = atr(df["high"], df["low"], df["close"], ATR_PERIOD)
    df["donchian_high"] = df["high"].rolling(DONCHIAN_PERIOD).max().shift(1)
    df["donchian_low"] = df["low"].rolling(DONCHIAN_PERIOD).min().shift(1)

    print("Baue ML-Features (TA-Basis + Trend-spezifisch) …")
    X = build_extended_features(df)
    # Trend-spezifische Features dazu
    X["dist_to_donchian_high"] = (df["donchian_high"] - df["close"]) / df["close"]
    X["dist_to_sma200"] = (df["close"] - df["sma_200"]) / df["sma_200"]
    X["atr_rel"] = df["atr"] / df["close"]
    X["sma_slope_20"] = df["sma_fast"].pct_change(10)
    X["sma_slope_50"] = df["sma_slow"].pct_change(20)
    print(f"  {X.shape[1]} Features")

    # Label: ist der Trend 30 Tage spaeter um >5 % gestiegen?
    forward_return = df["close"].shift(-ML_HORIZON_DAYS) / df["close"] - 1
    y = (forward_return > ML_TREND_THRESHOLD).astype(int)

    data = X.copy()
    data["label"] = y
    data["close"] = df["close"]
    # behalte OHLCV + Trend-Indikatoren fuer Backtest
    for col in ["open", "high", "low", "volume", "atr", "sma_fast",
                "donchian_high", "donchian_low"]:
        data[col] = df[col]
    data = data.dropna()
    data["label"] = data["label"].astype(int)
    features = list(X.columns)

    print(f"\nLabel-Verteilung: {data['label'].mean()*100:.1f} % der Bars haben echten Trend "
          f"(>{ML_TREND_THRESHOLD*100:.0f} % in {ML_HORIZON_DAYS}d)")
    print(f"Datenmenge: {len(data)} Bars, ML-Horizon = {ML_HORIZON_DAYS} Tage")

    # Walk-Forward Predictions
    n = len(data)
    train_end_init = int(n * TRAIN_END_FRAC)
    remaining = n - train_end_init
    fold_size = remaining // WALK_FORWARD_FOLDS

    print(f"\nWalk-Forward: erste {train_end_init} Bars als Initial-Train, "
          f"danach {WALK_FORWARD_FOLDS} Folds × {fold_size} Test-Bars …")

    all_test = []
    last_model = None
    for fold in range(WALK_FORWARD_FOLDS):
        train_end = train_end_init + fold * fold_size
        test_end = train_end + fold_size if fold < WALK_FORWARD_FOLDS - 1 else n
        # Kalibrierung: letzte 15 % des Trainings als Hold-out
        cal_split = int(train_end * 0.85)
        train = data.iloc[:cal_split]
        cal = data.iloc[cal_split:train_end]
        test = data.iloc[train_end:test_end].copy()

        base = make_model()
        base.fit(train[features], train["label"])
        model = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
        model.fit(cal[features], cal["label"])

        classes = list(model.classes_)
        test["p_trend"] = (
            model.predict_proba(test[features])[:, classes.index(1)] if 1 in classes else 0.0
        )
        all_test.append(test)
        last_model = model
        print(f"  Fold {fold+1}/{WALK_FORWARD_FOLDS}: "
              f"train→{data.index[cal_split-1].date()}  "
              f"test {test.index[0].date()}→{test.index[-1].date()}  "
              f"({len(test)} Bars)  max P(trend)={test['p_trend'].max():.3f}")

    test_all = pd.concat(all_test).sort_index()

    # OOS-Backtest-Df: muss alle fuer den Backtest notwendigen Spalten haben
    backtest_df = test_all[["open", "high", "low", "close", "volume", "atr",
                            "sma_fast", "donchian_high", "donchian_low",
                            "p_trend"]].copy()

    # Signale auf dem OOS-Set
    long_entry_pure = backtest_df["close"] >= backtest_df["donchian_high"]
    long_entry_ml = long_entry_pure & (backtest_df["p_trend"] >= ML_PROB_THRESHOLD)
    short_entry = pd.Series(False, index=backtest_df.index)

    print(f"\nSignal-Statistik auf OOS:")
    print(f"  Donchian-Breakouts (PUR):      {int(long_entry_pure.sum())}")
    print(f"  davon mit ML-Bestaetigung:     {int(long_entry_ml.sum())}")
    print(f"  ML-Filter blockiert:           {int(long_entry_pure.sum() - long_entry_ml.sum())}")

    # --- Backtest A: PUR ---
    print(f"\n--- Backtest A: PUR (ohne ML-Filter) ---")
    res_pure = trend_following_backtest(
        df=backtest_df, long_entry=long_entry_pure, short_entry=short_entry,
        atr=backtest_df["atr"], starting_capital=STARTING_CAPITAL_CHF,
        position_fraction=POSITION_FRAC, fee_rate=FEE_RATE, leverage=LEVERAGE,
        atr_mult=ATR_TRAILING_MULT, sma_exit=None,
    )

    # --- Backtest B: MIT ML-Filter ---
    print(f"--- Backtest B: MIT ML-Filter (P_trend ≥ {ML_PROB_THRESHOLD}) ---")
    res_ml = trend_following_backtest(
        df=backtest_df, long_entry=long_entry_ml, short_entry=short_entry,
        atr=backtest_df["atr"], starting_capital=STARTING_CAPITAL_CHF,
        position_fraction=POSITION_FRAC, fee_rate=FEE_RATE, leverage=LEVERAGE,
        atr_mult=ATR_TRAILING_MULT, sma_exit=None,
    )

    bh_final = STARTING_CAPITAL_CHF * (backtest_df["close"].iloc[-1] / backtest_df["close"].iloc[0])

    def report(name, res):
        trades = res["trades"]
        eq = res["equity"]
        max_dd = (eq / eq.cummax() - 1).min() if len(eq) else 0
        wr = (trades["trade_net_return"] > 0).mean() if len(trades) else 0
        winners = trades[trades["trade_net_return"] > 0]
        losers = trades[trades["trade_net_return"] < 0]
        avg_w = winners["trade_net_return"].mean() if len(winners) else 0
        avg_l = losers["trade_net_return"].mean() if len(losers) else 0
        rr = abs(avg_w / avg_l) if avg_l < 0 else float("inf")
        print(f"\n  === {name} ===")
        print(f"    Endkapital:    {res['final_capital']:>10.2f} CHF  ({res['total_return']*100:+.2f} %)")
        print(f"    Trades:        {len(trades):>5d}    Win-Rate: {wr*100:.1f} %    R:R: {rr:.2f}x")
        print(f"    Max Drawdown:  {max_dd*100:>+9.2f} %")
        if len(trades) > 0:
            print(f"    Avg Win/Loss:  +{avg_w*100:.2f} % / {avg_l*100:.2f} %")
            print(f"    Best/Worst:    {trades['trade_net_return'].max()*100:+.2f} % / "
                  f"{trades['trade_net_return'].min()*100:+.2f} %")
        return trades

    print("\n" + "=" * 78)
    print(f"=== OOS-VERGLEICH ({backtest_df.index[0].date()} → {backtest_df.index[-1].date()}) ===")
    print("=" * 78)
    print(f"  Buy & Hold (1x):  {bh_final:>10.2f} CHF  "
          f"({(bh_final/STARTING_CAPITAL_CHF-1)*100:+.2f} %)")
    trades_pure = report("PUR (Donchian only)", res_pure)
    trades_ml = report(f"MIT ML-Filter (P ≥ {ML_PROB_THRESHOLD})", res_ml)

    # ML-Detail: welche Trades hat ML blockiert/durchgelassen?
    if len(trades_pure) > 0:
        print(f"\n--- TRADE-DETAIL: was hat ML gemacht? ---")
        # Map jeden pure-trade auf "haben wir ihn mit ML auch gemacht?"
        ml_entry_set = set(trades_ml["entry_time"]) if len(trades_ml) else set()
        for _, t in trades_pure.iterrows():
            taken_by_ml = t["entry_time"] in ml_entry_set
            ml_p = backtest_df.loc[t["entry_time"], "p_trend"] if t["entry_time"] in backtest_df.index else float("nan")
            marker = "✓ ML auch" if taken_by_ml else "✗ ML blockiert"
            print(f"    {str(t['entry_time'])[:10]}  Trade brutto {t['gross_return']*100:>+7.2f}%  "
                  f"P_trend={ml_p:.2f}  {marker}")

    # Feature Importance
    if last_model is not None:
        last_test = all_test[-1]
        try:
            perm = permutation_importance(
                last_model, last_test[features], last_test["label"],
                n_repeats=3, random_state=42, n_jobs=-1,
            )
            imp = pd.DataFrame({"feature": features, "importance": perm.importances_mean}).sort_values("importance", ascending=False)
            print(f"\n--- Top-10 wichtigste Features fuer Trend-Vorhersage ---")
            for _, r in imp.head(10).iterrows():
                print(f"    {r['feature']:<28s} {r['importance']*1000:>+8.2f} e-3")
        except Exception as e:
            print(f"  (Permutation Importance fehlgeschlagen: {e})")


if __name__ == "__main__":
    main()
