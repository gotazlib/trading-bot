"""Multi-Asset Trend-Following + ML auf BTC, ETH, SOL, BNB, XRP.

Architektur:
- Pro Coin: 1d-Daten, Donchian-Breakout-Signal, kalibriertes ML-Modell
- Portfolio: max 20 % pro Coin, max 80 % Gesamt-Exposure
- 0.5 % Daily Portfolio-Loss-Limit → keine neuen Entries an dem Tag
- 2x Hebel auf jede Position
- 5x ATR Chandelier-Trailing-Stop
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
from src.backtest.trend import multi_asset_trend_backtest

warnings.filterwarnings("ignore", category=UserWarning)

EXCHANGE = "binance"
SYMBOLS = ["BTC/USDT", "ETH/USDT", "TRX/USDT"]   # Top-3 mit messbarem Edge
INPUT_TIMEFRAME = "1h"
RESAMPLE_TO = "1D"
STARTING_CAPITAL_CHF = 1000.0
FEE_RATE = 0.001

DONCHIAN_PERIOD = 55
ATR_PERIOD = 20
ATR_TRAILING_MULT = 5.0

LEVERAGE = 2.0
MAX_POSITION_PER_COIN = 0.30
MAX_TOTAL_EXPOSURE = 0.85
DAILY_LOSS_LIMIT = -0.005

ML_HORIZON_DAYS = 30
ML_TREND_THRESHOLD = 0.05
ML_PROB_THRESHOLD = 0.55          # Qualitaet > Quantitaet
ALLOW_SHORTS = True
SHORT_BIAS_EXTRA = 0.05           # Short braucht +0.05 mehr Konfidenz
TRAIN_END_FRAC = 0.50
N_FOLDS = 3


def aggregate_to(df, target):
    return df.resample(target).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=300, max_depth=5, learning_rate=0.05,
        min_samples_leaf=30, l2_regularization=1.0, random_state=42,
    )


def prepare_coin(symbol):
    """Laedt Coin-Daten, baut Indikatoren + ML-Features. Liefert (data_full)."""
    df_1h = load_ohlcv(EXCHANGE, symbol, INPUT_TIMEFRAME)
    if df_1h.empty:
        return None
    df = aggregate_to(df_1h, RESAMPLE_TO)
    df["atr"] = atr(df["high"], df["low"], df["close"], ATR_PERIOD)
    df["sma_fast"] = sma(df["close"], 20)
    df["sma_slow"] = sma(df["close"], 50)
    df["sma_200"] = sma(df["close"], 200)
    df["adx"] = adx(df["high"], df["low"], df["close"], ATR_PERIOD)
    df["donchian_high"] = df["high"].rolling(DONCHIAN_PERIOD).max().shift(1)
    df["donchian_low"] = df["low"].rolling(DONCHIAN_PERIOD).min().shift(1)

    X = build_extended_features(df)
    X["dist_to_donchian_high"] = (df["donchian_high"] - df["close"]) / df["close"]
    X["dist_to_sma200"] = (df["close"] - df["sma_200"]) / df["sma_200"]
    X["atr_rel"] = df["atr"] / df["close"]
    X["sma_slope_20"] = df["sma_fast"].pct_change(10)
    X["sma_slope_50"] = df["sma_slow"].pct_change(20)

    forward_return = df["close"].shift(-ML_HORIZON_DAYS) / df["close"] - 1
    y_long = (forward_return > ML_TREND_THRESHOLD).astype(int)
    y_short = (forward_return < -ML_TREND_THRESHOLD).astype(int)

    data = X.copy()
    data["label"] = y_long          # Long-Label (Backward-Compat)
    data["label_short"] = y_short   # Short-Label
    for c in ["close", "high", "low", "atr", "donchian_high", "donchian_low"]:
        data[c] = df[c]
    data = data.dropna()
    data["label"] = data["label"].astype(int)
    data["label_short"] = data["label_short"].astype(int)
    return data, list(X.columns)


def run_walk_forward_ml(data, features):
    """Walk-Forward: trainiert LONG- und SHORT-Modelle parallel pro Fold."""
    n = len(data)
    train_end_init = int(n * TRAIN_END_FRAC)
    if train_end_init < 200:
        return pd.DataFrame(), None
    remaining = n - train_end_init
    fold_size = max(50, remaining // N_FOLDS)
    all_test = []
    last_model = None
    for fold in range(N_FOLDS):
        train_end = train_end_init + fold * fold_size
        test_end = train_end + fold_size if fold < N_FOLDS - 1 else n
        if test_end > n: test_end = n
        cal_split = int(train_end * 0.85)
        if cal_split >= train_end - 20:
            cal_split = train_end - 20
        train = data.iloc[:cal_split]
        cal = data.iloc[cal_split:train_end]
        test = data.iloc[train_end:test_end].copy()
        if len(train) < 100 or len(cal) < 20 or len(test) < 10:
            continue
        try:
            # Long-Modell
            base_l = make_model()
            base_l.fit(train[features], train["label"])
            m_long = CalibratedClassifierCV(FrozenEstimator(base_l), method="isotonic")
            m_long.fit(cal[features], cal["label"])
            lc = list(m_long.classes_)
            test["p_trend"] = (m_long.predict_proba(test[features])[:, lc.index(1)]
                               if 1 in lc else 0.0)

            # Short-Modell
            if train["label_short"].sum() >= 10 and cal["label_short"].sum() >= 3:
                base_s = make_model()
                base_s.fit(train[features], train["label_short"])
                m_short = CalibratedClassifierCV(FrozenEstimator(base_s), method="isotonic")
                m_short.fit(cal[features], cal["label_short"])
                sc = list(m_short.classes_)
                test["p_short"] = (m_short.predict_proba(test[features])[:, sc.index(1)]
                                   if 1 in sc else 0.0)
            else:
                test["p_short"] = 0.0
            all_test.append(test)
            last_model = m_long
        except Exception as e:
            print(f"    [Fold {fold+1}] Fehler: {e}")
            continue
    if not all_test:
        return pd.DataFrame(), None
    return pd.concat(all_test).sort_index(), last_model


def main():
    print("Bereite Coins vor …")
    coins_prepared = {}
    common_features = None
    for sym in SYMBOLS:
        result = prepare_coin(sym)
        if result is None:
            print(f"  {sym}: keine Daten")
            continue
        data, features = result
        coins_prepared[sym] = (data, features)
        common_features = features
        print(f"  {sym}: {len(data)} 1d-Bars von {data.index.min().date()} bis {data.index.max().date()}, "
              f"Label-Anteil: {data['label'].mean()*100:.1f} %")

    print(f"\nWalk-Forward ML pro Coin ({N_FOLDS} Folds) …")
    coins_oos = {}
    coin_models = {}
    coin_features = {}
    for sym, (data, features) in coins_prepared.items():
        oos, model = run_walk_forward_ml(data, features)
        if oos.empty:
            print(f"  {sym}: keine OOS-Predictions")
            continue
        oos["signal"] = (
            (oos["close"] >= oos["donchian_high"])
            & (oos["p_trend"] >= ML_PROB_THRESHOLD)
        )
        oos["short_signal"] = (
            (oos["close"] <= oos["donchian_low"])
            & (oos["p_short"] >= ML_PROB_THRESHOLD + SHORT_BIAS_EXTRA)
        )
        coins_oos[sym] = oos[["close", "high", "low", "atr",
                              "p_trend", "p_short", "signal", "short_signal"]]
        coin_models[sym] = model
        coin_features[sym] = features
        print(f"  {sym}: OOS {oos.index[0].date()} → {oos.index[-1].date()} "
              f"({len(oos)} Bars), Long-Signale: {int(oos['signal'].sum())}, "
              f"Short-Signale: {int(oos['short_signal'].sum())}, "
              f"max P_long={oos['p_trend'].max():.2f}, max P_short={oos['p_short'].max():.2f}")

    # Per-Coin Feature Importance auf der OOS-Periode
    print(f"\n=== Per-Coin Top-5 Features (Permutation Importance) ===")
    for sym, (data, features) in coins_prepared.items():
        if sym not in coin_models or coin_models[sym] is None:
            continue
        try:
            test_subset = data.iloc[-200:][features + ["label"]].dropna()
            if len(test_subset) < 50:
                continue
            perm = permutation_importance(
                coin_models[sym], test_subset[features], test_subset["label"],
                n_repeats=2, random_state=42, n_jobs=-1,
            )
            imp = pd.DataFrame({"f": features, "imp": perm.importances_mean}).sort_values("imp", ascending=False)
            top5 = ", ".join(f"{r['f']}({r['imp']*1000:.1f})" for _, r in imp.head(5).iterrows())
            print(f"  {sym}: {top5}")
        except Exception as e:
            print(f"  {sym}: ({e})")

    if not coins_oos:
        print("Keine OOS-Daten — Abbruch.")
        return

    # Backtest
    print(f"\nMulti-Asset Backtest:")
    print(f"  Hebel:              {LEVERAGE}x")
    print(f"  Max pro Coin:       {MAX_POSITION_PER_COIN*100:.0f} %")
    print(f"  Max Gesamt:         {MAX_TOTAL_EXPOSURE*100:.0f} %")
    print(f"  Daily Loss Limit:   {DAILY_LOSS_LIMIT*100:+.1f} %")
    print(f"  Trailing-Stop:      {ATR_TRAILING_MULT}× ATR")

    res = multi_asset_trend_backtest(
        coins_data=coins_oos,
        starting_capital=STARTING_CAPITAL_CHF,
        fee_rate=FEE_RATE, leverage=LEVERAGE,
        atr_mult=ATR_TRAILING_MULT,
        max_position_per_coin=MAX_POSITION_PER_COIN,
        max_total_exposure=MAX_TOTAL_EXPOSURE,
        daily_loss_limit=DAILY_LOSS_LIMIT,
        allow_shorts=ALLOW_SHORTS,
    )

    # Equal-Weight B&H als Vergleich
    bh_returns = []
    for sym, df in coins_oos.items():
        bh_returns.append(df["close"].iloc[-1] / df["close"].iloc[0])
    bh_avg = np.mean(bh_returns)
    bh_final = STARTING_CAPITAL_CHF * bh_avg

    trades = res["trades"]
    equity = res["equity"]
    day_log = res["day_log"]

    print(f"\n{'='*78}")
    print(f"=== MULTI-ASSET ERGEBNIS ===")
    print(f"{'='*78}")
    print(f"  Strategie ({LEVERAGE}x):  {res['final_capital']:>10.2f} CHF   "
          f"({res['total_return']*100:+.2f} %)")
    print(f"  B&H (Equal-Weight):  {bh_final:>10.2f} CHF   "
          f"({(bh_avg-1)*100:+.2f} %)")
    print(f"  Anzahl Trades:       {len(trades):>10d}")

    if len(trades) > 0:
        win_rate = (trades["net_return"] > 0).mean()
        max_dd = (equity / equity.cummax() - 1).min()
        liquidations = (trades["reason"] == "LIQUIDATED").sum()
        print(f"  Win-Rate:            {win_rate*100:>9.1f} %")
        print(f"  Max Drawdown:        {max_dd*100:>9.2f} %")
        print(f"  Liquidationen:       {liquidations}")

        print(f"\n--- Trades pro Coin ---")
        for sym in coins_oos.keys():
            sym_trades = trades[trades["symbol"] == sym]
            if len(sym_trades) == 0:
                print(f"  {sym:>10s}: 0 Trades")
                continue
            wr = (sym_trades["net_return"] > 0).mean()
            total = sym_trades["pnl_chf"].sum()
            print(f"  {sym:>10s}: {len(sym_trades)} Trades, "
                  f"WR {wr*100:.1f}%, PnL +{total:.2f} CHF")

        print(f"\n--- Trade-Liste ---")
        for _, t in trades.sort_values("entry_time").iterrows():
            dirstr = "LONG" if t.get("direction", 1) == 1 else "SHORT"
            print(f"  {str(t['entry_time'])[:10]} → {str(t['exit_time'])[:10]}  "
                  f"{t['symbol']:>10s}  {dirstr:>5s}  pos={t['position_frac']*100:.0f}%  "
                  f"brutto {t['gross_return']*100:>+6.2f}%  "
                  f"netto {t['net_return']*100:>+6.2f}%  "
                  f"PnL {t['pnl_chf']:>+7.2f} CHF  {t['reason']}")

    if len(day_log) > 0:
        n_d = len(day_log)
        active = (day_log["day_return"] != 0).sum()
        win_d = (day_log["day_return"] > 0).sum()
        loss_d = (day_log["day_return"] < 0).sum()
        hit_limit = (day_log["day_return"] <= DAILY_LOSS_LIMIT).sum()
        print(f"\n--- TAGES-STATISTIK ({n_d} Tage) ---")
        print(f"  Aktive Tage:           {active}  ({active/n_d*100:.1f} %)")
        print(f"  Gewinn-Tage:           {win_d}  ({win_d/n_d*100:.1f} %)")
        print(f"  Verlust-Tage:          {loss_d}  ({loss_d/n_d*100:.1f} %)")
        print(f"  Tage mit ≤-0.5%-Limit: {hit_limit}")
        print(f"  Bester Tag:           {day_log['day_return'].max()*100:>+6.2f} %")
        print(f"  Schlechtester Tag:    {day_log['day_return'].min()*100:>+6.2f} %")


if __name__ == "__main__":
    main()
