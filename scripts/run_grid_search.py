"""Systematische ML-Grid-Search ueber alle relevanten Trading-Parameter.

Vorgehen (Schutz vor Multiple-Testing-Bias):
1. Daten in Optimization-Set (60 %) und Holdout-Set (40 %) trennen.
2. Auf dem Optimization-Set: 3 Walk-Forward-Folds, fuer JEDE Parameter-Kombi
   Long+Short-Modelle trainieren, kalibrieren und backtesten.
3. Beste Konfigurationen ranken (nach Sharpe-Ratio).
4. Top-1-Konfiguration FINAL auf dem ungesehenen Holdout-Set testen.
5. Wenn Holdout-Performance nahe Optimization-Performance → echter Edge.
   Wenn deutlich schlechter → war nur Glueck (Overfitting).

Erweiterte Features (17 statt 10) inklusive ATR, Stoch, ROC, CCI, Williams %R,
OBV, ADX. HistGradientBoosting lernt Interaktionen automatisch.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator

from src.data.storage import load_ohlcv
from src.ml.features import (
    build_extended_features,
    build_label_triple_barrier,
    build_label_triple_barrier_short,
)
from src.backtest.engine import triple_barrier_backtest_ls

warnings.filterwarnings("ignore", category=UserWarning)

EXCHANGE, SYMBOL, TIMEFRAME = "binance", "BTC/USDT", "15m"
STARTING_CAPITAL_CHF = 1000.0
FEE_RATE = 0.001

# Splits
OPT_FRAC = 0.60          # Optimization-Set (Grid-Search hier)
# Holdout = 1 - OPT_FRAC

# Walk-Forward innerhalb Optimization
N_FOLDS_OPT = 3
CAL_FRAC_OF_FOLD = 0.30

# Grid (jede Kombination wird getestet)
TP_GRID = [0.010, 0.015, 0.020]
SL_GRID = [0.003, 0.005]
HOLD_GRID = [24, 48]              # 6h, 12h
THRESHOLD_GRID = [0.20, 0.30]

# Sizing & Filter (fix)
SHORT_THRESHOLD_EXTRA = 0.10
MIN_POSITION_FRAC = 0.10
MAX_POSITION_FRAC = 0.50
KELLY_FACTOR = 1.0

REGIME_LOW_PCT = 25
REGIME_HIGH_PCT = 75
VOL_WINDOW_BARS = 96

DAILY_PROFIT_TARGET = 0.01
DAILY_LOSS_LIMIT = -0.01
COOLDOWN_BARS = 1


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=250, max_depth=6, learning_rate=0.06,
        min_samples_leaf=150, l2_regularization=1.0, random_state=42,
    )


def kelly_fraction(p_tp, net_tp, net_sl, threshold,
                   kelly_factor=KELLY_FACTOR, cap=MAX_POSITION_FRAC, floor=MIN_POSITION_FRAC):
    b = net_tp / abs(net_sl)
    f = p_tp * (1.0 + 1.0 / b) - 1.0 / b
    f = np.maximum(f, 0.0) * kelly_factor
    f = np.minimum(f, cap)
    return np.where(p_tp > threshold, np.maximum(f, floor), 0.0)


def train_and_calibrate(train, cal, features, label_col):
    base = make_model()
    base.fit(train[features], train[label_col])
    model = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    model.fit(cal[features], cal[label_col])
    return model


def evaluate_config(data_split, features, tp, sl, hold, threshold, n_folds, fold_size_target):
    """Walk-Forward Backtest fuer eine Konfiguration. Liefert Metriken-Dict."""
    n = len(data_split)
    fold_size = max(2000, n // n_folds)
    if fold_size * n_folds > n:
        fold_size = n // n_folds

    all_test = []
    for fold in range(n_folds):
        end = min((fold + 1) * fold_size, n)
        train_end = end - fold_size + int(fold_size * (1 - CAL_FRAC_OF_FOLD))
        if fold == 0:
            train_end = int(fold_size * 0.5)
        cal_end = train_end + int(fold_size * CAL_FRAC_OF_FOLD * 0.5)
        # Pragmatisch: jedes Fold ist eigen — train auf allem davor, cal+test im Fold
        fold_start = fold * fold_size
        fold_train_end = fold_start + int(fold_size * 0.55)
        fold_cal_end = fold_start + int(fold_size * 0.70)
        fold_end = end

        train = data_split.iloc[:fold_train_end]
        cal = data_split.iloc[fold_train_end:fold_cal_end]
        test = data_split.iloc[fold_cal_end:fold_end].copy()
        if len(cal) < 200 or len(test) < 200 or len(train) < 1000:
            continue

        m_long = train_and_calibrate(train, cal, features, "label_long")
        m_short = train_and_calibrate(train, cal, features, "label_short")
        long_classes = list(m_long.classes_)
        short_classes = list(m_short.classes_)
        test["p_long"] = m_long.predict_proba(test[features])[:, long_classes.index(1)] if 1 in long_classes else 0.0
        test["p_short"] = m_short.predict_proba(test[features])[:, short_classes.index(1)] if 1 in short_classes else 0.0

        vol_lo = float(train["regime_vol"].quantile(REGIME_LOW_PCT / 100))
        vol_hi = float(train["regime_vol"].quantile(REGIME_HIGH_PCT / 100))
        in_regime = (test["regime_vol"] >= vol_lo) & (test["regime_vol"] <= vol_hi)
        test["p_long_f"] = np.where(in_regime, test["p_long"], 0.0)
        test["p_short_f"] = np.where(
            (test["p_short"] > threshold + SHORT_THRESHOLD_EXTRA) & in_regime,
            test["p_short"], 0.0,
        )
        all_test.append(test)

    if not all_test:
        return None

    test_all = pd.concat(all_test).sort_index()
    net_tp = (1 + tp) * (1 - FEE_RATE) ** 2 - 1
    net_sl = (1 - sl) * (1 - FEE_RATE) ** 2 - 1
    p_used = np.maximum(test_all["p_long_f"].values, test_all["p_short_f"].values)
    pos_frac = kelly_fraction(p_used, net_tp, abs(net_sl), threshold)

    res = triple_barrier_backtest_ls(
        close=test_all["close"],
        p_tp_long=test_all["p_long_f"],
        p_tp_short=test_all["p_short_f"],
        entry_threshold=threshold,
        tp_pct=tp, sl_pct=sl, max_hold=hold,
        fee_rate=FEE_RATE, starting_capital=STARTING_CAPITAL_CHF,
        cooldown=COOLDOWN_BARS, position_fraction=pos_frac,
        daily_profit_target=DAILY_PROFIT_TARGET, daily_loss_limit=DAILY_LOSS_LIMIT,
    )

    trades = res["trades"]
    equity = res["equity"]
    eq_returns = equity.pct_change().dropna()
    sharpe = (eq_returns.mean() / eq_returns.std() * np.sqrt(96 * 365)) if eq_returns.std() > 0 else 0.0
    max_dd = (equity / equity.cummax() - 1).min() if len(equity) else 0.0
    win_rate = (trades["trade_net_return"] > 0).mean() if len(trades) else 0.0
    n_trades = len(trades)
    bh_return = float(test_all["close"].iloc[-1] / test_all["close"].iloc[0] - 1) if len(test_all) else 0.0

    return {
        "tp": tp, "sl": sl, "hold": hold, "threshold": threshold,
        "total_return": res["total_return"],
        "sharpe": sharpe, "max_dd": max_dd,
        "win_rate": win_rate, "n_trades": n_trades,
        "bh_return": bh_return,
        "vs_bh": res["total_return"] - bh_return,
    }


def main():
    print("Lade 15m-Daten …")
    df = load_ohlcv(EXCHANGE, SYMBOL, TIMEFRAME)
    print(f"  {len(df)} Kerzen von {df.index.min()} bis {df.index.max()}")

    print("\nBaue erweiterte Features (17 Indikatoren) + Long/Short-Labels …")
    X = build_extended_features(df)
    print(f"  Features: {X.shape[1]} — {list(X.columns)}")

    rets = df["close"].pct_change()
    regime_vol = rets.rolling(VOL_WINDOW_BARS).std()

    # Labels brauchen wir fuer JEDE (tp, sl, hold) Kombination → bauen on-the-fly
    data_base = X.copy()
    data_base["close"] = df["close"]
    data_base["high"] = df["high"]
    data_base["low"] = df["low"]
    data_base["regime_vol"] = regime_vol
    data_base = data_base.dropna()
    features = list(X.columns)

    # Split in Optimization vs Holdout
    n = len(data_base)
    opt_end = int(n * OPT_FRAC)
    print(f"\nSplit:")
    print(f"  Optimization-Set: {data_base.index[0].date()} → {data_base.index[opt_end-1].date()} ({opt_end} Kerzen)")
    print(f"  Holdout-Set:      {data_base.index[opt_end].date()} → {data_base.index[-1].date()} ({n - opt_end} Kerzen)")

    configs = list(itertools.product(TP_GRID, SL_GRID, HOLD_GRID, THRESHOLD_GRID))
    print(f"\nGrid-Search ueber {len(configs)} Konfigurationen "
          f"(TP × SL × Hold × Threshold) …")
    print(f"  {'TP':>5s} {'SL':>5s} {'Hold':>5s} {'Thr':>5s}  "
          f"{'Return':>9s} {'Sharpe':>7s} {'MaxDD':>7s} {'WR':>6s} {'Trades':>7s} {'vs B&H':>8s}")

    opt_results = []
    for ci, (tp, sl, hold, threshold) in enumerate(configs, 1):
        # Labels mit aktuellen tp/sl/hold bauen
        opt_split = data_base.iloc[:opt_end].copy()
        y_long = build_label_triple_barrier(
            df.loc[opt_split.index], tp_pct=tp, sl_pct=sl, max_hold=hold
        )
        y_short = build_label_triple_barrier_short(
            df.loc[opt_split.index], tp_pct=tp, sl_pct=sl, max_hold=hold
        )
        opt_split["label_long"] = y_long.astype("float").astype("Int64")
        opt_split["label_short"] = y_short.astype("float").astype("Int64")
        opt_split = opt_split.dropna()
        opt_split["label_long"] = opt_split["label_long"].astype(int)
        opt_split["label_short"] = opt_split["label_short"].astype(int)

        res = evaluate_config(opt_split, features, tp, sl, hold, threshold,
                              n_folds=N_FOLDS_OPT, fold_size_target=None)
        if res is None:
            print(f"  [{ci}/{len(configs)}] tp={tp:.3f} sl={sl:.3f} h={hold:2d} t={threshold:.2f}  → skipped")
            continue
        opt_results.append(res)
        print(f"  [{ci:2d}/{len(configs)}] "
              f"{tp*100:>4.1f}% {sl*100:>4.1f}% {hold:>4d}m {threshold:>4.2f}  "
              f"{res['total_return']*100:>+7.2f}% {res['sharpe']:>+6.2f} "
              f"{res['max_dd']*100:>+6.1f}% {res['win_rate']*100:>5.1f}% "
              f"{res['n_trades']:>7d} {res['vs_bh']*100:>+6.1f}pp")

    results_df = pd.DataFrame(opt_results).sort_values("sharpe", ascending=False)
    print(f"\n{'='*78}\nTOP-5 KONFIGURATIONEN auf OPTIMIZATION-SET (nach Sharpe):\n{'='*78}")
    print(results_df.head(5).to_string(index=False))

    # Beste Config auf Holdout testen
    if len(results_df) == 0:
        print("Keine validen Configs gefunden.")
        return

    best = results_df.iloc[0]
    print(f"\n{'='*78}\nHOLDOUT-TEST mit Top-1-Config:\n{'='*78}")
    print(f"  Parameter: TP={best['tp']*100:.1f}%, SL={best['sl']*100:.1f}%, "
          f"Hold={int(best['hold'])} bars, Threshold={best['threshold']:.2f}")

    holdout_data = data_base.iloc[opt_end:].copy()
    y_long = build_label_triple_barrier(
        df.loc[holdout_data.index], tp_pct=best["tp"], sl_pct=best["sl"], max_hold=int(best["hold"])
    )
    y_short = build_label_triple_barrier_short(
        df.loc[holdout_data.index], tp_pct=best["tp"], sl_pct=best["sl"], max_hold=int(best["hold"])
    )
    holdout_data["label_long"] = y_long
    holdout_data["label_short"] = y_short
    holdout_data = holdout_data.dropna()
    holdout_data["label_long"] = holdout_data["label_long"].astype(int)
    holdout_data["label_short"] = holdout_data["label_short"].astype(int)

    # Train auf gesamtem Opt-Set, dann auf Holdout testen
    opt_full = data_base.iloc[:opt_end].copy()
    y_long_full = build_label_triple_barrier(
        df.loc[opt_full.index], tp_pct=best["tp"], sl_pct=best["sl"], max_hold=int(best["hold"])
    )
    y_short_full = build_label_triple_barrier_short(
        df.loc[opt_full.index], tp_pct=best["tp"], sl_pct=best["sl"], max_hold=int(best["hold"])
    )
    opt_full["label_long"] = y_long_full
    opt_full["label_short"] = y_short_full
    opt_full = opt_full.dropna()
    opt_full["label_long"] = opt_full["label_long"].astype(int)
    opt_full["label_short"] = opt_full["label_short"].astype(int)

    cal_split = int(len(opt_full) * 0.85)
    train = opt_full.iloc[:cal_split]
    cal = opt_full.iloc[cal_split:]
    m_long = train_and_calibrate(train, cal, features, "label_long")
    m_short = train_and_calibrate(train, cal, features, "label_short")

    long_classes = list(m_long.classes_)
    short_classes = list(m_short.classes_)
    holdout_data["p_long"] = m_long.predict_proba(holdout_data[features])[:, long_classes.index(1)] if 1 in long_classes else 0.0
    holdout_data["p_short"] = m_short.predict_proba(holdout_data[features])[:, short_classes.index(1)] if 1 in short_classes else 0.0

    vol_lo = float(opt_full["regime_vol"].quantile(REGIME_LOW_PCT / 100))
    vol_hi = float(opt_full["regime_vol"].quantile(REGIME_HIGH_PCT / 100))
    in_regime = (holdout_data["regime_vol"] >= vol_lo) & (holdout_data["regime_vol"] <= vol_hi)
    holdout_data["p_long_f"] = np.where(in_regime, holdout_data["p_long"], 0.0)
    holdout_data["p_short_f"] = np.where(
        (holdout_data["p_short"] > best["threshold"] + SHORT_THRESHOLD_EXTRA) & in_regime,
        holdout_data["p_short"], 0.0,
    )

    net_tp = (1 + best["tp"]) * (1 - FEE_RATE) ** 2 - 1
    net_sl = (1 - best["sl"]) * (1 - FEE_RATE) ** 2 - 1
    p_used = np.maximum(holdout_data["p_long_f"].values, holdout_data["p_short_f"].values)
    pos_frac = kelly_fraction(p_used, net_tp, abs(net_sl), best["threshold"])

    hres = triple_barrier_backtest_ls(
        close=holdout_data["close"],
        p_tp_long=holdout_data["p_long_f"],
        p_tp_short=holdout_data["p_short_f"],
        entry_threshold=best["threshold"],
        tp_pct=best["tp"], sl_pct=best["sl"], max_hold=int(best["hold"]),
        fee_rate=FEE_RATE, starting_capital=STARTING_CAPITAL_CHF,
        cooldown=COOLDOWN_BARS, position_fraction=pos_frac,
        daily_profit_target=DAILY_PROFIT_TARGET, daily_loss_limit=DAILY_LOSS_LIMIT,
    )

    htrades = hres["trades"]
    heq = hres["equity"]
    h_returns = heq.pct_change().dropna()
    h_sharpe = (h_returns.mean() / h_returns.std() * np.sqrt(96 * 365)) if h_returns.std() > 0 else 0.0
    h_dd = (heq / heq.cummax() - 1).min()
    h_wr = (htrades["trade_net_return"] > 0).mean() if len(htrades) else 0.0
    bh_h = holdout_data["close"].iloc[-1] / holdout_data["close"].iloc[0] - 1

    print(f"\n  HOLDOUT-Zeitraum: {holdout_data.index[0].date()} → {holdout_data.index[-1].date()}")
    print(f"  {'':25s}  {'Optimization':>14s}  {'Holdout (unsichtbar)':>22s}")
    print(f"  {'Total Return':25s}  {best['total_return']*100:>+13.2f}%  {hres['total_return']*100:>+21.2f}%")
    print(f"  {'Sharpe':25s}  {best['sharpe']:>14.2f}  {h_sharpe:>22.2f}")
    print(f"  {'Max Drawdown':25s}  {best['max_dd']*100:>+13.2f}%  {h_dd*100:>+21.2f}%")
    print(f"  {'Win-Rate':25s}  {best['win_rate']*100:>13.1f}%  {h_wr*100:>21.1f}%")
    print(f"  {'Anzahl Trades':25s}  {int(best['n_trades']):>14d}  {len(htrades):>22d}")
    print(f"  {'Buy & Hold (Vergleich)':25s}  {best['bh_return']*100:>+13.2f}%  {bh_h*100:>+21.2f}%")
    print(f"  {'vs Buy & Hold':25s}  {best['vs_bh']*100:>+12.1f}pp  "
          f"{(hres['total_return']-bh_h)*100:>+20.1f}pp")

    drift = hres['total_return'] - best['total_return']
    print(f"\nOverfitting-Check (Opt vs Holdout):")
    print(f"  Return-Drift: {drift*100:+.2f} pp")
    if abs(drift) < 0.05:
        print(f"  → Holdout-Performance bestaetigt Optimization. Edge ist plausibel real.")
    elif drift < -0.05:
        print(f"  → Holdout deutlich schlechter. Wahrscheinlich teilweise Overfitting.")
    else:
        print(f"  → Holdout sogar besser. Glueck oder echter Edge — Vorsicht trotzdem.")


if __name__ == "__main__":
    main()
