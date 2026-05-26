"""5m-BTC Backtest mit Funding-Rate-Features.

Vergleicht zwei Modelle auf demselben Holdout:
  A) Baseline: nur TA-Indikatoren (17 Features + 4 Zeit-Features)
  B) Erweitert: TA + 6 Funding-Rate-Features

So sehen wir EHRLICH, ob Funding tatsaechlich Edge bringt.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance

from src.data.storage import load_ohlcv, load_funding
from src.ml.features import (
    build_features_with_time,
    build_funding_features,
    build_label_triple_barrier,
    build_label_triple_barrier_short,
)
from src.backtest.engine import triple_barrier_backtest_ls

warnings.filterwarnings("ignore", category=UserWarning)

EXCHANGE, SYMBOL, TIMEFRAME = "binance", "BTC/USDT", "5m"
BAR_MINUTES = 5
STARTING_CAPITAL_CHF = 1000.0
FEE_RATE = 0.001

# 5m-spezifisch: kleineres TP/SL, kuerzerer Horizont
TP_PCT = 0.010                # +1.0 %
SL_PCT = 0.003                # -0.3 %
MAX_HOLD_BARS = 36            # 36 × 5m = 3h

ENTRY_PROB_THRESHOLD = 0.25
SHORT_THRESHOLD_EXTRA = 0.10

REGIME_LOW_PCT = 25
REGIME_HIGH_PCT = 75
VOL_WINDOW_BARS = 12 * 24      # 24h in 5m-Bars

DAILY_PROFIT_TARGET = 0.01
DAILY_LOSS_LIMIT = -0.01
COOLDOWN_BARS = 2              # 10 Min

# Walk-Forward
N_FOLDS = 4
INITIAL_TRAIN_FRAC = 0.50
CAL_FRAC_OF_FOLD = 0.30

# Position-Sizing
MIN_POSITION_FRAC = 0.10
MAX_POSITION_FRAC = 0.40
KELLY_FACTOR = 1.0


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.06,
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


def run_walk_forward(data, features, name):
    """Walk-Forward fuer eine Feature-Set-Variante. Liefert konkateniertes Test-DF."""
    n = len(data)
    initial_train = int(n * INITIAL_TRAIN_FRAC)
    remaining = n - initial_train
    fold_size = remaining // N_FOLDS
    test_size = int(fold_size * (1 - CAL_FRAC_OF_FOLD))
    cal_size = fold_size - test_size

    all_test = []
    for fold in range(N_FOLDS):
        train_end = initial_train + fold * fold_size
        cal_end = train_end + cal_size
        test_end = cal_end + test_size if fold < N_FOLDS - 1 else n
        train = data.iloc[:train_end]
        cal = data.iloc[train_end:cal_end]
        test = data.iloc[cal_end:test_end].copy()

        m_long = train_and_calibrate(train, cal, features, "label_long")
        m_short = train_and_calibrate(train, cal, features, "label_short")
        lc = list(m_long.classes_)
        sc = list(m_short.classes_)
        test["p_long"] = m_long.predict_proba(test[features])[:, lc.index(1)] if 1 in lc else 0.0
        test["p_short"] = m_short.predict_proba(test[features])[:, sc.index(1)] if 1 in sc else 0.0

        vol_lo = float(train["regime_vol"].quantile(REGIME_LOW_PCT / 100))
        vol_hi = float(train["regime_vol"].quantile(REGIME_HIGH_PCT / 100))
        in_regime = (test["regime_vol"] >= vol_lo) & (test["regime_vol"] <= vol_hi)
        test["p_long_f"] = np.where(in_regime, test["p_long"], 0.0)
        test["p_short_f"] = np.where(
            (test["p_short"] > ENTRY_PROB_THRESHOLD + SHORT_THRESHOLD_EXTRA) & in_regime,
            test["p_short"], 0.0,
        )
        test["fold"] = fold
        all_test.append(test)
        print(f"  [{name}] Fold {fold+1}/{N_FOLDS}: "
              f"{test.index[0].date()}→{test.index[-1].date()}  "
              f"max p_long={test['p_long_f'].max():.3f} max p_short={test['p_short_f'].max():.3f}")

    return pd.concat(all_test).sort_index(), m_long, m_short


def backtest_and_report(test_all, label):
    net_tp = (1 + TP_PCT) * (1 - FEE_RATE) ** 2 - 1
    net_sl = (1 - SL_PCT) * (1 - FEE_RATE) ** 2 - 1
    be_win = abs(net_sl) / (net_tp + abs(net_sl))
    p_used = np.maximum(test_all["p_long_f"].values, test_all["p_short_f"].values)
    pos_frac = kelly_fraction(p_used, net_tp, abs(net_sl), ENTRY_PROB_THRESHOLD)

    res = triple_barrier_backtest_ls(
        close=test_all["close"],
        p_tp_long=test_all["p_long_f"],
        p_tp_short=test_all["p_short_f"],
        entry_threshold=ENTRY_PROB_THRESHOLD,
        tp_pct=TP_PCT, sl_pct=SL_PCT, max_hold=MAX_HOLD_BARS,
        fee_rate=FEE_RATE, starting_capital=STARTING_CAPITAL_CHF,
        cooldown=COOLDOWN_BARS, position_fraction=pos_frac,
        daily_profit_target=DAILY_PROFIT_TARGET, daily_loss_limit=DAILY_LOSS_LIMIT,
    )

    trades = res["trades"]
    equity = res["equity"]
    day_log = res["day_log"]
    bh_final = STARTING_CAPITAL_CHF * (test_all["close"].iloc[-1] / test_all["close"].iloc[0])

    longs = trades[trades["direction"] == 1] if len(trades) else trades
    shorts = trades[trades["direction"] == -1] if len(trades) else trades
    win_rate = (trades["trade_net_return"] > 0).mean() if len(trades) else 0.0
    max_dd = (equity / equity.cummax() - 1).min() if len(equity) else 0.0
    win_long = (longs["trade_net_return"] > 0).mean() if len(longs) else 0
    win_short = (shorts["trade_net_return"] > 0).mean() if len(shorts) else 0

    win_days = (day_log["day_return"] > 0).sum() if len(day_log) else 0
    loss_days = (day_log["day_return"] < 0).sum() if len(day_log) else 0
    active_days = (day_log["day_return"] != 0).sum() if len(day_log) else 0
    n_days = len(day_log) if len(day_log) else 0

    print(f"\n=== {label} ===")
    print(f"  Endkapital:        {res['final_capital']:>10.2f} CHF  ({res['total_return']*100:+.2f} %)")
    print(f"  Buy & Hold:        {bh_final:>10.2f} CHF  ({(bh_final/STARTING_CAPITAL_CHF-1)*100:+.2f} %)")
    print(f"  Trades:            {len(trades):>5d}   "
          f"Long: {len(longs):>4d} (WR {win_long*100:.1f}%)   "
          f"Short: {len(shorts):>4d} (WR {win_short*100:.1f}%)")
    print(f"  Win-Rate gesamt:   {win_rate*100:>6.1f} %   (Break-Even: {be_win*100:.1f} %)")
    print(f"  Max Drawdown:      {max_dd*100:>6.2f} %")
    if n_days > 0:
        print(f"  Tage: {n_days} total, {active_days} aktiv ({active_days/n_days*100:.1f}%), "
              f"{win_days} Gewinn, {loss_days} Verlust  "
              f"Bester {day_log['day_return'].max()*100:+.2f} %, Schlechtester {day_log['day_return'].min()*100:+.2f} %")
    return res


def main():
    print("Lade 5m-OHLCV …")
    df = load_ohlcv(EXCHANGE, SYMBOL, TIMEFRAME)
    print(f"  {len(df)} 5m-Kerzen von {df.index.min()} bis {df.index.max()}")

    print("Lade Funding-Rates …")
    fund = load_funding(EXCHANGE, SYMBOL)
    print(f"  {len(fund)} Funding-Datenpunkte von {fund.index.min()} bis {fund.index.max()}")

    print("Baue Features …")
    X_ta = build_features_with_time(df)
    X_fund = build_funding_features(df, fund, bar_minutes=BAR_MINUTES)
    print(f"  TA-Features: {X_ta.shape[1]}  ({list(X_ta.columns)})")
    print(f"  Funding-Features: {X_fund.shape[1]}  ({list(X_fund.columns)})")

    rets = df["close"].pct_change()
    regime_vol = rets.rolling(VOL_WINDOW_BARS).std()
    y_long = build_label_triple_barrier(df, tp_pct=TP_PCT, sl_pct=SL_PCT, max_hold=MAX_HOLD_BARS)
    y_short = build_label_triple_barrier_short(df, tp_pct=TP_PCT, sl_pct=SL_PCT, max_hold=MAX_HOLD_BARS)

    base = pd.concat([X_ta, X_fund], axis=1)
    base["label_long"] = y_long
    base["label_short"] = y_short
    base["close"] = df["close"]
    base["regime_vol"] = regime_vol
    base = base.dropna()
    base["label_long"] = base["label_long"].astype(int)
    base["label_short"] = base["label_short"].astype(int)

    ta_features = list(X_ta.columns)
    full_features = list(X_ta.columns) + list(X_fund.columns)

    net_tp = (1 + TP_PCT) * (1 - FEE_RATE) ** 2 - 1
    net_sl = (1 - SL_PCT) * (1 - FEE_RATE) ** 2 - 1
    be_win = abs(net_sl) / (net_tp + abs(net_sl))
    print(f"\nMathematik: net_TP={net_tp*100:+.2f}%, net_SL={net_sl*100:+.2f}%, "
          f"Break-Even Win-Rate={be_win*100:.1f}%")
    print(f"Threshold: Long {ENTRY_PROB_THRESHOLD:.2f} / Short {ENTRY_PROB_THRESHOLD+SHORT_THRESHOLD_EXTRA:.2f}")

    print(f"\nA) Walk-Forward mit BASELINE (nur TA, {len(ta_features)} Features) …")
    test_ta, _, _ = run_walk_forward(base, ta_features, "TA")

    print(f"\nB) Walk-Forward mit FUNDING ({len(full_features)} Features) …")
    test_full, m_long_full, m_short_full = run_walk_forward(base, full_features, "TA+FUND")

    print("\n" + "=" * 78)
    print("VERGLEICH BEIDER VARIANTEN auf demselben Walk-Forward-Test-Zeitraum")
    print("=" * 78)
    backtest_and_report(test_ta, "Baseline (nur TA)")
    res_full = backtest_and_report(test_full, "Erweitert (TA + Funding)")

    # Funding-Feature-Wichtigkeit per Permutation auf letztem Fold
    last_test = test_full[test_full["fold"] == test_full["fold"].max()].copy()
    print(f"\n=== Permutation Importance (LONG-Modell, letztes Fold) ===")
    try:
        perm = permutation_importance(
            m_long_full, last_test[full_features], last_test["label_long"],
            n_repeats=3, random_state=42, n_jobs=-1,
        )
        imp = pd.DataFrame({"feature": full_features, "importance": perm.importances_mean,
                            "std": perm.importances_std}).sort_values("importance", ascending=False)
        print(f"  {'Feature':<28s} {'Importance':>12s}   ±std")
        for _, r in imp.head(15).iterrows():
            marker = " ← FUNDING" if r["feature"] in X_fund.columns else ""
            print(f"  {r['feature']:<28s} {r['importance']*1000:>+9.2f} e-3  ±{r['std']*1000:.2f}{marker}")
    except Exception as e:
        print(f"  Permutation Importance fehlgeschlagen: {e}")


if __name__ == "__main__":
    main()
