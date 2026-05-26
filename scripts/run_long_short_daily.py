"""Long/Short Walk-Forward Bot mit realistischen Daily-Targets.

Aenderungen gegenueber long-only:
- Trainiert ZWEI Modelle: Long-Modell + Short-Modell
- Beide kalibriert (isotonic regression)
- Pro Bar: nehme die staerkere Richtung (Long oder Short), wenn > Threshold
- Daily-Targets realistisch: +1 % Profit-Target, -1 % Loss-Limit
- Threshold etwas niedriger (0.20) — weil wir doppelt so viele Setups haben
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
    build_label_triple_barrier_short,
)
from src.backtest.engine import triple_barrier_backtest_ls

warnings.filterwarnings("ignore", category=UserWarning)

EXCHANGE, SYMBOL, TIMEFRAME = "binance", "BTC/USDT", "15m"
STARTING_CAPITAL_CHF = 1000.0
FEE_RATE = 0.001

TP_PCT = 0.015
SL_PCT = 0.004
MAX_HOLD_BARS = 24

ENTRY_PROB_THRESHOLD = 0.18           # gilt fuer Long
SHORT_THRESHOLD_EXTRA = 0.12          # Short braucht zusaetzlich 0.12 mehr (= 0.30 effektiv)

# REALISTISCHE Daily Targets (statt unrealistischer 5 %)
DAILY_PROFIT_TARGET = 0.01    # +1 % pro Tag → Stopp (erreichbares Ziel)
DAILY_LOSS_LIMIT = -0.01      # -1 % pro Tag → Stopp (engerer Schutz)

REGIME_LOW_PCT = 25
REGIME_HIGH_PCT = 75
VOL_WINDOW_BARS = 96

N_FOLDS = 5
INITIAL_TRAIN_FRAC = 0.50
CAL_FRAC_OF_TEST = 0.5

COOLDOWN_BARS = 1

# Position-Sizing: Kelly mit Floor (mindestens 8 % wenn ueberhaupt traden)
KELLY_FACTOR = 1.0
MAX_POSITION_FRAC = 0.50
MIN_POSITION_FRAC = 0.10


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=400, max_depth=6, learning_rate=0.05,
        min_samples_leaf=100, l2_regularization=1.0, random_state=42,
    )


def kelly_fraction(p_tp, net_tp, net_sl, kelly_factor=KELLY_FACTOR,
                   cap=MAX_POSITION_FRAC, floor=MIN_POSITION_FRAC, threshold=ENTRY_PROB_THRESHOLD):
    """Half-Kelly mit Floor: wenn Signal ueber Threshold, mindestens floor %."""
    b = net_tp / abs(net_sl)
    f = p_tp * (1.0 + 1.0 / b) - 1.0 / b
    f = np.maximum(f, 0.0) * kelly_factor
    f = np.minimum(f, cap)
    # Floor nur, wenn Signal aktiv (sonst 0)
    # Floor wird AUCH dann angewandt, wenn Kelly = 0 (z.B. weil p_tp knapp
    # unter Break-Even liegt). Das ist eine bewusste Aktivitaets-Entscheidung,
    # akzeptiert leicht negative Erwartung im Tausch fuer Trade-Frequenz.
    f = np.where(p_tp > threshold, np.maximum(f, floor), 0.0)
    return f


def train_and_calibrate(train, cal, features, label_col):
    """Trainiert HistGB + kalibriert isotonic. Gibt das kalibrierte Modell zurueck."""
    base = make_model()
    base.fit(train[features], train[label_col])
    model = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    model.fit(cal[features], cal[label_col])
    return model


def main():
    print("Lade 15m-Daten …")
    df = load_ohlcv(EXCHANGE, SYMBOL, TIMEFRAME)
    print(f"  {len(df)} Kerzen von {df.index.min()} bis {df.index.max()}")

    print("Baue Features + Long- und Short-Labels …")
    X = build_features_with_interactions(df)
    rets = df["close"].pct_change()
    regime_vol = rets.rolling(VOL_WINDOW_BARS).std()
    y_long = build_label_triple_barrier(df, tp_pct=TP_PCT, sl_pct=SL_PCT, max_hold=MAX_HOLD_BARS)
    y_short = build_label_triple_barrier_short(df, tp_pct=TP_PCT, sl_pct=SL_PCT, max_hold=MAX_HOLD_BARS)

    data = X.copy()
    data["label_long"] = y_long
    data["label_short"] = y_short
    data["close"] = df["close"]
    data["regime_vol"] = regime_vol
    data = data.dropna()
    data["label_long"] = data["label_long"].astype(int)
    data["label_short"] = data["label_short"].astype(int)
    features = list(X.columns)

    print("\nLabel-Verteilung Long-Modell:")
    cl = data["label_long"].value_counts().sort_index()
    for cls, n in cl.items():
        print(f"  {cls:+d}: {n:>6d}  ({n/len(data)*100:4.1f} %)")

    print("\nLabel-Verteilung Short-Modell:")
    cs = data["label_short"].value_counts().sort_index()
    for cls, n in cs.items():
        print(f"  {cls:+d}: {n:>6d}  ({n/len(data)*100:4.1f} %)")

    net_tp = (1 + TP_PCT) * (1 - FEE_RATE) ** 2 - 1
    net_sl = (1 - SL_PCT) * (1 - FEE_RATE) ** 2 - 1
    be_win = abs(net_sl) / (net_tp + abs(net_sl))
    print(f"\nGebuehrenmathematik: net_TP={net_tp*100:+.2f}%, net_SL={net_sl*100:+.2f}%, "
          f"Break-Even Win-Rate={be_win*100:.1f}%")
    print(f"\nDaily Profit Target:  +{DAILY_PROFIT_TARGET*100:.1f} %  (realistisch)")
    print(f"Daily Loss Limit:    {DAILY_LOSS_LIMIT*100:+.1f} %")
    print(f"Entry Threshold:      {ENTRY_PROB_THRESHOLD:.2f}")

    # Walk-Forward
    n = len(data)
    initial_train = int(n * INITIAL_TRAIN_FRAC)
    remaining = n - initial_train
    fold_size = remaining // N_FOLDS
    test_size = int(fold_size * (1 - CAL_FRAC_OF_TEST))
    cal_size = fold_size - test_size

    print(f"\nWalk-Forward {N_FOLDS} Folds, je {test_size} Test-Kerzen …")

    all_test = []
    for fold in range(N_FOLDS):
        train_end = initial_train + fold * fold_size
        cal_end = train_end + cal_size
        test_end = cal_end + test_size if fold < N_FOLDS - 1 else n
        train = data.iloc[:train_end]
        cal = data.iloc[train_end:cal_end]
        test = data.iloc[cal_end:test_end].copy()

        # Long-Modell
        m_long = train_and_calibrate(train, cal, features, "label_long")
        long_classes = list(m_long.classes_)
        proba_long = m_long.predict_proba(test[features])
        test["p_long"] = proba_long[:, long_classes.index(1)] if 1 in long_classes else 0.0

        # Short-Modell
        m_short = train_and_calibrate(train, cal, features, "label_short")
        short_classes = list(m_short.classes_)
        proba_short = m_short.predict_proba(test[features])
        test["p_short"] = proba_short[:, short_classes.index(1)] if 1 in short_classes else 0.0

        # Regime-Filter
        vol_lo = float(train["regime_vol"].quantile(REGIME_LOW_PCT / 100))
        vol_hi = float(train["regime_vol"].quantile(REGIME_HIGH_PCT / 100))
        in_regime = (test["regime_vol"] >= vol_lo) & (test["regime_vol"] <= vol_hi)
        test["p_long_f"] = np.where(in_regime, test["p_long"], 0.0)
        test["p_short_f"] = np.where(in_regime, test["p_short"], 0.0)
        test["fold"] = fold

        all_test.append(test)
        print(f"  Fold {fold+1}: test {test.index[0].date()}→{test.index[-1].date()}  "
              f"max p_long={test['p_long_f'].max():.3f}  max p_short={test['p_short_f'].max():.3f}")

    test_all = pd.concat(all_test).sort_index()

    # Asymmetrischer Threshold: Short-Signale muessen deutlich staerker sein als Long
    short_threshold = ENTRY_PROB_THRESHOLD + SHORT_THRESHOLD_EXTRA
    print(f"\nAsymmetrische Schwellen:")
    print(f"  Long  signal > {ENTRY_PROB_THRESHOLD:.2f}")
    print(f"  Short signal > {short_threshold:.2f}  (haerter, weil Short-Modell schwaecher)")
    test_all["p_short_f"] = np.where(
        test_all["p_short_f"] > short_threshold,
        test_all["p_short_f"],
        0.0,
    )

    # Position-Sizing: Kelly basierend auf der jeweils GEWAEHLTEN Richtung
    p_used = np.where(
        (test_all["p_long_f"] > test_all["p_short_f"]) & (test_all["p_long_f"] > ENTRY_PROB_THRESHOLD),
        test_all["p_long_f"],
        np.where(
            (test_all["p_short_f"] > ENTRY_PROB_THRESHOLD),
            test_all["p_short_f"],
            0.0
        )
    )
    test_all["pos_frac"] = kelly_fraction(p_used, net_tp, abs(net_sl))

    print(f"\nLaufe Long/Short-Backtest …")
    result = triple_barrier_backtest_ls(
        close=test_all["close"],
        p_tp_long=test_all["p_long_f"],
        p_tp_short=test_all["p_short_f"],
        entry_threshold=ENTRY_PROB_THRESHOLD,
        tp_pct=TP_PCT, sl_pct=SL_PCT,
        max_hold=MAX_HOLD_BARS,
        fee_rate=FEE_RATE,
        starting_capital=STARTING_CAPITAL_CHF,
        cooldown=COOLDOWN_BARS,
        position_fraction=test_all["pos_frac"].values,
        daily_profit_target=DAILY_PROFIT_TARGET,
        daily_loss_limit=DAILY_LOSS_LIMIT,
    )

    trades = result["trades"]
    equity = result["equity"]
    final = result["final_capital"]
    day_log = result["day_log"]
    bh_final = STARTING_CAPITAL_CHF * (test_all["close"].iloc[-1] / test_all["close"].iloc[0])

    print("\n" + "=" * 78)
    print(f"=== LONG/SHORT ERGEBNIS {test_all.index.min().date()} → {test_all.index.max().date()} ===")
    print("=" * 78)
    print(f"  Strategie:        {final:>10.2f} CHF   ({result['total_return']*100:+.2f} %)")
    print(f"  Buy & Hold:       {bh_final:>10.2f} CHF   ({(bh_final/STARTING_CAPITAL_CHF-1)*100:+.2f} %)")
    print(f"  Anzahl Trades:    {len(trades):>10d}")

    if len(trades) > 0:
        longs = trades[trades["direction"] == 1]
        shorts = trades[trades["direction"] == -1]
        win_rate = (trades["trade_net_return"] > 0).mean()
        win_long = (longs["trade_net_return"] > 0).mean() if len(longs) else 0
        win_short = (shorts["trade_net_return"] > 0).mean() if len(shorts) else 0
        avg_pos = trades["pos_frac"].mean() * 100
        avg_hold_min = trades["hold_bars"].mean() * 15
        total_fees = sum(abs(t["capital_before"]) * t["pos_frac"] * FEE_RATE * 2
                         for _, t in trades.iterrows())
        max_dd = (equity / equity.cummax() - 1).min()
        reasons = trades["reason"].value_counts().to_dict()

        print(f"  Win-Rate gesamt:  {win_rate*100:>9.1f} %  (Break-Even: {be_win*100:.1f} %)")
        print(f"  Long-Trades:      {len(longs):>4d}   Win-Rate: {win_long*100:.1f} %")
        print(f"  Short-Trades:     {len(shorts):>4d}   Win-Rate: {win_short*100:.1f} %")
        print(f"  Avg. Position:    {avg_pos:>9.1f} %  "
              f"(min {trades['pos_frac'].min()*100:.1f}, max {trades['pos_frac'].max()*100:.1f})")
        print(f"  Avg. Haltedauer:  {avg_hold_min:>9.0f} Minuten")
        print(f"  Gebuehren total:  {total_fees:>9.2f} CHF")
        print(f"  Max Drawdown:     {max_dd*100:>9.2f} %")
        print(f"  Exits:            TP={reasons.get('TP',0)}  "
              f"SL={reasons.get('SL',0)}  TIME={reasons.get('TIME',0)}")

    # --- DAILY ANALYSE ---
    print("\n--- DAILY TARGET ANALYSE ---")
    if len(day_log) > 0:
        n_days = len(day_log)
        hit_target = (day_log["day_return"] >= DAILY_PROFIT_TARGET).sum()
        hit_limit = (day_log["day_return"] <= DAILY_LOSS_LIMIT).sum()
        win_days = (day_log["day_return"] > 0).sum()
        loss_days = (day_log["day_return"] < 0).sum()
        flat_days = (day_log["day_return"] == 0).sum()
        active_days = (day_log["day_return"] != 0).sum()

        print(f"  Trading-Tage gesamt:          {n_days}")
        print(f"  Tage mit Trade-Aktivitaet:    {active_days}  ({active_days/n_days*100:.1f} %)")
        print(f"")
        print(f"  Tage mit ≥+{DAILY_PROFIT_TARGET*100:.0f}% erreicht:      "
              f"{hit_target}  ({hit_target/n_days*100:.2f} %)")
        print(f"  Tage mit ≤{DAILY_LOSS_LIMIT*100:.0f}% erreicht:      "
              f"{hit_limit}  ({hit_limit/n_days*100:.2f} %)")
        print(f"")
        print(f"  Gewinn-Tage:                  {win_days}  ({win_days/n_days*100:.1f} %)")
        print(f"  Verlust-Tage:                 {loss_days}  ({loss_days/n_days*100:.1f} %)")
        print(f"  Flache Tage:                  {flat_days}  ({flat_days/n_days*100:.1f} %)")
        print(f"  Bester Tag:                  {day_log['day_return'].max()*100:>+6.2f} %")
        print(f"  Schlechtester Tag:           {day_log['day_return'].min()*100:>+6.2f} %")
        print(f"  Mittl. Tagesrendite:         {day_log['day_return'].mean()*100:>+6.3f} %")
        print(f"  Median Tagesrendite:         {day_log['day_return'].median()*100:>+6.3f} %")
        # Streak
        wins = (day_log["day_return"] > 0).values
        losses = (day_log["day_return"] < 0).values
        max_win_streak = max([sum(1 for _ in g) for k, g in __import__("itertools").groupby(wins) if k] or [0])
        max_loss_streak = max([sum(1 for _ in g) for k, g in __import__("itertools").groupby(losses) if k] or [0])
        print(f"  Laengste Gewinn-Serie:        {max_win_streak} Tage")
        print(f"  Laengste Verlustserie:        {max_loss_streak} Tage")

        active = day_log[day_log["day_return"] != 0].sort_values("day_return", ascending=False)
        if len(active) > 0:
            print(f"\n  Top-5 beste Tage:")
            for _, d in active.head(5).iterrows():
                print(f"    {d['date']}  {d['day_return']*100:>+6.2f} %  "
                      f"({d['start_capital']:.2f} → {d['end_capital']:.2f} CHF)")
            print(f"\n  Top-5 schlechteste Tage:")
            for _, d in active.tail(5).iloc[::-1].iterrows():
                print(f"    {d['date']}  {d['day_return']*100:>+6.2f} %  "
                      f"({d['start_capital']:.2f} → {d['end_capital']:.2f} CHF)")


if __name__ == "__main__":
    main()
