"""Walk-Forward 15m Bot mit Daily Profit Target + Daily Loss Limit.

User-Wunsch umgesetzt:
- Daily Profit Target +5 %: erreicht → Trading fuer den Tag aus
- Daily Loss Limit -2 %: erreicht → Trading fuer den Tag aus (Schutz)
- Niedrigerer Threshold 0.10 → mehr Aktivitaet, mehr Trades pro Tag
- Per-Trade Stop-Loss bleibt (Triple-Barrier)
- ML + Kalibrierung + Walk-Forward + Kelly bleiben

Realitaetscheck: Das System WILL traden, aber das +5%-Target wird sehr selten
erreicht. Das ist mathematisch eingebaut — die Anzahl Target-Hits ist der
ehrliche Indikator dafuer, wie unrealistisch das Ziel ist.
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

EXCHANGE, SYMBOL, TIMEFRAME = "binance", "BTC/USDT", "15m"
STARTING_CAPITAL_CHF = 1000.0
FEE_RATE = 0.001

TP_PCT = 0.015
SL_PCT = 0.004
MAX_HOLD_BARS = 24

# AGGRESSIVER: niedriger Threshold → mehr Aktivitaet
ENTRY_PROB_THRESHOLD = 0.10

# Daily targets
DAILY_PROFIT_TARGET = 0.05    # +5 % pro Tag → Stopp
DAILY_LOSS_LIMIT = -0.02      # -2 % pro Tag → Stopp

REGIME_LOW_PCT = 25
REGIME_HIGH_PCT = 75
VOL_WINDOW_BARS = 96

N_FOLDS = 5
INITIAL_TRAIN_FRAC = 0.50
CAL_FRAC_OF_TEST = 0.5

COOLDOWN_BARS = 1   # nur 15 Min Pause nach Trade → mehr Aktivitaet moeglich

KELLY_FACTOR = 1.0
MAX_POSITION_FRAC = 0.75


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=400, max_depth=6, learning_rate=0.05,
        min_samples_leaf=100, l2_regularization=1.0, random_state=42,
    )


def kelly_fraction(p_tp, net_tp, net_sl, kelly_factor=KELLY_FACTOR, cap=MAX_POSITION_FRAC):
    b = net_tp / abs(net_sl)
    f = p_tp * (1.0 + 1.0 / b) - 1.0 / b
    f = np.maximum(f, 0.0) * kelly_factor
    return np.minimum(f, cap)


def main():
    print("Lade 15m-Daten …")
    df = load_ohlcv(EXCHANGE, SYMBOL, TIMEFRAME)
    print(f"  {len(df)} Kerzen von {df.index.min()} bis {df.index.max()}")

    X = build_features_with_interactions(df)
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

    net_tp = (1 + TP_PCT) * (1 - FEE_RATE) ** 2 - 1
    net_sl = (1 - SL_PCT) * (1 - FEE_RATE) ** 2 - 1
    be_win = abs(net_sl) / (net_tp + abs(net_sl))
    print(f"\nGebuehrenmathematik: net_TP={net_tp*100:+.2f}%, "
          f"net_SL={net_sl*100:+.2f}%, Break-Even Win-Rate={be_win*100:.1f}%")
    print(f"\nDaily Profit Target:  +{DAILY_PROFIT_TARGET*100:.1f} %")
    print(f"Daily Loss Limit:    {DAILY_LOSS_LIMIT*100:+.1f} %")
    print(f"Entry Threshold:      {ENTRY_PROB_THRESHOLD:.2f}  (aggressiv)")

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

        base = make_model()
        base.fit(train[features], train["label"])
        model = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
        model.fit(cal[features], cal["label"])

        classes = list(model.classes_)
        proba = model.predict_proba(test[features])
        test["pred"] = model.predict(test[features])
        test["p_tp"] = proba[:, classes.index(1)] if 1 in classes else 0.0
        test["fold"] = fold

        vol_lo = float(train["regime_vol"].quantile(REGIME_LOW_PCT / 100))
        vol_hi = float(train["regime_vol"].quantile(REGIME_HIGH_PCT / 100))
        in_regime = (test["regime_vol"] >= vol_lo) & (test["regime_vol"] <= vol_hi)
        test["p_tp_filtered"] = np.where(in_regime, test["p_tp"], 0.0)

        all_test.append(test)
        print(f"  Fold {fold+1}: test {test.index[0].date()}→{test.index[-1].date()}  "
              f"max p_tp_filt={test['p_tp_filtered'].max():.3f}")

    test_all = pd.concat(all_test).sort_index()
    acc = accuracy_score(test_all["label"], test_all["pred"])
    print(f"\nGesamt Out-of-Sample Trefferquote: {acc*100:.1f} %")

    # Fixed 25 % Position pro Trade — ignoriert Kelly, weil Kelly bei diesen
    # niedrigen Wahrscheinlichkeiten auf 0 setzt. Aggressivere Variante zur
    # Demonstration: das System WIRD jetzt traden.
    test_all["kelly_frac"] = np.where(test_all["p_tp_filtered"] > 0, 0.25, 0.0)

    print(f"\nLaufe Backtest mit Daily-Limits …")
    result = triple_barrier_backtest(
        close=test_all["close"],
        p_tp=test_all["p_tp_filtered"],
        entry_threshold=ENTRY_PROB_THRESHOLD,
        tp_pct=TP_PCT,
        sl_pct=SL_PCT,
        max_hold=MAX_HOLD_BARS,
        fee_rate=FEE_RATE,
        starting_capital=STARTING_CAPITAL_CHF,
        cooldown=COOLDOWN_BARS,
        position_fraction=test_all["kelly_frac"].values,
        daily_profit_target=DAILY_PROFIT_TARGET,
        daily_loss_limit=DAILY_LOSS_LIMIT,
    )

    trades = result["trades"]
    equity = result["equity"]
    final = result["final_capital"]
    day_log = result["day_log"]
    bh_final = STARTING_CAPITAL_CHF * (test_all["close"].iloc[-1] / test_all["close"].iloc[0])

    print("\n" + "=" * 78)
    print(f"=== ERGEBNIS {test_all.index.min().date()} → {test_all.index.max().date()} ===")
    print("=" * 78)
    print(f"  Strategie:        {final:>10.2f} CHF   ({result['total_return']*100:+.2f} %)")
    print(f"  Buy & Hold:       {bh_final:>10.2f} CHF   ({(bh_final/STARTING_CAPITAL_CHF-1)*100:+.2f} %)")
    print(f"  Anzahl Trades:    {len(trades):>10d}")

    if len(trades) > 0:
        win_rate = (trades["trade_net_return"] > 0).mean()
        avg_pos = trades["pos_frac"].mean() * 100
        avg_hold_min = trades["hold_bars"].mean() * 15
        total_fees = sum(abs(t["capital_before"]) * t["pos_frac"] * FEE_RATE * 2
                         for _, t in trades.iterrows())
        max_dd = (equity / equity.cummax() - 1).min()
        reasons = trades["reason"].value_counts().to_dict()
        print(f"  Win-Rate:         {win_rate*100:>9.1f} %  (Break-Even: {be_win*100:.1f} %)")
        print(f"  Avg. Position:    {avg_pos:>9.1f} %  "
              f"(min {trades['pos_frac'].min()*100:.1f}, max {trades['pos_frac'].max()*100:.1f})")
        print(f"  Avg. Haltedauer:  {avg_hold_min:>9.0f} Minuten")
        print(f"  Gebuehren total:  {total_fees:>9.2f} CHF")
        print(f"  Max Drawdown:     {max_dd*100:>9.2f} %")
        print(f"  Exits:            TP={reasons.get('TP',0)}  "
              f"SL={reasons.get('SL',0)}  TIME={reasons.get('TIME',0)}")

    # --- DAILY TARGET ANALYSE ---
    print("\n--- DAILY TARGET ANALYSE ---")
    if len(day_log) == 0:
        print("  Keine vollstaendigen Trading-Tage geloggt.")
    else:
        n_days = len(day_log)
        hit_target = (day_log["day_return"] >= DAILY_PROFIT_TARGET).sum()
        hit_limit = (day_log["day_return"] <= DAILY_LOSS_LIMIT).sum()
        win_days = (day_log["day_return"] > 0).sum()
        loss_days = (day_log["day_return"] < 0).sum()
        flat_days = (day_log["day_return"] == 0).sum()
        best = day_log["day_return"].max()
        worst = day_log["day_return"].min()

        print(f"  Trading-Tage gesamt:          {n_days}")
        print(f"  Tage mit Trade-Aktivitaet:    {(day_log['day_return'] != 0).sum()}  "
              f"({(day_log['day_return']!=0).mean()*100:.1f} %)")
        print(f"")
        print(f"  Tage mit ≥+5 % erreicht:      {hit_target}  ({hit_target/n_days*100:.2f} %)")
        print(f"  Tage mit ≤-2 % erreicht:      {hit_limit}  ({hit_limit/n_days*100:.2f} %)")
        print(f"")
        print(f"  Gewinn-Tage:                  {win_days}  ({win_days/n_days*100:.1f} %)")
        print(f"  Verlust-Tage:                 {loss_days}  ({loss_days/n_days*100:.1f} %)")
        print(f"  Flache Tage:                  {flat_days}  ({flat_days/n_days*100:.1f} %)")
        print(f"  Bester Tag:                  {best*100:>+6.2f} %")
        print(f"  Schlechtester Tag:           {worst*100:>+6.2f} %")
        print(f"  Mittl. Tagesrendite:         {day_log['day_return'].mean()*100:>+6.3f} %")
        print(f"  Median Tagesrendite:         {day_log['day_return'].median()*100:>+6.3f} %")

        active = day_log[day_log["day_return"] != 0].sort_values("day_return", ascending=False)
        if len(active) > 0:
            print(f"\n  Top-5 beste Tage (aktiv):")
            for _, d in active.head(5).iterrows():
                print(f"    {d['date']}  {d['day_return']*100:>+6.2f} %  "
                      f"({d['start_capital']:.2f} → {d['end_capital']:.2f} CHF)")
            print(f"\n  Top-5 schlechteste Tage (aktiv):")
            for _, d in active.tail(5).iloc[::-1].iterrows():
                print(f"    {d['date']}  {d['day_return']*100:>+6.2f} %  "
                      f"({d['start_capital']:.2f} → {d['end_capital']:.2f} CHF)")

    print("\n--- WAS HAETTE +5 %/Tag bedeutet? ---")
    n_days_simple = (test_all.index.max() - test_all.index.min()).days
    target_compound = (1 + DAILY_PROFIT_TARGET) ** n_days_simple
    print(f"  {DAILY_PROFIT_TARGET*100:.0f}% × {n_days_simple} Tage (compound) = "
          f"{target_compound:,.0f}× Startkapital = {target_compound*STARTING_CAPITAL_CHF:,.0f} CHF")
    print(f"  Realer Strategie-Multiplikator:   {final/STARTING_CAPITAL_CHF:.3f}× = {final:,.2f} CHF")


if __name__ == "__main__":
    main()
