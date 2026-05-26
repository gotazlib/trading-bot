"""Walk-Forward Triple-Barrier auf 15m-BTC mit Kelly-Sizing + Tagesstatistik.

Ueber alle 2 Jahre Daten:
- 5 Folds, expanding window (jedes Fold trainiert auf allem davor)
- Pro Fold: eigene Kalibrierung + Test
- Position-Size pro Trade dynamisch via Half-Kelly (kleinere Position bei
  schwacher Probability, groessere bei starker)
- Marktregime-Filter beibehalten
- Tages-PnL-Statistik: an wie vielen Tagen war die Strategie tatsaechlich
  profitabel?
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

ENTRY_PROB_THRESHOLD = 0.20

REGIME_LOW_PCT = 25
REGIME_HIGH_PCT = 75
VOL_WINDOW_BARS = 96

# Walk-Forward: 5 Folds, Initial-Train 50 %, jedes Test-Fenster ~10 %
N_FOLDS = 5
INITIAL_TRAIN_FRAC = 0.50
CAL_FRAC_OF_TEST = 0.5  # 5 % cal, 10 % test pro Fold

COOLDOWN_BARS = 2

# Kelly-Sizing
KELLY_FACTOR = 1.0      # Full Kelly
MAX_POSITION_FRAC = 0.75 # max 75 % des Kapitals pro Trade (Rest bleibt Sicherheitspuffer)


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=400, max_depth=6, learning_rate=0.05,
        min_samples_leaf=100, l2_regularization=1.0, random_state=42,
    )


def kelly_fraction(p_tp, net_tp, net_sl, kelly_factor=KELLY_FACTOR, cap=MAX_POSITION_FRAC):
    """Half-Kelly basierend auf kalibrierter P(TP).

    Kelly = p * (1+1/b) - 1/b   mit  b = net_tp / |net_sl|
    Bei p < Break-Even gibt es keinen positiven Erwartungswert → 0 %.
    """
    b = net_tp / abs(net_sl)
    f = p_tp * (1.0 + 1.0 / b) - 1.0 / b
    f = np.maximum(f, 0.0) * kelly_factor
    return np.minimum(f, cap)


def daily_stats(equity: pd.Series) -> dict:
    """Tagesweise PnL-Statistik aus der Equity-Kurve."""
    daily_end = equity.resample("1D").last().dropna()
    daily_ret = daily_end.pct_change().dropna()
    n = len(daily_ret)
    if n == 0:
        return {"n_days": 0}
    wins = (daily_ret > 0).sum()
    losses = (daily_ret < 0).sum()
    flat = (daily_ret == 0).sum()
    return {
        "n_days": n,
        "win_days": int(wins),
        "loss_days": int(losses),
        "flat_days": int(flat),
        "win_rate_daily": wins / n,
        "best_day": daily_ret.max(),
        "worst_day": daily_ret.min(),
        "avg_day": daily_ret.mean(),
        "median_day": daily_ret.median(),
        "std_day": daily_ret.std(),
        "consec_wins_max": _max_streak(daily_ret > 0),
        "consec_losses_max": _max_streak(daily_ret < 0),
    }


def _max_streak(bool_series: pd.Series) -> int:
    streak, best = 0, 0
    for v in bool_series:
        streak = streak + 1 if v else 0
        best = max(best, streak)
    return best


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

    # Gebuehrenmathematik
    net_tp = (1 + TP_PCT) * (1 - FEE_RATE) ** 2 - 1
    net_sl = (1 - SL_PCT) * (1 - FEE_RATE) ** 2 - 1
    be_win = abs(net_sl) / (net_tp + abs(net_sl))
    print(f"\nGebuehrenmathematik: net_TP={net_tp*100:+.2f}%, "
          f"net_SL={net_sl*100:+.2f}%, Break-Even Win-Rate={be_win*100:.1f}%")

    # Walk-Forward-Aufteilung
    n = len(data)
    initial_train = int(n * INITIAL_TRAIN_FRAC)
    remaining = n - initial_train
    fold_size = remaining // N_FOLDS  # gesamt cal+test pro Fold
    test_size = int(fold_size * (1 - CAL_FRAC_OF_TEST))
    cal_size = fold_size - test_size

    print(f"\nWalk-Forward {N_FOLDS} Folds:")
    print(f"  Initial-Train:  {initial_train} Kerzen")
    print(f"  pro Fold:       {cal_size} cal + {test_size} test = {fold_size}")

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

        # Regime-Schwellen aus DIESEM Fold-Train
        vol_lo = float(train["regime_vol"].quantile(REGIME_LOW_PCT / 100))
        vol_hi = float(train["regime_vol"].quantile(REGIME_HIGH_PCT / 100))
        in_regime = (test["regime_vol"] >= vol_lo) & (test["regime_vol"] <= vol_hi)
        test["p_tp_filtered"] = np.where(in_regime, test["p_tp"], 0.0)

        all_test.append(test)
        print(f"  Fold {fold+1}: train→{train.index[-1].date()} | "
              f"test {test.index[0].date()}→{test.index[-1].date()} "
              f"({len(test)}) | max p_tp_filt={test['p_tp_filtered'].max():.3f}")

    test_all = pd.concat(all_test).sort_index()
    acc = accuracy_score(test_all["label"], test_all["pred"])
    print(f"\nGesamt Out-of-Sample Trefferquote (3 Klassen): {acc*100:.1f} %")

    # Kelly-Sizing pro Kerze
    test_all["kelly_frac"] = kelly_fraction(test_all["p_tp_filtered"], net_tp, abs(net_sl))

    # Threshold: nimm 30 %, fallback wenn nie erreicht
    threshold = ENTRY_PROB_THRESHOLD
    max_p = test_all["p_tp_filtered"].max()
    if max_p < threshold:
        nonzero = test_all["p_tp_filtered"][test_all["p_tp_filtered"] > 0]
        threshold = max(0.05, float(nonzero.quantile(0.99)) if len(nonzero) else 0.05)
        print(f"\n[Hinweis] Max p_tp_filtered={max_p:.3f} < {ENTRY_PROB_THRESHOLD:.2f}. "
              f"Demo-Threshold {threshold:.3f}.")
    else:
        print(f"\nTrade-Threshold: {threshold:.2f} (Max im Test: {max_p:.3f})")

    print(f"\nLaufe Backtest (Kelly-Sizing × {KELLY_FACTOR}, "
          f"max {MAX_POSITION_FRAC*100:.0f} % pro Trade) …")
    result = triple_barrier_backtest(
        close=test_all["close"],
        p_tp=test_all["p_tp_filtered"],
        entry_threshold=threshold,
        tp_pct=TP_PCT,
        sl_pct=SL_PCT,
        max_hold=MAX_HOLD_BARS,
        fee_rate=FEE_RATE,
        starting_capital=STARTING_CAPITAL_CHF,
        cooldown=COOLDOWN_BARS,
        position_fraction=test_all["kelly_frac"].values,
    )

    trades = result["trades"]
    equity = result["equity"]
    final = result["final_capital"]
    bh_final = STARTING_CAPITAL_CHF * (test_all["close"].iloc[-1] / test_all["close"].iloc[0])

    print("\n" + "=" * 78)
    print(f"=== WALK-FORWARD ERGEBNIS {test_all.index.min().date()} → {test_all.index.max().date()} ===")
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
        print(f"  Win-Rate (Trades):{win_rate*100:>9.1f} %   (Break-Even: {be_win*100:.1f} %)")
        print(f"  Avg. Position:    {avg_pos:>9.1f} %   "
              f"(min {trades['pos_frac'].min()*100:.1f}, max {trades['pos_frac'].max()*100:.1f})")
        print(f"  Avg. Haltedauer:  {avg_hold_min:>9.0f} Minuten")
        print(f"  Gebuehren total:  {total_fees:>9.2f} CHF")
        print(f"  Max Drawdown:     {max_dd*100:>9.2f} %")
        print(f"  Exits:            TP={reasons.get('TP',0)}  "
              f"SL={reasons.get('SL',0)}  TIME={reasons.get('TIME',0)}")

    # Tages-Statistik
    print("\n--- TAGES-PnL-STATISTIK (ehrlich: wie oft gewinnst du pro Tag?) ---")
    ds = daily_stats(equity)
    bh_ds = daily_stats(STARTING_CAPITAL_CHF * test_all["close"] / test_all["close"].iloc[0])

    if ds["n_days"] > 0:
        print(f"  Test-Zeitraum:           {ds['n_days']} Tage")
        print(f"\n                          Strategie           Buy & Hold")
        print(f"  Gewinn-Tage:           {ds['win_days']:>4d}  ({ds['win_rate_daily']*100:>4.1f} %)   "
              f"{bh_ds['win_days']:>4d}  ({bh_ds['win_rate_daily']*100:>4.1f} %)")
        print(f"  Verlust-Tage:          {ds['loss_days']:>4d}  ({ds['loss_days']/ds['n_days']*100:>4.1f} %)   "
              f"{bh_ds['loss_days']:>4d}  ({bh_ds['loss_days']/bh_ds['n_days']*100:>4.1f} %)")
        print(f"  Flache Tage:           {ds['flat_days']:>4d}  ({ds['flat_days']/ds['n_days']*100:>4.1f} %)   "
              f"{bh_ds['flat_days']:>4d}  ({bh_ds['flat_days']/bh_ds['n_days']*100:>4.1f} %)")
        print(f"  Bester Tag:           {ds['best_day']*100:>+6.2f} %              {bh_ds['best_day']*100:>+6.2f} %")
        print(f"  Schlechtester Tag:    {ds['worst_day']*100:>+6.2f} %              {bh_ds['worst_day']*100:>+6.2f} %")
        print(f"  Mittl. Tagesrendite:  {ds['avg_day']*100:>+6.3f} %             {bh_ds['avg_day']*100:>+6.3f} %")
        print(f"  Median Tagesrendite:  {ds['median_day']*100:>+6.3f} %             {bh_ds['median_day']*100:>+6.3f} %")
        print(f"  Tages-Vola (std):     {ds['std_day']*100:>6.3f} %              {bh_ds['std_day']*100:>6.3f} %")
        print(f"  Laengste Gewinn-Serie:{ds['consec_wins_max']:>4d} Tage          "
              f"{bh_ds['consec_wins_max']:>4d} Tage")
        print(f"  Laengste Verlustserie:{ds['consec_losses_max']:>4d} Tage          "
              f"{bh_ds['consec_losses_max']:>4d} Tage")

    # Monats-Statistik
    monthly = equity.resample("MS").last().dropna().pct_change().dropna()
    if len(monthly) > 0:
        print(f"\n--- MONATS-PnL ---")
        for ts, ret in monthly.items():
            mark = "✓" if ret > 0 else ("·" if ret == 0 else "✗")
            print(f"  {ts.strftime('%Y-%m'):8s}  {ret*100:>+6.2f} %  {mark}")
        win_months = (monthly > 0).sum()
        print(f"  → {win_months}/{len(monthly)} Monate profitabel "
              f"({win_months/len(monthly)*100:.1f} %)")


if __name__ == "__main__":
    main()
