"""Explizite Indikator-Paar-Signale + Backtest + 2x Hebel.

Vorgehen:
1. Fuer jeden Indikator: definiere eine "bullish-Regel" (z.B. RSI < 35 = bullish).
2. Berechne alle Paare (A AND B) als Buy-Signal-Generatoren.
3. Backteste jedes Paar einzeln auf Trainings-Daten (welches Paar liefert das
   beste Sharpe-Verhaeltnis?).
4. Top-N-Paare bilden ein Voting-Ensemble fuer den finalen Backtest.
5. Final-Backtest auf Holdout-Set mit Hebel.
"""
import itertools
import warnings

import numpy as np
import pandas as pd

from src.data.storage import load_ohlcv, load_funding
from src.ml.features import build_features_with_time, build_funding_features
from src.backtest.engine import triple_barrier_backtest_ls

warnings.filterwarnings("ignore", category=UserWarning)

EXCHANGE, SYMBOL, TIMEFRAME = "binance", "BTC/USDT", "15m"
BAR_MINUTES = 15
STARTING_CAPITAL_CHF = 1000.0
FEE_RATE = 0.001

TP_PCT = 0.015
SL_PCT = 0.005
MAX_HOLD_BARS = 24      # 6h

LEVERAGE = 2.0          # 2x Hebel auf Binance Futures
COOLDOWN_BARS = 2

# Ensemble: wie viele Top-Paare und wie viele muessen zustimmen?
TOP_N_PAIRS = 10
MIN_VOTES_TO_TRADE = 8   # 80 % der Top-Paare muessen zustimmen

OPT_FRAC = 0.60          # Optimization-Set fuer Paar-Auswahl
# Rest = Holdout

DAILY_PROFIT_TARGET = 0.02   # 2 % bei 2x Hebel
DAILY_LOSS_LIMIT = -0.02

REGIME_LOW_PCT = 25
REGIME_HIGH_PCT = 75
VOL_WINDOW_BARS = 96


# --- Bull-Regeln pro Indikator ---
# Jede Regel gibt True/False zurueck pro Bar.
BULL_RULES = {
    # Mean-Reversion (oversold = bullish)
    "rsi_14":             lambda x: x < 35,
    "bb_pct":             lambda x: x < 0.2,
    "stoch_k":            lambda x: x < 25,
    "williams_r_14":      lambda x: x < -80,
    "cci_20":             lambda x: x < -100,
    # Trend (positive = bullish)
    "macd_hist":          lambda x: x > 0,
    "ema_ratio":          lambda x: x > 0,
    "sma_ratio":          lambda x: x > 0,
    "roc_10":             lambda x: x > 0,
    "roc_30":             lambda x: x > 0,
    "ret_1":              lambda x: x > 0,
    "ret_3":              lambda x: x > 0,
    "ret_6":              lambda x: x > 0,
    # Volumen
    "vol_change":         lambda x: x > 0,
    "obv_slope":          lambda x: x > 0,
    # Funding (negativ = Shorts ueberfuellt = Long-Squeeze-Chance)
    "funding_now":        lambda x: x < 0,
    "funding_z":          lambda x: x < -1,
}

BEAR_RULES = {
    # Mean-Reversion (overbought = bearish)
    "rsi_14":             lambda x: x > 65,
    "bb_pct":             lambda x: x > 0.8,
    "stoch_k":            lambda x: x > 75,
    "williams_r_14":      lambda x: x > -20,
    "cci_20":             lambda x: x > 100,
    "macd_hist":          lambda x: x < 0,
    "ema_ratio":          lambda x: x < 0,
    "sma_ratio":          lambda x: x < 0,
    "roc_10":             lambda x: x < 0,
    "roc_30":             lambda x: x < 0,
    "ret_1":              lambda x: x < 0,
    "ret_3":              lambda x: x < 0,
    "ret_6":              lambda x: x < 0,
    "vol_change":         lambda x: x > 0,    # Volumen-Anstieg bei Fall = Panic
    "obv_slope":          lambda x: x < 0,
    "funding_now":        lambda x: x > 0,
    "funding_z":          lambda x: x > 1,
}


def evaluate_pair(close, regime_ok, bull_signal_a, bull_signal_b,
                  bear_signal_a=None, bear_signal_b=None):
    """Backtest fuer ein Indikator-Paar.

    Buy: bull_a AND bull_b AND regime_ok
    Sell (Short): bear_a AND bear_b AND regime_ok  (falls Bear-Regeln gegeben)

    Liefert Sharpe + n_trades + total_return.
    """
    # Konstruiere "P(TP)"-aequivalente Pseudo-Wahrscheinlichkeiten 0/1
    p_long = (bull_signal_a & bull_signal_b & regime_ok).astype(float)
    if bear_signal_a is not None:
        p_short = (bear_signal_a & bear_signal_b & regime_ok).astype(float)
    else:
        p_short = pd.Series(0.0, index=close.index)

    # Fixed 20% Position pro Trade
    pos_frac = np.where((p_long > 0.5) | (p_short > 0.5), 0.20, 0.0)

    res = triple_barrier_backtest_ls(
        close=close,
        p_tp_long=p_long, p_tp_short=p_short,
        entry_threshold=0.5,    # Signal ist 0 oder 1
        tp_pct=TP_PCT, sl_pct=SL_PCT, max_hold=MAX_HOLD_BARS,
        fee_rate=FEE_RATE, starting_capital=STARTING_CAPITAL_CHF,
        cooldown=COOLDOWN_BARS, position_fraction=pos_frac,
        leverage=1.0,    # Paar-Ranking ohne Hebel — sauberer Vergleich
    )
    eq = res["equity"]
    eq_ret = eq.pct_change().dropna()
    sharpe = (eq_ret.mean() / eq_ret.std() * np.sqrt(96 * 365)) if eq_ret.std() > 0 else 0.0
    return {
        "total_return": res["total_return"],
        "sharpe": sharpe,
        "n_trades": len(res["trades"]),
        "win_rate": (res["trades"]["trade_net_return"] > 0).mean() if len(res["trades"]) else 0.0,
    }


def main():
    print("Lade Daten …")
    df = load_ohlcv(EXCHANGE, SYMBOL, TIMEFRAME)
    fund = load_funding(EXCHANGE, SYMBOL)
    print(f"  {len(df)} 15m-Kerzen, {len(fund)} Funding-Datenpunkte")

    print("Baue Features …")
    X_ta = build_features_with_time(df)
    X_fund = build_funding_features(df, fund, bar_minutes=BAR_MINUTES)
    X = pd.concat([X_ta, X_fund], axis=1)
    rets = df["close"].pct_change()
    regime_vol = rets.rolling(VOL_WINDOW_BARS).std()

    data = X.copy()
    data["close"] = df["close"]
    data["regime_vol"] = regime_vol
    data = data.dropna()

    # Split
    n = len(data)
    opt_end = int(n * OPT_FRAC)
    opt_data = data.iloc[:opt_end]
    holdout_data = data.iloc[opt_end:]
    print(f"\nSplit:")
    print(f"  Optimization: {opt_data.index[0].date()} → {opt_data.index[-1].date()}  ({len(opt_data)})")
    print(f"  Holdout:      {holdout_data.index[0].date()} → {holdout_data.index[-1].date()}  ({len(holdout_data)})")

    # Regime auf Opt-Set
    vol_lo = float(opt_data["regime_vol"].quantile(REGIME_LOW_PCT / 100))
    vol_hi = float(opt_data["regime_vol"].quantile(REGIME_HIGH_PCT / 100))
    regime_opt = (opt_data["regime_vol"] >= vol_lo) & (opt_data["regime_vol"] <= vol_hi)

    # Bull-Signale pro Indikator auf Opt-Set
    bull_signals = {}
    bear_signals = {}
    for ind, rule in BULL_RULES.items():
        if ind in opt_data.columns:
            bull_signals[ind] = rule(opt_data[ind]).fillna(False)
    for ind, rule in BEAR_RULES.items():
        if ind in opt_data.columns:
            bear_signals[ind] = rule(opt_data[ind]).fillna(False)
    indicators = list(bull_signals.keys())
    print(f"\n{len(indicators)} Indikatoren mit Bull-Regeln")

    # --- Schritt 1: Backtest jedes Paares auf Opt-Set ---
    pairs = list(itertools.combinations(indicators, 2))
    print(f"\nTeste {len(pairs)} Indikator-Paare auf Optimization-Set …")

    pair_results = []
    for ind_a, ind_b in pairs:
        result = evaluate_pair(
            opt_data["close"], regime_opt,
            bull_signals[ind_a], bull_signals[ind_b],
            bear_signals.get(ind_a), bear_signals.get(ind_b),
        )
        if result["n_trades"] < 10:    # min 10 trades fuer Robustheit
            continue
        pair_results.append({
            "pair": f"{ind_a} + {ind_b}",
            "ind_a": ind_a, "ind_b": ind_b,
            **result,
        })

    pair_df_raw = pd.DataFrame(pair_results)
    # KRITISCHER FIX: nur Paare mit POSITIVEM Return im Optimization-Set
    # — Sharpe alleine ist verfuehrerisch (siehe vorheriger Run)
    profitable = pair_df_raw[pair_df_raw["total_return"] > 0].sort_values("sharpe", ascending=False)
    print(f"\nFilter: Paare mit POSITIVEM Return im Optimization-Set:")
    print(f"  {len(profitable)} von {len(pair_df_raw)} Paaren bleiben uebrig.")
    if len(profitable) == 0:
        print("\n  Kein einziges Paar war auf dem Opt-Set profitabel.")
        print("  Das ist die ehrliche Antwort: kein Edge in Paar-Voting-Regeln.")
        return

    print(f"\nTOP-15 PAARE (nur profitable, Ranking nach Sharpe):")
    print(f"  {'Pair':<35s} {'Return':>8s} {'Sharpe':>7s} {'WR':>6s} {'Trades':>7s}")
    for _, r in profitable.head(15).iterrows():
        print(f"  {r['pair']:<35s} {r['total_return']*100:>+7.2f}% "
              f"{r['sharpe']:>6.2f} {r['win_rate']*100:>5.1f}% {int(r['n_trades']):>7d}")

    if len(profitable) < TOP_N_PAIRS:
        print(f"\n  Nur {len(profitable)} profitable Paare gefunden — verwende alle.")
        top_pairs = profitable
    else:
        top_pairs = profitable.head(TOP_N_PAIRS)
    print(f"\nVerwende Top-{TOP_N_PAIRS}-Paare als Voting-Ensemble.")
    print(f"Trade wenn mindestens {MIN_VOTES_TO_TRADE} von {TOP_N_PAIRS} Paaren 'Buy' sagen.")

    # --- Schritt 2: Ensemble-Voting auf Holdout-Set ---
    regime_h = (holdout_data["regime_vol"] >= vol_lo) & (holdout_data["regime_vol"] <= vol_hi)
    bull_h = {ind: BULL_RULES[ind](holdout_data[ind]).fillna(False) for ind in indicators}
    bear_h = {ind: BEAR_RULES[ind](holdout_data[ind]).fillna(False) for ind in indicators}

    n_pairs_used = len(top_pairs)
    long_votes = pd.Series(0, index=holdout_data.index)
    short_votes = pd.Series(0, index=holdout_data.index)
    for _, r in top_pairs.iterrows():
        a, b = r["ind_a"], r["ind_b"]
        long_votes = long_votes + (bull_h[a] & bull_h[b]).astype(int)
        short_votes = short_votes + (bear_h[a] & bear_h[b]).astype(int)

    p_long_h = (long_votes / n_pairs_used).where(regime_h, 0.0)
    p_short_h = (short_votes / n_pairs_used).where(regime_h, 0.0)
    # 80 % Voting-Regel: an die tatsaechliche Paar-Anzahl angepasst
    entry_threshold = min(MIN_VOTES_TO_TRADE, n_pairs_used) / n_pairs_used

    # Aggressivere Position bei sehr hohem Vote-Anteil — bis 50 %
    pos_frac = np.where(
        (p_long_h > entry_threshold) | (p_short_h > entry_threshold),
        np.maximum(p_long_h, p_short_h).values * 0.50,
        0.0
    )

    n_signals = int(((p_long_h > entry_threshold) | (p_short_h > entry_threshold)).sum())
    print(f"\nSignale ueber {entry_threshold*100:.0f}%-Voting-Schwelle: {n_signals} von {len(holdout_data)} Kerzen "
          f"({n_signals/len(holdout_data)*100:.3f} %)")

    print(f"\n--- Backtest Top-{TOP_N_PAIRS}-Ensemble auf HOLDOUT mit Hebel={LEVERAGE}x ---")
    res = triple_barrier_backtest_ls(
        close=holdout_data["close"],
        p_tp_long=p_long_h, p_tp_short=p_short_h,
        entry_threshold=entry_threshold,
        tp_pct=TP_PCT, sl_pct=SL_PCT, max_hold=MAX_HOLD_BARS,
        fee_rate=FEE_RATE, starting_capital=STARTING_CAPITAL_CHF,
        cooldown=COOLDOWN_BARS, position_fraction=pos_frac,
        daily_profit_target=DAILY_PROFIT_TARGET, daily_loss_limit=DAILY_LOSS_LIMIT,
        leverage=LEVERAGE,
    )

    trades = res["trades"]
    equity = res["equity"]
    final = res["final_capital"]
    day_log = res["day_log"]
    bh_final = STARTING_CAPITAL_CHF * (holdout_data["close"].iloc[-1] / holdout_data["close"].iloc[0])

    print(f"\n{'='*78}")
    print(f"HOLDOUT-ERGEBNIS {holdout_data.index[0].date()} → {holdout_data.index[-1].date()}")
    print(f"{'='*78}")
    print(f"  Strategie ({LEVERAGE}x):  {final:>10.2f} CHF   ({res['total_return']*100:+.2f} %)")
    print(f"  Buy & Hold (1x):    {bh_final:>10.2f} CHF   ({(bh_final/STARTING_CAPITAL_CHF-1)*100:+.2f} %)")
    print(f"  Anzahl Trades:      {len(trades):>10d}")

    if len(trades) > 0:
        longs = trades[trades["direction"] == 1]
        shorts = trades[trades["direction"] == -1]
        win_rate = (trades["trade_net_return"] > 0).mean()
        liquidated = (trades["reason"] == "LIQUIDATED").sum()
        max_dd = (equity / equity.cummax() - 1).min()
        reasons = trades["reason"].value_counts().to_dict()
        net_tp = (1 + TP_PCT) * (1 - FEE_RATE) ** 2 - 1
        net_sl = (1 - SL_PCT) * (1 - FEE_RATE) ** 2 - 1
        be_win = abs(net_sl) / (net_tp + abs(net_sl))
        print(f"  Win-Rate:           {win_rate*100:>9.1f} %   (Break-Even: {be_win*100:.1f} %)")
        print(f"  Long-Trades:        {len(longs):>4d}   Short-Trades: {len(shorts):>4d}")
        print(f"  Max Drawdown:       {max_dd*100:>9.2f} %")
        print(f"  Liquidationen:      {liquidated:>4d}  ← Achtung wenn > 0!")
        print(f"  Exits:              TP={reasons.get('TP',0)}  "
              f"SL={reasons.get('SL',0)}  TIME={reasons.get('TIME',0)}  "
              f"LIQ={reasons.get('LIQUIDATED',0)}")

    if len(day_log) > 0:
        n_d = len(day_log)
        active = (day_log["day_return"] != 0).sum()
        win_d = (day_log["day_return"] > 0).sum()
        loss_d = (day_log["day_return"] < 0).sum()
        print(f"\n--- TAGES-STATISTIK ---")
        print(f"  Tage gesamt:         {n_d}")
        print(f"  Aktive Tage:         {active}  ({active/n_d*100:.1f} %)")
        print(f"  Gewinn-Tage:         {win_d}  ({win_d/n_d*100:.1f} %)")
        print(f"  Verlust-Tage:        {loss_d}  ({loss_d/n_d*100:.1f} %)")
        print(f"  Bester Tag:         {day_log['day_return'].max()*100:>+6.2f} %")
        print(f"  Schlechtester Tag:  {day_log['day_return'].min()*100:>+6.2f} %")


if __name__ == "__main__":
    main()
