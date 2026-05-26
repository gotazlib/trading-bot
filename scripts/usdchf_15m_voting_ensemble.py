"""USD/CHF 15m Voting-Ensemble basierend auf 10-Agent-Konsens.

Architektur:
- ANCHOR (roc_10): MUSS zustimmen — Hauptindikator
- CONFIRMER (rsi_14, keltner_pct, macd_hist, ema_ratio): ≥N müssen zustimmen
- EXCLUDED (cci_20, williams_r_14, sma_ratio): ignoriert (Agents-Konsens)

Tests:
- 70/30 Split (Opt-Set vs Holdout)
- Verschiedene MIN_CONFIRMERS (2, 3, 4)
- Verschiedene Hebel (1x, 2x, 5x)
"""
import warnings
import numpy as np
import pandas as pd

from src.data.storage import load_ohlcv
from src.backtest.engine import triple_barrier_backtest_ls
from scripts.usdchf_15m_pair_matrix import (
    BULL_RULES, BEAR_RULES, compute_indicators,
    TP_PCT, SL_PCT, MAX_HOLD_BARS, FEE_RATE, COOLDOWN,
    EXCHANGE, SYMBOL, TIMEFRAME, STARTING_CAPITAL_CHF, OPT_FRAC,
)

warnings.filterwarnings("ignore", category=UserWarning)

ANCHOR = "roc_10"
CONFIRMERS = ["rsi_14", "keltner_pct", "macd_hist", "ema_ratio"]
POSITION_PCT = 0.40           # 40 % pro Trade (für Hebel-Skalierung)

LEVERAGES_TO_TEST = [1.0, 2.0, 5.0]
MIN_CONF_TO_TEST = [2, 3, 4]


def build_voting_signals(df, anchor, confirmers, min_conf):
    """Liefert (long_signal, short_signal) als bool Series."""
    long_anchor = BULL_RULES[anchor](df[anchor]).fillna(False)
    short_anchor = BEAR_RULES[anchor](df[anchor]).fillna(False)
    long_conf = sum(BULL_RULES[c](df[c]).fillna(False).astype(int) for c in confirmers)
    short_conf = sum(BEAR_RULES[c](df[c]).fillna(False).astype(int) for c in confirmers)
    long_sig = long_anchor & (long_conf >= min_conf)
    short_sig = short_anchor & (short_conf >= min_conf)
    return long_sig, short_sig


def run_backtest(df, long_sig, short_sig, leverage):
    p_long = long_sig.astype(float)
    p_short = short_sig.astype(float)
    pos_frac = np.where((p_long > 0.5) | (p_short > 0.5), POSITION_PCT, 0.0)
    return triple_barrier_backtest_ls(
        close=df["close"], p_tp_long=p_long, p_tp_short=p_short,
        entry_threshold=0.5,
        tp_pct=TP_PCT, sl_pct=SL_PCT, max_hold=MAX_HOLD_BARS,
        fee_rate=FEE_RATE, starting_capital=STARTING_CAPITAL_CHF,
        cooldown=COOLDOWN, position_fraction=pos_frac, leverage=leverage,
    )


def main():
    print(f"Lade {SYMBOL} {TIMEFRAME} …")
    df = load_ohlcv(EXCHANGE, SYMBOL, TIMEFRAME)
    ind = compute_indicators(df)
    df = pd.concat([df, ind], axis=1).dropna()
    n = len(df)
    opt_end = int(n * OPT_FRAC)
    df_opt = df.iloc[:opt_end]
    df_h = df.iloc[opt_end:].copy()
    bh_opt = STARTING_CAPITAL_CHF * (df_opt["close"].iloc[-1] / df_opt["close"].iloc[0])
    bh_h = STARTING_CAPITAL_CHF * (df_h["close"].iloc[-1] / df_h["close"].iloc[0])

    print(f"\nSplit:")
    print(f"  Opt-Set:  {df_opt.index[0]} → {df_opt.index[-1]}  ({len(df_opt)} Bars)")
    print(f"  Holdout:  {df_h.index[0]} → {df_h.index[-1]}  ({len(df_h)} Bars)")
    print(f"\n  B&H Opt:     {bh_opt:.2f} CHF ({(bh_opt/STARTING_CAPITAL_CHF-1)*100:+.2f}%)")
    print(f"  B&H Holdout: {bh_h:.2f} CHF ({(bh_h/STARTING_CAPITAL_CHF-1)*100:+.2f}%)")

    print(f"\nVoting-Architektur:")
    print(f"  Anchor:     {ANCHOR}")
    print(f"  Confirmer:  {CONFIRMERS}")
    print(f"  Excluded:   cci_20, williams_r_14, sma_ratio")
    print(f"  TP/SL:      {TP_PCT*100:.2f}% / {SL_PCT*100:.2f}%, Hold max {MAX_HOLD_BARS} Bars = {MAX_HOLD_BARS/4:.0f}h")

    # Signal-Count auf beiden Sets
    print(f"\n{'='*90}")
    print(f"{'Min Conf':>8s} {'Set':>8s} {'Long':>5s} {'Short':>5s} | "
          f"{'Hebel':>5s} {'Endkap.':>10s} {'Return':>9s} {'Trades':>6s} {'WR':>5s} {'MaxDD':>8s}")
    print(f"{'-'*90}")

    summary = []
    for min_conf in MIN_CONF_TO_TEST:
        long_o, short_o = build_voting_signals(df_opt, ANCHOR, CONFIRMERS, min_conf)
        long_h, short_h = build_voting_signals(df_h, ANCHOR, CONFIRMERS, min_conf)
        for lev in LEVERAGES_TO_TEST:
            res_o = run_backtest(df_opt, long_o, short_o, lev)
            res_h = run_backtest(df_h, long_h, short_h, lev)
            for label, set_name, sig_l, sig_s, res in [
                ("OPT", "OPT", long_o, short_o, res_o),
                ("HOLD", "HOLD", long_h, short_h, res_h),
            ]:
                trades = res["trades"]
                eq = res["equity"]
                max_dd = (eq / eq.cummax() - 1).min() if len(eq) else 0
                wr = (trades["trade_net_return"] > 0).mean() if len(trades) else 0
                print(f"{min_conf:>8d} {set_name:>8s} {int(sig_l.sum()):>5d} {int(sig_s.sum()):>5d} | "
                      f"{lev:>4.1f}x {res['final_capital']:>10.2f} "
                      f"{res['total_return']*100:>+8.2f}% {len(trades):>6d} "
                      f"{wr*100:>4.1f}% {max_dd*100:>+7.2f}%")
                if label == "HOLD":
                    summary.append({
                        "min_conf": min_conf, "lev": lev,
                        "ret": res["total_return"], "wr": wr,
                        "trades": len(trades), "max_dd": max_dd,
                    })
        print(f"{'-'*90}")

    # Bester Holdout-Lauf
    pos = [s for s in summary if s["ret"] > 0]
    print(f"\n=== HOLDOUT-FAZIT ===")
    print(f"  Total getestet:    {len(summary)} Konfigurationen")
    print(f"  Positiv:           {len(pos)}")
    print(f"  B&H Holdout:       {(bh_h/STARTING_CAPITAL_CHF-1)*100:+.2f}%")
    if len(summary):
        best = max(summary, key=lambda x: x["ret"])
        print(f"  Bester Lauf:       min_conf={best['min_conf']}, Hebel={best['lev']:.0f}x "
              f"→ {best['ret']*100:+.2f}% (Trades: {best['trades']}, WR: {best['wr']*100:.1f}%)")
        beats_bh = sum(1 for s in summary if s["ret"] > (bh_h/STARTING_CAPITAL_CHF-1))
        print(f"  Schlagen B&H:      {beats_bh} von {len(summary)} Konfigurationen")


if __name__ == "__main__":
    main()
