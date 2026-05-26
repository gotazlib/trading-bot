"""Volatility-Breakout-Strategie auf 5 Forex-Majors (2010-2026).

ALTERNATIVE zur Pair-Voting-Strategie (run_forex_10pairs.py), die mit
Ensemble-Voting auf den Holdout-Daten ca. -1 % machte. Hier wird ein
volatilitaetsadaptiver ATR-Breakout getestet:

Strategie pro Pair (daily):
  - ATR-Periode:   20
  - SMA-Filter:    50 (Regimedetektion)
  - Breakout:      |close - yesterday_close| > 1.5 * ATR(20)
  - Long-Entry:    close > yesterday_close + 1.5 * ATR(20)  UND  close > SMA(50)
  - Short-Entry:   close < yesterday_close - 1.5 * ATR(20)  UND  close < SMA(50)
  - Exit:          Chandelier-Trailing-Stop (3 * ATR vom Hoch/Tief seit Entry)
  - Hebel:         5x
  - Fees:          0.02 % je Seite (Forex-Spread)
  - Position:      30 % des Pair-Kapitals pro Trade (mit 5x Hebel = 1.5x Notional)

Engine: src/backtest/trend.py::trend_following_backtest (eines Pair nach dem
anderen, wie run_trend_following.py). Anschliessend Aggregation aller Pairs
zu einem Portfolio.
"""
import warnings

import numpy as np
import pandas as pd

from src.data.storage import load_ohlcv
from src.indicators.technical import sma, atr
from src.backtest.trend import trend_following_backtest

warnings.filterwarnings("ignore", category=UserWarning)

EXCHANGE = "forex"
TIMEFRAME = "1d"
PAIRS = ["USD/CHF", "EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD"]

STARTING_CAPITAL_CHF = 1000.0
CAPITAL_PER_PAIR = STARTING_CAPITAL_CHF / len(PAIRS)   # 200 CHF je Pair
FEE_RATE = 0.0002              # 0.02 % Forex-Spread
LEVERAGE = 5.0
POSITION_FRAC = 0.30           # 30 % je Trade
ATR_PERIOD = 20
SMA_PERIOD = 50
BREAKOUT_K = 1.5               # 1.5 x ATR Breakout-Schwelle
ATR_TRAIL_MULT = 3.0           # 3 x ATR Chandelier-Stop

START_DATE = "2010-01-01"
END_DATE = "2026-12-31"


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet Indikatoren und Long/Short-Entry-Signale fuer ein Pair."""
    out = df.copy()
    out["atr"] = atr(out["high"], out["low"], out["close"], ATR_PERIOD)
    out["sma50"] = sma(out["close"], SMA_PERIOD)
    out["prev_close"] = out["close"].shift(1)
    out["upper_break"] = out["prev_close"] + BREAKOUT_K * out["atr"]
    out["lower_break"] = out["prev_close"] - BREAKOUT_K * out["atr"]
    out["long_entry"] = (out["close"] > out["upper_break"]) & (out["close"] > out["sma50"])
    out["short_entry"] = (out["close"] < out["lower_break"]) & (out["close"] < out["sma50"])
    return out.dropna()


def run_pair(symbol: str) -> dict | None:
    df = load_ohlcv(EXCHANGE, symbol, TIMEFRAME)
    if df.empty:
        print(f"  {symbol}: keine Daten in DB, ueberspringe")
        return None
    df = df.loc[START_DATE:END_DATE]
    if df.empty:
        print(f"  {symbol}: keine Daten im Zeitraum")
        return None

    df = build_signals(df)
    if len(df) < SMA_PERIOD + 5:
        print(f"  {symbol}: zu wenige Bars nach Indikatoren ({len(df)})")
        return None

    res = trend_following_backtest(
        df=df,
        long_entry=df["long_entry"],
        short_entry=df["short_entry"],
        atr=df["atr"],
        starting_capital=CAPITAL_PER_PAIR,
        position_fraction=POSITION_FRAC,
        fee_rate=FEE_RATE,
        leverage=LEVERAGE,
        atr_mult=ATR_TRAIL_MULT,
        sma_exit=None,
    )

    trades = res["trades"].copy()
    if len(trades) > 0:
        trades["symbol"] = symbol
    bh = CAPITAL_PER_PAIR * (df["close"].iloc[-1] / df["close"].iloc[0])
    return {
        "symbol": symbol,
        "df": df,
        "trades": trades,
        "equity": res["equity"],
        "final": res["final_capital"],
        "bh_final": bh,
        "start": df.index.min(),
        "end": df.index.max(),
        "n_bars": len(df),
        "long_signals": int(df["long_entry"].sum()),
        "short_signals": int(df["short_entry"].sum()),
    }


def main():
    print(f"=== Volatility-Breakout auf {len(PAIRS)} Forex-Pairs ===")
    print(f"   ATR({ATR_PERIOD}) x {BREAKOUT_K} Breakout, SMA({SMA_PERIOD}) Trend-Filter")
    print(f"   Trailing-Stop {ATR_TRAIL_MULT}x ATR, Hebel {LEVERAGE}x")
    print(f"   Startkapital {STARTING_CAPITAL_CHF:.0f} CHF (= {CAPITAL_PER_PAIR:.0f} je Pair)")
    print()

    results = []
    for sym in PAIRS:
        print(f"-> {sym}")
        r = run_pair(sym)
        if r is not None:
            results.append(r)
            print(f"   {r['n_bars']} Bars  {r['start'].date()} → {r['end'].date()}  "
                  f"Long-Sig {r['long_signals']}  Short-Sig {r['short_signals']}")

    if not results:
        print("Keine Ergebnisse — Datenbank scheint leer fuer diese Pairs.")
        return

    print()
    print("=" * 84)
    print(f"=== ERGEBNIS PRO PAIR (Hebel {LEVERAGE}x) ===")
    print("=" * 84)
    header = f"  {'Pair':<9s} {'Endkap.':>10s} {'Return':>9s} {'B&H':>10s} {'Trades':>6s} {'WR':>6s} {'MaxDD':>7s}"
    print(header)
    print("-" * 84)

    total_final = 0.0
    total_bh = 0.0
    all_trades = []
    for r in results:
        trades = r["trades"]
        eq = r["equity"]
        wr = (trades["trade_net_return"] > 0).mean() if len(trades) else 0.0
        max_dd = (eq / eq.cummax() - 1).min() if len(eq) else 0.0
        ret_pct = (r["final"] / CAPITAL_PER_PAIR - 1) * 100
        bh_pct = (r["bh_final"] / CAPITAL_PER_PAIR - 1) * 100
        total_final += r["final"]
        total_bh += r["bh_final"]
        all_trades.append(trades)
        print(f"  {r['symbol']:<9s} {r['final']:>10.2f} {ret_pct:>+8.2f}% "
              f"{r['bh_final']:>10.2f} {len(trades):>6d} {wr*100:>5.1f}% {max_dd*100:>+6.2f}%")

    print("-" * 84)
    port_ret = (total_final / STARTING_CAPITAL_CHF - 1) * 100
    bh_ret = (total_bh / STARTING_CAPITAL_CHF - 1) * 100
    print(f"  {'PORTFOL.':<9s} {total_final:>10.2f} {port_ret:>+8.2f}% "
          f"{total_bh:>10.2f}  (B&H {bh_ret:+.2f}%)")

    # Vergleich mit Pair-Voting-Baseline
    pair_voting_ret = -1.0
    edge = port_ret - pair_voting_ret
    print()
    print(f"  Pair-Voting-Baseline: {pair_voting_ret:+.2f}%")
    print(f"  Vol-Breakout:         {port_ret:+.2f}%")
    print(f"  Edge ggue Pair-Vot.:  {edge:+.2f} pp  "
          f"({'BESSER' if edge > 0 else 'SCHLECHTER'})")

    # Trade-Aggregation
    combined = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if len(combined) > 0:
        wins = (combined["trade_net_return"] > 0).sum()
        losses = (combined["trade_net_return"] < 0).sum()
        print()
        print(f"  Trades gesamt:  {len(combined)}  (Wins {wins} / Losses {losses})")
        print(f"  Win-Rate:       {wins/len(combined)*100:.1f} %")
        print(f"  Avg PnL/Trade:  {combined['pnl_chf'].mean():+.3f} CHF")
        print(f"  Bester PnL:     {combined['pnl_chf'].max():+.2f} CHF")
        print(f"  Schlimmster:    {combined['pnl_chf'].min():+.2f} CHF")

        print()
        print("=== TOP-3 TRADES NACH PnL (CHF) ===")
        top3 = combined.reindex(combined["pnl_chf"].abs().sort_values(ascending=False).index).head(3)
        for _, t in top3.iterrows():
            d = "LONG" if t["direction"] == 1 else "SHORT"
            print(f"  {t['symbol']:<9s} {d:<5s} "
                  f"{str(t['entry_time'])[:10]} → {str(t['exit_time'])[:10]}  "
                  f"hold {int(t['hold_bars']):>3d}d  "
                  f"gross {t['gross_return']*100:>+6.2f}%  "
                  f"PnL {t['pnl_chf']:>+7.2f} CHF")


if __name__ == "__main__":
    main()
