"""Trend-Following auf 4h-BTC mit Hebel.

Strategie:
- Entry Long:  Schlusskurs >= 20-Bar-Hoch UND SMA20 > SMA50 UND ADX > 25
- Entry Short: Schlusskurs <= 20-Bar-Tief UND SMA20 < SMA50 UND ADX > 25
- Exit:        Chandelier-Trailing-Stop (3 × ATR vom hoechsten/tiefsten Punkt seit Entry)
               oder SMA-Crossover gegen die Trade-Richtung

Test ueber 2023-01 bis 2026-05: erfasst sowohl Bullenmarkt 2024 als auch
Baerenmarkt 2025-2026.
"""
import warnings

import numpy as np
import pandas as pd

from src.data.storage import load_ohlcv
from src.indicators.technical import sma, atr, adx
from src.backtest.trend import trend_following_backtest

warnings.filterwarnings("ignore", category=UserWarning)

EXCHANGE, SYMBOL = "binance", "BTC/USDT"
INPUT_TIMEFRAME = "1h"
RESAMPLE_TO = "1D"
STARTING_CAPITAL_CHF = 1000.0
FEE_RATE = 0.001

# Turtle System 2: 55-Tage-Breakout, langer Trailing-Stop, LONG-ONLY
# Shorts haben in unseren Tests 0/8 Win-Rate — Krypto ist strukturell long-biased.
DONCHIAN_PERIOD = 55         # 55 Tage Breakout (langfristig)
SMA_FAST = 20
SMA_SLOW = 50
ADX_THRESHOLD = 0            # KEIN ADX-Filter mehr (zu restriktiv)
ATR_PERIOD = 20
ATR_TRAILING_MULT = 4.0      # weiter Stop, weniger Whipsaws
DISABLE_SMA_EXIT = True      # nur ATR-Stop, kein SMA-Crossover-Exit
LONG_ONLY = True
DISABLE_SMA_FILTER = True    # auch kein SMA-Filter beim Entry

POSITION_FRAC = 0.50
LEVERAGE = 2.0


def aggregate_to(df: pd.DataFrame, target: str) -> pd.DataFrame:
    return df.resample(target).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()


def main():
    print("Lade 1h-Daten …")
    df_1h = load_ohlcv(EXCHANGE, SYMBOL, INPUT_TIMEFRAME)
    print(f"  {len(df_1h)} 1h-Kerzen von {df_1h.index.min()} bis {df_1h.index.max()}")

    print(f"Aggregiere zu {RESAMPLE_TO} …")
    df = aggregate_to(df_1h, RESAMPLE_TO)
    print(f"  {len(df)} {RESAMPLE_TO}-Kerzen")

    print("\nBerechne Indikatoren …")
    df["sma_fast"] = sma(df["close"], SMA_FAST)
    df["sma_slow"] = sma(df["close"], SMA_SLOW)
    df["adx"] = adx(df["high"], df["low"], df["close"], ATR_PERIOD)
    df["atr"] = atr(df["high"], df["low"], df["close"], ATR_PERIOD)
    df["donchian_high"] = df["high"].rolling(DONCHIAN_PERIOD).max().shift(1)
    df["donchian_low"] = df["low"].rolling(DONCHIAN_PERIOD).min().shift(1)

    df = df.dropna()
    print(f"  Nach dropna: {len(df)} Kerzen")
    print(f"  Zeitraum: {df.index.min().date()} → {df.index.max().date()}")

    # Signale
    long_entry = df["close"] >= df["donchian_high"]
    if not DISABLE_SMA_FILTER:
        long_entry = long_entry & (df["sma_fast"] > df["sma_slow"])
    if ADX_THRESHOLD > 0:
        long_entry = long_entry & (df["adx"] > ADX_THRESHOLD)

    if LONG_ONLY:
        short_entry = pd.Series(False, index=df.index)
    else:
        short_entry = df["close"] <= df["donchian_low"]
        if not DISABLE_SMA_FILTER:
            short_entry = short_entry & (df["sma_fast"] < df["sma_slow"])
        if ADX_THRESHOLD > 0:
            short_entry = short_entry & (df["adx"] > ADX_THRESHOLD)

    print(f"\nSignal-Statistik:")
    print(f"  Long-Entry-Bars:   {int(long_entry.sum()):>5d}  "
          f"({long_entry.mean()*100:.2f} %)")
    print(f"  Short-Entry-Bars:  {int(short_entry.sum()):>5d}  "
          f"({short_entry.mean()*100:.2f} %)")
    print(f"  ADX > {ADX_THRESHOLD}:        {int((df['adx']>ADX_THRESHOLD).sum()):>5d}  "
          f"({(df['adx']>ADX_THRESHOLD).mean()*100:.1f} %)")

    print(f"\nBacktest-Konfiguration:")
    print(f"  Position pro Trade:  {POSITION_FRAC*100:.0f} %")
    print(f"  Hebel:               {LEVERAGE}x")
    print(f"  Trailing-Stop:       {ATR_TRAILING_MULT}x ATR")
    print(f"  SMA-Exit-Filter:     {SMA_FAST}-Bar SMA Cross")

    res = trend_following_backtest(
        df=df,
        long_entry=long_entry,
        short_entry=short_entry,
        atr=df["atr"],
        starting_capital=STARTING_CAPITAL_CHF,
        position_fraction=POSITION_FRAC,
        fee_rate=FEE_RATE,
        leverage=LEVERAGE,
        atr_mult=ATR_TRAILING_MULT,
        sma_exit=None if DISABLE_SMA_EXIT else df["sma_fast"],
    )

    trades = res["trades"]
    equity = res["equity"]
    final = res["final_capital"]
    day_log = res["day_log"]
    bh_final = STARTING_CAPITAL_CHF * (df["close"].iloc[-1] / df["close"].iloc[0])

    print(f"\n{'='*78}")
    print(f"=== TREND-FOLLOWING ERGEBNIS ({df.index.min().date()} → {df.index.max().date()}) ===")
    print(f"{'='*78}")
    print(f"  Strategie ({LEVERAGE}x):  {final:>12.2f} CHF   ({res['total_return']*100:+.2f} %)")
    print(f"  Buy & Hold (1x):       {bh_final:>12.2f} CHF   ({(bh_final/STARTING_CAPITAL_CHF-1)*100:+.2f} %)")
    print(f"  Anzahl Trades:         {len(trades):>10d}")

    if len(trades) > 0:
        longs = trades[trades["direction"] == 1]
        shorts = trades[trades["direction"] == -1]
        win_rate = (trades["trade_net_return"] > 0).mean()
        avg_hold_bars = trades["hold_bars"].mean()
        winners = trades[trades["trade_net_return"] > 0]
        losers = trades[trades["trade_net_return"] < 0]
        avg_win = winners["trade_net_return"].mean() * 100 if len(winners) else 0
        avg_loss = losers["trade_net_return"].mean() * 100 if len(losers) else 0
        reward_risk = abs(avg_win / avg_loss) if avg_loss < 0 else float("inf")
        max_dd = (equity / equity.cummax() - 1).min()
        liquidations = (trades["reason"] == "LIQUIDATED").sum()
        best_trade = trades["trade_net_return"].max() * 100
        worst_trade = trades["trade_net_return"].min() * 100

        print(f"  Win-Rate:              {win_rate*100:>9.1f} %")
        print(f"  Long-Trades:           {len(longs):>4d}   "
              f"Win-Rate: {(longs['trade_net_return']>0).mean()*100:.1f} %"
              if len(longs) else f"  Long-Trades:           {len(longs)}")
        print(f"  Short-Trades:          {len(shorts):>4d}   "
              f"Win-Rate: {(shorts['trade_net_return']>0).mean()*100:.1f} %"
              if len(shorts) else f"  Short-Trades:          {len(shorts)}")
        print(f"  Avg Haltedauer:        {avg_hold_bars:>9.1f} Bars (≈ {avg_hold_bars:.1f} Tage)")
        print(f"  Avg Win:               {avg_win:>+8.2f} %  (vor Hebel)")
        print(f"  Avg Loss:              {avg_loss:>+8.2f} %  (vor Hebel)")
        print(f"  Reward:Risk:           {reward_risk:>9.2f}x")
        print(f"  Bester Trade:          {best_trade:>+8.2f} %")
        print(f"  Schlechtester Trade:   {worst_trade:>+8.2f} %")
        print(f"  Max Drawdown:          {max_dd*100:>+8.2f} %")
        print(f"  Liquidationen:         {liquidations:>4d}")

        # Trade-Log
        print(f"\n--- ALLE TRADES ---")
        print(f"  {'Entry':<11s} {'Exit':<11s} {'Tage':>5s} {'Dir':>4s} "
              f"{'Brutto':>8s} {'Netto':>8s} {'Kapital':>12s}")
        for _, t in trades.iterrows():
            print(f"  {str(t['entry_time'])[:10]} {str(t['exit_time'])[:10]} "
                  f"{t['hold_bars']:>5.0f} "
                  f"{'LONG' if t['direction']==1 else 'SHORT':>5s} "
                  f"{t['gross_return']*100:>+7.2f}% "
                  f"{t['net_return']*100:>+7.2f}% "
                  f"{t['capital_after']:>11.2f}")

    if len(day_log) > 0:
        n_d = len(day_log)
        active = (day_log["day_return"] != 0).sum()
        win_d = (day_log["day_return"] > 0).sum()
        loss_d = (day_log["day_return"] < 0).sum()
        print(f"\n--- TAGES-STATISTIK ({n_d} Tage) ---")
        print(f"  Aktive Tage:    {active}  ({active/n_d*100:.1f} %)")
        print(f"  Gewinn-Tage:    {win_d}  ({win_d/n_d*100:.1f} %)")
        print(f"  Verlust-Tage:   {loss_d}  ({loss_d/n_d*100:.1f} %)")
        print(f"  Bester Tag:    {day_log['day_return'].max()*100:>+6.2f} %")
        print(f"  Schlimmster:   {day_log['day_return'].min()*100:>+6.2f} %")

    # Bullenmarkt vs Baerenmarkt aufschluesseln
    print(f"\n--- PHASEN-ANALYSE ---")
    # Heuristik: 200-SMA-Trend
    df["sma200"] = sma(df["close"], 200)
    df["is_bull"] = df["close"] > df["sma200"]
    bull_periods = df[df["is_bull"]].index
    bear_periods = df[~df["is_bull"]].index
    if len(bull_periods) > 0:
        bull_eq = equity.reindex(bull_periods)
        # PnL nur waehrend Bullenphasen
        bull_returns = bull_eq.pct_change().dropna()
        bh_bull = df.loc[bull_periods, "close"]
        bh_bull_ret = bh_bull.iloc[-1] / bh_bull.iloc[0] - 1 if len(bh_bull) > 1 else 0
        strat_bull_ret = (1 + bull_returns).prod() - 1 if len(bull_returns) else 0
        print(f"  Bullenphasen ({len(bull_periods)} Bars):")
        print(f"     Strategie kumuliert:  {strat_bull_ret*100:>+7.2f} %")
        print(f"     B&H kumuliert:        {bh_bull_ret*100:>+7.2f} %")
        print(f"     Vorteil Strategie:    {(strat_bull_ret-bh_bull_ret)*100:>+7.2f} pp")


if __name__ == "__main__":
    main()
