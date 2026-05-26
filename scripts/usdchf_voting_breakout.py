"""USD/CHF Daily — Vol-Breakout + Voting-Confirmation 90 %.

Strategie:
- Basis-Signal: Vol-Breakout (close > yest + 1.5×ATR UND close > SMA50)
- Zusatz-Filter: ≥90 % der Voting-Indikatoren müssen zustimmen
- Position: 80 % bei Signal (User-Wunsch: mehr Volumen pro Trade)
- Hebel: 5x
- Trailing-Stop: 3×ATR
- Daten: 16 Jahre USD/CHF Daily (2010-2026)

Test verschiedener Voting-Schwellen (90/80/70/60 %), damit wir sehen,
ob 90 % zu restriktiv ist oder genau richtig.
"""
import warnings

import numpy as np
import pandas as pd

from src.data.storage import load_ohlcv
from src.indicators.technical import sma, atr
from src.backtest.trend import trend_following_backtest
from scripts.run_forex_brute_force import compute_indicators

# TREND-orientierte Bull/Bear-Regeln passend zu Vol-Breakout
# (die alten BULL_RULES waren Mean-Reversion und passen nicht zu Breakout-Setups)
BULL_RULES = {
    "rsi_14":          lambda x: x > 50,
    "bb_pct":          lambda x: x > 0.5,
    "stoch_k":         lambda x: x > 50,
    "williams_r_14":   lambda x: x > -50,
    "cci_20":          lambda x: x > 0,
    "macd_hist":       lambda x: x > 0,
    "ema_ratio":       lambda x: x > 0,
    "sma_ratio":       lambda x: x > 0,
    "roc_10":          lambda x: x > 0,
    "roc_30":          lambda x: x > 0,
}
BEAR_RULES = {
    "rsi_14":          lambda x: x < 50,
    "bb_pct":          lambda x: x < 0.5,
    "stoch_k":         lambda x: x < 50,
    "williams_r_14":   lambda x: x < -50,
    "cci_20":          lambda x: x < 0,
    "macd_hist":       lambda x: x < 0,
    "ema_ratio":       lambda x: x < 0,
    "sma_ratio":       lambda x: x < 0,
    "roc_10":          lambda x: x < 0,
    "roc_30":          lambda x: x < 0,
}

warnings.filterwarnings("ignore", category=UserWarning)

EXCHANGE, SYMBOL, TIMEFRAME = "forex", "USD/CHF", "1d"
STARTING_CAPITAL_CHF = 1000.0
FEE_RATE = 0.0002       # 0.02 % Forex-Spread

# Vol-Breakout
ATR_PERIOD = 20
BREAKOUT_MULT = 1.5
TRAILING_MULT = 3.0
SMA_PERIOD = 50

POSITION_FRAC = 0.40    # konservativer
LEVERAGE = 3.0          # konservativer

# Voting
VOTING_INDICATORS = [
    "rsi_14", "bb_pct", "stoch_k", "williams_r_14", "cci_20",
    "macd_hist", "ema_ratio", "sma_ratio", "roc_10", "roc_30",
]
VOTING_THRESHOLDS_TO_TEST = [0.90, 0.80, 0.70, 0.60]


def main():
    print(f"Lade {SYMBOL} Daily …")
    df = load_ohlcv(EXCHANGE, SYMBOL, TIMEFRAME)
    print(f"  {len(df)} Bars von {df.index.min().date()} bis {df.index.max().date()}")

    # Indikatoren berechnen
    ind = compute_indicators(df)
    df = pd.concat([df, ind], axis=1)
    df["atr"] = atr(df["high"], df["low"], df["close"], ATR_PERIOD)
    df["sma_fast"] = sma(df["close"], SMA_PERIOD)
    df["prev_close"] = df["close"].shift(1)
    df = df.dropna()

    # Vol-Breakout
    long_breakout = (
        (df["close"] > df["prev_close"] + BREAKOUT_MULT * df["atr"])
        & (df["close"] > df["sma_fast"])
    )
    short_breakout = (
        (df["close"] < df["prev_close"] - BREAKOUT_MULT * df["atr"])
        & (df["close"] < df["sma_fast"])
    )

    # Voting: wie viele Indikatoren sind bullish / bearish?
    bull_count = sum(
        BULL_RULES[ind](df[ind]).fillna(False).astype(int)
        for ind in VOTING_INDICATORS if ind in df.columns
    )
    bear_count = sum(
        BEAR_RULES[ind](df[ind]).fillna(False).astype(int)
        for ind in VOTING_INDICATORS if ind in df.columns
    )
    n_voters = len(VOTING_INDICATORS)
    bull_pct = bull_count / n_voters
    bear_pct = bear_count / n_voters

    print(f"\nBreakout-Statistik:")
    print(f"  Long-Breakout-Bars:  {int(long_breakout.sum())}  ({long_breakout.mean()*100:.1f} %)")
    print(f"  Short-Breakout-Bars: {int(short_breakout.sum())}  ({short_breakout.mean()*100:.1f} %)")

    print(f"\n{'Voting-Schwelle':<16s} {'Long-Sig':>9s} {'Short-Sig':>10s} {'Trades':>7s} "
          f"{'Endkap.':>9s} {'Return':>8s} {'MaxDD':>8s} {'WR':>5s}")

    bh_final = STARTING_CAPITAL_CHF * (df["close"].iloc[-1] / df["close"].iloc[0])
    print(f"  B&H USD/CHF: {bh_final:.2f} CHF ({(bh_final/STARTING_CAPITAL_CHF-1)*100:+.2f} %)\n")

    for thresh in VOTING_THRESHOLDS_TO_TEST:
        long_signal = long_breakout & (bull_pct >= thresh)
        short_signal = short_breakout & (bear_pct >= thresh)

        res = trend_following_backtest(
            df=df,
            long_entry=long_signal,
            short_entry=short_signal,
            atr=df["atr"],
            starting_capital=STARTING_CAPITAL_CHF,
            position_fraction=POSITION_FRAC,
            fee_rate=FEE_RATE,
            leverage=LEVERAGE,
            atr_mult=TRAILING_MULT,
            sma_exit=None,
        )
        trades = res["trades"]
        eq = res["equity"]
        max_dd = (eq / eq.cummax() - 1).min() if len(eq) else 0
        wr = (trades["trade_net_return"] > 0).mean() if len(trades) else 0
        print(f"  {thresh*100:>3.0f} %         "
              f"{int(long_signal.sum()):>9d} {int(short_signal.sum()):>10d} "
              f"{len(trades):>7d} "
              f"{res['final_capital']:>9.2f} "
              f"{res['total_return']*100:>+7.2f}% "
              f"{max_dd*100:>+7.2f}% "
              f"{wr*100:>4.1f}%")

    # Detail fuer 90 % (User-Wunsch)
    long_signal = long_breakout & (bull_pct >= 0.90)
    short_signal = short_breakout & (bear_pct >= 0.90)
    res = trend_following_backtest(
        df=df, long_entry=long_signal, short_entry=short_signal,
        atr=df["atr"], starting_capital=STARTING_CAPITAL_CHF,
        position_fraction=POSITION_FRAC, fee_rate=FEE_RATE,
        leverage=LEVERAGE, atr_mult=TRAILING_MULT, sma_exit=None,
    )
    trades = res["trades"]
    n_years = (df.index[-1] - df.index[0]).days / 365.25
    annual = (res["final_capital"] / STARTING_CAPITAL_CHF) ** (1 / n_years) - 1
    print(f"\n=== DETAIL bei 90 % Voting-Threshold ({n_years:.1f} Jahre) ===")
    print(f"  Endkapital:   {res['final_capital']:.2f} CHF")
    print(f"  Total Return: {res['total_return']*100:+.2f} %")
    print(f"  Jährlich (CAGR): {annual*100:+.2f} % p. a.")

    if len(trades) > 0:
        print(f"\n--- Alle {len(trades)} Trades ---")
        print(f"  {'Entry':<11s} {'Exit':<11s} {'Dir':>5s} {'Dauer':>5s} "
              f"{'Brutto':>8s} {'Netto':>8s} {'Kapital':>10s}")
        for _, t in trades.iterrows():
            dirstr = "LONG" if t["direction"] == 1 else "SHORT"
            print(f"  {str(t['entry_time'])[:10]} {str(t['exit_time'])[:10]} "
                  f"{dirstr:>5s} {t['hold_bars']:>4d}d "
                  f"{t['gross_return']*100:>+7.2f}% {t['net_return']*100:>+7.2f}% "
                  f"{t['capital_after']:>9.2f}")


if __name__ == "__main__":
    main()
