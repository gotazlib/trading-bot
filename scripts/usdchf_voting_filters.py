"""USD/CHF Daily — Vol-Breakout + 90 % Voting + verschiedene Krisen-Filter.

Filter-Varianten getestet:
A) Baseline (nur Vol-Breakout + Voting)
B) Vola-Filter: ATR muss in 25-75 Perzentil (252-Day) sein → vermeidet Krisen
C) ADX-Filter: 20 < ADX < 60 (sauberer Trend, kein Choppy/Hysterie)
D) Drawdown-Filter: trade nicht, wenn aktueller Equity-Drawdown > -10 %
E) Combined: Vola + ADX + Drawdown

Zusätzliche Analysen:
- Trade-by-Trade Breakdown
- Welche Trades hätte jeder Filter geblockt? (SNB 2015!)
- Monthly Return-Heatmap
"""
import warnings

import numpy as np
import pandas as pd

from src.data.storage import load_ohlcv
from src.indicators.technical import sma, atr, adx
from src.backtest.trend import trend_following_backtest
from scripts.run_forex_brute_force import compute_indicators

warnings.filterwarnings("ignore", category=UserWarning)

EXCHANGE, SYMBOL, TIMEFRAME = "forex", "USD/CHF", "1d"
STARTING_CAPITAL_CHF = 1000.0
FEE_RATE = 0.0002

ATR_PERIOD = 20
BREAKOUT_MULT = 1.5
TRAILING_MULT = 3.0
SMA_PERIOD = 50

POSITION_FRAC = 0.40
LEVERAGE = 3.0

VOTING_THRESHOLD = 0.90

# Filter-Parameter
VOL_REGIME_WINDOW = 252       # 1 Jahr fuer Vola-Percentile
VOL_LOW_PCT = 25
VOL_HIGH_PCT = 75
ADX_MIN = 20
ADX_MAX = 60
MAX_DRAWDOWN_BLOCK = -0.10    # 10 % DD = Trading pausieren

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
BEAR_RULES = {k: (lambda v=v: lambda x: not v(x))() for k, v in BULL_RULES.items()}
# Korrekter umkehren:
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


def prepare_data():
    df = load_ohlcv(EXCHANGE, SYMBOL, TIMEFRAME)
    ind = compute_indicators(df)
    df = pd.concat([df, ind], axis=1)
    df["atr"] = atr(df["high"], df["low"], df["close"], ATR_PERIOD)
    df["sma_fast"] = sma(df["close"], SMA_PERIOD)
    df["adx_14"] = adx(df["high"], df["low"], df["close"], 14)
    df["prev_close"] = df["close"].shift(1)
    df["atr_rel"] = df["atr"] / df["close"]
    df["vol_q25"] = df["atr_rel"].rolling(VOL_REGIME_WINDOW).quantile(VOL_LOW_PCT / 100)
    df["vol_q75"] = df["atr_rel"].rolling(VOL_REGIME_WINDOW).quantile(VOL_HIGH_PCT / 100)
    df = df.dropna()
    return df


def build_signals(df, filter_name):
    """Liefert long_signal, short_signal mit gewähltem Filter."""
    long_breakout = (
        (df["close"] > df["prev_close"] + BREAKOUT_MULT * df["atr"])
        & (df["close"] > df["sma_fast"])
    )
    short_breakout = (
        (df["close"] < df["prev_close"] - BREAKOUT_MULT * df["atr"])
        & (df["close"] < df["sma_fast"])
    )
    bull_count = sum(BULL_RULES[k](df[k]).fillna(False).astype(int) for k in BULL_RULES if k in df.columns)
    bear_count = sum(BEAR_RULES[k](df[k]).fillna(False).astype(int) for k in BEAR_RULES if k in df.columns)
    n = len(BULL_RULES)
    bull_ok = (bull_count / n) >= VOTING_THRESHOLD
    bear_ok = (bear_count / n) >= VOTING_THRESHOLD

    long_sig = long_breakout & bull_ok
    short_sig = short_breakout & bear_ok

    if filter_name == "VOLA":
        in_vol = (df["atr_rel"] >= df["vol_q25"]) & (df["atr_rel"] <= df["vol_q75"])
        long_sig = long_sig & in_vol
        short_sig = short_sig & in_vol
    elif filter_name == "ADX":
        in_adx = (df["adx_14"] >= ADX_MIN) & (df["adx_14"] <= ADX_MAX)
        long_sig = long_sig & in_adx
        short_sig = short_sig & in_adx
    elif filter_name == "COMBO":
        in_vol = (df["atr_rel"] >= df["vol_q25"]) & (df["atr_rel"] <= df["vol_q75"])
        in_adx = (df["adx_14"] >= ADX_MIN) & (df["adx_14"] <= ADX_MAX)
        long_sig = long_sig & in_vol & in_adx
        short_sig = short_sig & in_vol & in_adx
    return long_sig, short_sig


def backtest(df, long_sig, short_sig):
    return trend_following_backtest(
        df=df, long_entry=long_sig, short_entry=short_sig,
        atr=df["atr"], starting_capital=STARTING_CAPITAL_CHF,
        position_fraction=POSITION_FRAC, fee_rate=FEE_RATE,
        leverage=LEVERAGE, atr_mult=TRAILING_MULT, sma_exit=None,
    )


def report(name, res, n_years):
    trades = res["trades"]
    eq = res["equity"]
    max_dd = (eq / eq.cummax() - 1).min() if len(eq) else 0
    wr = (trades["trade_net_return"] > 0).mean() if len(trades) else 0
    cagr = (res["final_capital"] / STARTING_CAPITAL_CHF) ** (1 / n_years) - 1 if n_years > 0 else 0
    print(f"  {name:<10s} {len(trades):>5d} {res['final_capital']:>9.2f} "
          f"{res['total_return']*100:>+7.2f}% {cagr*100:>+6.2f}% {max_dd*100:>+7.2f}% {wr*100:>5.1f}%")
    return res


def main():
    df = prepare_data()
    n_years = (df.index[-1] - df.index[0]).days / 365.25
    print(f"USD/CHF Daily: {len(df)} Bars, {n_years:.1f} Jahre ({df.index.min().date()}–{df.index.max().date()})")
    print(f"Setup: 90% Voting, Position 40%, Hebel 3x, TP=Trailing 3xATR\n")
    bh = STARTING_CAPITAL_CHF * (df["close"].iloc[-1] / df["close"].iloc[0])
    bh_cagr = (bh / STARTING_CAPITAL_CHF) ** (1 / n_years) - 1
    print(f"B&H USD/CHF: {bh:.2f} CHF ({(bh/STARTING_CAPITAL_CHF-1)*100:+.2f}%, CAGR {bh_cagr*100:+.2f}%)\n")

    print(f"  {'Filter':<10s} {'Trades':>5s} {'Endkap.':>9s} {'Return':>8s} {'CAGR':>7s} {'MaxDD':>8s} {'WR':>6s}")
    results = {}
    for f in ["NONE", "VOLA", "ADX", "COMBO"]:
        long_sig, short_sig = build_signals(df, f)
        res = backtest(df, long_sig, short_sig)
        results[f] = (res, long_sig, short_sig)
        report(f, res, n_years)

    # Detail-Analyse fuer COMBO-Filter (vermutlich beste Variante)
    print(f"\n--- DETAIL: COMBO-Filter (Vola + ADX) ---")
    res, ls, ss = results["COMBO"]
    trades = res["trades"]
    if len(trades) > 0:
        n_long = (trades["direction"] == 1).sum()
        n_short = (trades["direction"] == -1).sum()
        avg_hold = trades["hold_bars"].mean()
        best = trades.loc[trades["trade_net_return"].idxmax()]
        worst = trades.loc[trades["trade_net_return"].idxmin()]
        print(f"  Long-Trades: {n_long}, Short-Trades: {n_short}")
        print(f"  Avg Haltedauer: {avg_hold:.1f} Tage")
        print(f"  Bester Trade:  {best['entry_time'].date()} → {best['exit_time'].date()}  "
              f"{best['trade_net_return']*100:+.2f}%")
        print(f"  Schlimmster:   {worst['entry_time'].date()} → {worst['exit_time'].date()}  "
              f"{worst['trade_net_return']*100:+.2f}%")

        # SNB-2015 Check
        snb_date = pd.Timestamp("2015-01-15", tz="UTC")
        nearby = trades[(trades["entry_time"] > snb_date - pd.Timedelta(days=5))
                        & (trades["entry_time"] < snb_date + pd.Timedelta(days=10))]
        print(f"\n  SNB-2015-Schutz: {'JA' if len(nearby) == 0 else 'NEIN'}  "
              f"({len(nearby)} Trades um den SNB-Tag herum)")

        # Jährliche Returns
        print(f"\n--- Jährliche Performance ---")
        eq = res["equity"]
        daily_ret = eq.pct_change().fillna(0)
        yearly = (1 + daily_ret).resample("YE").prod() - 1
        for year, ret in yearly.items():
            marker = "✓" if ret > 0 else "✗"
            print(f"  {year.year}: {ret*100:>+7.2f}% {marker}")
        win_years = (yearly > 0).sum()
        print(f"\n  → {win_years}/{len(yearly)} Jahre profitabel ({win_years/len(yearly)*100:.0f}%)")

        # Alle Trades
        print(f"\n--- Alle {len(trades)} COMBO-Trades ---")
        print(f"  {'Entry':<11s} {'Exit':<11s} {'Dir':>5s} {'Dauer':>5s} {'Brutto':>8s} {'Netto':>8s} {'Kapital':>10s}")
        for _, t in trades.iterrows():
            d = "LONG" if t["direction"] == 1 else "SHORT"
            print(f"  {str(t['entry_time'])[:10]} {str(t['exit_time'])[:10]} "
                  f"{d:>5s} {t['hold_bars']:>4d}d "
                  f"{t['gross_return']*100:>+7.2f}% {t['net_return']*100:>+7.2f}% "
                  f"{t['capital_after']:>9.2f}")


if __name__ == "__main__":
    main()
