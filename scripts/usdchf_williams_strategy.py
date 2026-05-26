"""USD/CHF Williams %R Mean-Reversion-Strategie.

Basierend auf der Signal-Analyse: Williams %R Extrem-Signale haben
70-74 % Hit-Rate über 20 Jahre. Pure Implementierung:
- Williams<-90 → Long Position T+0 close, Exit T+1 close
- Williams>-10 → Short Position T+0 close, Exit T+1 close
- Hebel 5x, Position 30 %
- Spread 0.02 %
"""
import warnings
import numpy as np
import pandas as pd

from src.data.storage import load_ohlcv
from src.indicators.technical import williams_r

warnings.filterwarnings("ignore", category=UserWarning)

STARTING_CAPITAL_CHF = 1000.0
FEE_RATE = 0.0002             # 0.02 % Spread
POSITION_FRAC = 0.30
LEVERAGE = 5.0

# Williams-Schwellen
WR_OVERSOLD = -90    # Long-Signal
WR_OVERBOUGHT = -10  # Short-Signal
HOLD_DAYS = 1        # 1 Tag halten (Best-Horizon laut Analyse)


def main():
    df = load_ohlcv("forex", "USD/CHF", "1d")
    df["wr"] = williams_r(df["high"], df["low"], df["close"], 14)
    df = df.dropna()

    # Signale (Cross-Events, nicht "ist im Zustand")
    long_signal = (df["wr"].shift(1) >= WR_OVERSOLD) & (df["wr"] < WR_OVERSOLD)
    short_signal = (df["wr"].shift(1) <= WR_OVERBOUGHT) & (df["wr"] > WR_OVERBOUGHT)

    print(f"USD/CHF Williams %R Strategie")
    print(f"  Zeitraum: {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Bars: {len(df)}")
    print(f"  Long-Signale (WR<{WR_OVERSOLD}): {int(long_signal.sum())}")
    print(f"  Short-Signale (WR>{WR_OVERBOUGHT}): {int(short_signal.sum())}")

    # Backtest: Position eröffnen bei Signal-Bar close, schließen nach HOLD_DAYS
    capital = STARTING_CAPITAL_CHF
    trades = []
    in_position = 0
    entry_idx = None
    entry_price = 0.0

    close = df["close"].values
    n = len(df)

    for i in range(n - 1):
        # Position schließen falls Halt-Periode erreicht
        if in_position != 0 and (i - entry_idx) >= HOLD_DAYS:
            exit_price = close[i]
            price_return = (exit_price / entry_price - 1.0) * in_position
            trade_net = (1 + price_return) * (1 - FEE_RATE) ** 2 - 1.0
            portfolio_return = POSITION_FRAC * trade_net * LEVERAGE
            if portfolio_return <= -0.99:
                portfolio_return = -0.99
            new_capital = capital * (1 + portfolio_return)
            trades.append({
                "entry_time": df.index[entry_idx],
                "exit_time": df.index[i],
                "direction": in_position,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": price_return,
                "net_return": portfolio_return,
                "capital_before": capital,
                "capital_after": new_capital,
            })
            capital = new_capital
            in_position = 0

        # Neue Position eröffnen
        if in_position == 0:
            if long_signal.iloc[i]:
                in_position = 1
                entry_idx = i
                entry_price = close[i]
            elif short_signal.iloc[i]:
                in_position = -1
                entry_idx = i
                entry_price = close[i]

    trades_df = pd.DataFrame(trades)
    n_years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (capital / STARTING_CAPITAL_CHF) ** (1 / n_years) - 1 if n_years > 0 else 0
    bh = STARTING_CAPITAL_CHF * (df["close"].iloc[-1] / df["close"].iloc[0])

    print(f"\n=== ERGEBNIS ({n_years:.1f} Jahre) ===")
    print(f"  Startkapital:    {STARTING_CAPITAL_CHF:.2f} CHF")
    print(f"  Endkapital:      {capital:.2f} CHF")
    print(f"  Total Return:    {(capital/STARTING_CAPITAL_CHF-1)*100:+.2f} %")
    print(f"  CAGR:            {cagr*100:+.2f} % p. a.")
    print(f"  B&H USD/CHF:     {bh:.2f} CHF ({(bh/STARTING_CAPITAL_CHF-1)*100:+.2f} %)")
    print(f"  Anzahl Trades:   {len(trades_df)}")

    if len(trades_df) > 0:
        wr = (trades_df["net_return"] > 0).mean()
        n_long = (trades_df["direction"] == 1).sum()
        n_short = (trades_df["direction"] == -1).sum()
        wr_long = (trades_df[trades_df["direction"] == 1]["net_return"] > 0).mean() if n_long else 0
        wr_short = (trades_df[trades_df["direction"] == -1]["net_return"] > 0).mean() if n_short else 0
        avg_w = trades_df[trades_df["net_return"] > 0]["net_return"].mean() if (trades_df["net_return"] > 0).any() else 0
        avg_l = trades_df[trades_df["net_return"] < 0]["net_return"].mean() if (trades_df["net_return"] < 0).any() else 0
        best = trades_df["net_return"].max()
        worst = trades_df["net_return"].min()

        # Equity-Curve
        equity = pd.Series([STARTING_CAPITAL_CHF] + trades_df["capital_after"].tolist(),
                           index=[df.index[0]] + trades_df["exit_time"].tolist())
        max_dd = (equity / equity.cummax() - 1).min()

        print(f"  Win-Rate:        {wr*100:.1f} %  (Long: {wr_long*100:.1f}%, Short: {wr_short*100:.1f}%)")
        print(f"  Long/Short:      {n_long} / {n_short}")
        print(f"  Avg Win/Loss:    +{avg_w*100:.3f}% / {avg_l*100:.3f}%")
        print(f"  Best/Worst:      {best*100:+.2f}% / {worst*100:+.2f}%")
        print(f"  Max Drawdown:    {max_dd*100:+.2f} %")

        # Yearly Returns
        eq_idx = pd.DatetimeIndex([df.index[0]] + trades_df["exit_time"].tolist())
        eq_full = pd.Series([STARTING_CAPITAL_CHF] + trades_df["capital_after"].tolist(), index=eq_idx)
        yearly = eq_full.resample("YE").last().pct_change().dropna()
        win_y = (yearly > 0).sum()
        print(f"\n  Jährliche Performance:")
        for y, r in yearly.items():
            mark = "✓" if r > 0 else "✗"
            print(f"    {y.year}: {r*100:>+7.2f}% {mark}")
        print(f"\n  → {win_y}/{len(yearly)} Jahre profitabel ({win_y/len(yearly)*100:.0f}%)")


if __name__ == "__main__":
    main()
