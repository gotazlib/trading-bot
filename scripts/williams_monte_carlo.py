"""Monte-Carlo Permutation Test fuer USD/CHF Williams %R Strategie.

Testet, ob der Edge der Mean-Reversion-Strategie statistisch signifikant ist
oder reines Glueck. Vorgehen:
  1. Reale Strategie auf echten Daten -> CAGR_real
  2. 1000x: Permutiere tägliche Returns, rekonstruiere Close-Serie,
     berechne synthetische High/Low/Close, run dieselbe Strategie -> CAGR_i
  3. p-Wert = Anteil der Permutationen mit CAGR_i >= CAGR_real
"""
import warnings
import numpy as np
import pandas as pd

from src.data.storage import load_ohlcv
from src.indicators.technical import williams_r

warnings.filterwarnings("ignore", category=UserWarning)

STARTING_CAPITAL_CHF = 1000.0
FEE_RATE = 0.0002
POSITION_FRAC = 0.30
LEVERAGE = 5.0

WR_OVERSOLD = -90
WR_OVERBOUGHT = -10
HOLD_DAYS = 1
WR_PERIOD = 14

N_PERMUTATIONS = 1000
SEED = 42


def run_strategy(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 days: float) -> float:
    """Williams-Strategie -> CAGR. Erwartet plain numpy arrays."""
    h = pd.Series(high)
    l = pd.Series(low)
    c = pd.Series(close)
    wr = williams_r(h, l, c, WR_PERIOD)

    valid = wr.notna()
    wr = wr[valid].reset_index(drop=True)
    close_arr = c[valid].reset_index(drop=True).values
    n = len(close_arr)
    if n < 5:
        return 0.0

    wr_prev = wr.shift(1)
    long_signal = ((wr_prev >= WR_OVERSOLD) & (wr < WR_OVERSOLD)).values
    short_signal = ((wr_prev <= WR_OVERBOUGHT) & (wr > WR_OVERBOUGHT)).values

    capital = STARTING_CAPITAL_CHF
    in_position = 0
    entry_idx = None
    entry_price = 0.0

    for i in range(n - 1):
        if in_position != 0 and (i - entry_idx) >= HOLD_DAYS:
            exit_price = close_arr[i]
            price_return = (exit_price / entry_price - 1.0) * in_position
            trade_net = (1 + price_return) * (1 - FEE_RATE) ** 2 - 1.0
            portfolio_return = POSITION_FRAC * trade_net * LEVERAGE
            if portfolio_return <= -0.99:
                portfolio_return = -0.99
            capital *= (1 + portfolio_return)
            in_position = 0

        if in_position == 0:
            if long_signal[i]:
                in_position = 1
                entry_idx = i
                entry_price = close_arr[i]
            elif short_signal[i]:
                in_position = -1
                entry_idx = i
                entry_price = close_arr[i]

    n_years = days / 365.25
    if n_years <= 0 or capital <= 0:
        return 0.0
    return (capital / STARTING_CAPITAL_CHF) ** (1 / n_years) - 1


def permute_ohlc(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                 rng: np.random.Generator):
    """Permutiere log-Returns und rekonstruiere High/Low/Close.

    Echte High/Low/Close-Spreads des jeweiligen Tages werden mit-permutiert,
    so dass die Tagesrange-Verteilung erhalten bleibt.
    """
    log_ret = np.diff(np.log(close))
    # Pro-Tag-Spreads relativ zum Close (additiv in log space)
    high_spread = np.log(high[1:] / close[1:])   # >= 0
    low_spread = np.log(low[1:] / close[1:])     # <= 0

    idx = rng.permutation(len(log_ret))
    log_ret_p = log_ret[idx]
    high_spread_p = high_spread[idx]
    low_spread_p = low_spread[idx]

    close_p = np.empty_like(close)
    close_p[0] = close[0]
    close_p[1:] = close[0] * np.exp(np.cumsum(log_ret_p))

    high_p = np.empty_like(high)
    low_p = np.empty_like(low)
    high_p[0] = high[0]
    low_p[0] = low[0]
    high_p[1:] = close_p[1:] * np.exp(high_spread_p)
    low_p[1:] = close_p[1:] * np.exp(low_spread_p)

    return high_p, low_p, close_p


def main():
    df = load_ohlcv("forex", "USD/CHF", "1d").dropna()
    print(f"USD/CHF Daily: {df.index.min().date()} -> {df.index.max().date()} "
          f"({len(df)} Bars)")

    days = (df.index[-1] - df.index[0]).days
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)

    # 1) Echtes CAGR
    cagr_real = run_strategy(high, low, close, days)
    print(f"\nEchtes CAGR: {cagr_real * 100:+.3f} % p.a.")

    # 2) Permutation Test
    print(f"\nLaufe {N_PERMUTATIONS} Monte-Carlo-Permutationen ...")
    rng = np.random.default_rng(SEED)
    cagrs = np.empty(N_PERMUTATIONS)
    for k in range(N_PERMUTATIONS):
        h_p, l_p, c_p = permute_ohlc(close, high, low, rng)
        cagrs[k] = run_strategy(h_p, l_p, c_p, days)
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{N_PERMUTATIONS} done "
                  f"(running p={(cagrs[:k+1] >= cagr_real).mean():.3f})")

    # 3) Statistik
    p_value = float((cagrs >= cagr_real).mean())
    better = int((cagrs >= cagr_real).sum())

    print(f"\n=== PERMUTATIONS-VERTEILUNG (N={N_PERMUTATIONS}) ===")
    print(f"  Min:    {cagrs.min() * 100:+.3f} %")
    print(f"  P05:    {np.percentile(cagrs, 5) * 100:+.3f} %")
    print(f"  Median: {np.median(cagrs) * 100:+.3f} %")
    print(f"  Mean:   {cagrs.mean() * 100:+.3f} %")
    print(f"  P95:    {np.percentile(cagrs, 95) * 100:+.3f} %")
    print(f"  Max:    {cagrs.max() * 100:+.3f} %")
    print(f"  Std:    {cagrs.std(ddof=1) * 100:.3f} pp")

    print(f"\n=== SIGNIFIKANZ ===")
    print(f"  Echtes CAGR:       {cagr_real * 100:+.3f} %")
    print(f"  Permutationen >=:  {better} / {N_PERMUTATIONS}")
    print(f"  p-Wert:            {p_value:.4f}")

    if p_value < 0.01:
        verdict = "SEHR SIGNIFIKANT (p<0.01) -> Edge ist real"
    elif p_value < 0.05:
        verdict = "SIGNIFIKANT (p<0.05) -> Edge wahrscheinlich real"
    elif p_value < 0.10:
        verdict = "SCHWACH (p<0.10) -> Edge moeglich, vorsichtig sein"
    else:
        verdict = "NICHT SIGNIFIKANT -> Performance vermutlich Glueck"
    print(f"  Verdikt:           {verdict}")


if __name__ == "__main__":
    main()
