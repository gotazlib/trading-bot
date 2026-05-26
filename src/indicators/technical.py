def sma(series, window):
    return series.rolling(window=window).mean()

def ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line

def bollinger_pct(series, window=20, n_std=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper, lower = ma + n_std * std, ma - n_std * std
    return (series - lower) / (upper - lower)


def atr(high, low, close, period=14):
    """Average True Range — Volatilitaet auf Basis von High/Low/Close."""
    import pandas as pd
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator (K und D Linien)."""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, 1e-12)
    d = k.rolling(d_period).mean()
    return k, d


def roc(series, period=10):
    """Rate of Change."""
    return (series / series.shift(period) - 1.0) * 100


def cci(high, low, close, period=20):
    """Commodity Channel Index."""
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mean_dev = tp.rolling(period).apply(lambda x: (x - x.mean()).abs().mean(), raw=False)
    return (tp - sma_tp) / (0.015 * mean_dev.replace(0, 1e-12))


def williams_r(high, low, close, period=14):
    """Williams %R."""
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, 1e-12)


def obv(close, volume):
    """On-Balance Volume."""
    import numpy as np
    sign = np.sign(close.diff()).fillna(0)
    return (sign * volume).cumsum()


def keltner_pct(high, low, close, period=20, mult=2.0):
    """Position innerhalb Keltner-Channel (0=untere Bande, 1=obere Bande)."""
    typical = (high + low + close) / 3
    ma = typical.rolling(period).mean()
    atr_local = atr(high, low, close, period)
    upper = ma + mult * atr_local
    lower = ma - mult * atr_local
    return (close - lower) / (upper - lower).replace(0, 1e-12)


def aroon(high, low, period=25):
    """Aroon Oszillator: misst wie lange seit letztem Hoch/Tief.
    Liefert (aroon_up, aroon_down, aroon_osc)."""
    import pandas as pd
    aroon_up = high.rolling(period + 1).apply(
        lambda x: ((period - x.argmax()) / period) * 100, raw=False
    )
    aroon_down = low.rolling(period + 1).apply(
        lambda x: ((period - x.argmin()) / period) * 100, raw=False
    )
    return aroon_up, aroon_down, aroon_up - aroon_down


def chaikin_money_flow(high, low, close, volume, period=20):
    """Chaikin Money Flow: Volumen-gewichteter Kaufdruck."""
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, 1e-12)
    mfv = mfm * volume
    return mfv.rolling(period).sum() / volume.rolling(period).sum().replace(0, 1e-12)


def trend_strength(close, short=10, long=50):
    """Trend-Staerke als (close/long_ma - 1) skaliert mit Vola."""
    long_ma = sma(close, long)
    ret_vola = close.pct_change().rolling(20).std()
    return ((close / long_ma) - 1) / ret_vola.replace(0, 1e-12)


def adx(high, low, close, period=14):
    """Average Directional Index — Trendstaerke (0-100)."""
    import pandas as pd
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_local = tr.rolling(period).mean().replace(0, 1e-12)
    plus_di = 100 * plus_dm.rolling(period).mean() / atr_local
    minus_di = 100 * minus_dm.rolling(period).mean() / atr_local
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-12)
    return dx.rolling(period).mean()
