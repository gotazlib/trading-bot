import numpy as np
import pandas as pd
from src.indicators.technical import (
    sma, ema, rsi, macd, bollinger_pct,
    atr, stochastic, roc, cci, williams_r, obv, adx,
    keltner_pct, aroon, chaikin_money_flow, trend_strength,
)


def build_features(df):
    """Baut aus OHLCV eine Tabelle technischer Indikatoren (die 'Features')."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]

    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_6"] = close.pct_change(6)
    out["volatility"] = close.pct_change().rolling(24).std()
    out["rsi_14"] = rsi(close, 14)
    out["macd_hist"] = macd(close)[2]
    out["bb_pct"] = bollinger_pct(close, 20)
    out["sma_ratio"] = sma(close, 10) / sma(close, 50) - 1
    out["ema_ratio"] = ema(close, 12) / ema(close, 26) - 1
    out["vol_change"] = df["volume"].pct_change().rolling(6).mean()
    return out


def build_label(df, horizon=1):
    """Einfaches Vorzeichen-Label: 1 = Kurs in 'horizon' Kerzen hoeher, sonst 0."""
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    return (future_return > 0).astype(int)


def build_extended_features(df):
    """Erweiterte Indikator-Bibliothek: 17 Features ohne Paar-Produkte.

    HistGradientBoosting lernt Interaktionen automatisch, daher keine
    expliziten Paar-Features mehr — Modell ist trotzdem maechtig genug.
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Momentum / Returns
    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_6"] = close.pct_change(6)
    out["roc_10"] = roc(close, 10)
    out["roc_30"] = roc(close, 30)

    # Volatilitaet
    out["volatility"] = close.pct_change().rolling(24).std()
    out["atr_14"] = atr(high, low, close, 14) / close

    # Oszillatoren
    out["rsi_14"] = rsi(close, 14)
    stoch_k, stoch_d = stochastic(high, low, close, 14, 3)
    out["stoch_k"] = stoch_k
    out["stoch_d_diff"] = stoch_k - stoch_d
    out["williams_r_14"] = williams_r(high, low, close, 14)
    out["cci_20"] = cci(high, low, close, 20)

    # Trend
    out["macd_hist"] = macd(close)[2]
    out["bb_pct"] = bollinger_pct(close, 20)
    out["sma_ratio"] = sma(close, 10) / sma(close, 50) - 1
    out["ema_ratio"] = ema(close, 12) / ema(close, 26) - 1
    out["adx_14"] = adx(high, low, close, 14)

    # Volumen
    out["vol_change"] = volume.pct_change().rolling(6).mean()
    obv_series = obv(close, volume)
    out["obv_slope"] = obv_series.pct_change(10)
    out["cmf_20"] = chaikin_money_flow(high, low, close, volume, 20)

    # Channels
    out["keltner_pct"] = keltner_pct(high, low, close, 20)

    # Aroon
    a_up, a_dn, a_osc = aroon(high, low, 25)
    out["aroon_up"] = a_up
    out["aroon_down"] = a_dn
    out["aroon_osc"] = a_osc

    # Trend-Strength
    out["trend_strength"] = trend_strength(close, 10, 50)
    out["trend_strength_long"] = trend_strength(close, 20, 200)

    # Highs/Lows distance (Donchian-light)
    out["dist_to_20d_high"] = (high.rolling(20).max() - close) / close
    out["dist_to_20d_low"] = (close - low.rolling(20).min()) / close

    return out


def build_funding_features(df_ohlcv: pd.DataFrame, df_funding: pd.DataFrame, bar_minutes: int = 5):
    """Funding-Rate-Features fuer jeden OHLCV-Bar.

    Funding kommt alle 8h — wir forward-fillen auf den OHLCV-Index.
    Features:
      - funding_now: aktuelle Funding-Rate (zuletzt bekannter Wert)
      - funding_3p: Summe ueber 3 Perioden (24h Funding) — Sentiment-Last
      - funding_z: 30-Tage Z-Score — wie extrem ist die aktuelle Rate?
      - funding_momentum: Aenderung zur Vorperiode
      - funding_sign: +1/-1/0 (Long-Squeeze-Druck vs Short-Squeeze-Druck)
      - funding_sign_flip_recent: Anzahl Vorzeichenwechsel in 24h
    """
    if df_funding.empty:
        return pd.DataFrame(index=df_ohlcv.index)

    funding = df_funding["rate"].reindex(
        df_ohlcv.index.union(df_funding.index)
    ).sort_index().ffill().reindex(df_ohlcv.index)

    out = pd.DataFrame(index=df_ohlcv.index)
    out["funding_now"] = funding
    out["funding_3p"] = funding.rolling(3, min_periods=1).sum()
    n_bars_30d = max(96, 30 * 24 * 60 // bar_minutes)
    n_bars_24h = max(8, 24 * 60 // bar_minutes)
    rolling_mean = funding.rolling(n_bars_30d, min_periods=10).mean()
    rolling_std = funding.rolling(n_bars_30d, min_periods=10).std()
    out["funding_z"] = (funding - rolling_mean) / rolling_std.replace(0, np.nan)
    out["funding_momentum"] = funding.diff()
    out["funding_sign"] = np.sign(funding)
    sign_flip = (out["funding_sign"].diff().abs() > 1).astype(int)
    out["funding_sign_flip_recent"] = sign_flip.rolling(n_bars_24h, min_periods=1).sum()
    return out


def build_features_with_time(df):
    """Erweiterte Features + Zeit-Encoding fuer KNN-Pattern-Matching.

    Zeit als sin/cos: damit ist 23:00 'nahe' an 01:00 (zyklisch), und
    Sonntag-Abend nahe an Montag-Morgen. Wichtig fuer Distanzmessung.
    """
    out = build_extended_features(df)
    idx = df.index
    hour = idx.hour + idx.minute / 60.0
    weekday = idx.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    return out


def build_features_with_interactions(df):
    """Wie build_features, plus alle paarweisen Produkte der Basis-Features.

    Damit kann ein Modell direkt 'lernen', welche Indikator-Kombinationen
    am informativsten sind: jedes Paar bekommt ein eigenes Feature, und
    Permutation Importance kann jedem Paar einen Beitrag zuordnen.
    """
    base = build_features(df)
    out = base.copy()
    cols = list(base.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            out[f"{a} X {b}"] = base[a] * base[b]
    return out


def build_label_vol_scaled(df, horizon=6, vol_window=24, k=0.5):
    """Vol-skaliertes Label (vereinfachte Triple-Barrier-Variante).

    1  = zukuenftige Rendite ueber +k * rolling_vol   (klares Aufwaertssignal)
    -1 = zukuenftige Rendite unter -k * rolling_vol  (klares Abwaertssignal)
    0  = im Rauschen drumherum

    Macht das Lernziel anspruchsvoller: das Modell soll nicht beliebige
    leichte Aufwaertsbewegungen vorhersagen, sondern signifikante.
    """
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    vol = df["close"].pct_change().rolling(vol_window).std()
    threshold = k * vol
    label = pd.Series(0, index=df.index, dtype=int)
    label[future_return > threshold] = 1
    label[future_return < -threshold] = -1
    # Letzte 'horizon' Eintraege haben keinen sinnvollen zukuenftigen Wert
    label.iloc[-horizon:] = np.nan
    return label


def build_label_triple_barrier(df, tp_pct=0.005, sl_pct=0.003, max_hold=60):
    """Triple-Barrier-Label nach Lopez de Prado.

    Fuer jede Kerze i wird in die naechsten max_hold Kerzen geschaut:
      label = +1  wenn close zuerst entry*(1+tp_pct) erreicht  (Take-Profit zuerst)
      label = -1  wenn close zuerst entry*(1-sl_pct) erreicht  (Stop-Loss zuerst)
      label =  0  wenn keiner der beiden bis max_hold getroffen wurde (Zeit-Stop)

    Das ist GENAU das, was der Backtest spaeter umsetzt: das Modell lernt
    direkt 'gewinnt der Trade in der naechsten Stunde oder geht er kaputt?'.
    """
    close = df["close"].values
    n = len(close)
    labels = np.zeros(n, dtype=np.float64)
    labels[:] = np.nan

    for i in range(n - max_hold):
        entry = close[i]
        tp = entry * (1.0 + tp_pct)
        sl = entry * (1.0 - sl_pct)
        path = close[i + 1 : i + max_hold + 1]
        hit_tp_mask = path >= tp
        hit_sl_mask = path <= sl
        first_tp = hit_tp_mask.argmax() if hit_tp_mask.any() else -1
        first_sl = hit_sl_mask.argmax() if hit_sl_mask.any() else -1
        if first_tp >= 0 and (first_sl < 0 or first_tp < first_sl):
            labels[i] = 1.0
        elif first_sl >= 0:
            labels[i] = -1.0
        else:
            labels[i] = 0.0

    return pd.Series(labels, index=df.index)


def build_label_triple_barrier_short(df, tp_pct=0.015, sl_pct=0.004, max_hold=24):
    """Triple-Barrier-Label fuer Short-Position.

    +1 = Preis faellt zuerst um tp_pct  (Short-Gewinn)
    -1 = Preis steigt zuerst um sl_pct  (Short-Verlust = Long-Gewinn)
     0 = Time-Stop

    Achtung: Long- und Short-Labels sind NICHT komplementaer, weil die
    Schwellen asymmetrisch sind (TP weiter als SL). Beides muss separat
    trainiert werden.
    """
    close = df["close"].values
    n = len(close)
    labels = np.zeros(n, dtype=np.float64)
    labels[:] = np.nan

    for i in range(n - max_hold):
        entry = close[i]
        tp_short = entry * (1.0 - tp_pct)   # Preis MUSS fallen um TP
        sl_short = entry * (1.0 + sl_pct)   # Preis darf nicht so weit steigen
        path = close[i + 1 : i + max_hold + 1]
        hit_tp = path <= tp_short
        hit_sl = path >= sl_short
        first_tp = hit_tp.argmax() if hit_tp.any() else -1
        first_sl = hit_sl.argmax() if hit_sl.any() else -1
        if first_tp >= 0 and (first_sl < 0 or first_tp < first_sl):
            labels[i] = 1.0
        elif first_sl >= 0:
            labels[i] = -1.0
        else:
            labels[i] = 0.0

    return pd.Series(labels, index=df.index)
