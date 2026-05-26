"""Trend-Following-Backtest mit Chandelier-Trailing-Stop."""
from collections import defaultdict

import numpy as np
import pandas as pd


def trend_following_backtest(
    df: pd.DataFrame,
    long_entry: pd.Series,
    short_entry: pd.Series,
    atr: pd.Series,
    starting_capital: float,
    position_fraction: float = 0.25,
    fee_rate: float = 0.001,
    leverage: float = 1.0,
    atr_mult: float = 3.0,
    sma_exit: pd.Series | None = None,
):
    """Trend-Following mit Chandelier-Stop.

    df braucht: open, high, low, close, volume Spalten.
    long_entry/short_entry: boolean Series (welcher Bar gibt das Signal).
    atr: Average True Range (gleicher Index wie df).
    sma_exit (optional): zusaetzliche Exit-Bedingung bei Crossover unter SMA.

    Trailing-Stop:
      Long:  stop = max(stop_so_far, close - atr_mult * ATR)
             Exit wenn close <= stop
      Short: stop = min(stop_so_far, close + atr_mult * ATR)
             Exit wenn close >= stop
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    timestamps = df.index
    atr_arr = atr.values
    long_entry_arr = long_entry.fillna(False).values
    short_entry_arr = short_entry.fillna(False).values
    sma_arr = sma_exit.values if sma_exit is not None else None

    n = len(close)
    capital = starting_capital
    equity = np.full(n, capital, dtype=np.float64)
    trades = []
    day_log = []
    current_date = None
    day_start_cap = capital

    in_position = 0           # 0=cash, +1=long, -1=short
    entry_idx = None
    entry_price = 0.0
    trail_stop = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0

    for i in range(n):
        ts = timestamps[i]
        date_i = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
        if date_i != current_date:
            if current_date is not None:
                day_log.append({
                    "date": current_date,
                    "start_capital": day_start_cap,
                    "end_capital": capital,
                    "day_return": capital / day_start_cap - 1.0,
                })
            current_date = date_i
            day_start_cap = capital

        equity[i] = capital

        if in_position == 0:
            # Suche neuen Einstieg
            if long_entry_arr[i] and not np.isnan(atr_arr[i]):
                in_position = 1
                entry_idx = i
                entry_price = close[i]
                highest_since_entry = close[i]
                trail_stop = close[i] - atr_mult * atr_arr[i]
            elif short_entry_arr[i] and not np.isnan(atr_arr[i]):
                in_position = -1
                entry_idx = i
                entry_price = close[i]
                lowest_since_entry = close[i]
                trail_stop = close[i] + atr_mult * atr_arr[i]
        else:
            # Trailing-Stop aktualisieren und Exit pruefen
            if in_position == 1:
                if close[i] > highest_since_entry:
                    highest_since_entry = close[i]
                    new_stop = highest_since_entry - atr_mult * atr_arr[i]
                    if new_stop > trail_stop:
                        trail_stop = new_stop
                exit_now = (low[i] <= trail_stop)
                if sma_arr is not None and not np.isnan(sma_arr[i]) and close[i] < sma_arr[i]:
                    exit_now = True
            else:  # short
                if close[i] < lowest_since_entry:
                    lowest_since_entry = close[i]
                    new_stop = lowest_since_entry + atr_mult * atr_arr[i]
                    if new_stop < trail_stop:
                        trail_stop = new_stop
                exit_now = (high[i] >= trail_stop)
                if sma_arr is not None and not np.isnan(sma_arr[i]) and close[i] > sma_arr[i]:
                    exit_now = True

            if exit_now or i == n - 1:
                exit_price = trail_stop if (i < n - 1) else close[i]
                # Bei Endbar: schliesse zum close
                if i == n - 1 and not exit_now:
                    exit_price = close[i]
                price_return = (exit_price / entry_price - 1.0) * in_position
                trade_net = (1 + price_return) * (1 - fee_rate) * (1 - fee_rate) - 1.0
                portfolio_return = position_fraction * trade_net * leverage
                if portfolio_return <= -0.99:
                    portfolio_return = -0.99
                    reason = "LIQUIDATED"
                else:
                    reason = "STOP"
                new_cap = capital * (1 + portfolio_return)
                trades.append({
                    "entry_time": timestamps[entry_idx],
                    "exit_time":  timestamps[i],
                    "hold_bars":  i - entry_idx,
                    "direction":  in_position,
                    "entry_price": entry_price,
                    "exit_price":  exit_price,
                    "gross_return": price_return,
                    "trade_net_return": trade_net,
                    "net_return": portfolio_return,
                    "capital_before": capital,
                    "capital_after":  new_cap,
                    "pnl_chf": new_cap - capital,
                    "reason": reason,
                })
                capital = new_cap
                in_position = 0

    equity[i:] = capital
    if current_date is not None:
        day_log.append({
            "date": current_date,
            "start_capital": day_start_cap,
            "end_capital": capital,
            "day_return": capital / day_start_cap - 1.0,
        })

    return {
        "trades": pd.DataFrame(trades),
        "equity": pd.Series(equity, index=df.index),
        "final_capital": capital,
        "total_return": capital / starting_capital - 1.0,
        "day_log": pd.DataFrame(day_log),
    }


def multi_asset_trend_backtest(
    coins_data: dict,
    starting_capital: float,
    fee_rate: float = 0.001,
    leverage: float = 1.0,
    atr_mult: float = 5.0,
    max_position_per_coin: float = 0.20,
    max_total_exposure: float = 0.80,
    daily_loss_limit: float = -0.005,
    allow_shorts: bool = False,
):
    """Multi-Asset Trend-Following (Long und Short).

    coins_data: dict symbol -> DataFrame mit Spalten:
      close, high, low, atr,
      signal (bool: Long-Entry-Signal),
      short_signal (bool: optional, nur wenn allow_shorts=True)
    """
    all_ts = sorted(set().union(*[set(df.index) for df in coins_data.values()]))
    if not all_ts:
        return {"trades": pd.DataFrame(), "equity": pd.Series(dtype=float),
                "final_capital": starting_capital, "total_return": 0.0,
                "day_log": pd.DataFrame()}

    capital = starting_capital
    open_pos = {}    # symbol -> dict
    trades = []
    equity_records = []
    day_log = []
    current_date = None
    day_start_capital = capital

    for ts in all_ts:
        date_i = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
        if date_i != current_date:
            if current_date is not None:
                day_log.append({
                    "date": current_date,
                    "start_capital": day_start_capital,
                    "end_capital": capital,
                    "day_return": capital / day_start_capital - 1.0,
                })
            current_date = date_i
            day_start_capital = capital

        day_return = capital / day_start_capital - 1.0
        daily_paused = day_return <= daily_loss_limit

        # 1) Update offene Positionen & Exit-Checks
        for symbol in list(open_pos.keys()):
            df = coins_data[symbol]
            if ts not in df.index:
                continue
            row = df.loc[ts]
            if pd.isna(row.get("close")) or pd.isna(row.get("atr")):
                continue
            pos = open_pos[symbol]
            direction = pos.get("direction", 1)

            if direction == 1:
                if row["close"] > pos["highest"]:
                    pos["highest"] = row["close"]
                new_stop = pos["highest"] - atr_mult * row["atr"]
                if new_stop > pos["trail_stop"]:
                    pos["trail_stop"] = new_stop
                exit_now = row["low"] <= pos["trail_stop"]
            else:  # short
                if row["close"] < pos["lowest"]:
                    pos["lowest"] = row["close"]
                new_stop = pos["lowest"] + atr_mult * row["atr"]
                if new_stop < pos["trail_stop"]:
                    pos["trail_stop"] = new_stop
                exit_now = row["high"] >= pos["trail_stop"]

            if exit_now or ts == all_ts[-1]:
                exit_price = pos["trail_stop"] if exit_now else row["close"]
                price_return = (exit_price / pos["entry_price"] - 1.0) * direction
                trade_net = (1 + price_return) * (1 - fee_rate) ** 2 - 1.0
                portfolio_return_contribution = pos["position_frac"] * trade_net * leverage
                if portfolio_return_contribution <= -0.99:
                    portfolio_return_contribution = -0.99
                    reason = "LIQUIDATED"
                else:
                    reason = "STOP" if exit_now else "END"
                pnl_chf = capital * portfolio_return_contribution
                capital = capital + pnl_chf
                trades.append({
                    "symbol": symbol,
                    "direction": direction,
                    "entry_time": pos["entry_ts"], "exit_time": ts,
                    "hold_bars": (ts - pos["entry_ts"]).days,
                    "entry_price": pos["entry_price"], "exit_price": exit_price,
                    "position_frac": pos["position_frac"],
                    "gross_return": price_return,
                    "net_return": portfolio_return_contribution,
                    "pnl_chf": pnl_chf, "reason": reason,
                })
                del open_pos[symbol]

        # 2) Neue Entries — nur wenn nicht pausiert
        if not daily_paused:
            current_exposure = sum(p["position_frac"] for p in open_pos.values())
            for symbol, df in coins_data.items():
                if symbol in open_pos or ts not in df.index:
                    continue
                row = df.loc[ts]
                if pd.isna(row.get("atr")):
                    continue
                long_sig = bool(row.get("signal", False))
                short_sig = allow_shorts and bool(row.get("short_signal", False))
                if not (long_sig or short_sig):
                    continue
                available = max_total_exposure - current_exposure
                if available < 0.01:
                    break
                pos_frac = min(max_position_per_coin, available)
                if long_sig:
                    open_pos[symbol] = {
                        "entry_ts": ts, "direction": 1,
                        "entry_price": row["close"],
                        "highest": row["close"],
                        "trail_stop": row["close"] - atr_mult * row["atr"],
                        "position_frac": pos_frac,
                    }
                else:  # short
                    open_pos[symbol] = {
                        "entry_ts": ts, "direction": -1,
                        "entry_price": row["close"],
                        "lowest": row["close"],
                        "trail_stop": row["close"] + atr_mult * row["atr"],
                        "position_frac": pos_frac,
                    }
                current_exposure += pos_frac

        equity_records.append((ts, capital))

    eq_index, eq_values = zip(*equity_records)
    equity = pd.Series(eq_values, index=pd.Index(eq_index))

    if current_date is not None:
        day_log.append({
            "date": current_date, "start_capital": day_start_capital,
            "end_capital": capital, "day_return": capital / day_start_capital - 1.0,
        })

    return {
        "trades": pd.DataFrame(trades), "equity": equity,
        "final_capital": capital,
        "total_return": capital / starting_capital - 1.0,
        "day_log": pd.DataFrame(day_log),
    }
