import numpy as np
import pandas as pd

BARS_PER_YEAR = 24 * 365  # 1h-Kerzen


def evaluate(positions: pd.Series, market_returns: pd.Series, fee_rate: float):
    """Bewertet eine Strategie aus Positionsreihe (-1/0/1) und Marktrenditen.

    Position wird auf der NAECHSTEN Kerze umgesetzt (keine Look-ahead-Verzerrung).
    Gebuehren werden pro Positionswechsel abgezogen.
    """
    position = positions.shift(1).fillna(0)
    strat_ret = position * market_returns
    trades = position.diff().abs().fillna(0)
    strat_ret = strat_ret - trades * fee_rate

    equity = (1 + strat_ret).cumprod()
    drawdown = (equity / equity.cummax() - 1).min()
    std = strat_ret.std()
    sharpe = (strat_ret.mean() / std * np.sqrt(BARS_PER_YEAR)) if std > 0 else 0.0
    exposure = (position != 0).mean()

    return {
        "total_return": equity.iloc[-1] - 1,
        "max_drawdown": drawdown,
        "sharpe": sharpe,
        "num_trades": int(trades.sum()),
        "exposure": exposure,
        "equity": equity,
        "strategy_returns": strat_ret,
    }


def triple_barrier_backtest(
    close: pd.Series,
    p_tp: pd.Series,
    entry_threshold: float,
    tp_pct: float,
    sl_pct: float,
    max_hold: int,
    fee_rate: float,
    starting_capital: float,
    cooldown: int = 0,
    position_fraction=None,
    daily_profit_target: float | None = None,
    daily_loss_limit: float | None = None,
):
    """Realistischer Triple-Barrier-Backtest mit Equity-Tracking in absoluter Waehrung.

    - Position wird auf der NAECHSTEN Kerze nach Signal eroeffnet (kein Look-ahead)
    - Pro Trade gesamtes Kapital eingesetzt
    - Trade endet, sobald TP, SL oder max_hold zuerst eintritt
    - Gebuehr fae llt zweimal an (einmal Entry, einmal Exit)
    - cooldown: nach jedem Trade so viele Kerzen aussetzen
    """
    close_arr = close.values
    p_arr = p_tp.values
    timestamps = close.index
    n = len(close_arr)

    if position_fraction is None:
        pos_frac_arr = np.ones(n, dtype=np.float64)
    elif np.isscalar(position_fraction):
        pos_frac_arr = np.full(n, float(position_fraction), dtype=np.float64)
    else:
        pos_frac_arr = np.asarray(position_fraction, dtype=np.float64)

    capital = starting_capital
    equity_curve = np.full(n, capital, dtype=np.float64)
    trades = []
    i = 0

    current_date = None
    day_start_capital = capital
    day_status_log = []

    while i < n - 1:
        ts = timestamps[i]
        date_i = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
        if date_i != current_date:
            if current_date is not None:
                day_status_log.append({
                    "date": current_date,
                    "start_capital": day_start_capital,
                    "end_capital": capital,
                    "day_return": capital / day_start_capital - 1.0,
                })
            current_date = date_i
            day_start_capital = capital

        day_return = capital / day_start_capital - 1.0
        daily_paused = (
            (daily_profit_target is not None and day_return >= daily_profit_target)
            or (daily_loss_limit is not None and day_return <= daily_loss_limit)
        )

        if daily_paused or p_arr[i] <= entry_threshold:
            equity_curve[i] = capital
            i += 1
            continue

        entry_idx = i + 1
        if entry_idx >= n:
            break
        entry_price = close_arr[entry_idx]
        tp_price = entry_price * (1.0 + tp_pct)
        sl_price = entry_price * (1.0 - sl_pct)

        exit_idx = min(entry_idx + max_hold, n - 1)
        exit_reason = "TIME"
        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, n)):
            if close_arr[j] >= tp_price:
                exit_idx = j
                exit_reason = "TP"
                break
            if close_arr[j] <= sl_price:
                exit_idx = j
                exit_reason = "SL"
                break

        gross_return = close_arr[exit_idx] / entry_price - 1.0
        trade_net_return = (1 + gross_return) * (1 - fee_rate) * (1 - fee_rate) - 1.0
        pos_frac = max(0.0, min(1.0, pos_frac_arr[i]))
        # Nur ein Teil des Kapitals war im Markt; der Rest blieb Cash (0 % Zins)
        portfolio_return = pos_frac * trade_net_return
        new_capital = capital * (1 + portfolio_return)

        trades.append({
            "entry_time": timestamps[entry_idx],
            "exit_time":  timestamps[exit_idx],
            "hold_bars":  exit_idx - entry_idx,
            "entry_price": entry_price,
            "exit_price":  close_arr[exit_idx],
            "p_tp": p_arr[i],
            "pos_frac": pos_frac,
            "reason": exit_reason,
            "gross_return":      gross_return,
            "trade_net_return":  trade_net_return,
            "net_return":        portfolio_return,
            "capital_before": capital,
            "capital_after":  new_capital,
            "pnl_chf": new_capital - capital,
        })

        # Fuelle equity_curve vom Signal-Index bis zum Exit linear mit altem Kapital,
        # am Exit-Index dann das neue Kapital
        equity_curve[i:exit_idx] = capital
        equity_curve[exit_idx] = new_capital
        capital = new_capital
        i = exit_idx + 1 + cooldown

    # Reste auffuellen
    equity_curve[i:] = capital
    if current_date is not None:
        day_status_log.append({
            "date": current_date,
            "start_capital": day_start_capital,
            "end_capital": capital,
            "day_return": capital / day_start_capital - 1.0,
        })

    return {
        "trades": pd.DataFrame(trades),
        "equity": pd.Series(equity_curve, index=close.index),
        "final_capital": capital,
        "total_return": capital / starting_capital - 1.0,
        "day_log": pd.DataFrame(day_status_log),
    }


def triple_barrier_backtest_ls(
    close: pd.Series,
    p_tp_long: pd.Series,
    p_tp_short: pd.Series,
    entry_threshold: float,
    tp_pct: float,
    sl_pct: float,
    max_hold: int,
    fee_rate: float,
    starting_capital: float,
    cooldown: int = 0,
    position_fraction=None,
    daily_profit_target: float | None = None,
    daily_loss_limit: float | None = None,
    leverage: float = 1.0,
):
    """Long/Short Triple-Barrier-Backtest.

    Bei jedem Bar:
      direction = +1 wenn p_tp_long > p_tp_short und > entry_threshold
      direction = -1 wenn p_tp_short > p_tp_long und > entry_threshold
      direction =  0 sonst (cash)
    """
    close_arr = close.values
    p_long = p_tp_long.values
    p_short = p_tp_short.values
    timestamps = close.index
    n = len(close_arr)

    if position_fraction is None:
        pos_frac_arr = np.ones(n, dtype=np.float64)
    elif np.isscalar(position_fraction):
        pos_frac_arr = np.full(n, float(position_fraction), dtype=np.float64)
    else:
        pos_frac_arr = np.asarray(position_fraction, dtype=np.float64)

    capital = starting_capital
    equity_curve = np.full(n, capital, dtype=np.float64)
    trades = []
    day_status_log = []
    current_date = None
    day_start_capital = capital
    i = 0

    while i < n - 1:
        ts = timestamps[i]
        date_i = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
        if date_i != current_date:
            if current_date is not None:
                day_status_log.append({
                    "date": current_date,
                    "start_capital": day_start_capital,
                    "end_capital": capital,
                    "day_return": capital / day_start_capital - 1.0,
                })
            current_date = date_i
            day_start_capital = capital

        day_return = capital / day_start_capital - 1.0
        daily_paused = (
            (daily_profit_target is not None and day_return >= daily_profit_target)
            or (daily_loss_limit is not None and day_return <= daily_loss_limit)
        )

        # Richtungs-Entscheidung
        pl, ps = p_long[i], p_short[i]
        if pl > entry_threshold and pl >= ps:
            direction = 1
            p_signal = pl
        elif ps > entry_threshold and ps > pl:
            direction = -1
            p_signal = ps
        else:
            direction = 0
            p_signal = 0.0

        if daily_paused or direction == 0:
            equity_curve[i] = capital
            i += 1
            continue

        entry_idx = i + 1
        if entry_idx >= n:
            break
        entry_price = close_arr[entry_idx]

        if direction == 1:
            tp_price = entry_price * (1.0 + tp_pct)
            sl_price = entry_price * (1.0 - sl_pct)
        else:
            tp_price = entry_price * (1.0 - tp_pct)
            sl_price = entry_price * (1.0 + sl_pct)

        exit_idx = min(entry_idx + max_hold, n - 1)
        exit_reason = "TIME"
        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, n)):
            if direction == 1:
                if close_arr[j] >= tp_price:
                    exit_idx, exit_reason = j, "TP"
                    break
                if close_arr[j] <= sl_price:
                    exit_idx, exit_reason = j, "SL"
                    break
            else:  # short
                if close_arr[j] <= tp_price:
                    exit_idx, exit_reason = j, "TP"
                    break
                if close_arr[j] >= sl_price:
                    exit_idx, exit_reason = j, "SL"
                    break

        price_return = close_arr[exit_idx] / entry_price - 1.0
        gross_return = direction * price_return  # Short profitiert von negativem Preis-Return
        trade_net_return = (1 + gross_return) * (1 - fee_rate) * (1 - fee_rate) - 1.0
        pos_frac = max(0.0, min(1.0, pos_frac_arr[i]))
        # Hebel: skaliert sowohl Gewinn als auch Verlust
        portfolio_return = pos_frac * trade_net_return * leverage
        # Liquidation: bei -100 % Verlust ist Kapital weg
        if portfolio_return <= -0.99:
            portfolio_return = -0.99
            exit_reason = "LIQUIDATED"
        new_capital = capital * (1 + portfolio_return)

        trades.append({
            "entry_time": timestamps[entry_idx],
            "exit_time":  timestamps[exit_idx],
            "hold_bars":  exit_idx - entry_idx,
            "entry_price": entry_price,
            "exit_price":  close_arr[exit_idx],
            "direction":   direction,
            "p_signal":    p_signal,
            "pos_frac":    pos_frac,
            "reason":      exit_reason,
            "gross_return":     gross_return,
            "trade_net_return": trade_net_return,
            "net_return":       portfolio_return,
            "capital_before":   capital,
            "capital_after":    new_capital,
            "pnl_chf":          new_capital - capital,
        })

        equity_curve[i:exit_idx] = capital
        equity_curve[exit_idx] = new_capital
        capital = new_capital
        i = exit_idx + 1 + cooldown

    equity_curve[i:] = capital
    if current_date is not None:
        day_status_log.append({
            "date": current_date,
            "start_capital": day_start_capital,
            "end_capital": capital,
            "day_return": capital / day_start_capital - 1.0,
        })

    return {
        "trades": pd.DataFrame(trades),
        "equity": pd.Series(equity_curve, index=close.index),
        "final_capital": capital,
        "total_return": capital / starting_capital - 1.0,
        "day_log": pd.DataFrame(day_status_log),
    }


def print_row(name: str, res: dict):
    print(
        f"  {name:28s} "
        f"Rendite {res['total_return']*100:>7.1f} %   "
        f"DD {res['max_drawdown']*100:>6.1f} %   "
        f"Sharpe {res['sharpe']:>5.2f}   "
        f"Trades {res['num_trades']:>5d}   "
        f"Exposure {res['exposure']*100:>5.1f} %"
    )
