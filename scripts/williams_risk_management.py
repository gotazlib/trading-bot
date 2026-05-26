"""Williams Multi-Asset Portfolio + Risk-Management-Layer.

Zusätzliche Sicherheits-Regeln:
- STOP-LOSS: wenn Equity 5% unter aktuellem Peak → ALLE Positionen schließen + pausiere N Tage
- TAKE-PROFIT: wenn Equity 20% über letzter Take-Profit-Baseline → realisiere, neue Baseline
- Vergleich verschiedener Schwellen
"""
import warnings
import numpy as np
import pandas as pd

from src.data.storage import load_ohlcv
from src.indicators.technical import williams_r

warnings.filterwarnings("ignore", category=UserWarning)

TOTAL_CAPITAL_CHF = 5000.0
FEE_RATE = 0.0002
HOLD_DAYS = 1
WR_OVERSOLD = -85
WR_OVERBOUGHT = -10
POSITION_PER_PAIR = 0.10
MAX_TOTAL_EXPOSURE = 0.60
LEVERAGE = 7.0

PAIRS = ["USD/CHF", "EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD"]


def run_with_risk_mgmt(coins, all_ts, stop_loss_pct, take_profit_pct, pause_days):
    """
    stop_loss_pct: z.B. 0.05 = wenn equity 5% unter Peak → schließen, pause
    take_profit_pct: z.B. 0.20 = wenn equity 20% über baseline → schließen, neue baseline
    pause_days: nach Stop wie viele Tage keine neuen Trades
    """
    capital = TOTAL_CAPITAL_CHF
    open_pos = {}
    n_trades = 0
    n_wins = 0
    n_stops = 0
    n_takes = 0
    equity_curve = []

    peak = capital            # höchster Equity-Punkt für Stop-Loss
    baseline = capital        # für Take-Profit-Tracking
    pause_until = -1          # Index in all_ts

    for idx, ts in enumerate(all_ts):
        # Exits (HOLD_DAYS oder forced)
        for symbol in list(open_pos.keys()):
            pos = open_pos[symbol]
            df = coins[symbol]
            if ts not in df.index:
                continue
            try:
                entry_idx = df.index.get_loc(pos["entry_ts"])
                cur_idx = df.index.get_loc(ts)
            except KeyError:
                continue
            force_exit = pos.get("force_exit", False)
            if (cur_idx - entry_idx) >= HOLD_DAYS or force_exit:
                exit_price = df.loc[ts, "close"]
                price_ret = (exit_price / pos["entry_price"] - 1.0) * pos["direction"]
                trade_net = (1 + price_ret) * (1 - FEE_RATE) ** 2 - 1.0
                port_ret = pos["position_frac"] * trade_net * LEVERAGE
                if port_ret <= -0.99:
                    port_ret = -0.99
                pnl = capital * port_ret
                capital += pnl
                if pnl > 0:
                    n_wins += 1
                n_trades += 1
                del open_pos[symbol]

        # Mark-to-market
        unreal = 0.0
        for symbol, pos in open_pos.items():
            df = coins[symbol]
            if ts not in df.index:
                continue
            cp = df.loc[ts, "close"]
            pr = (cp / pos["entry_price"] - 1.0) * pos["direction"]
            ur = pos["position_frac"] * pr * LEVERAGE
            if ur <= -0.99:
                ur = -0.99
            unreal += capital * ur
        equity_now = capital + unreal
        equity_curve.append(equity_now)

        # Peak update
        if equity_now > peak:
            peak = equity_now

        # STOP-LOSS Check
        dd_from_peak = equity_now / peak - 1
        if stop_loss_pct > 0 and dd_from_peak <= -stop_loss_pct and len(open_pos) > 0:
            # Forciere Exit aller Positionen im nächsten Bar
            for sym in open_pos:
                open_pos[sym]["force_exit"] = True
            pause_until = idx + pause_days
            n_stops += 1

        # TAKE-PROFIT Check
        profit_from_baseline = equity_now / baseline - 1
        if take_profit_pct > 0 and profit_from_baseline >= take_profit_pct:
            # Realisiere: alle Positionen schließen, baseline neu setzen
            for sym in open_pos:
                open_pos[sym]["force_exit"] = True
            baseline = equity_now
            n_takes += 1

        # Entries — nur wenn nicht in Pause
        if idx > pause_until and len(open_pos) == 0:
            cur_exp = 0.0
            for symbol in PAIRS:
                df = coins[symbol]
                if ts not in df.index:
                    continue
                row = df.loc[ts]
                avail = MAX_TOTAL_EXPOSURE - cur_exp
                if avail < POSITION_PER_PAIR * 0.5:
                    break
                pos_frac = min(POSITION_PER_PAIR, avail)
                if row["long_sig"]:
                    open_pos[symbol] = {"entry_ts": ts, "direction": 1,
                                        "entry_price": row["close"], "position_frac": pos_frac}
                    cur_exp += pos_frac
                elif row["short_sig"]:
                    open_pos[symbol] = {"entry_ts": ts, "direction": -1,
                                        "entry_price": row["close"], "position_frac": pos_frac}
                    cur_exp += pos_frac

    eq = pd.Series(equity_curve)
    max_dd = (eq / eq.cummax() - 1).min() if len(eq) > 1 else 0
    return capital, n_trades, n_wins, n_stops, n_takes, max_dd


def main():
    print("Lade Daten …")
    coins = {}
    for sym in PAIRS:
        df = load_ohlcv("forex", sym, "1d")
        df["wr"] = williams_r(df["high"], df["low"], df["close"], 14)
        df = df.dropna()
        df["long_sig"] = (df["wr"].shift(1) >= WR_OVERSOLD) & (df["wr"] < WR_OVERSOLD)
        df["short_sig"] = (df["wr"].shift(1) <= WR_OVERBOUGHT) & (df["wr"] > WR_OVERBOUGHT)
        coins[sym] = df
    common_start = max(df.index.min() for df in coins.values())
    common_end = min(df.index.max() for df in coins.values())
    for sym in PAIRS:
        coins[sym] = coins[sym].loc[common_start:common_end]
    all_ts = sorted(set().union(*[set(df.index) for df in coins.values()]))
    n_years = (common_end - common_start).days / 365.25
    print(f"Zeitraum: {common_start.date()} → {common_end.date()} ({n_years:.1f} Jahre)\n")

    print(f"=== TESTS VERSCHIEDENER RISK-MGMT-PARAMETER ===")
    print(f"  {'Stop%':>5s} {'TP%':>5s} {'Pause':>5s} {'Endkap.':>15s} {'CAGR':>8s} "
          f"{'MaxDD':>8s} {'Trades':>6s} {'WR':>5s} {'Stops':>6s} {'Takes':>6s}")

    configs = [
        (0.00, 0.00, 0, "Baseline (kein Risk-Mgmt)"),
        (0.05, 0.20, 5, "User-Setup: -5% Stop, +20% TP, 5d Pause"),
        (0.05, 0.20, 14, "-5% Stop, +20% TP, 14d Pause"),
        (0.05, 0.20, 30, "-5% Stop, +20% TP, 30d Pause"),
        (0.03, 0.10, 7, "Eng: -3% Stop, +10% TP"),
        (0.10, 0.30, 7, "Weit: -10% Stop, +30% TP"),
        (0.05, 0.00, 14, "Nur Stop-Loss -5%, kein TP"),
        (0.00, 0.20, 0, "Nur Take-Profit +20%, kein Stop"),
    ]
    results = []
    for sl, tp, pause, label in configs:
        cap, ntr, nw, nstop, ntake, dd = run_with_risk_mgmt(coins, all_ts, sl, tp, pause)
        cagr = (cap / TOTAL_CAPITAL_CHF) ** (1 / n_years) - 1
        wr = nw / ntr if ntr else 0
        print(f"  {sl*100:>4.0f}% {tp*100:>4.0f}% {pause:>4d}d {cap:>14,.0f} "
              f"{cagr*100:>+7.2f}% {dd*100:>+7.2f}% {ntr:>6d} {wr*100:>4.1f}% "
              f"{nstop:>6d} {ntake:>6d}")
        results.append({"label": label, "cap": cap, "cagr": cagr, "dd": dd, "stops": nstop, "takes": ntake})

    print(f"\n=== KOMMENTAR ===")
    for r in results:
        ratio = r["cagr"] / abs(r["dd"]) if r["dd"] != 0 else 0
        print(f"  {r['label']}")
        print(f"    Endkapital: {r['cap']:,.0f} CHF | CAGR {r['cagr']*100:+.2f}% | "
              f"MaxDD {r['dd']*100:+.2f}% | Risk-Adj {ratio:.2f}")
        print(f"    Stops getriggert: {r['stops']} | Take-Profits: {r['takes']}")


if __name__ == "__main__":
    main()
