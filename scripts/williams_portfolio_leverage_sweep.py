"""Hebel-Sweep 1x bis 100x auf dem Multi-Asset Williams-Portfolio.

Zeigt empirisch:
- Wo ist der Sweet Spot?
- Ab wann fängt es an zu liquidieren?
- Was passiert mit MaxDD?
- Realistisches CAGR vs Risiko
"""
import warnings
import numpy as np
import pandas as pd

from src.data.storage import load_ohlcv
from src.indicators.technical import williams_r

warnings.filterwarnings("ignore", category=UserWarning)

STARTING_CAPITAL_CHF = 1000.0
FEE_RATE = 0.0002
HOLD_DAYS = 1
WR_OVERSOLD = -85
WR_OVERBOUGHT = -10
POSITION_PER_PAIR = 0.10
MAX_TOTAL_EXPOSURE = 0.60

PAIRS = ["USD/CHF", "EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD"]
LEVERAGES = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]


def run_portfolio(coins, all_ts, leverage):
    capital = STARTING_CAPITAL_CHF
    open_pos = {}
    n_liquidations = 0
    n_trades = 0
    n_wins = 0
    equity_curve = []   # Track jeder Tag fuer korrekten MaxDD

    for ts in all_ts:
        # Exits
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
            if (cur_idx - entry_idx) >= HOLD_DAYS:
                exit_price = df.loc[ts, "close"]
                price_ret = (exit_price / pos["entry_price"] - 1.0) * pos["direction"]
                trade_net = (1 + price_ret) * (1 - FEE_RATE) ** 2 - 1.0
                port_ret = pos["position_frac"] * trade_net * leverage
                if port_ret <= -0.99:
                    port_ret = -0.99
                    n_liquidations += 1
                pnl = capital * port_ret
                capital += pnl
                if pnl > 0:
                    n_wins += 1
                n_trades += 1
                del open_pos[symbol]

        # Mark-to-market: aktuelles Equity inkl. offener Positionen
        unrealized = 0.0
        for symbol, pos in open_pos.items():
            df = coins[symbol]
            if ts not in df.index:
                continue
            cur_price = df.loc[ts, "close"]
            price_ret = (cur_price / pos["entry_price"] - 1.0) * pos["direction"]
            # Unrealized PnL inkl. Hebel
            unreal_port_ret = pos["position_frac"] * price_ret * leverage
            if unreal_port_ret <= -0.99:
                unreal_port_ret = -0.99
            unrealized += capital * unreal_port_ret

        equity_now = capital + unrealized
        equity_curve.append(equity_now)

        # Bail wenn Equity aufgebraucht
        if equity_now < 1.0:
            break

        # Entries
        current_exposure = sum(p["position_frac"] for p in open_pos.values())
        for symbol in PAIRS:
            if symbol in open_pos:
                continue
            df = coins[symbol]
            if ts not in df.index:
                continue
            row = df.loc[ts]
            available = MAX_TOTAL_EXPOSURE - current_exposure
            if available < POSITION_PER_PAIR * 0.5:
                break
            pos_frac = min(POSITION_PER_PAIR, available)
            if row["long_sig"]:
                open_pos[symbol] = {
                    "entry_ts": ts, "direction": 1,
                    "entry_price": row["close"], "position_frac": pos_frac,
                }
                current_exposure += pos_frac
            elif row["short_sig"]:
                open_pos[symbol] = {
                    "entry_ts": ts, "direction": -1,
                    "entry_price": row["close"], "position_frac": pos_frac,
                }
                current_exposure += pos_frac

    # Korrekter MaxDD via Equity-Kurve
    eq = pd.Series(equity_curve)
    max_dd = (eq / eq.cummax() - 1).min() if len(eq) > 1 else 0
    wr = n_wins / n_trades if n_trades else 0
    return capital, n_trades, wr, max_dd, n_liquidations


def main():
    # Lade Daten einmal
    print("Lade 5 Pairs …")
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

    print(f"  Zeitraum: {common_start.date()} → {common_end.date()} ({n_years:.1f} Jahre)\n")

    print(f"{'='*82}")
    print(f"HEBEL-SWEEP MULTI-ASSET-PORTFOLIO (Williams Mean-Reversion, 10% Pos/Pair, Hold 1d)")
    print(f"{'='*82}")
    print(f"  {'Hebel':>5s} | {'Endkap.':>15s} {'Return':>11s} {'CAGR':>9s} "
          f"{'MaxDD':>8s} {'Trades':>6s} {'WR':>5s} {'Liq':>4s}")
    print(f"  {'-'*78}")

    results = []
    for lev in LEVERAGES:
        capital, n_trades, wr, max_dd, n_liq = run_portfolio(coins, all_ts, lev)
        cagr = (capital / STARTING_CAPITAL_CHF) ** (1 / n_years) - 1 if capital > 0 else -1
        warning = " ⚠ LIQUIDIERT!" if n_liq > 0 else ""
        print(f"  {lev:>4d}x | {capital:>14,.2f} CHF "
              f"{(capital/STARTING_CAPITAL_CHF-1)*100:>+10.2f}% "
              f"{cagr*100:>+8.2f}% {max_dd*100:>+7.2f}% "
              f"{n_trades:>6d} {wr*100:>4.1f}% {n_liq:>4d}{warning}")
        results.append({"lev": lev, "capital": capital, "cagr": cagr,
                        "max_dd": max_dd, "liq": n_liq, "risk_adj": cagr/abs(max_dd) if max_dd != 0 else 0})

    # Sweet Spot Analyse
    print(f"\n=== SWEET SPOT ANALYSE ===")
    best_abs = max(results, key=lambda x: x["cagr"])
    no_liq = [r for r in results if r["liq"] == 0]
    best_no_liq = max(no_liq, key=lambda x: x["cagr"]) if no_liq else None
    best_risk_adj = max(results, key=lambda x: x.get("risk_adj", 0))

    print(f"  Maximaler CAGR absolut:    Hebel {best_abs['lev']}x → {best_abs['cagr']*100:+.2f}% "
          f"(MaxDD {best_abs['max_dd']*100:.1f}%, Liq {best_abs['liq']})")
    if best_no_liq:
        print(f"  Bester ohne Liquidation:   Hebel {best_no_liq['lev']}x → {best_no_liq['cagr']*100:+.2f}% "
              f"(MaxDD {best_no_liq['max_dd']*100:.1f}%)")
    print(f"  Best Risk-Adjusted:        Hebel {best_risk_adj['lev']}x → {best_risk_adj['cagr']*100:+.2f}% / "
          f"|DD| {abs(best_risk_adj['max_dd'])*100:.1f}% = Ratio {best_risk_adj['risk_adj']:.2f}")

    # Tabelle: Liquidations-Schwelle
    first_liq = next((r for r in results if r["liq"] > 0), None)
    if first_liq:
        print(f"\n  Erste Liquidation tritt bei Hebel {first_liq['lev']}x auf")
    else:
        print(f"\n  Keine Liquidationen in irgendeinem Hebel-Test!")


if __name__ == "__main__":
    main()
