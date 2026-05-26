"""Hebel-Mix-Simulation: Kapital auf verschiedene Hebel-Tranchen aufteilen.

5000 CHF Startkapital, verschiedene Mix-Strategien getestet:
- Conservative: Schwerpunkt auf niedrige Hebel
- Balanced: gleiche Verteilung
- Aggressive: Schwerpunkt auf hohe Hebel
- Custom: vom User definierbar

Jede Tranche läuft separat mit ihrem eigenen Hebel,
am Ende werden alle aggregiert.
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

PAIRS = ["USD/CHF", "EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD"]

# Mix-Strategien: List of (leverage, capital_share)
MIX_STRATEGIES = {
    "All 1x (Cash-ähnlich)":         [(1, 1.00)],
    "All 5x (klassisch)":            [(5, 1.00)],
    "All 7x (Sweet Spot)":           [(7, 1.00)],
    "All 10x (aggressiv)":           [(10, 1.00)],
    "Konservativ Mix (Avg 3.2x)":   [(2, 0.40), (3, 0.30), (5, 0.20), (7, 0.10)],
    "Balanced Mix (Avg 6.25x)":     [(3, 0.25), (5, 0.25), (7, 0.25), (10, 0.25)],
    "Aggressive Mix (Avg 9.5x)":    [(5, 0.10), (7, 0.20), (10, 0.30), (15, 0.40)],
    "Barbell (sicher + Spekulation)": [(2, 0.60), (15, 0.40)],
    "Wachstums-Pyramide":           [(3, 0.20), (5, 0.30), (7, 0.30), (10, 0.20)],
}


def run_tranche(coins, all_ts, leverage, start_capital):
    """Ein Hebel-Konto, läuft die Multi-Asset Williams Strategy."""
    capital = start_capital
    open_pos = {}
    n_liquidations = 0
    n_trades = 0
    n_wins = 0
    equity_curve = []

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

        # Mark-to-market
        unrealized = 0.0
        for symbol, pos in open_pos.items():
            df = coins[symbol]
            if ts not in df.index:
                continue
            cur_price = df.loc[ts, "close"]
            price_ret = (cur_price / pos["entry_price"] - 1.0) * pos["direction"]
            unreal = pos["position_frac"] * price_ret * leverage
            if unreal <= -0.99:
                unreal = -0.99
            unrealized += capital * unreal
        equity_curve.append(capital + unrealized)

        if capital + unrealized < start_capital * 0.001:
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
                open_pos[symbol] = {"entry_ts": ts, "direction": 1,
                                    "entry_price": row["close"], "position_frac": pos_frac}
                current_exposure += pos_frac
            elif row["short_sig"]:
                open_pos[symbol] = {"entry_ts": ts, "direction": -1,
                                    "entry_price": row["close"], "position_frac": pos_frac}
                current_exposure += pos_frac

    eq = pd.Series(equity_curve)
    max_dd = (eq / eq.cummax() - 1).min() if len(eq) > 1 else 0
    wr = n_wins / n_trades if n_trades else 0
    return capital, n_trades, wr, max_dd, n_liquidations, eq


def main():
    print(f"Hebel-Mix-Simulation — {TOTAL_CAPITAL_CHF:.0f} CHF Startkapital\n")
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

    # Cache pro Hebel — jeder Hebel wird nur einmal berechnet
    print(f"Berechne Hebel-Tranchen einzeln …")
    unique_levs = sorted(set(lev for strat in MIX_STRATEGIES.values() for lev, _ in strat))
    lev_results = {}
    for lev in unique_levs:
        # Pro Hebel mit 1000 CHF baseline simulieren — dann skalieren
        cap, _, _, dd, liq, eq = run_tranche(coins, all_ts, lev, 1000.0)
        lev_results[lev] = {"final_per_1000": cap, "max_dd": dd, "liq": liq, "eq": eq}
        cagr_lev = (cap / 1000.0) ** (1 / n_years) - 1
        print(f"  Hebel {lev}x: 1000 CHF → {cap:,.0f} CHF (CAGR {cagr_lev*100:+.2f}%, DD {dd*100:+.1f}%)")

    print(f"\n{'='*88}")
    print(f"MIX-STRATEGIEN ({TOTAL_CAPITAL_CHF:.0f} CHF Startkapital)")
    print(f"{'='*88}")
    print(f"  {'Strategie':<32s} {'Avg Hebel':>9s} {'Endkap.':>14s} {'CAGR':>8s} {'MaxDD':>8s}")

    strategy_results = []
    for name, allocations in MIX_STRATEGIES.items():
        # Avg Hebel berechnen
        avg_lev = sum(lev * share for lev, share in allocations)

        # Pro Tranche: skaliere lev_result mit tatsaechlichem Kapital
        # Trick: Returns sind linear in initial capital
        total_final = 0
        # Auch Portfolio-Equity-Kurve aggregieren
        max_len = max(len(lev_results[lev]["eq"]) for lev, _ in allocations)
        combined_eq = np.zeros(max_len)
        for lev, share in allocations:
            tranche_capital = TOTAL_CAPITAL_CHF * share
            scaling = tranche_capital / 1000.0
            final_tranche = lev_results[lev]["final_per_1000"] * scaling
            total_final += final_tranche
            # Equity-Kurve aufaddieren (gepadded)
            eq = lev_results[lev]["eq"].values * scaling
            if len(eq) < max_len:
                # Pad mit letztem Wert
                eq = np.concatenate([eq, np.full(max_len - len(eq), eq[-1] if len(eq) > 0 else 0)])
            combined_eq += eq

        total_return = total_final / TOTAL_CAPITAL_CHF - 1
        cagr = (total_final / TOTAL_CAPITAL_CHF) ** (1 / n_years) - 1
        combined_eq_s = pd.Series(combined_eq)
        max_dd = (combined_eq_s / combined_eq_s.cummax() - 1).min()
        strategy_results.append({
            "name": name, "avg_lev": avg_lev, "final": total_final,
            "cagr": cagr, "max_dd": max_dd, "allocations": allocations,
        })
        print(f"  {name:<32s} {avg_lev:>7.2f}x {total_final:>13,.0f} CHF "
              f"{cagr*100:>+7.2f}% {max_dd*100:>+7.2f}%")

    print(f"\n=== DETAIL: Beste Mix-Strategien ===")
    by_cagr = sorted(strategy_results, key=lambda x: -x["cagr"])
    by_risk_adj = sorted(strategy_results, key=lambda x: -(x["cagr"] / abs(x["max_dd"]) if x["max_dd"] != 0 else 0))

    print(f"\n  Top 3 nach absolutem CAGR:")
    for r in by_cagr[:3]:
        print(f"    {r['name']:<35s} CAGR {r['cagr']*100:+6.2f}%, MaxDD {r['max_dd']*100:+6.2f}%, "
              f"End {r['final']:>12,.0f} CHF")
        for lev, share in r["allocations"]:
            tr_cap = TOTAL_CAPITAL_CHF * share
            tr_end = lev_results[lev]["final_per_1000"] * (tr_cap / 1000)
            print(f"      → {tr_cap:>6,.0f} CHF @ {lev}x → {tr_end:>13,.0f} CHF")

    print(f"\n  Top 3 nach Risk-Adjusted (CAGR/|DD|):")
    for r in by_risk_adj[:3]:
        ratio = r["cagr"] / abs(r["max_dd"]) if r["max_dd"] != 0 else 0
        print(f"    {r['name']:<35s} CAGR {r['cagr']*100:+6.2f}%, DD {r['max_dd']*100:+6.2f}%, "
              f"Ratio {ratio:.2f}")


if __name__ == "__main__":
    main()
