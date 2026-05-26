"""Paper-Trading Status-Dashboard.

Zeigt komplette Performance-Uebersicht ohne neuen Trade auszufuehren.
Jederzeit aufrufbar:
    python -m scripts.paper_status
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

STATE_DIR = Path("results")
STATE_FILE = STATE_DIR / "paper_state.json"
TRADES_FILE = STATE_DIR / "paper_trades.csv"
EQUITY_FILE = STATE_DIR / "paper_equity.csv"


def main():
    print("=" * 78)
    print("PAPER-TRADING STATUS-DASHBOARD")
    print("=" * 78)

    if not STATE_FILE.exists():
        print("\n  Kein State gefunden. Starte zuerst:")
        print("    python -m scripts.paper_trading")
        return

    with open(STATE_FILE) as f:
        state = json.load(f)

    print(f"\n  Gestartet:        {state['started_at'][:10]}")
    print(f"  Letzter Run:      {state.get('last_run', 'nie')}")
    print(f"  Startkapital:     {state['initial_capital']:.2f} CHF")

    # Tranchen-Status
    print(f"\n  TRANCHEN-STATUS:")
    print(f"    {'Name':<14s} {'Kapital':>10s} {'Anteil':>7s} {'Hebel':>5s} {'Offen':>6s}")
    for tc_name, tc in state["tranches"].items():
        start_cap = state["initial_capital"] * tc["capital_share"]
        ret = tc["capital"] / start_cap - 1
        print(f"    {tc_name:<14s} {tc['capital']:>9.2f}  {tc['capital_share']*100:>5.0f}% "
              f"{tc['leverage']:>4.1f}x {len(tc['open_positions']):>5d}")
        for sym, pos in tc["open_positions"].items():
            d = "LONG" if pos["direction"] == 1 else "SHORT"
            print(f"        → {d} {sym} @ {pos['entry_price']:.5f} seit {pos['entry_date']}")

    # Gesamt-Equity
    if EQUITY_FILE.exists():
        eq = pd.read_csv(EQUITY_FILE)
        eq["date"] = pd.to_datetime(eq["date"])
        eq = eq.sort_values("date").reset_index(drop=True)

        last_eq = eq.iloc[-1]["total_equity"]
        first_eq = eq.iloc[0]["total_equity"]
        n_days = len(eq)
        total_ret = last_eq / first_eq - 1

        print(f"\n  PERFORMANCE-ÜBERSICHT:")
        print(f"    Aktuelle Gesamt-Equity: {last_eq:>10.2f} CHF")
        print(f"    Total Return:           {total_ret*100:>+8.2f}%")
        print(f"    Anzahl Trading-Tage:    {n_days}")
        if n_days >= 30:
            month_ret = last_eq / eq.iloc[-30]["total_equity"] - 1
            print(f"    Letzte 30 Tage:         {month_ret*100:>+8.2f}%")
        if n_days >= 7:
            week_ret = last_eq / eq.iloc[-7]["total_equity"] - 1
            print(f"    Letzte 7 Tage:          {week_ret*100:>+8.2f}%")
        # Annualisiert
        if n_days >= 30:
            years = n_days / 252
            cagr = (last_eq / first_eq) ** (1 / years) - 1 if years > 0 else 0
            print(f"    Hochgerechnet CAGR:     {cagr*100:>+8.2f}% p.a. (extrapoliert)")
        # MaxDD
        if n_days >= 2:
            running_max = eq["total_equity"].cummax()
            dd = (eq["total_equity"] / running_max - 1)
            max_dd = dd.min()
            print(f"    Max Drawdown:           {max_dd*100:>+8.2f}%")

    # Trades
    if TRADES_FILE.exists():
        trades = pd.read_csv(TRADES_FILE)
        if len(trades) > 0:
            wins = trades[trades["pnl_chf"] > 0]
            losses = trades[trades["pnl_chf"] < 0]
            print(f"\n  TRADE-STATISTIK:")
            print(f"    Trades gesamt:    {len(trades)}")
            print(f"    Gewinner:         {len(wins)} ({len(wins)/len(trades)*100:.1f}%)")
            print(f"    Verlierer:        {len(losses)} ({len(losses)/len(trades)*100:.1f}%)")
            print(f"    Avg Gewinn:       {wins['pnl_chf'].mean():>+8.2f} CHF" if len(wins) else "")
            print(f"    Avg Verlust:      {losses['pnl_chf'].mean():>+8.2f} CHF" if len(losses) else "")
            print(f"    Bester Trade:     {trades['pnl_chf'].max():>+8.2f} CHF")
            print(f"    Schlimmster:      {trades['pnl_chf'].min():>+8.2f} CHF")
            print(f"    Total PnL:        {trades['pnl_chf'].sum():>+8.2f} CHF")

            # Letzte 10 Trades
            print(f"\n  LETZTE 10 TRADES:")
            print(f"    {'Datum':<11s} {'Tranche':<13s} {'Dir':>5s} {'Pair':>8s} "
                  f"{'Brutto':>8s} {'Netto':>8s} {'PnL CHF':>10s}")
            for _, t in trades.tail(10).iterrows():
                print(f"    {str(t['date']):<11s} {str(t['tranche']):<13s} "
                      f"{str(t['direction']):>5s} {str(t['pair']):>8s} "
                      f"{t['gross_return_pct']:>+7.2f}% {t['net_return_pct']:>+7.2f}% "
                      f"{t['pnl_chf']:>+9.2f}")

            # Pro Tranche
            print(f"\n  PRO TRANCHE:")
            for tc_name in trades["tranche"].unique():
                sub = trades[trades["tranche"] == tc_name]
                w = (sub["pnl_chf"] > 0).sum()
                wr = w / len(sub) * 100 if len(sub) else 0
                print(f"    {tc_name:<14s} {len(sub):>3d} Trades  WR {wr:>4.1f}%  "
                      f"Total PnL {sub['pnl_chf'].sum():>+8.2f} CHF")

    print(f"\n  Dateien:")
    print(f"    State:   {STATE_FILE}")
    print(f"    Trades:  {TRADES_FILE}")
    print(f"    Equity:  {EQUITY_FILE}")
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
