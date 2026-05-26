"""Paper-Trading-Bot Williams Strategie + Barbell-Allokation.

Datenquelle: yfinance (kostenlos, kein Broker-Account).
Allokation: 60 % @ 2× Hebel + 40 % @ 15× Hebel (Barbell).
Speichert State persistent — täglich ausführbar.

Erster Run: initialisiert mit 5.000 CHF Demo-Kapital.
Folgende Runs: aktualisiert State, schliesst alte Positionen, eroeffnet neue.

Manueller Run:
    source venv/bin/activate && python -m scripts.paper_trading

Cron-Setup (macOS, jeden Werktag 22:00):
    0 22 * * 1-5 cd /Users/gorkem/trading-bot && \
        /Users/gorkem/trading-bot/venv/bin/python -m scripts.paper_trading

Reset (von vorne anfangen):
    rm results/paper_state.json results/paper_trades.csv results/paper_equity.csv
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ============== KONFIGURATION ==============
PAIRS = {
    "USD/CHF":  "CHF=X",
    "EUR/USD":  "EURUSD=X",
    "USD/JPY":  "JPY=X",
    "GBP/USD":  "GBPUSD=X",
    "AUD/USD":  "AUDUSD=X",
}

TRANCHES_CONFIG = [
    {"name": "Tranche_2x",  "leverage": 2.0,  "capital_share": 0.60},
    {"name": "Tranche_15x", "leverage": 15.0, "capital_share": 0.40},
]

INITIAL_CAPITAL = 5000.0
POSITION_FRAC = 0.10          # 10 % pro Pair je Tranche
MAX_TOTAL_EXPOSURE = 0.60     # max 60 % der Tranche im Markt
WR_PERIOD = 14
WR_OVERSOLD = -85
WR_OVERBOUGHT = -10
HOLD_DAYS = 1
FEE_RATE = 0.0002

STATE_DIR = Path("results")
STATE_FILE = STATE_DIR / "paper_state.json"
TRADES_FILE = STATE_DIR / "paper_trades.csv"
EQUITY_FILE = STATE_DIR / "paper_equity.csv"


# ============== DATENLADUNG ==============
def fetch_recent_data(yf_symbol, days=90):
    """Laedt letzte N Tage Daily-OHLCV via yfinance."""
    raw = yf.download(yf_symbol, period=f"{days}d", interval="1d",
                      progress=False, auto_adjust=False)
    if raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.copy()
    df.index.name = "timestamp"
    df = df.reset_index()
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[["timestamp", "open", "high", "low", "close"]].dropna()
    df = df.set_index("timestamp").sort_index()
    return df


def williams_r(high, low, close, period=14):
    highest = high.rolling(period).max()
    lowest = low.rolling(period).min()
    return -100 * (highest - close) / (highest - lowest).replace(0, 1e-12)


# ============== STATE-MANAGEMENT ==============
def load_state():
    if not STATE_FILE.exists():
        return None
    with open(STATE_FILE) as f:
        return json.load(f)


def save_state(state):
    STATE_DIR.mkdir(exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def init_state():
    state = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "initial_capital": INITIAL_CAPITAL,
        "last_run": None,
        "tranches": {},
    }
    for tc in TRANCHES_CONFIG:
        state["tranches"][tc["name"]] = {
            "leverage": tc["leverage"],
            "capital_share": tc["capital_share"],
            "capital": INITIAL_CAPITAL * tc["capital_share"],
            "open_positions": {},   # symbol -> position dict
        }
    return state


def append_trade(trade_record):
    STATE_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame([trade_record])
    if TRADES_FILE.exists():
        df.to_csv(TRADES_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(TRADES_FILE, index=False)


def append_equity_snapshot(date, total_equity, per_tranche):
    STATE_DIR.mkdir(exist_ok=True)
    row = {"date": date, "total_equity": total_equity, **per_tranche}
    df = pd.DataFrame([row])
    if EQUITY_FILE.exists():
        df.to_csv(EQUITY_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(EQUITY_FILE, index=False)


# ============== TRADING-LOGIK ==============
def process_tranche(tranche, latest_data, today_str):
    """
    latest_data: dict symbol -> df mit close + wr (letzte ~90 Tage)
    Liefert: list of trade_records + updated tranche
    """
    leverage = tranche["leverage"]
    capital = tranche["capital"]
    open_pos = tranche["open_positions"]
    trades = []

    # 1) Exits — alle Positionen die HOLD_DAYS alt sind
    for symbol in list(open_pos.keys()):
        pos = open_pos[symbol]
        df = latest_data.get(symbol)
        if df is None or len(df) == 0:
            continue
        entry_date = pd.Timestamp(pos["entry_date"])
        today = df.index[-1]
        # Anzahl Bars seit Entry (Tagesgranularitaet)
        try:
            entry_idx = df.index.get_indexer([entry_date], method="nearest")[0]
        except Exception:
            continue
        days_held = len(df) - 1 - entry_idx
        if days_held >= HOLD_DAYS:
            exit_price = float(df["close"].iloc[-1])
            price_ret = (exit_price / pos["entry_price"] - 1.0) * pos["direction"]
            trade_net = (1 + price_ret) * (1 - FEE_RATE) ** 2 - 1.0
            port_ret = pos["position_frac"] * trade_net * leverage
            if port_ret <= -0.99:
                port_ret = -0.99
            pnl = capital * port_ret
            capital += pnl
            trades.append({
                "date": today_str,
                "tranche": tranche.get("name", ""),
                "pair": symbol,
                "direction": "LONG" if pos["direction"] == 1 else "SHORT",
                "entry_date": pos["entry_date"],
                "entry_price": pos["entry_price"],
                "exit_date": today_str,
                "exit_price": exit_price,
                "gross_return_pct": round(price_ret * 100, 4),
                "net_return_pct": round(port_ret * 100, 4),
                "pnl_chf": round(pnl, 2),
                "capital_after": round(capital, 2),
            })
            del open_pos[symbol]

    # 2) Entries — pruefe Signale
    current_exposure = sum(p["position_frac"] for p in open_pos.values())
    for symbol, df in latest_data.items():
        if symbol in open_pos or df is None or len(df) < WR_PERIOD + 2:
            continue
        wr_now = float(df["wr"].iloc[-1])
        wr_prev = float(df["wr"].iloc[-2])
        long_sig = (wr_prev >= WR_OVERSOLD) and (wr_now < WR_OVERSOLD)
        short_sig = (wr_prev <= WR_OVERBOUGHT) and (wr_now > WR_OVERBOUGHT)
        available = MAX_TOTAL_EXPOSURE - current_exposure
        if available < POSITION_FRAC * 0.5:
            break
        if long_sig:
            pos_frac = min(POSITION_FRAC, available)
            open_pos[symbol] = {
                "entry_date": today_str,
                "entry_price": float(df["close"].iloc[-1]),
                "direction": 1,
                "position_frac": pos_frac,
            }
            current_exposure += pos_frac
        elif short_sig:
            pos_frac = min(POSITION_FRAC, available)
            open_pos[symbol] = {
                "entry_date": today_str,
                "entry_price": float(df["close"].iloc[-1]),
                "direction": -1,
                "position_frac": pos_frac,
            }
            current_exposure += pos_frac

    tranche["capital"] = capital
    tranche["open_positions"] = open_pos
    return trades, tranche


def calc_unrealized(tranche, latest_data):
    """Mark-to-Market der offenen Positionen."""
    leverage = tranche["leverage"]
    capital = tranche["capital"]
    unrealized = 0.0
    for symbol, pos in tranche["open_positions"].items():
        df = latest_data.get(symbol)
        if df is None or len(df) == 0:
            continue
        cur_price = float(df["close"].iloc[-1])
        price_ret = (cur_price / pos["entry_price"] - 1.0) * pos["direction"]
        unreal = pos["position_frac"] * price_ret * leverage
        if unreal <= -0.99:
            unreal = -0.99
        unrealized += capital * unreal
    return capital + unrealized


# ============== MAIN ==============
def main(reset=False):
    print("=" * 78)
    print(f"PAPER-TRADING-BOT — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 78)

    if reset:
        for f in [STATE_FILE, TRADES_FILE, EQUITY_FILE]:
            if f.exists():
                f.unlink()
                print(f"  Geloescht: {f}")

    state = load_state()
    if state is None:
        state = init_state()
        print(f"\n  Initialisiert mit {INITIAL_CAPITAL:.0f} CHF")
        for tc_name, tc in state["tranches"].items():
            print(f"    {tc_name}: {tc['capital']:.0f} CHF @ {tc['leverage']}x Hebel "
                  f"({tc['capital_share']*100:.0f} %)")
    else:
        print(f"\n  Bestehender State geladen (Start: {state['started_at'][:10]})")

    # Lade aktuelle Daten
    print(f"\n  Lade aktuelle Forex-Daten via yfinance …")
    latest_data = {}
    for symbol, yf_sym in PAIRS.items():
        df = fetch_recent_data(yf_sym, days=90)
        if df is None or len(df) < WR_PERIOD + 2:
            print(f"    {symbol}: keine Daten")
            continue
        df["wr"] = williams_r(df["high"], df["low"], df["close"], WR_PERIOD)
        df = df.dropna()
        latest_data[symbol] = df
        wr_now = float(df["wr"].iloc[-1])
        wr_prev = float(df["wr"].iloc[-2])
        close_now = float(df["close"].iloc[-1])
        print(f"    {symbol}: close={close_now:.5f}, WR(prev→now): {wr_prev:.1f} → {wr_now:.1f}")

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Verarbeite jede Tranche
    all_trades = []
    for tc_name, tc in state["tranches"].items():
        tc["name"] = tc_name   # fuer Trade-Records
        trades, tc_updated = process_tranche(tc, latest_data, today_str)
        all_trades.extend(trades)
        state["tranches"][tc_name] = tc_updated

    # Trades persistieren
    if all_trades:
        print(f"\n  TRADES heute ({len(all_trades)}):")
        for t in all_trades:
            print(f"    [{t['tranche']:>12s}] {t['direction']} {t['pair']}  "
                  f"{t['gross_return_pct']:>+6.2f}%  Netto {t['net_return_pct']:>+6.2f}%  "
                  f"PnL {t['pnl_chf']:>+7.2f} CHF  → Kapital {t['capital_after']:.2f}")
            append_trade(t)
    else:
        print(f"\n  Keine geschlossenen Trades heute.")

    # Aktuelle Equity (inkl. unrealized)
    total_equity = 0
    per_tranche_equity = {}
    print(f"\n  AKTUELLER STATUS:")
    for tc_name, tc in state["tranches"].items():
        eq = calc_unrealized(tc, latest_data)
        per_tranche_equity[tc_name] = round(eq, 2)
        total_equity += eq
        n_open = len(tc["open_positions"])
        ret = eq / (INITIAL_CAPITAL * tc["capital_share"]) - 1
        print(f"    {tc_name}: {eq:>9.2f} CHF ({ret*100:>+6.2f}%) "
              f"| {n_open} offene Positionen | Hebel {tc['leverage']}x")
        for sym, pos in tc["open_positions"].items():
            df = latest_data.get(sym)
            if df is not None:
                cp = float(df["close"].iloc[-1])
                pr = (cp / pos["entry_price"] - 1.0) * pos["direction"]
                d = "LONG" if pos["direction"] == 1 else "SHORT"
                print(f"        {d} {sym}: entry {pos['entry_price']:.5f} → now {cp:.5f}  "
                      f"({pr*100:+.2f}% gross, seit {pos['entry_date']})")

    print(f"\n  GESAMT: {total_equity:>9.2f} CHF "
          f"({(total_equity/INITIAL_CAPITAL-1)*100:+.2f}% seit Start)")

    append_equity_snapshot(today_str, round(total_equity, 2), per_tranche_equity)

    state["last_run"] = today_str
    save_state(state)

    # Auto-generiere Excel + HTML Reports
    try:
        from scripts.paper_report import main as generate_report
        print(f"\n  Generiere Reports …")
        generate_report()
    except Exception as e:
        print(f"\n  (Report-Generation fehlgeschlagen: {e})")

    # Performance-Übersicht aus History
    if EQUITY_FILE.exists():
        eq_hist = pd.read_csv(EQUITY_FILE)
        n_days = len(eq_hist)
        if n_days > 1:
            print(f"\n  --- HISTORIE ({n_days} Tage) ---")
            first = eq_hist.iloc[0]["total_equity"]
            last = eq_hist.iloc[-1]["total_equity"]
            print(f"    Total Return seit Start: {(last/first-1)*100:+.2f}%")
            if n_days > 30:
                month_ago = eq_hist.iloc[-30]["total_equity"]
                print(f"    Letzte 30 Tage:          {(last/month_ago-1)*100:+.2f}%")
            print(f"    State-Dateien:           {STATE_DIR.absolute()}/paper_*.csv")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="State zuruecksetzen")
    args = parser.parse_args()
    main(reset=args.reset)
