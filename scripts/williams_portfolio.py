"""Multi-Asset Portfolio Williams Strategy.

Alle 5 Forex Major Pairs gleichzeitig im Portfolio:
- Pro Pair Position 10% (5×10% = max 50% Gesamt)
- Hebel 7x pro Position
- Williams-Strategie (Hold 1d) auf jeden Pair
- Pro Tag: pruefe alle Pairs auf Signale, eroeffne falls Kapital verfügbar
- Korrelationen real (alle USD-Pairs reagieren auf USD-Bewegungen)
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
POSITION_PER_PAIR = 0.10     # 10% pro Pair (5 Pairs × 10% = 50% max)
LEVERAGE = 7.0
MAX_TOTAL_EXPOSURE = 0.60    # max 60% Gesamt — Sicherheits-Cap

PAIRS = ["USD/CHF", "EUR/USD", "USD/JPY", "GBP/USD", "AUD/USD"]


def main():
    print(f"Williams Multi-Asset Portfolio")
    print(f"  {len(PAIRS)} Pairs × 10% Position × 7x Hebel")
    print(f"  Max Gesamt-Exposure: {MAX_TOTAL_EXPOSURE*100:.0f}%\n")

    # Lade alle Pairs
    coins = {}
    for sym in PAIRS:
        df = load_ohlcv("forex", sym, "1d")
        df["wr"] = williams_r(df["high"], df["low"], df["close"], 14)
        df = df.dropna()
        df["long_sig"] = (df["wr"].shift(1) >= WR_OVERSOLD) & (df["wr"] < WR_OVERSOLD)
        df["short_sig"] = (df["wr"].shift(1) <= WR_OVERBOUGHT) & (df["wr"] > WR_OVERBOUGHT)
        coins[sym] = df

    # Gemeinsamer Zeitraum
    common_start = max(df.index.min() for df in coins.values())
    common_end = min(df.index.max() for df in coins.values())
    for sym in PAIRS:
        coins[sym] = coins[sym].loc[common_start:common_end]

    all_ts = sorted(set().union(*[set(df.index) for df in coins.values()]))
    n_years = (common_end - common_start).days / 365.25
    print(f"Zeitraum: {common_start.date()} → {common_end.date()} ({n_years:.1f} Jahre)")
    print(f"Bars: {len(all_ts)}\n")

    # Portfolio Backtest
    capital = STARTING_CAPITAL_CHF
    open_pos = {}    # symbol -> dict
    trades = []
    equity_records = []
    max_cap = capital
    min_cap_after_peak = capital

    for ts in all_ts:
        # 1) Exits — alle Positionen nach HOLD_DAYS schließen
        for symbol in list(open_pos.keys()):
            pos = open_pos[symbol]
            df = coins[symbol]
            if ts not in df.index:
                continue
            # entry_idx im DataFrame
            try:
                entry_pos = df.index.get_loc(pos["entry_ts"])
                cur_pos = df.index.get_loc(ts)
            except KeyError:
                continue
            if (cur_pos - entry_pos) >= HOLD_DAYS:
                exit_price = df.loc[ts, "close"]
                price_ret = (exit_price / pos["entry_price"] - 1.0) * pos["direction"]
                trade_net = (1 + price_ret) * (1 - FEE_RATE) ** 2 - 1.0
                port_ret = pos["position_frac"] * trade_net * LEVERAGE
                if port_ret <= -0.99:
                    port_ret = -0.99
                pnl = capital * port_ret
                capital += pnl
                trades.append({
                    "symbol": symbol, "entry_time": pos["entry_ts"], "exit_time": ts,
                    "direction": pos["direction"],
                    "entry_price": pos["entry_price"], "exit_price": exit_price,
                    "gross_return": price_ret, "net_return": port_ret,
                    "pnl": pnl, "position_frac": pos["position_frac"],
                })
                del open_pos[symbol]
                if capital > max_cap:
                    max_cap = capital; min_cap_after_peak = capital
                elif capital < min_cap_after_peak:
                    min_cap_after_peak = capital

        # 2) Entries
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

        equity_records.append((ts, capital))

    final = capital
    eq = pd.Series([c for _, c in equity_records], index=[t for t, _ in equity_records])
    max_dd = (eq / eq.cummax() - 1).min()
    cagr = (final / STARTING_CAPITAL_CHF) ** (1 / n_years) - 1

    print(f"=== ERGEBNIS ===")
    print(f"  Endkapital:        {final:.2f} CHF")
    print(f"  Total Return:      {(final/STARTING_CAPITAL_CHF-1)*100:+.2f}%")
    print(f"  CAGR:              {cagr*100:+.2f}% p. a.")
    print(f"  Max Drawdown:      {max_dd*100:+.2f}%")
    print(f"  Anzahl Trades:     {len(trades)}")

    if trades:
        tdf = pd.DataFrame(trades)
        wr = (tdf["net_return"] > 0).mean()
        print(f"  Win-Rate:          {wr*100:.1f}%")
        # Pro Coin
        print(f"\n  Trades pro Pair:")
        for sym in PAIRS:
            sub = tdf[tdf["symbol"] == sym]
            if len(sub) == 0:
                continue
            wr_s = (sub["net_return"] > 0).mean()
            pnl_s = sub["pnl"].sum()
            print(f"    {sym:>8s}: {len(sub):>4d} Trades, WR {wr_s*100:>4.1f}%, "
                  f"PnL {pnl_s:>+10.2f} CHF")

        # Jährliche Returns
        eq_yearly = eq.resample("YE").last().pct_change().dropna()
        print(f"\n  Jährliche Performance:")
        for y, r in eq_yearly.items():
            mark = "✓" if r > 0 else "✗"
            print(f"    {y.year}: {r*100:>+7.2f}% {mark}")
        win_y = (eq_yearly > 0).sum()
        print(f"\n  → {win_y}/{len(eq_yearly)} Jahre profitabel ({win_y/len(eq_yearly)*100:.0f}%)")


if __name__ == "__main__":
    main()
