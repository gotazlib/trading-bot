"""Laedt historische Funding-Rates fuer BTC/USDT von Binance Futures.

Funding-Rate kommt alle 8 Stunden — drei Datenpunkte pro Tag.
Bei 2 Jahren = 2190 Datenpunkte. Schnell.
"""
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd
from sqlalchemy import text

from src.database.connection import engine

EXCHANGE = "binance"
SYMBOL_CCXT = "BTC/USDT:USDT"   # Linear perpetual
SYMBOL_STORE = "BTC/USDT"
DAYS_BACK = 730


INSERT_SQL = text("""
    INSERT INTO funding_rates (exchange, symbol, timestamp, rate)
    VALUES (:exchange, :symbol, :timestamp, :rate)
    ON CONFLICT (exchange, symbol, timestamp) DO NOTHING
""")


def main():
    init_db()
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)).timestamp() * 1000)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

    print(f"Lade Funding-Rates fuer {SYMBOL_CCXT} ab {datetime.fromtimestamp(start_ms/1000, tz=timezone.utc).date()} …")

    all_rows = []
    since = start_ms
    while since < end_ms:
        batch = exchange.fetch_funding_rate_history(SYMBOL_CCXT, since=since, limit=1000)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1]["timestamp"]
        since = last_ts + 1
        print(f"  ... {len(all_rows)} Datenpunkte (bis {pd.to_datetime(last_ts, unit='ms', utc=True)})")
        if len(batch) < 1000:
            break

    if not all_rows:
        print("Keine Daten gefunden.")
        return

    rows = [
        {
            "exchange": EXCHANGE,
            "symbol": SYMBOL_STORE,
            "timestamp": pd.to_datetime(r["timestamp"], unit="ms", utc=True).to_pydatetime(),
            "rate": float(r["fundingRate"]),
        }
        for r in all_rows
    ]
    with engine.begin() as conn:
        conn.execute(INSERT_SQL, rows)

    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT COUNT(*) AS n, MIN(timestamp) AS von, MAX(timestamp) AS bis,
                   MIN(rate) AS rmin, MAX(rate) AS rmax, AVG(rate) AS ravg
            FROM funding_rates WHERE exchange=:e AND symbol=:s
        """), {"e": EXCHANGE, "s": SYMBOL_STORE}).first()
    print(f"\nIn DB: {row.n} Funding-Rates  ({row.von} -> {row.bis})")
    print(f"  Range: min={row.rmin:.6f}, max={row.rmax:.6f}, mean={row.ravg:.6f}")


def init_db():
    """Stellt sicher, dass die funding_rates Tabelle existiert."""
    DDL = """
    CREATE TABLE IF NOT EXISTS funding_rates (
        exchange   TEXT        NOT NULL,
        symbol     TEXT        NOT NULL,
        timestamp  TIMESTAMPTZ NOT NULL,
        rate       DOUBLE PRECISION,
        PRIMARY KEY (exchange, symbol, timestamp)
    );
    """
    SELECT_HT = """
    SELECT create_hypertable('funding_rates', 'timestamp', if_not_exists => TRUE);
    """
    with engine.begin() as conn:
        conn.execute(text(DDL))
        try:
            conn.execute(text(SELECT_HT))
        except Exception as e:
            print(f"(Hypertable bereits konvertiert oder TimescaleDB-fehler: {e})")


if __name__ == "__main__":
    main()
