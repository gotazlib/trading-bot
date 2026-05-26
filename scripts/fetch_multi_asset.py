"""Laedt 1h-Daten fuer mehrere Top-Coins seit 2023-01-01."""
from sqlalchemy import text
from src.data.fetcher import fetch_ohlcv_history
from src.data.storage import store_ohlcv
from src.database.connection import engine

EXCHANGE = "binance"
TIMEFRAME = "1h"
START_DATE = "2023-01-01"
SYMBOLS = ["DOGE/USDT", "ADA/USDT", "TRX/USDT", "AVAX/USDT", "LINK/USDT"]


def main():
    for sym in SYMBOLS:
        print(f"\n=== {sym} {TIMEFRAME} ab {START_DATE} ===")
        df = fetch_ohlcv_history(EXCHANGE, sym, TIMEFRAME, START_DATE)
        n = store_ohlcv(df, EXCHANGE, sym, TIMEFRAME)
        print(f"{n} Kerzen verarbeitet (Duplikate ignoriert).")

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT symbol, COUNT(*) AS n, MIN(timestamp) AS von, MAX(timestamp) AS bis
            FROM ohlcv WHERE exchange='binance' AND timeframe='1h'
            GROUP BY symbol ORDER BY symbol
        """)).fetchall()
    print("\n--- 1h-Daten in DB ---")
    for r in rows:
        print(f"  {r.symbol:>10s}  {r.n:>6d}  ({r.von} -> {r.bis})")


if __name__ == "__main__":
    main()
