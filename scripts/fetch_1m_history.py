"""Laedt 1m-Kerzen fuer BTC/USDT seit START_DATE in TimescaleDB.

Bei 180 Tagen sind das ~260k Kerzen / ~260 API-Requests.
"""
from datetime import datetime, timedelta, timezone
from sqlalchemy import text

from src.data.fetcher import fetch_ohlcv_history
from src.data.storage import store_ohlcv
from src.database.connection import engine

EXCHANGE = "binance"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
DAYS_BACK = 180


def main():
    start = (datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)).date().isoformat()
    print(f"=== {SYMBOL} {TIMEFRAME} ab {start} ({DAYS_BACK} Tage) ===")
    df = fetch_ohlcv_history(EXCHANGE, SYMBOL, TIMEFRAME, start)
    n = store_ohlcv(df, EXCHANGE, SYMBOL, TIMEFRAME)
    print(f"\n{n} Kerzen verarbeitet (Duplikate ignoriert).")

    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT COUNT(*) AS n, MIN(timestamp) AS von, MAX(timestamp) AS bis
            FROM ohlcv
            WHERE exchange=:e AND symbol=:s AND timeframe=:tf
        """), {"e": EXCHANGE, "s": SYMBOL, "tf": TIMEFRAME}).first()
    print(f"In DB: {row.n} 1m-Kerzen  ({row.von} -> {row.bis})")


if __name__ == "__main__":
    main()
