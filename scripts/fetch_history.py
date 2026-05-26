from sqlalchemy import text
from src.data.fetcher import fetch_ohlcv_history
from src.data.storage import store_ohlcv
from src.database.connection import engine

EXCHANGE = "binance"
SYMBOLS = ["BTC/USDT"]
TIMEFRAMES = ["1h"]
START_DATE = "2023-01-01"

def main():
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            print(f"\n=== {symbol} {timeframe} ab {START_DATE} ===")
            df = fetch_ohlcv_history(EXCHANGE, symbol, timeframe, START_DATE)
            n = store_ohlcv(df, EXCHANGE, symbol, timeframe)
            print(f"{n} Kerzen gespeichert.")

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT symbol, timeframe, COUNT(*) AS n,
                   MIN(timestamp) AS von, MAX(timestamp) AS bis
            FROM ohlcv
            GROUP BY symbol, timeframe
            ORDER BY symbol, timeframe
        """)).fetchall()

    print("\n--- Inhalt der Datenbank ---")
    for r in rows:
        print(f"{r.symbol} {r.timeframe}: {r.n} Kerzen  ({r.von}  ->  {r.bis})")

if __name__ == "__main__":
    main()
