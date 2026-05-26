import ccxt
import pandas as pd
from sqlalchemy import text
from src.database.connection import engine

EXCHANGE = "binance"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
LIMIT = 1000

INSERT_SQL = text("""
    INSERT INTO ohlcv (exchange, symbol, timeframe, timestamp,
                       open, high, low, close, volume)
    VALUES (:exchange, :symbol, :timeframe, :timestamp,
            :open, :high, :low, :close, :volume)
    ON CONFLICT (exchange, symbol, timeframe, timestamp) DO NOTHING
""")

def fetch():
    exchange = getattr(ccxt, EXCHANGE)()
    print(f"Lade {LIMIT} {TIMEFRAME}-Kerzen für {SYMBOL} ...")
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=LIMIT)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

def store(df):
    rows = [
        {
            "exchange": EXCHANGE, "symbol": SYMBOL, "timeframe": TIMEFRAME,
            "timestamp": r.timestamp.to_pydatetime(),
            "open": r.open, "high": r.high, "low": r.low,
            "close": r.close, "volume": r.volume,
        }
        for r in df.itertuples(index=False)
    ]
    with engine.begin() as conn:
        conn.execute(INSERT_SQL, rows)
    return len(rows)

def main():
    df = fetch()
    n = store(df)
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM ohlcv")).scalar()
    print(f"{n} Kerzen verarbeitet (Duplikate ignoriert). Gesamt in DB: {total}")

if __name__ == "__main__":
    main()
