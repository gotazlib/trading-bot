"""Laedt 5 ZUSAETZLICHE Forex-Pairs (Cross/Crosses) via yfinance seit 2010."""
import pandas as pd
import yfinance as yf
from sqlalchemy import text

from src.data.storage import store_ohlcv
from src.database.connection import engine

EXCHANGE = "forex"
TIMEFRAME = "1d"
START_DATE = "2010-01-01"

# yfinance Symbole fuer 5 weitere Pairs (Diversifikation)
PAIRS = {
    "USD/CAD": "CAD=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "EUR/GBP": "EURGBP=X",
}


def main():
    for pair_name, yf_symbol in PAIRS.items():
        print(f"\n=== {pair_name} ({yf_symbol}) ab {START_DATE} ===")
        raw = yf.download(yf_symbol, start=START_DATE, progress=False, auto_adjust=False)
        if raw.empty:
            print("  Keine Daten erhalten.")
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.copy()
        df.index.name = "timestamp"
        df = df.reset_index()
        df = df.rename(columns={c: c.lower() for c in df.columns})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df["volume"] = df["volume"].fillna(0).astype(float)
        df = df.dropna(subset=["close"])
        print(f"  {len(df)} Bars von {df['timestamp'].min()} bis {df['timestamp'].max()}")
        n = store_ohlcv(df, EXCHANGE, pair_name, TIMEFRAME)
        print(f"  {n} gespeichert (Duplikate ignoriert).")

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT symbol, COUNT(*) AS n, MIN(timestamp) AS von, MAX(timestamp) AS bis
            FROM ohlcv WHERE exchange='forex'
            GROUP BY symbol ORDER BY symbol
        """)).fetchall()
    print("\n--- Forex in DB ---")
    for r in rows:
        print(f"  {r.symbol:>10s}: {r.n:>5d} Bars  ({r.von} -> {r.bis})")


if __name__ == "__main__":
    main()
