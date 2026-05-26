"""Laedt 20 Jahre Forex Daily-Daten via yfinance (2005-2026)."""
import pandas as pd
import yfinance as yf
from sqlalchemy import text

from src.data.storage import store_ohlcv
from src.database.connection import engine

EXCHANGE = "forex"
TIMEFRAME = "1d"
START_DATE = "2005-01-01"

PAIRS = {
    "USD/CHF": "CHF=X",
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
    "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X",
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
            FROM ohlcv WHERE exchange='forex' AND timeframe='1d'
            GROUP BY symbol ORDER BY symbol
        """)).fetchall()
    print("\n--- Forex 1d in DB (20-Jahre-Ziel) ---")
    for r in rows:
        years = (pd.Timestamp(r.bis) - pd.Timestamp(r.von)).days / 365
        print(f"  {r.symbol:>10s}: {r.n:>5d} Bars, {years:.1f} Jahre ({r.von.year}-{r.bis.year})")


if __name__ == "__main__":
    main()
