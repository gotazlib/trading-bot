"""Laedt USD/CHF Intraday-Daten via yfinance — soviel wie kostenlos verfuegbar."""
import pandas as pd
import yfinance as yf
from sqlalchemy import text
from src.data.storage import store_ohlcv
from src.database.connection import engine

EXCHANGE = "forex"
SYMBOL = "USD/CHF"

# yfinance Limits: 15m → 60d, 1h → 2y, 5m → 30d
CONFIGS = [
    ("15m", "15m", "60d"),
    ("1h",  "1h",  "730d"),
]


def main():
    for tf_label, yf_interval, yf_period in CONFIGS:
        print(f"\n=== USD/CHF {tf_label} (period={yf_period}) ===")
        raw = yf.download("CHF=X", interval=yf_interval, period=yf_period,
                          progress=False, auto_adjust=False)
        if raw.empty:
            print("  Keine Daten!")
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
        n = store_ohlcv(df, EXCHANGE, SYMBOL, tf_label)
        print(f"  {n} gespeichert.")

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT timeframe, COUNT(*) AS n, MIN(timestamp) AS von, MAX(timestamp) AS bis
            FROM ohlcv WHERE exchange='forex' AND symbol='USD/CHF'
            GROUP BY timeframe ORDER BY timeframe
        """)).fetchall()
    print("\n--- USD/CHF in DB ---")
    for r in rows:
        print(f"  {r.timeframe:>4s}: {r.n:>6d} Bars  ({r.von} -> {r.bis})")


if __name__ == "__main__":
    main()
