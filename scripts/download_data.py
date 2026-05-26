import ccxt
import pandas as pd
from datetime import datetime

# --- Konfiguration ---
EXCHANGE = "binance"
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
LIMIT = 1000  # ca. 41 Tage stündliche Kerzen in einem Abruf

def download():
    exchange = getattr(ccxt, EXCHANGE)()
    print(f"Lade {LIMIT} {TIMEFRAME}-Kerzen für {SYMBOL} von {EXCHANGE} ...")

    # Öffentliche Marktdaten – kein API-Key nötig
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=LIMIT)

    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    # Millisekunden-Timestamp in lesbares Datum umwandeln
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    filename = f"data/{SYMBOL.replace('/', '_')}_{TIMEFRAME}.csv"
    df.to_csv(filename, index=False)

    print(f"\nGespeichert: {filename}  ({len(df)} Zeilen)")
    print(f"Zeitraum: {df['datetime'].min()}  bis  {df['datetime'].max()}\n")
    print("Letzte 5 Kerzen:")
    print(df[["datetime", "open", "high", "low", "close", "volume"]].tail())

if __name__ == "__main__":
    download()