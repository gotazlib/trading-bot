from sqlalchemy import text
from src.database.connection import engine

DDL = """
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS ohlcv (
    exchange   TEXT        NOT NULL,
    symbol     TEXT        NOT NULL,
    timeframe  TEXT        NOT NULL,
    timestamp  TIMESTAMPTZ NOT NULL,
    open       DOUBLE PRECISION,
    high       DOUBLE PRECISION,
    low        DOUBLE PRECISION,
    close      DOUBLE PRECISION,
    volume     DOUBLE PRECISION,
    PRIMARY KEY (exchange, symbol, timeframe, timestamp)
);

SELECT create_hypertable('ohlcv', 'timestamp', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS funding_rates (
    exchange   TEXT        NOT NULL,
    symbol     TEXT        NOT NULL,
    timestamp  TIMESTAMPTZ NOT NULL,
    rate       DOUBLE PRECISION,
    PRIMARY KEY (exchange, symbol, timestamp)
);

SELECT create_hypertable('funding_rates', 'timestamp', if_not_exists => TRUE);
"""

def main():
    with engine.begin() as conn:
        for statement in DDL.split(";"):
            if statement.strip():
                conn.execute(text(statement))
    print("Datenbank initialisiert: Tabelle 'ohlcv' + Hypertable bereit.")

if __name__ == "__main__":
    main()
