from sqlalchemy import create_engine
from config.settings import DATABASE_URL

# Die "engine" ist unser Tor zur Datenbank.
# pool_pre_ping prüft vor jeder Nutzung, ob die Verbindung noch lebt.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
