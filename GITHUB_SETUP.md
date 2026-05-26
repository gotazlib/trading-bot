# GitHub Actions Setup — Paper-Trading-Bot

Der Bot läuft automatisch täglich auf GitHub-Servern (kostenlos), committed Resultate
ins Repo, du siehst sie direkt im Browser.

## Setup in 8 Schritten

### 1. GitHub-Konto erstellen (falls nicht vorhanden)

https://github.com/signup — kostenlos.

### 2. Neues Repository erstellen

- https://github.com/new
- Name: `trading-bot` (oder beliebig)
- **Private** wählen (damit deine Trade-Daten nicht öffentlich sind)
- KEIN README, .gitignore oder License vorinitialisieren

### 3. Lokales Repo initialisieren

Im Terminal:

```bash
cd /Users/gorkem/trading-bot
git init
git branch -M main
git remote add origin https://github.com/DEIN-USERNAME/trading-bot.git
```

(Ersetze `DEIN-USERNAME` mit deinem GitHub-Namen.)

### 4. Files vorbereiten

```bash
# .gitignore prüfen — sollte venv und .env ausschließen
cat .gitignore

# .env Datei sichern (NICHT committen)
mv .env .env.local 2>/dev/null || true
```

### 5. Erstes Commit + Push

```bash
git add .
git commit -m "Initial: Paper-Trading-Bot Setup"
git push -u origin main
```

GitHub fragt dich nach **Authentication**:
- Username: dein GitHub-Username
- Password: **kein Passwort, sondern Personal Access Token** (PAT)

### 6. Personal Access Token erstellen

- https://github.com/settings/tokens/new
- Note: "trading-bot"
- Expiration: 1 Jahr (oder no expiration)
- Scopes: **`repo`** ankreuzen
- Generate Token → **kopieren** (wird nur einmal angezeigt!)
- Bei Git-Push-Frage als Password einfügen

Token speichern (z.B. in Keychain):
```bash
git config --global credential.helper osxkeychain
```

### 7. GitHub Actions aktivieren

Actions ist standardmäßig aktiv bei neuen Repos. Verifizieren:
- Repo auf github.com öffnen → Tab **Actions**
- Falls Hinweis "Workflows are disabled": auf "I understand my workflows, enable them" klicken

### 8. Manuell ersten Lauf triggern

- Repo → Tab **Actions**
- Workflow **"Paper-Trading Bot"** auswählen
- Button **"Run workflow"** → grüner Run-Button
- Warten ~30 Sekunden → Lauf erscheint in der Liste
- Klick auf Run → sieh Logs in Echtzeit

## Wo siehst du die Resultate?

### Direkt im GitHub-Browser

```
https://github.com/DEIN-USERNAME/trading-bot/tree/main/results
```

Klick auf:
- **`paper_report.xlsx`** → Download-Knopf rechts oben → in Numbers/Excel öffnen
- **`paper_report.html`** → siehe Tipp unten
- **`paper_trades.csv`** → wird direkt im Browser als Tabelle angezeigt
- **`paper_log.txt`** → Console-Output des letzten Runs

### HTML-Dashboard im Browser

GitHub zeigt HTML als raw text. Du hast 2 Optionen:

**A) HTMLPreview-Service**:
Öffne diese URL (ersetze DEIN-USERNAME und ggf. branch):
```
https://htmlpreview.github.io/?https://github.com/DEIN-USERNAME/trading-bot/blob/main/results/paper_report.html
```

**B) Lokal mit aktuellem Stand**:
```bash
cd /Users/gorkem/trading-bot
git pull
open results/paper_report.html
```

**C) GitHub Pages aktivieren** (öffentlich, dauerhafte URL):
- Repo Settings → Pages → Source: `main` Branch, `/` (root)
- Speichern → nach 1 Min ist HTML erreichbar unter:
  `https://DEIN-USERNAME.github.io/trading-bot/results/paper_report.html`

### Email-Benachrichtigung bei Run

GitHub schickt dir Mail wenn ein Workflow fehlschlägt. Standardmäßig aktiv.

Falls du Mail auch bei Erfolg willst:
- GitHub Settings → Notifications → Actions → "Send me notifications for: All workflow runs"

## Wann läuft der Bot?

Workflow-Datei `.github/workflows/paper_trading.yml`:
- **Schedule**: Werktags 22:00 UTC
  - 23:00 Schweizer Zeit Winter (CET)
  - 00:00 Schweizer Zeit Sommer (CEST)
- **Manuell** jederzeit triggerbar via "Run workflow"

## Bot-Aktualisierungen pushen

Wenn du den Code änderst (z.B. Williams-Parameter):

```bash
cd /Users/gorkem/trading-bot
git add scripts/paper_trading.py
git commit -m "Adjust strategy parameters"
git push
```

Der nächste Cron-Run nutzt automatisch den neuen Code.

## Reset des Paper-Trading-State

Falls du von vorne anfangen willst:

```bash
git rm results/paper_*.json results/paper_*.csv results/paper_*.html results/paper_*.xlsx 2>/dev/null
git commit -m "Reset paper trading state"
git push
```

Beim nächsten Run wird neu initialisiert.

## Troubleshooting

### "git push" verlangt Login

Personal Access Token (Schritt 6) ist abgelaufen oder fehlt. Neuen erstellen.

### Workflow läuft nicht

- Repo Actions Tab prüfen — Workflow muss "enabled" sein
- Repo darf nicht "leer" sein (mind. 1 Commit)
- Cron-Schedule braucht 1-2 Tage um anzulaufen bei neuen Repos

### "yfinance" Error in Logs

Manchmal sind Forex-Symbole temporär nicht erreichbar. Bot retry-t automatisch beim
nächsten Lauf. Wenn dauerhaft: Symbole in `scripts/paper_trading.py` checken.

### State wurde nicht aktualisiert

Schau in `paper_log.txt` ob der Bot durchgelaufen ist. Falls "git push" failed:
in GitHub Actions Logs nach Permission-Error suchen → `permissions: contents: write`
in Workflow-File muss da sein.

## Kosten

- **GitHub-Konto**: kostenlos
- **Private Repos**: kostenlos
- **GitHub Actions**: 2.000 Minuten/Monat kostenlos. Bot braucht <1 Min pro Run → unlimitiert
- **GitHub Pages**: kostenlos

Komplett kostenlos.

## Sicherheit

- Repo auf **Private** stellen — sonst sind deine Trade-Daten öffentlich sichtbar
- `.env` Datei NIE committen (steht in `.gitignore`)
- Personal Access Token NIE im Code speichern

## Nächste Schritte

Nach 4-8 Wochen Live-Daten:
- HTML-Report durchschauen: Performance, Win-Rate, MaxDD
- Vergleichen mit Backtest-Erwartung (+9-22 % p. a., 60-66 % WR)
- Wenn ähnlich → Live-Trading mit Dukascopy/Swissquote

Bei Fragen: Issues im Repo aufmachen → ich kann lesen und antworten.
