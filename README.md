# Barkeeper – ein kleines neuronales Netz zum Verstehen

Ein Barkeeper, der Sätze wie "ein bier", "kein wein" oder "zahlen bitte" versteht und dazulernt.
Kein echtes LLM, sondern ein neuronales Netz mit knapp 4000 Gewichten. Es nutzt dieselben
Grundideen wie ein LLM (Zahlen, Gewichte, Softmax, Gradientenabstieg), ist aber klein genug,
um jeden Schritt nachzuvollziehen.

## Dateien

| Datei | Zweck |
|---|---|
| `barkeeper.py` | Erste Version: eine Matrix (Wörter → Antwort) |
| `barkeeper2.py` | Zwei Schichten (Wörter → Konzepte → Antwort), Konsole |
| `barkeeper_server.py` | Webserver mit Chat, Teachen und Entscheidungs-Log |
| `barkeeper_chat.html` | Browser-Oberfläche (wird vom Server ausgeliefert) |
| `barkeeper_seed.json` | Antworten, Preise, Kontexte und Start-Beispiele |
| `barkeeper2.db` | SQLite: Trainingsbeispiele und trainiertes Modell (als JSON) |

Voraussetzung: Python 3 mit `numpy`. Der Webserver nutzt sonst nur die Standardbibliothek.

## Starten

Browser-Chat:

```
python3 barkeeper_server.py
```

Dann http://localhost:8765 öffnen (nicht die HTML-Datei direkt, sonst erreicht sie den Server nicht).

Konsole:

```
python3 barkeeper2.py
```

## Bedienung

| Eingabe | Wirkung |
|---|---|
| `ein bier`, `kein wein`, `zahlen bitte` | normal bestellen |
| `noch eins = bier` | direkt beibringen (Kontext = aktueller Kontext) |
| `[wein] noch eins = wein` | beibringen mit explizitem Kontext |
| `falsch` | letzte Antwort korrigieren (Liste, Nummer oder neue Antwort wählen) |
| `skip` | Teach-Frage überspringen |
| `reset` | neuer Gast (Kontext und Rechnung zurücksetzen) |

Versteht der Barkeeper einen Satz nicht ("Hä? Wie bitte?"), fragt er automatisch, was er hätte tun sollen.

Nur in der Konsole: `data` (Trainingsdaten prüfen), `vergleich` (eine Matrix gegen zwei Schichten),
`log` (Entscheidungs-Log an/aus), `dashboard` (HTML-Übersicht nach `/tmp/barkeeper_dashboard.html`).

Im Browser steht unter jeder Antwort ein aufklappbares Panel
"Wie hat der Barkeeper entschieden?" mit den aktivsten Konzepten, ihren Treibern und den Stimmen.

## Wie es funktioniert

### 1. Text wird zu Zahlen
Aus "ein bier" im Kontext `start` werden die Merkmale `ein`, `bier`, `[start]`.
Jedes der bekannten Merkmale (aktuell 133) bekommt eine 1, wenn es vorkommt, sonst 0.
Dieser Zahlenvektor ist die Eingabe `x`.

### 2. Zwei Schichten

```
h = tanh(W1 · x + b1)       Schicht 1: Wörter + Kontext → 24 Konzepte
p = softmax(W2 · h + b2)    Schicht 2: Konzepte → Wahrscheinlichkeit je Antwort
```

- `W1` ist eine 24 × 133 Tabelle. Jedes Konzept summiert gewichtete Stimmen der aktiven Wörter.
- `tanh` drückt die Summe auf einen Wert zwischen -1 und +1 (die "Aktivierung" im Log).
- `W2` sammelt für jede Antwort die Stimmen der Konzepte. `softmax` macht daraus Prozente (Summe 100 %).
- Die höchste Antwort gewinnt. Unter 40 % (oder bei zu vielen unbekannten Wörtern) fragt der Barkeeper nach.

### 3. Warum zwei Schichten?
Eine einzelne Matrix addiert nur unabhängige Wortstimmen: `kein` und `bier` können nicht zusammenwirken,
"kein bier" würde weiter für Bier stimmen. Ein verstecktes Konzept kann nur dann ausschlagen, wenn
`kein` und `bier` zusammen vorkommen, und dann für Wein stimmen. Der Konsolenbefehl `vergleich` zeigt den Unterschied.

### 4. Training
Wiederholt (1500 Runden, über alle Beispiele gleichzeitig):

1. Antwort raten.
2. Fehler messen: `P - Y` (Vermutung minus richtige Antwort).
3. Schuld rückwärts an jedes Gewicht verteilen (Backpropagation).
4. Jedes Gewicht ein Stück gegen seine Schuld verschieben (Gradientenabstieg, mit Momentum).

### 5. Wie der Fehler die richtige Stelle in der Matrix findet
Jedes Gewicht hat eine Adresse (Ausgang `i`, Eingang `j`):

```
Änderung von W[i,j]  =  Fehler an Ausgang i  ×  Wert von Eingang j
```

Im Code: `D2.T @ Hd` für `W2` und `D1.T @ X` für `W1`.

- **Schicht 2:** Gradient = Fehler der Antwort × Aktivierung des Konzepts. Aktive Konzepte, die zur richtigen Antwort passen, stimmen künftig stärker dafür.
- **Weitergabe an Schicht 1:** `D1 = (D2 @ W2) * (1 - Hd**2)`.
  `D2 @ W2` fragt jedes Konzept, wie viel es zur falschen Antwort beigetragen hat.
  `(1 - Hd**2)` ist die Steigung von `tanh`; ein Konzept, das schon bei ±1 klebt, ändert sich kaum.
- **Schicht 1:** Gradient = Schuld des Konzepts × `x[j]`. Nur Wörter, die im Satz vorkamen (`x[j] = 1`), werden verändert; alle anderen Spalten bekommen genau 0.

Ein Beispiel verschiebt also nur die Spalten seiner eigenen Wörter, und zwar wenig (gemittelt über alle Beispiele).
"bier" und "kein bier" schieben die Spalte `bier` in entgegengesetzte Richtungen. Dadurch müssen sich Konzepte
auf die Kombination spezialisieren.

### 6. Woher kommen die Konzepte?
Niemand entwirft sie. Jedes der 24 Konzepte ist eine Zeile in `W1`, die mit Zufallszahlen startet
(`np.random.randn`, fester Seed = wiederholbar). Das Training formt aus dem Zufall Spezialisten.

- Zufälliger Start ist nötig, weil sonst alle Zeilen denselben Gradienten bekämen und für immer identische Kopien blieben.
- Die Bedeutung ("Bier bestellen") ist unsere Interpretation anhand der treibenden Wörter im Log. Das Modell kennt nur Nummern.
- Die Nummern sind willkürlich und können sich nach dem Neutrainieren verschieben. Ein Konzept mischt oft mehrere Muster.
- Festgelegt sind nur die Anzahl (`HIDDEN = 24`) und die Rechenform (`tanh`).

### 7. Teachen
`learn()` löscht ein altes Beispiel mit gleichem Kontext und Satz, fügt das neue ein und trainiert alles neu
(gleicher Seed). Die Spalten des neuen Satzes werden verschoben, bis er stimmt; die übrigen Beispiele halten den Rest in Position.

## Mögliche Weiterentwicklung

```
Angular (Chat-UI, Log-Panel, Dashboard)
   │  REST/JSON
NestJS (Gateway: Auth, Sessions, Validierung, Rate-Limiting)
   │  REST/JSON (nur intern)
Python-Kern (FastAPI): Modell, Training, Teachen, Erklärung, SQLite
```

Sinnvolle Reihenfolge:

1. Modellcode in ein importierbares `barkeeper_core.py` auslagern (aktuell ist er in Konsole und Server doppelt).
2. FastAPI mit Zustand pro Sitzung (aktuell teilen sich alle Browser-Tabs einen Gast).
3. Angular-Oberfläche gegen diese API.
4. NestJS-Gateway, sobald Login oder öffentliches Hosting nötig sind.
