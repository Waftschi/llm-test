# Barkeeper – ein kleines neuronales Netz zum Verstehen

Ein Barkeeper, der Sätze wie "ein bier", "kein wein" oder "zahlen bitte" versteht und dazulernt.
Kein echtes LLM, sondern ein neuronales Netz mit knapp 4000 Gewichten. Es nutzt dieselben
Grundideen wie ein LLM (Zahlen, Gewichte, Softmax, Gradientenabstieg), ist aber klein genug,
um jeden Schritt nachzuvollziehen.

English version (default): [README.md](README.md)

**Kurzanleitung für Kolleginnen und Kollegen** (in 20–30 Minuten erklärbar): [docs/README.de.md](docs/README.de.md)

**Präsentation:** [Deutsch](https://waftschi.github.io/llm-test/praesentation/presentation.html) · [Englisch](https://waftschi.github.io/llm-test/praesentation_en/presentation.html) (Folien, die das Netz Schritt für Schritt erklären)

## Dateien

| Datei | Zweck |
|---|---|
| `barkeeper_core.py` | Das Modell: zwei Schichten (Wörter → Konzepte → Antwort), Training, Teachen, Entscheidungs-Log und Blame als Daten. Kein `print`/`input` |
| `barkeeper2.py` | Konsole auf dem Kern |
| `server.py` | Web-UI: kleine JSON-API (nur Standardbibliothek), die auch `web/index.html` ausliefert |
| `web/index.html` | Die Webseite (reines HTML/JS, kein Build): links der Chat, rechts das Innenleben des Netzes |
| `show_matrix.py` | Zeigt die gelernten Gewichte (Wörter, Konzepte, Antworten, CSV) |
| `barkeeper_seed.json` | Antworten, Preise, Kontexte, Start-Beispiele und `fallback` (Antwort und Mindest-Sicherheit, wenn nichts verstanden wird) |
| `barkeeper_simple_seed.json`, `barkeeper_simple_en_seed.json` | Mini-Modelle auf Deutsch und Englisch (3 Antworten, 4 Konzepte). Enthalten auch `suggestions`: Beispielsätze als Buttons in der Web-UI und als "Try:" in der Konsole |
| `barkeeper2.db`, `barkeeper_simple*.db` | SQLite: Trainingsbeispiele und trainiertes Modell (als JSON), eine Datenbank pro Profil. Wird beim ersten Start erzeugt, liegt nicht im Repo |
| `praesentation/`, `praesentation_en/` | Folien auf Deutsch und Englisch (`build_presentation.py` erzeugt sie) |

Voraussetzung: Python 3 mit `numpy`. Sonst wird nur die Standardbibliothek genutzt.

```
pip3 install -r requirements.txt
```

## Starten

### Profile (Sprache und Größe)

| Profil | Modell | Sprache |
|---|---|---|
| *(keins)* | großes Modell, 24 Konzepte | Deutsch |
| `simple` | Mini-Modell, 4 Konzepte (siehe README_SIMPLE.md) | Deutsch |
| `simple-en` | Mini-Modell, 4 Konzepte | Englisch |

Jedes Profil hat eine eigene Seed-Datei und eine eigene Datenbank, sie vermischen sich nie. Das Profil wird als Argument oder über die Umgebungsvariable `BARKEEPER_PROFILE` angegeben.

### Konsole

```
python3 barkeeper2.py              # großes Modell
python3 barkeeper2.py simple       # Mini-Modell, Deutsch
python3 barkeeper2.py simple-en    # Mini-Modell, Englisch
python3 show_matrix.py             # gelernte Matrix ansehen
```

### Web-UI

```
python3 server.py                  # http://localhost:8000
python3 server.py simple-en        # Profil als Argument
python3 server.py simple-en 8080   # und/oder anderer Port
```

- **Links:** der Chat wie in der Konsole: Beispiel-Buttons, `falsch`, `reset`, `Satz = Antwort`. Versteht der Barkeeper etwas nicht, bietet ein Lehrer-Panel die bekannten Antworten an (oder eine neue eintippen).
- **Rechts (Tabs):** *Decision* (das Log durch beide Schichten, mit rotem Banner für unbekannte Wörter), *Blame* (beim Teachen), *Parameters* (Größen, Lernrate, Momentum, Runden, Seed, Fallback-Schwelle, Antworten, Vokabular), *Training data* und *1 matrix vs 2 layers*.
- Gelerntes gilt für alle Browser-Tabs (ein Modell, eine Datenbank). Nur Kontext und Rechnung sind pro Tab getrennt.

Noch nicht in der Web-UI: ein Tab mit den Konzept-Erklärungen aus den Folien.

## Bedienung

| Eingabe | Wirkung |
|---|---|
| `ein bier`, `kein wein`, `zahlen bitte` | normal bestellen |
| `noch eins = bier` | direkt beibringen (Kontext = aktueller Kontext). Ein Beispiel pro Zeile, nur ein `=`: `lambrusco = wein` und `kein lambrusco = bier` als zwei getrennte Zeilen eingeben |
| `[wein] noch eins = wein` | beibringen mit explizitem Kontext |
| `falsch` | letzte Antwort korrigieren (Liste, Nummer oder neue Antwort wählen) |
| `skip` | Teach-Frage überspringen |
| `reset` | neuer Gast (Kontext und Rechnung zurücksetzen) |

Versteht der Barkeeper einen Satz nicht ("Hä? Wie bitte?"), fragt er automatisch, was er hätte tun sollen.

Weitere Befehle: `data` (Trainingsdaten prüfen), `vergleich` (eine Matrix gegen zwei Schichten),
`log` (Entscheidungs-Log an/aus).

Das Entscheidungs-Log steht unter jeder Antwort und zeigt die 4 Konzepte mit der größten Stimme für die gewählte Antwort, ihre Treiber,
die Stimmen, die Summen, tanh und die Softmax-Rechnung.

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

Heute: `barkeeper_core.py` (Modell) → Konsole und `server.py` + `web/index.html` (reines JS). Die API bleibt gleich, wenn die Schichten wachsen:

```
Angular (Chat-UI, Log-Panel, Dashboard)        ← ersetzt web/index.html, wenn die UI für reines JS zu groß wird
   │  REST/JSON
NestJS (Gateway: Auth, Sessions, Validierung, Rate-Limiting)   ← erst bei Login oder öffentlichem Hosting
   │  REST/JSON (nur intern)
Python-Kern: barkeeper_core.py + server.py
```

Ideen: ein Konzepte-Tab mit den Erklärungen aus den Folien, Trainingsdaten in der UI bearbeiten, Sprachumschalter in der UI.

## Lizenz

[MIT](LICENSE): frei nutzbar, veränderbar und weitergebbar, solange der Copyright-Hinweis erhalten bleibt.
