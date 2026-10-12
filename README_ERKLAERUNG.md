# Wie der Barkeeper lernt – die Erklärung

Diese Datei erklärt Schritt für Schritt, was in `barkeeper2.py` passiert. Die Bedienung steht in `README.md`.

## Vorab: Das ist kein LLM

Der Barkeeper ist ein kleines neuronales Netz mit 3.891 Gewichten, das einen von 27 Antworten auswählt.
Ein LLM funktioniert nach denselben Prinzipien (Zahlen, Gewichte, Softmax, Gradientenabstieg),
hat aber Milliarden Gewichte, sagt das nächste Wort voraus und lernt aus riesigen Textmengen.
Gerade weil unser Netz so klein ist, kann man jede Zahl ansehen.

## 1. Text wird zu Zahlen (`features`, `to_matrix`)

"ein bier" im Kontext `start` ergibt die Merkmale `ein`, `bier`, `[start]`.
Jedes der bekannten Merkmale (aktuell 133) bekommt eine 1, wenn es vorkommt, sonst eine 0.
Diese Liste aus 133 Zahlen ist die Eingabe `x`.

## 2. Schicht 1: Wörter werden zu Konzepten (`forward`)

```
h = tanh(W1 · x + b1)
```

- `W1` ist eine Tabelle mit 24 Zeilen (Konzepte) und 133 Spalten (Wörter).
- Jedes Konzept addiert die gewichteten Stimmen der aktiven Wörter.
- `tanh` drückt die Summe auf einen Wert zwischen -1 und +1. Das ist die "Aktivierung" im Log.

## 3. Schicht 2: Konzepte werden zu einer Antwort

```
p = softmax(W2 · h + b2)
```

- `W2` hat 27 Zeilen (Antworten) und 24 Spalten (Konzepte).
- Jede Antwort sammelt die Stimmen der Konzepte.
- `softmax` macht daraus Prozente, die zusammen 100 % ergeben. Die höchste Antwort gewinnt.
- Unter 40 % (oder bei zu vielen unbekannten Wörtern) sagt der Barkeeper "Hä? Wie bitte?" und fragt nach.

## 4. Warum zwei Schichten?

Eine einzelne Matrix addiert nur unabhängige Wortstimmen. `kein` und `bier` können dort nicht zusammenwirken,
"kein bier" würde weiter für Bier stimmen.

Ein verstecktes Konzept kann so gelernt werden, dass es nur ausschlägt, wenn `kein` **und** `bier` zusammen vorkommen,
und dann für Wein stimmt. Der Konsolenbefehl `vergleich` zeigt das: Das Ein-Matrix-Modell scheitert an neuen
"kein"-Sätzen, das Zwei-Schichten-Modell nicht.

## 5. Training (`train`)

Alle Beispiele (Satz + Kontext → richtige Antwort) werden gleichzeitig verarbeitet. Diese Schleife läuft 1500-mal:

1. **Raten:** Das Netz berechnet für jedes Beispiel `P`.
2. **Fehler messen:** `P - Y` (Vermutung minus richtige Antwort).
3. **Schuld verteilen:** Der Fehler wird rückwärts an jedes Gewicht weitergegeben (Backpropagation).
4. **Korrigieren:** Jedes Gewicht wird ein Stück gegen seine Schuld verschoben (Gradientenabstieg).
   Momentum behält einen Teil des letzten Schritts bei, damit das Lernen schneller "bergab rollt".

## 6. Wie der Fehler die richtige Stelle in der Matrix findet

Jedes Gewicht hat eine Adresse: (Ausgang `i`, Eingang `j`). Die Regel lautet:

```
Änderung von W[i,j]  =  Fehler an Ausgang i  ×  Wert von Eingang j
```

Im Code sind das die Zeilen `D2.T @ Hd` (für `W2`) und `D1.T @ X` (für `W1`).
Das sind Matrixprodukte, die diese Rechnung für alle Adressen auf einmal ausführen.

Ein Gewicht ändert sich also nur, wenn **beides** zutrifft:
der Fehler kommt an seinem Ausgang an, **und** sein Eingang war aktiv.

### Kleines Beispiel: "ein bier", richtige Antwort = bier

Zur Veranschaulichung hat Schicht 2 hier nur drei Antworten (gruss, bier, wein) und zwei Konzepte.
Die Aktivierungen sind `h = [+0.9, -0.5]`.

1. Der Barkeeper rät `P = [0.2, 0.3, 0.5]`. Das Ziel ist `Y = [0, 1, 0]`.
2. Fehler `P - Y = [+0.2, -0.7, +0.5]`.
   Negativ bedeutet "diese Antwort müsste höher sein", positiv "niedriger".
3. Gradient für `W2[bier, Konzept 1]` = `-0.7 × 0.9 = -0.63`.
   Das Update zieht den Gradienten ab, das Gewicht **steigt**.
   Konzept 1 ist aktiv und positiv, stimmt also künftig stärker für bier.
4. Gradient für `W2[bier, Konzept 2]` = `-0.7 × -0.5 = +0.35`, das Gewicht **sinkt**.
   Konzept 2 ist negativ, und ein kleineres Gewicht mal ein negativer Wert ergibt einen höheren Bier-Wert.
   Auch hier wird die richtige Antwort gestärkt.
5. Die Zeile wein hat Fehler +0.5. Konzepte, die für wein gestimmt haben, werden schwächer.

### Den Fehler an Schicht 1 weitergeben

```
D1 = (D2 @ W2) * (1 - Hd**2)
```

- `D2 @ W2` fragt jedes Konzept: "Wie viel hast du zur falschen Antwort beigetragen?"
  Es summiert über alle Antworten den Ausgangsfehler mal das Gewicht des Konzepts in diese Antwort.
- `(1 - Hd**2)` ist die Steigung von `tanh`.
  Ein Konzept, das schon bei ±1 klebt, ändert sich kaum. Ein Konzept nahe 0 ändert sich leicht.

Danach bekommt `W1[k, j]` den Wert `D1[k] × x[j]`. Das ist der Adress-Schritt:

- Die **Zeile** `k` ist das Konzept, dem die Schuld gegeben wurde.
- Die **Spalte** `j` ist ein Wort.
- Wörter im Satz (`x[j] = 1`) werden verändert.
- Wörter, die nicht im Satz stehen (`x[j] = 0`), haben den Gradienten genau 0 und bleiben unberührt.

"ein bier" verschiebt also nur die Spalten `ein`, `bier` und `[start]`.
Die Gewichte für `wein`, `zahlen` und alle anderen Wörter lässt dieses Beispiel in Ruhe.

### Warum das mit vielen Beispielen funktioniert

Jedes Beispiel schiebt nur seine eigenen Spalten, und nur ein kleines Stück (gemittelt über alle Beispiele durch `/ len(examples)`).
Das wird 1500-mal wiederholt. "bier" und "kein bier" schieben die Spalte `bier` in entgegengesetzte Richtungen.
Deshalb müssen sich einzelne Konzepte auf die Kombination `kein` + `bier` spezialisieren.

## 7. Woher kommen die Konzepte?

Niemand entwirft sie. Sie entstehen im Training.

- **Am Anfang sind sie Zufall.** Jedes der 24 Konzepte ist eine Zeile in `W1`, die mit Zufallszahlen startet:
  `np.random.randn(HIDDEN, len(vocab)) * 0.5`. Der feste Seed (`np.random.seed(0)`) macht das Training wiederholbar.
- **Das Training formt sie.** Eine Zeile, die zufällig schon etwas bei `bier` + `ein` ausschlägt, wird dafür weiter verstärkt
  und stimmt dann für bier. Eine andere wird von `kein` + `bier` geformt und stimmt für wein.
- **Warum zufällig starten?** Würden alle 24 Zeilen bei 0 beginnen, bekämen sie bei jedem Schritt denselben Gradienten
  und blieben für immer identische Kopien. Der Zufall bricht diese Symmetrie. Das Ein-Matrix-Modell darf bei 0 starten,
  weil es keine versteckte Schicht hat.
- **Die Namen sind unsere Interpretation.** Das Modell kennt nur "Konzept 15". Dass es "Bier bestellen" bedeuten könnte,
  lesen wir im Log an den treibenden Wörtern ab.
- **Die Nummern sind willkürlich** und können sich nach dem Neutrainieren verschieben.
  Ein Konzept mischt oft mehrere Muster, und mehrere Konzepte teilen sich eine Aufgabe.
- **Festgelegt sind nur** die Anzahl (`HIDDEN = 24`) und die Rechenform (`tanh`).
  Zu wenige Konzepte, und Kombinationen wie "kein bier" lassen sich nicht trennen.
  Zu viele, und das Netz lernt die Beispiele eher auswendig.

## 8. Teachen (`learn`)

Wenn du `noch eins = bier` eingibst oder `falsch` korrigierst:

1. Ein altes Beispiel mit gleichem Kontext und Satz wird gelöscht.
2. Das neue Beispiel kommt in die Datenbank.
3. Alles wird mit demselben Seed neu trainiert.
4. Die Spalten des neuen Satzes werden verschoben, bis er stimmt. Die übrigen Beispiele halten den Rest an seinem Platz.

## 9. Selbst ausprobieren (Konsole)

1. `python3 barkeeper2.py` starten, `ein bier` eingeben und das Entscheidungs-Log lesen (`log` schaltet es an oder aus):
   Mehrere Konzepte schlagen wegen `bier` und `ein` aus.
2. `kein bier` eingeben und vergleichen: Das stärkste Konzept hat jetzt `kein` als größten Treiber, Wein gewinnt.
3. Etwas Unbekanntes wie `ein weißbier` eingeben, beibringen und erneut fragen.

## Gleiche Ideen in echten LLMs

Zufallsstart, Daten, Fehlerkorrektur über Gradienten und Bedeutungen, die niemand programmiert hat,
sondern die sich im Training bilden, gelten auch für LLMs. Dort sind es nur Milliarden Gewichte statt knapp 4000.
