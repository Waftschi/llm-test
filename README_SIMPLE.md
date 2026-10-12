# Mini-Modell zum Mitverfolgen

Ein winziger Start, damit du sehen kannst, wie sich Konzepte und Wörter beim Lernen verändern.
Statt 4258 Gewichten hat das Mini-Modell nur 43.

## Was ist drin (`barkeeper_simple_seed.json`)

- 3 Antworten: `gruss`, `bier`, `wein`
- 6 Beispiele: hallo, hi, ein bier, bier, ein wein, wein
- 6 Merkmale: hallo, hi, ein, bier, wein, `[start]`
- nur **4 Konzepte** statt 24 (`"hidden": 4`)
- kein "kein"-Beispiel – das bringst du später selbst bei

Das Mini-Modell hat eine eigene Datenbank (`barkeeper_simple.db`). Deine bisherigen Daten (`barkeeper2.db`) bleiben unberührt.

## Starten

```
python3 barkeeper2.py simple                       # Konsole
BARKEEPER_PROFILE=simple python3 show_matrix.py    # Matrix ansehen (nur 4 x 6)
```

Ohne `simple` läuft alles wie bisher mit dem großen Modell.

## Experiment

1. `python3 barkeeper2.py simple` starten und `ein bier` eingeben.
   Im Log siehst du nur vier Konzepte, die klar von `bier` getrieben werden.
2. `BARKEEPER_PROFILE=simple python3 show_matrix.py` ausführen und die Matrix notieren.
3. `kein bier = wein` beibringen und `show_matrix.py` erneut ausführen.
   Vergleiche die Zahlen, besonders die Spalten `kein` und `bier`.
4. Danach `kein wein = bier` und `heute bier = bier` beibringen und wieder vergleichen.

## Hinweise

- Mit nur 4 Konzepten lernt das Modell "kein bier" eventuell nicht sauber.
  Das zeigt, warum die Anzahl der Konzepte wichtig ist.
- Mehr Konzepte: `"hidden"` in `barkeeper_simple_seed.json` auf 6 oder 8 setzen
  und `barkeeper_simple.db` löschen, damit neu trainiert wird.
- Zum Zurücksetzen auf den Anfangszustand reicht es, `barkeeper_simple.db` zu löschen.

Die Erklärung dazu, wie die Gewichte lernen, steht in `README_ERKLAERUNG.md`.
