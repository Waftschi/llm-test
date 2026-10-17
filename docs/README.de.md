# Barkeeper in 25 Minuten

Kurze Anleitung für Kolleginnen und Kollegen: Was der Barkeeper ist, wie er funktioniert und wie man ihn in 20–30 Minuten erklärt.
Kein Mathe-Vorwissen nötig. Details stehen im [README](../README.de.md), die Folien sind die Sprechhilfe.

English version: [README.md](README.md)

## Was ist das?

Ein kleiner Chatbot, der einen Barkeeper spielt. Man tippt einen Satz ("ein bier", "kein wein", "zahlen bitte"),
und er antwortet. Es ist ein **neuronales Netz mit 4.258 Zahlen**, kein Sprachmodell.
Es nutzt dieselben Bausteine wie ein LLM (Zahlen, Gewichte, Softmax, Lernen aus Fehlern),
ist aber so klein, dass man jeden Schritt ansehen kann.

**Nicht** enthalten: freier Text, unbekannte Wörter verstehen, eine Weboberfläche (kommt später).

## Die Geschichte in fünf Gedanken

1. **Text wird zu Zahlen.** Jedes bekannte Wort bekommt eine 1, wenn es im Satz steht, sonst eine 0.
   Der Kontext (was der Gast zuletzt bestellt hat, z. B. `[wein]`) zählt wie ein weiteres Wort.
2. **Schicht 1: Wörter → Konzepte.** 24 „Konzepte“ sind Ja/Nein-Fragen an den Satz,
   zum Beispiel „Kommt *kein* zusammen mit *bier* vor?“. Niemand entwirft sie, das Netz findet sie selbst.
3. **Schicht 2: Konzepte → Antwort.** Jedes Konzept stimmt für oder gegen jede der 34 möglichen Antworten.
   Softmax macht aus den Punktzahlen Prozente. Die höchste gewinnt. Unter 40 % sagt der Barkeeper „Hä? Wie bitte?“.
4. **Lernen = raten, messen, Schuld verteilen, nachbessern.** Das Netz rät, misst den Fehler, gibt die Schuld
   rückwärts an jedes Gewicht weiter und verschiebt jedes Gewicht ein kleines Stück. 1.500-mal.
   Gelernt werden nur die Gewichte.
5. **Warum zwei Schichten?** „kein bier“ braucht eine *Kombination* aus zwei Wörtern. Eine Schicht kann nur Einzelwort-Stimmen addieren.
   `tanh` zwischen den Schichten macht Kombinationen möglich.

## Ablauf für 25 Minuten

Die Folien-Nummern gelten für das aktuelle Deck (`praesentation/presentation.html`, 27 Folien). Sie verschieben sich, wenn Folien dazukommen.

| Zeit | Thema | Folien | Kernaussage |
|---|---|---|---|
| 0–3 min | Die Aufgabe | 1–3 | Ein Barkeeper, der „kein bier“ versteht |
| 3–11 min | **Teil 1: Antworten** | 4–11 | Wie aus einem Satz eine Antwort wird (Zahlen → Schicht 1 → Schicht 2 → Softmax) |
| 11–18 min | **Teil 2: Lernen** | 13–19 | Woher die Gewichte kommen: raten, messen, Schuld verteilen, nachbessern |
| 18–21 min | **Teil 3: Einordnung** | 22, 25–27 | Warum zwei Schichten, und was ein LLM anders macht |
| 21–25 min | **Live-Demo** | – | siehe unten |

**Kurze Route (20 Minuten):** Folien 7–8 (Herkunft der Konzepte, W2/b2), 18–19 (Backpropagation im Detail) und 23–24 (Hürden-Beispiel, tanh-Beweis) weglassen.
Das sind die Vertiefungsfolien für Rückfragen.

## Live-Demo (etwa 5 Minuten)

Start: `pip3 install -r requirements.txt`, dann `python3 barkeeper2.py`.

| Eingabe | Erwartung | Aussage dazu |
|---|---|---|
| `ein bier` | bier, 100 % | Das Entscheidungslog zeigt Konzepte, Stimmen und Softmax |
| `reset`, dann `kein bier` | wein | „kein“ dreht das Getränk um, eine Kombination aus zwei Wörtern |
| `reset`, dann `kein wein` | bier | Dieselbe Regel in die andere Richtung |
| `log` | Entscheidungslog aus/an | Für eine ruhigere Demo ausschalten |
| `asdf qwer` | „Hä? Wie bitte?“ und eine Lernfrage | Es fragt nach, statt zu raten. Mit **Enter** überspringen |
| `vergleich` | eine Matrix 8/12, zwei Schichten 12/12 bei neuen Sätzen | Der Beweis, dass zwei Schichten nötig sind |
| `python3 show_matrix.py konzept 15` | die Wörter und Antworten, die Konzept 15 antreiben | Konzepte sind nur Zeilen aus Zahlen |

**Kontext zählt:** `ein bier` und direkt danach `kein bier` (ohne `reset`) ist sich bei wein weniger sicher als
`kein bier` bei einem neuen Gast, weil der Kontext `[bier]` in die andere Richtung zieht. Ein guter Gesprächsanlass.

**Beibringen:** `noch eins = bier` lehrt ein Beispiel (ein Beispiel pro Zeile, nur ein `=`: `lambrusco = wein` und `kein lambrusco = bier` sind zwei getrennte Eingaben). Dabei wird das *ganze* Modell mit zufälligen Gewichten neu trainiert
(einige Sekunden), und `barkeeper2.db` ändert sich. Für die Demo besser das Mini-Modell nehmen (`python3 barkeeper2.py simple`) oder die Datenbank vorher kopieren.

## Wichtige Zahlen

| | Großes Modell | Bedeutung |
|---|---|---|
| Trainingsbeispiele | 230 | Satz + Kontext → Antwort |
| Merkmale | 141 | bekannte Wörter und Kontext-Token |
| Konzepte | 24 | verdeckte Schicht |
| Antworten | 34 | mögliche Antworten |
| Gewichte und Bias | 4.258 | 24×141 + 24 + 34×24 + 34 |
| Training | 1.500 Runden | Lernrate 0.5, Momentum 0.9, Seed 0 |

## Glossar

| Begriff | In einfachen Worten |
|---|---|
| **Gewicht** | Eine gelernte Zahl: wie stark ein Eingang ein Ergebnis hoch- oder runterschiebt |
| **Bias** | Ein gelernter Grundwert, der unabhängig von der Eingabe gilt |
| **Konzept** | Eine der 24 verdeckten Ja/Nein-Fragen von Schicht 1 (eine Zeile aus Zahlen in `W1`) |
| **tanh** | Drückt eine Summe auf −1 … +1 (−1 = nein, +1 = ja) |
| **Softmax** | Macht aus Punktzahlen Prozente, die zusammen 100 % ergeben |
| **Fehler (P − Y)** | Vermutung minus richtige Antwort. Positiv = zu stark, negativ = zu schwach |
| **Backpropagation** | Den Fehler rückwärts weitergeben, um zu sehen, wie viel Schuld jedes Gewicht trägt |
| **Trainingsrunde** | Einmal raten, messen, Schuld verteilen, nachbessern über alle Beispiele |
| **Seed** | Fester Start der Zufallszahlen, damit das Training wiederholbar ist |

## Mögliche Fragen

- **Ist das ein LLM?** Nein. Es hat dieselben Zutaten, aber nur 4.258 Gewichte, einen festen Wortschatz und eine Liste vorbereiteter Antworten.
- **Was macht ein LLM anders?** Es sagt das nächste Wort voraus, hat Milliarden Gewichte, nutzt Attention, um Wörter aufeinander zu beziehen,
  und wurde mit riesigen Textmengen trainiert. Das Lernprinzip ist dasselbe.
- **Warum 24 Konzepte?** Unsere Wahl. Zu wenige sind zu grob, zu viele lernen die Beispiele auswendig.
- **Versteht es neue Wörter?** Nein. Unbekannte Wörter werden ignoriert. Sind mehr als die Hälfte unbekannt, fragt es nach.
- **Warum alles neu trainieren beim Beibringen?** Das hält den Code einfach und wiederholbar. Ein schnellerer Weg (nur ein paar Schritte mit dem neuen Beispiel) riskiert, Altes zu vergessen.

## Wo finde ich was?

| Was | Wo |
|---|---|
| Konsolen-App | `barkeeper2.py` (`python3 barkeeper2.py`, Mini-Modell: `python3 barkeeper2.py simple`) |
| Antworten, Preise, Beispiele, Fallback | `barkeeper_seed.json` |
| Gelernte Gewichte ansehen | `python3 show_matrix.py` |
| Folien | [Deutsch](https://waftschi.github.io/llm-test/praesentation/presentation.html) · [Englisch](https://waftschi.github.io/llm-test/praesentation_en/presentation.html) |
| Ausführliche Erklärung | [README](../README.de.md) |
