"""Zeigt die gelernten Gewichte aus barkeeper2.db (nur lesen).

python3 show_matrix.py                 Überblick: je Konzept stärkste Wörter und Antworten
python3 show_matrix.py wort bier       Spalte von W1: wie stark wirkt 'bier' auf jedes Konzept
python3 show_matrix.py konzept 15      Zeile von W1 + Spalte von W2 für Konzept 15
python3 show_matrix.py antwort wein    Zeile von W2: welche Konzepte stimmen für 'wein'
python3 show_matrix.py csv             schreibt W1.csv und W2.csv (mit Beschriftung)
"""
import csv, json, os, sqlite3, sys
import numpy as np

# BARKEEPER_PROFILE=simple python3 show_matrix.py  -> kleines Modell
db = sqlite3.connect("barkeeper_simple.db" if os.environ.get("BARKEEPER_PROFILE") == "simple" else "barkeeper2.db")
m = {k: json.loads(v) for k, v in db.execute("SELECT name, value FROM model")}
W1, W2 = np.array(m["W1"]), np.array(m["W2"])
vocab, intents = m["vocab"], m["intents"]
print(f"W1: {W1.shape} (Konzepte x Wörter)   W2: {W2.shape} (Antworten x Konzepte)\n")

def top(values, names, n):
    return "  ".join(f"{names[i]}({values[i]:+.1f})" for i in np.argsort(-np.abs(values))[:n])

args = sys.argv[1:]
if not args:
    for k in range(W1.shape[0]):
        print(f"Konzept {k:2d}  Wörter: {top(W1[k], vocab, 5)}")
        print(f"            Antworten: {top(W2[:, k], intents, 3)}")
elif args[0] == "wort":
    j = vocab.index(args[1])
    for k in np.argsort(-np.abs(W1[:, j])):
        print(f"Konzept {k:2d}  W1[{k},{args[1]}] = {W1[k, j]:+.2f}")
elif args[0] == "konzept":
    k = int(args[1])
    print(f"W1-Zeile (Wörter):   {top(W1[k], vocab, 10)}")
    print(f"W2-Spalte (Antworten): {top(W2[:, k], intents, 10)}")
elif args[0] == "antwort":
    i = intents.index(args[1])
    for k in np.argsort(-np.abs(W2[i])):
        print(f"Konzept {k:2d}  W2[{args[1]},{k}] = {W2[i, k]:+.2f}")
elif args[0] == "csv":
    for name, W, rows_, cols in (("W1", W1, [f"konzept_{k}" for k in range(W1.shape[0])], vocab),
                                 ("W2", W2, intents, [f"konzept_{k}" for k in range(W2.shape[1])])):
        with open(f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([""] + cols)
            for label, row in zip(rows_, W):
                w.writerow([label] + [f"{x:.3f}" for x in row])
        print(f"{name}.csv geschrieben")
else:
    print(__doc__)
