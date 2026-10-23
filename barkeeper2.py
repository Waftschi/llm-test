"""Barkeeper with TWO LAYERS (console).

A small intent classifier (seed file, chat commands), the model is
    h = tanh(W1 · x + b1)      layer 1: words + context -> hidden concepts
    p = softmax(W2 · h + b2)   layer 2: concepts -> what the barkeeper does
so it can learn combinations like "kein bier" (-> not beer!).

The model itself lives in barkeeper_core.py (also used by the web UI, server.py).
New command:  vergleich   trains the old one-matrix model on the same data and compares.
"""
import os, sys
from barkeeper_core import Barkeeper, Session

# "python3 barkeeper2.py simple" starts a tiny model with its own database
PROFILE = os.environ.get("BARKEEPER_PROFILE") or (sys.argv[1] if len(sys.argv) > 1 else "")

# ---------- decision log through both layers ----------
def print_log(d):
    print("    ═══ TWO-LAYER DECISION LOG ═══")
    if d["unknown"]:
        print(f"    ⚠️  unknown words (ignored): {', '.join(d['unknown'])}")

    print(f"\n    📥 LAYER 1: Words → Concepts   (the 4 with the biggest vote for '{d['layer1']['top_intent']}')")
    print(f"    {'─' * 70}")
    for c in d["layer1"]["concepts"]:
        meaning = {"strong": "↑ STRONG", "weak": "↓ WEAK", "neutral": "~ neutral"}[c["label"]]
        print(f"    Concept {c['concept']:<2d} [{c['activation']:+.2f} {meaning}]")
        print(f"              ← " + "  ".join(f"{t['word']}({t['weight']:+.1f})" for t in c["drivers"]))
        print(f"              Σ words {c['word_sum']:+.2f}  + bias {c['bias']:+.2f}  = {c['z']:+.2f}"
              f"   → tanh({c['z']:+.2f}) = {c['tanh']:+.2f}")

    l2 = d["layer2"]
    print(f"\n    📤 LAYER 2: Concepts → Answer")
    print(f"    {'─' * 70}")
    print(f"    Which concepts influenced each answer:")
    print(f"    {'':16s}  concepts:      " + " / ".join(f"c{k}" for k in l2["concepts"]))
    for a in l2["answers"]:
        print(f"    {a['intent']:16s}  concepts vote: " + " / ".join(f"{v:+.1f}" for v in a["votes"])
              + f"  = {a['shown']:+.1f}")
        print(f"    {'':16s}  + others {a['others']:+.1f}  + bias {a['bias']:+.1f}  → score {a['score']:+.1f}")

    sm = d["softmax"]
    print(f"\n    🧮 SOFTMAX: e^(score − best score), then share of the total")
    print(f"    {'─' * 70}")
    for a in sm["answers"]:
        print(f"    {a['intent']:16s}  {a['score']:+6.1f} − {sm['best_score']:+.1f} = {a['shifted']:+6.1f}"
              f"   e^ = {a['exp']:.4f}   share = {a['exp']:.4f} / {sm['total_exp']:.4f} = {a['share']:.1%}")
    print(f"    {'all other ' + str(sm['others_count']):16s}  {'':24s}e^ = {sm['others_exp']:.4f}   (sum of the remaining answers)")
    print(f"    {'total':16s}  {'':24s}e^ = {sm['total_exp']:.4f}")

    print(f"\n    🎯 FINAL DECISION")
    print(f"    {'─' * 70}")
    print("    " + " vs ".join(f"{f['intent']} = {f['p']:3.0%}" for f in d["final"]))

def print_blame(b):
    print("    ═══ BLAME (before retraining) ═══")
    if b["new_words"]:
        print(f"    🆕 new words (get their own column in W1): {', '.join(b['new_words'])}")
    if b["new_intent"]:
        print(f"    🆕 new answer '{b['new_intent']}' (gets its own row in W2)")
    print(f"\n    📉 ANSWERS with the most blame: error D2 = P − Y  (+ too strong, − too weak)")
    for a in b["answers"]:
        print(f"    {a['intent']:16s} P = {a['p']:6.1%}   right = {a['right']:.0f}   D2 = {a['d2']:+.2f}  "
              f"{'too strong' if a['d2'] > 0 else 'too weak'}")
    print(f"\n    🎯 CONCEPTS with the most blame: D1 = (D2 · W2) × (1 − h²)")
    for c in b["concepts"]:
        words = "  ".join(f"{t['word']}({t['weight']:+.1f})" for t in c["drivers"])
        print(f"    Concept {c['concept']:<2d} blame {c['blame']:+.3f}   h = {c['h']:+.2f}   ← {words}")
    print("    (the weights of the words in this sentence move, a little in each of the 1500 rounds)\n")

def print_learned(l):
    if show:
        print_blame(l["blame"])
    if l["replaced"]:
        print(f"  (replaced {l['replaced']} old example)")
    print(f"  🔄 retraining both layers ({l['examples']} examples total) ...")
    print(f"  ✅ learned: [{l['context']}] '{l['sentence']}' -> {l['intent']}")

def print_act(a):
    if a["blocked"]:
        print(f"\n  🍺 Barkeeper: {a['reply']}\n")
        return
    print(f"\n  🍺 Barkeeper: {a['reply']}   ({a['intent']}, {a['prob']:.0%})")
    print("     (wrong? type: falsch)\n")

def ask_teacher(choices):
    print("  🎓 What should I have done? Pick a number or type a new answer:")
    for n, c in enumerate(choices, 1):
        print(f"     {n:2d}. {c}")
    pick = input("  teach (Enter = skip)> ").strip()
    if not pick:
        print("  ok, skipped\n"); return None
    if pick.isdigit() and 1 <= int(pick) <= len(choices):
        return choices[int(pick) - 1]
    return pick

# ---------- Start ----------
bar = Barkeeper(PROFILE)
if bar.trained_on_start:
    p = bar.params()
    print("🔄 training two-layer neural network ...")
    print(f"   layer 1: {p['vocab_size']} words+context → {bar.hidden} hidden concepts")
    print(f"   layer 2: {bar.hidden} concepts → {p['n_intents']} possible answers")
    print(f"✅ model ready with {p['w1_size'] + p['w2_size']} trained weights")

p = bar.params()
print(f"🍺🍺 Two-layer barkeeper: {p['n_examples']} examples, "
      f"{p['vocab_size']} features -> {bar.hidden} concepts -> {p['n_intents']} answers "
      f"= {p['n_weights']} weights")
print("Try:    " + "  /  ".join(bar.suggestions))
print("Teach:  if I don't understand, I'll ask you;  wrong answer -> falsch")
print("        or directly: noch eins = bier  /  [wein] noch eins = wein")
print("Other:  data | vergleich | log | reset | exit\n")

s, show = Session(), True

def teach_pending(choices):
    intent = ask_teacher(choices)
    if not intent:
        s.pending = None
        return
    r = bar.teach(s, intent)
    print_learned(r["learned"])
    print_act(r["act"])

while True:
    said = input(f"you [{s.context}, {s.bill:.2f} €]> ").strip()
    cmd = said.lower()
    if cmd in ("exit", "quit", ""):
        break
    if cmd == "log":
        show = not show; print(f"  log {'on' if show else 'off'}\n"); continue

    if cmd in ("data", "daten"):
        d = bar.data()
        for r in d["rows"]:
            mark = "✓" if r["ok"] else f"✗ says {r['guess']}"
            print(f"  {r['n']:3d}. [{r['context']:9s}] {r['sentence']:28s} = {r['intent']:16s} {mark}")
        print(f"\n  {d['correct']}/{d['total']} correct\n"); continue

    if cmd in ("vergleich", "compare"):              # one matrix vs two layers, same data
        print("  training the old one-matrix model on the same examples ...")
        c = bar.compare()
        print(f"\n  ONE MATRIX:  {c['one_ok']}/{c['total']} correct")
        print(f"  TWO LAYERS:  {c['two_ok']}/{c['total']} correct\n")
        print("  NEW sentences (never trained):")
        print("    " + "sentence".ljust(24) + "should be".ljust(11) + "one matrix".ljust(14) + "two layers")
        for t in c["tests"]:
            print(f"    {t['sentence']:24s}{t['expected']:11s}{t['one']:9s}{'✓' if t['one'] == t['expected'] else '✗':5s}"
                  f"{t['two']:9s}{'✓' if t['two'] == t['expected'] else '✗'}")
        print(f"    {'':35s}{c['one_test_ok']:2d}/{len(c['tests'])}         {c['two_test_ok']:2d}/{len(c['tests'])}\n")
        continue

    r = bar.handle(s, said)
    if r["type"] == "reset":
        print("  (a new guest walks in)\n")
    elif r["type"] == "error":
        print(f"  {'⚠️  ' if '=' in r['message'] else ''}{r['message']}\n")
    elif r["type"] == "learned":
        print_learned(r["learned"]); print()
    elif r["type"] == "wrong":
        print(f"  you said '{r['sentence']}' in [{r['context']}], I did: {r['wrong_intent']}")
        teach_pending(r["choices"])
    elif r["type"] == "say":
        if show:
            print_log(r["log"])
        if r["understood"]:
            print_act(r["act"])
        else:
            print(f"\n  🍺 Barkeeper: {r['reply']}   (best guess: {r['intent']}, {r['confidence']:.0%})")
            teach_pending(r["choices"])
