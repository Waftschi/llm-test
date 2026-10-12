import numpy as np
import sqlite3
import random

DB_FILE = "barkeeper.db"

import json, os

SEED_FILE = "barkeeper_seed.json"

# ---------- 1+2. Intents and starting examples ----------
# Loaded from barkeeper_seed.json if it exists, otherwise these small defaults are used.
# INTENTS: name -> (possible replies, price, new context)   context None = keep current
INTENTS = {
    "gruss":           (["Servus! Was darf's sein?"], 0, None),
    "bier":            (["Ein Bier, gern!"], 4.50, "bier"),
    "zahlen":          (["Das macht {bill:.2f} €. Danke dir!"], 0, "start"),
    "nichts_bestellt": (["Du hast doch noch gar nichts bestellt!"], 0, None),
    "tschuess":        (["Mach's gut, bis bald!"], 0, "start"),
}
default_examples = [
    ("start", "hallo", "gruss"), ("start", "ein bier bitte", "bier"),
    ("bier", "noch eins", "bier"), ("start", "noch eins", "nichts_bestellt"),
    ("bier", "zahlen bitte", "zahlen"), ("start", "zahlen bitte", "nichts_bestellt"),
    ("start", "tschüss", "tschuess"),
]
if os.path.exists(SEED_FILE):
    seed = json.load(open(SEED_FILE, encoding="utf-8"))
    INTENTS = {k: (v["replies"], v["price"], v["context"]) for k, v in seed["intents"].items()}
    default_examples = [tuple(e) for e in seed["examples"]]
    print(f"Seed file {SEED_FILE}: {len(INTENTS)} intents, {len(default_examples)} examples")

# ---------- 3. The model (same as before, plus a context feature) ----------
def clean(text):
    text = text.casefold()
    for ch in "?!.,;:":
        text = text.replace(ch, "")
    return " ".join(text.split())

def features(context, sentence):           # words + context = input
    return clean(sentence).split() + [f"[{context}]"]

def to_vector(feats, vocab):
    return np.array([1.0 if f in feats else 0.0 for f in vocab])

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

def train(examples, rounds=300):
    vocab   = sorted({f for c, s, _ in examples for f in features(c, s)})
    intents = sorted({i for _, _, i in examples})
    W = np.zeros((len(intents), len(vocab)))
    for _ in range(rounds):
        for c, s, i in examples:
            x = to_vector(features(c, s), vocab)
            p = softmax(W @ x)
            target = np.zeros(len(intents)); target[intents.index(i)] = 1
            W -= 0.1 * np.outer(p - target, x)
    return W, vocab, intents

# ---------- 4. Database ----------
def open_db():
    db = sqlite3.connect(DB_FILE)
    db.execute("CREATE TABLE IF NOT EXISTS examples (id INTEGER PRIMARY KEY, context TEXT, sentence TEXT, intent TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS weights (intent TEXT, feature TEXT, value REAL)")
    if not db.execute("SELECT 1 FROM examples").fetchone():
        db.executemany("INSERT INTO examples (context, sentence, intent) VALUES (?, ?, ?)",
                       [(c, clean(s), i) for c, s, i in default_examples])
        db.commit()
    return db

def load_examples(db):
    return db.execute("SELECT context, sentence, intent FROM examples ORDER BY id").fetchall()

def save_weights(db, W, vocab, intents):
    db.execute("DELETE FROM weights")
    db.executemany("INSERT INTO weights VALUES (?, ?, ?)",
                   [(it, f, float(W[i, j])) for i, it in enumerate(intents) for j, f in enumerate(vocab)])
    db.commit()

def load_weights(db):
    rows = db.execute("SELECT intent, feature, value FROM weights").fetchall()
    if not rows:
        return None
    intents = sorted({r[0] for r in rows}); vocab = sorted({r[1] for r in rows})
    W = np.zeros((len(intents), len(vocab)))
    for it, f, v in rows:
        W[intents.index(it), vocab.index(f)] = v
    return W, vocab, intents

# ---------- 5. Decision log ----------
def show_log(feats, W, vocab, intents, p):
    top = np.argsort(p)[::-1][:3]                     # only the 3 best intents
    print("    " + "feature".ljust(16) + "".join(intents[i].rjust(16) for i in top))
    for f in feats:
        if f in vocab:
            j = vocab.index(f)
            print("    " + f.ljust(16) + "".join(f"{W[i, j]:+16.2f}" for i in top))
        else:
            print("    " + f.ljust(16) + "(unknown, ignored)")
    print("    " + "probability".ljust(16) + "".join(f"{p[i]:16.0%}" for i in top))

# ---------- 6. The bar ----------
db = open_db()
examples = load_examples(db)
model = load_weights(db)
if model:
    W, vocab, intents = model
else:
    W, vocab, intents = train(examples)
    save_weights(db, W, vocab, intents)

context, bill, show = "start", 0.0, True

# ---------- 7. Helpers for interactive teaching ----------
def learn(ctx, sentence, intent):
    global examples, W, vocab, intents
    for known in list(INTENTS) + intents:             # reuse known intent spelling
        if known.casefold() == intent.casefold():
            intent = known
    old = db.execute("DELETE FROM examples WHERE context=? AND sentence=?",
                     (ctx, clean(sentence))).rowcount   # a correction replaces the old example
    db.execute("INSERT INTO examples (context, sentence, intent) VALUES (?, ?, ?)",
               (ctx, clean(sentence), intent)); db.commit()
    if old:
        print(f"  (replaced {old} old example)")
    examples = load_examples(db)
    W, vocab, intents = train(examples)
    save_weights(db, W, vocab, intents)
    print(f"  ✅ learned: [{ctx}] '{clean(sentence)}' -> {intent}")
    return intent

def ask_teacher():
    """Show numbered choices; return an intent or a new text answer (None = skip)."""
    choices = list(INTENTS) + [i for i in intents if i not in INTENTS]
    print("  🎓 What should I have done? Pick a number or type a new answer:")
    for n, c in enumerate(choices, 1):
        print(f"     {n:2d}. {c}")
    pick = input("  teach (Enter = skip)> ").strip()
    if not pick:
        print("  ok, skipped\n"); return None
    if pick.isdigit() and 1 <= int(pick) <= len(choices):
        return choices[int(pick) - 1]
    return pick

def act(intent, prob):
    """Do what the intent says; return what is needed to undo it."""
    global context, bill
    before = (context, bill)
    replies, price, new_ctx = INTENTS.get(intent, ([intent], 0, None))  # text answer
    if intent == "tschuess" and bill > 0:             # a little rule outside the model
        print(f"\n  🍺 Barkeeper: Halt! Erst zahlen, {bill:.2f} € bitte!\n"); return before
    reply = random.choice(replies).format(bill=bill)
    if intent == "zahlen":
        bill = 0.0
    bill += price
    if new_ctx:
        context = new_ctx
    print(f"\n  🍺 Barkeeper: {reply}   ({intent}, {prob:.0%})")
    print("     (wrong? type: falsch)\n")
    return before

last = None                                           # (context, sentence, intent, state before)

print("🍺 Welcome to the bar! Talk to the barkeeper.")
print("Teach:  if I don't understand, I'll ask you.")
print("        if I answer wrong, type: falsch")
print("        or directly: noch eins = bier  /  [wein] noch eins = wein")
print("Other:  data | log (on/off) | reset (new guest) | exit\n")

while True:
    said = input(f"you [{context}, {bill:.2f} €]> ").strip()
    cmd = said.lower()
    if cmd in ("exit", "quit", ""):
        break
    if cmd == "log":
        show = not show
        print(f"  log {'on' if show else 'off'}\n"); continue
    if cmd == "reset":
        context, bill, last = "start", 0.0, None
        print("  (a new guest walks in)\n"); continue
    if cmd in ("data", "daten"):
        ok = 0
        for n, (c, s, i) in enumerate(examples, 1):
            p = softmax(W @ to_vector(features(c, s), vocab))
            guess = intents[p.argmax()]; ok += guess == i
            mark = "✓" if guess == i else f"✗ says {guess}"
            print(f"  {n:2d}. [{c:6s}] {s:28s} = {i:16s} {mark}")
        print(f"\n  {ok}/{len(examples)} correct\n"); continue

    if cmd in ("falsch", "wrong", "nein falsch"):     # ---- correct the last answer ----
        if not last:
            print("  nothing to correct yet\n"); continue
        ctx, sentence, wrong, (old_ctx, old_bill) = last
        print(f"  you said '{sentence}' in [{ctx}], I did: {wrong}")
        intent = ask_teacher()
        if intent:
            context, bill = old_ctx, old_bill         # undo the wrong action
            intent = learn(ctx, sentence, intent)
            last = (ctx, sentence, intent, act(intent, 1.0))
        continue

    if "=" in said:                                   # ---- direct teaching ----
        left, intent = [x.strip() for x in said.split("=", 1)]
        ctx = context
        if left.startswith("[") and "]" in left:      # explicit [context]
            ctx, left = left[1:left.index("]")].strip().lower(), left[left.index("]") + 1:]
        if not clean(left) or not intent:
            print("  please write: sentence = answer\n"); continue
        learn(ctx, left, intent); print(); continue

    # ---- normal talking ----
    feats = features(context, said)
    p = softmax(W @ to_vector(feats, vocab))
    intent = intents[p.argmax()]
    known_words = [f for f in feats[:-1] if f in vocab]

    if show:
        show_log(feats, W, vocab, intents, p)
    too_few = len(known_words) < len(feats[:-1]) / 2  # less than half the words known
    if not known_words or too_few or p.max() < 0.4:   # not sure enough -> ask the teacher
        print(f"\n  🍺 Barkeeper: Hä? Wie bitte?   (best guess: {intent}, {p.max():.0%})")
        intent = ask_teacher()
        if intent:
            intent = learn(context, said, intent)
            ctx = context
            last = (ctx, clean(said), intent, act(intent, 1.0))
        continue

    ctx = context
    last = (ctx, clean(said), intent, act(intent, p.max()))
