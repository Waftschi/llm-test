"""Barkeeper with TWO LAYERS.

A small intent classifier (seed file, chat commands), the model is
    h = tanh(W1 · x + b1)      layer 1: words + context -> hidden concepts
    p = softmax(W2 · h + b2)   layer 2: concepts -> what the barkeeper does
so it can learn combinations like "kein bier" (-> not beer!).

New command:  vergleich   trains the old one-matrix model on the same data and compares.
"""
import numpy as np
import sqlite3, random, json, os, sys

# "python3 barkeeper2.py simple" starts a tiny model with its own database
PROFILE   = os.environ.get("BARKEEPER_PROFILE") or (sys.argv[1] if len(sys.argv) > 1 else "")
SIMPLE    = PROFILE == "simple"
DB_FILE   = "barkeeper_simple.db" if SIMPLE else "barkeeper2.db"
SEED_FILE = "barkeeper_simple_seed.json" if SIMPLE else "barkeeper_seed.json"
HIDDEN    = 24                             # number of hidden concepts (layer 1 size)

# ---------- 1. Intents and seed examples ----------
INTENTS = {
    "gruss":           (["Servus! Was darf's sein?"], 0, None),
    "bier":            (["Ein Bier, gern!"], 4.50, "bier"),
    "wein":            (["Ein Glas Wein, bitte schön."], 6.00, "wein"),
    "zahlen":          (["Das macht {bill:.2f} €. Danke dir!"], 0, "start"),
    "nichts_bestellt": (["Du hast doch noch gar nichts bestellt!"], 0, None),
    "tschuess":        (["Mach's gut, bis bald!"], 0, "start"),
}
seed_examples = [
    ("start", "hallo", "gruss"), ("start", "ein bier bitte", "bier"), ("start", "ein wein", "wein"),
    ("bier", "noch eins", "bier"), ("start", "noch eins", "nichts_bestellt"),
    ("bier", "zahlen bitte", "zahlen"), ("start", "zahlen bitte", "nichts_bestellt"),
    ("start", "tschüss", "tschuess"),
]
if os.path.exists(SEED_FILE):
    seed = json.load(open(SEED_FILE, encoding="utf-8"))
    INTENTS = {k: (v["replies"], v["price"], v["context"]) for k, v in seed["intents"].items()}
    seed_examples = [tuple(e) for e in seed["examples"]]
    HIDDEN = seed.get("hidden", HIDDEN)

# Examples that need COMBINATIONS: "kein" flips the meaning of the drink.
# Our guest drinks only beer or wine, so "kein bier" means wine and vice versa.
# A single matrix cannot learn these, two layers can.
KEIN_EXAMPLES = [
    ("start", "kein bier", "wein"),       ("start", "kein wein", "bier"),
    ("start", "bitte kein bier", "wein"), ("start", "bitte kein wein", "bier"),
    ("start", "heute kein bier", "wein"), ("start", "heute kein wein", "bier"),
    ("bier",  "jetzt kein bier mehr", "wein"), ("wein", "jetzt kein wein mehr", "bier"),
    # balance: the plain words must also be learned WITHOUT "kein"
    ("start", "bier", "bier"),            ("start", "wein", "wein"),
    ("start", "heute bier", "bier"),      ("start", "heute wein", "wein"),
    ("start", "ein wein bitte", "wein"),  ("start", "bitte ein bier", "bier"),
]

# New sentences the model has NEVER seen. "vergleich" tests both models on them.
TEST_SENTENCES = [
    ("start", "kein bier bitte", "wein"),     ("start", "kein wein bitte", "bier"),
    ("start", "lieber kein bier", "wein"),    ("start", "lieber kein wein", "bier"),
    ("start", "ich will kein wein", "bier"),  ("start", "heute kein bier bitte", "wein"),
    ("start", "bier bitte", "bier"),          ("start", "wein bitte", "wein"),
    ("start", "ich will bier", "bier"),       ("start", "ich will wein", "wein"),
    ("start", "heute ein bier", "bier"),      ("start", "heute ein wein", "wein"),
]
if not SIMPLE:
    seed_examples += KEIN_EXAMPLES

# ---------- 2. Text -> numbers ----------
def clean(text):
    text = text.casefold()
    for ch in "?!.,;:":
        text = text.replace(ch, "")
    return " ".join(text.split())

def features(context, sentence):           # words + context = input
    return clean(sentence).split() + [f"[{context}]"]

def to_matrix(examples, vocab, intents):   # all examples at once: X (inputs), Y (targets)
    col = {f: j for j, f in enumerate(vocab)}
    X = np.zeros((len(examples), len(vocab)))
    Y = np.zeros((len(examples), len(intents)))
    for n, (c, s, i) in enumerate(examples):
        for f in features(c, s):
            if f in col:
                X[n, col[f]] = 1
        Y[n, intents.index(i)] = 1
    return X, Y

def softmax(Z):
    E = np.exp(Z - Z.max(axis=-1, keepdims=True))
    return E / E.sum(axis=-1, keepdims=True)

# ---------- 3. The two-layer model ----------
def forward(m, X):
    Hd = np.tanh(X @ m["W1"].T + m["b1"])            # layer 1: concepts
    P  = softmax(Hd @ m["W2"].T + m["b2"])           # layer 2: answers
    return Hd, P

def train(examples, rounds=1500, lr=0.5):
    vocab   = sorted({f for c, s, _ in examples for f in features(c, s)})
    intents = sorted({i for _, _, i in examples})
    X, Y = to_matrix(examples, vocab, intents)
    np.random.seed(0)                                # same start every time -> repeatable
    m = {"vocab": vocab, "intents": intents,
         "W1": np.random.randn(HIDDEN, len(vocab)) * 0.5, "b1": np.zeros(HIDDEN),
         "W2": np.random.randn(len(intents), HIDDEN) * 0.5, "b2": np.zeros(len(intents))}
    # "momentum": keep a bit of the last step, so learning rolls faster downhill
    v = {k: np.zeros_like(m[k]) for k in ("W1", "b1", "W2", "b2")}
    for _ in range(rounds):
        Hd, P = forward(m, X)                        # guess (for all examples at once)
        D2 = (P - Y) / len(examples)                 # error at the output
        D1 = (D2 @ m["W2"]) * (1 - Hd**2)            # error passed back to layer 1
        grads = {"W2": D2.T @ Hd, "b2": D2.sum(0),   # blame for every weight
                 "W1": D1.T @ X,  "b1": D1.sum(0)}
        for k in grads:                              # correct
            v[k] = 0.9 * v[k] - lr * grads[k]
            m[k] += v[k]
    return m

def train_one_matrix(examples, rounds=1500, lr=0.5):  # the old model, for "vergleich"
    vocab   = sorted({f for c, s, _ in examples for f in features(c, s)})
    intents = sorted({i for _, _, i in examples})
    X, Y = to_matrix(examples, vocab, intents)
    W, v = np.zeros((len(intents), len(vocab))), 0
    for _ in range(rounds):
        P = softmax(X @ W.T)
        v = 0.9 * v - lr * (P - Y).T @ X / len(examples)
        W += v
    return lambda X_: softmax(X_ @ W.T), vocab, intents

# ---------- 4. Database: model stored as readable JSON ----------
def open_db():
    db = sqlite3.connect(DB_FILE)
    db.execute("CREATE TABLE IF NOT EXISTS examples (id INTEGER PRIMARY KEY, context TEXT, sentence TEXT, intent TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS model (name TEXT PRIMARY KEY, value TEXT)")
    if not db.execute("SELECT 1 FROM examples").fetchone():
        db.executemany("INSERT INTO examples (context, sentence, intent) VALUES (?, ?, ?)",
                       [(c, clean(s), i) for c, s, i in seed_examples])
        db.commit()
    return db

def load_examples(db):
    return db.execute("SELECT context, sentence, intent FROM examples ORDER BY id").fetchall()

def save_model(db, m):
    db.execute("DELETE FROM model")
    db.executemany("INSERT INTO model VALUES (?, ?)",
                   [(k, json.dumps(v.tolist() if isinstance(v, np.ndarray) else v, ensure_ascii=False))
                    for k, v in m.items()])
    db.commit()

def load_model(db):
    rows = dict(db.execute("SELECT name, value FROM model").fetchall())
    if not rows:
        return None
    m = {k: json.loads(v) for k, v in rows.items()}
    for k in ("W1", "b1", "W2", "b2"):
        m[k] = np.array(m[k])
    return m

# ---------- 5. Decision log through both layers ----------
def show_log(feats, m, x, h, p):
    vocab, intents = m["vocab"], m["intents"]
    unknown = [f for f in feats if f not in vocab]
    top_i = np.argsort(p)[::-1][:2]                   # the 2 best answers
    strong = np.argsort(-np.abs(h))[:4]               # the 4 most active concepts

    print("    ═══ TWO-LAYER DECISION LOG ═══")
    if unknown:
        print(f"    ⚠️  unknown words (ignored): {', '.join(unknown)}")

    print(f"\n    📥 LAYER 1: Words → Concepts")
    print(f"    {'─' * 70}")
    for k in strong:
        drivers = sorted([(m["W1"][k, j], vocab[j]) for j in np.nonzero(x)[0]],
                         key=lambda t: -abs(t[0]))[:3]
        driven = "  ".join(f"{w}({v:+.1f})" for v, w in drivers)
        concept_activation = h[k]
        concept_meaning = "↑ STRONG" if concept_activation > 0.5 else ("↓ WEAK" if concept_activation < -0.5 else "~ neutral")
        print(f"    Concept {k:<2d} [{concept_activation:+.2f} {concept_meaning}]")
        print(f"              ← {driven}")
        word_sum = float(m["W1"][k] @ x)                    # all active word weights added up
        z = word_sum + m["b1"][k]
        print(f"              Σ words {word_sum:+.2f}  + bias {m['b1'][k]:+.2f}  = {z:+.2f}   → tanh({z:+.2f}) = {np.tanh(z):+.2f}")

    print(f"\n    📤 LAYER 2: Concepts → Answer")
    print(f"    {'─' * 70}")
    print(f"    Which concepts influenced each answer:")
    print(f"    {'':16s}  concepts:      " + " / ".join(f"c{k}" for k in strong))
    for i in top_i:
        votes = [m["W2"][i, k] * h[k] for k in strong]
        shown = sum(votes)
        others = float(m["W2"][i] @ h) - shown            # concepts not listed above
        score = shown + others + m["b2"][i]               # this goes into softmax
        print(f"    {intents[i]:16s}  concepts vote: " + " / ".join(f"{v:+.1f}" for v in votes)
              + f"  = {shown:+.1f}")
        print(f"    {'':16s}  + others {others:+.1f}  + bias {m['b2'][i]:+.1f}  → score {score:+.1f}")

    scores = m["W2"] @ h + m["b2"]                    # score of every answer
    E = np.exp(scores - scores.max())                 # softmax step 1 (shifted by the best score)
    print(f"\n    🧮 SOFTMAX: e^(score − best score), then share of the total")
    print(f"    {'─' * 70}")
    for i in top_i:
        print(f"    {intents[i]:16s}  {scores[i]:+6.1f} − {scores.max():+.1f} = {scores[i] - scores.max():+6.1f}"
              f"   e^ = {E[i]:.4f}   share = {E[i]:.4f} / {E.sum():.4f} = {E[i] / E.sum():.1%}")
    rest = E.sum() - sum(E[i] for i in top_i)
    print(f"    {'all other ' + str(len(intents) - len(top_i)):16s}  {'':24s}e^ = {rest:.4f}   (sum of the remaining answers)")
    print(f"    {'total':16s}  {'':24s}e^ = {E.sum():.4f}")

    print(f"\n    🎯 FINAL DECISION")
    print(f"    {'─' * 70}")
    print("    " + " vs ".join(f"{intents[i]} = {p[i]:3.0%}" for i in top_i))

# ---------- 6. Start ----------
db = open_db()
examples = load_examples(db)
m = load_model(db)
if m is None:
    print("🔄 training two-layer neural network ...")
    print(f"   layer 1: {len(set(f for c, s, _ in examples for f in features(c, s)))} words+context → {HIDDEN} hidden concepts")
    print(f"   layer 2: {HIDDEN} concepts → {len({i for _, _, i in examples})} possible answers")
    m = train(examples); save_model(db, m)
    print(f"✅ model ready with {m['W1'].size + m['W2'].size} trained weights")

def predict(context, sentence):
    feats = features(context, sentence)
    x = np.array([1.0 if f in feats else 0.0 for f in m["vocab"]])
    h, p = forward(m, x)
    return feats, x, h, p

context, bill, show, last = "start", 0.0, True, None

def learn(ctx, sentence, intent):
    global examples, m
    for known in list(INTENTS) + m["intents"]:        # reuse known intent spelling
        if known.casefold() == intent.casefold():
            intent = known
    old = db.execute("DELETE FROM examples WHERE context=? AND sentence=?",
                     (ctx, clean(sentence))).rowcount
    db.execute("INSERT INTO examples (context, sentence, intent) VALUES (?, ?, ?)",
               (ctx, clean(sentence), intent)); db.commit()
    if old:
        print(f"  (replaced {old} old example)")
    examples = load_examples(db)
    print(f"  🔄 retraining both layers ({len(examples)} examples total) ...")
    m = train(examples); save_model(db, m)
    print(f"  ✅ learned: [{ctx}] '{clean(sentence)}' -> {intent}")
    return intent

def ask_teacher():
    choices = list(INTENTS) + [i for i in m["intents"] if i not in INTENTS]
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
    global context, bill
    before = (context, bill)
    replies, price, new_ctx = INTENTS.get(intent, ([intent], 0, None))
    if intent == "tschuess" and bill > 0:
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

print(f"🍺🍺 Two-layer barkeeper: {len(examples)} examples, "
      f"{len(m['vocab'])} features -> {HIDDEN} concepts -> {len(m['intents'])} answers "
      f"= {m['W1'].size + m['W2'].size + HIDDEN + len(m['intents'])} weights")
print("Try:    kein bier  /  heute kein wein  /  ein weißbier  /  noch eins")
print("Teach:  if I don't understand, I'll ask you;  wrong answer -> falsch")
print("        or directly: noch eins = bier  /  [wein] noch eins = wein")
print("Other:  data | vergleich | log | reset | exit\n")

while True:
    said = input(f"you [{context}, {bill:.2f} €]> ").strip()
    cmd = said.lower()
    if cmd in ("exit", "quit", ""):
        break
    if cmd == "log":
        show = not show; print(f"  log {'on' if show else 'off'}\n"); continue
    if cmd == "reset":
        context, bill, last = "start", 0.0, None
        print("  (a new guest walks in)\n"); continue

    if cmd in ("data", "daten"):
        X, _ = to_matrix(examples, m["vocab"], m["intents"])
        _, P = forward(m, X)
        ok = 0
        for n, ((c, s, i), p) in enumerate(zip(examples, P), 1):
            guess = m["intents"][p.argmax()]; ok += guess == i
            mark = "✓" if guess == i else f"✗ says {guess}"
            print(f"  {n:3d}. [{c:9s}] {s:28s} = {i:16s} {mark}")
        print(f"\n  {ok}/{len(examples)} correct\n"); continue

    if cmd in ("vergleich", "compare"):              # one matrix vs two layers, same data
        print("  training the old one-matrix model on the same examples ...")
        one, v1, i1 = train_one_matrix(examples)
        X1, _ = to_matrix(examples, v1, i1)
        guess1 = [i1[k] for k in one(X1).argmax(1)]
        X2, _ = to_matrix(examples, m["vocab"], m["intents"])
        guess2 = [m["intents"][k] for k in forward(m, X2)[1].argmax(1)]
        ok1 = sum(g == i for g, (_, _, i) in zip(guess1, examples))
        ok2 = sum(g == i for g, (_, _, i) in zip(guess2, examples))
        print(f"\n  ONE MATRIX:  {ok1}/{len(examples)} correct")
        print(f"  TWO LAYERS:  {ok2}/{len(examples)} correct\n")
        print("  NEW sentences (never trained):")
        tests = [(c, clean(s), i) for c, s, i in TEST_SENTENCES]
        T1, _ = to_matrix(tests, v1, i1)
        T2, _ = to_matrix(tests, m["vocab"], m["intents"])
        t1 = [i1[k] for k in one(T1).argmax(1)]
        t2 = [m["intents"][k] for k in forward(m, T2)[1].argmax(1)]
        print("    " + "sentence".ljust(24) + "should be".ljust(11) + "one matrix".ljust(14) + "two layers")
        for (c, s, i), g1, g2 in zip(tests, t1, t2):
            print(f"    {s:24s}{i:11s}{g1:9s}{'✓' if g1 == i else '✗':5s}{g2:9s}{'✓' if g2 == i else '✗'}")
        print(f"    {'':35s}{sum(g == t[2] for g, t in zip(t1, tests)):2d}/{len(tests)}"
              f"         {sum(g == t[2] for g, t in zip(t2, tests)):2d}/{len(tests)}\n")
        continue

    if cmd in ("falsch", "wrong"):
        if not last:
            print("  nothing to correct yet\n"); continue
        ctx, sentence, wrong, (old_ctx, old_bill) = last
        print(f"  you said '{sentence}' in [{ctx}], I did: {wrong}")
        intent = ask_teacher()
        if intent:
            context, bill = old_ctx, old_bill
            intent = learn(ctx, sentence, intent)
            last = (ctx, sentence, intent, act(intent, 1.0))
        continue

    if "=" in said:
        left, intent = [s.strip() for s in said.split("=", 1)]
        ctx = context
        if left.startswith("[") and "]" in left:
            ctx, left = left[1:left.index("]")].strip().lower(), left[left.index("]") + 1:]
        if not clean(left) or not intent:
            print("  please write: sentence = answer\n"); continue
        learn(ctx, left, intent); print(); continue

    # ---- normal talking ----
    feats, x, h, p = predict(context, said)
    intent = m["intents"][p.argmax()]
    known_words = [f for f in feats[:-1] if f in m["vocab"]]
    if show:
        show_log(feats, m, x, h, p)
    too_few = len(known_words) < len(feats[:-1]) / 2
    if not known_words or too_few or p.max() < 0.4:
        print(f"\n  🍺 Barkeeper: Hä? Wie bitte?   (best guess: {intent}, {p.max():.0%})")
        intent = ask_teacher()
        if intent:
            intent = learn(context, said, intent)
            last = (context, clean(said), intent, act(intent, 1.0))
        continue
    last = (context, clean(said), intent, act(intent, p.max()))
