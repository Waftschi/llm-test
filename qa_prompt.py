import numpy as np
import sqlite3

DB_FILE = "model.db"                      # the whole model lives in this one file

# 1. Starting examples (only used if the database is empty)
default_qa = [
    ("wie heißt du",                    "Claude"),
    ("was ist dein name",               "Claude"),
    ("wo liegt frankfurt",              "Hessen"),
    ("in welchem land liegt frankfurt", "Deutschland"),
    ("wie alt bist du",                 "neu"),
]

def train(qa, rounds=300):
    words   = sorted({w for q, _ in qa for w in q.split()})
    answers = sorted({a for _, a in qa})
    W = np.zeros((len(answers), len(words)))
    for _ in range(rounds):
        for q, a in qa:
            x = to_vector(q, words)
            p = softmax(W @ x)
            target = np.zeros(len(answers)); target[answers.index(a)] = 1
            W -= 0.1 * np.outer(p - target, x)
    return W, words, answers

def clean(text):                          # ignore case and punctuation
    text = text.casefold()          # casefold: also "ß" == "ss"
    for ch in "?!.,;:":
        text = text.replace(ch, "")
    return " ".join(text.split())

def to_vector(question, words):
    return np.array([1.0 if w in question.split() else 0.0 for w in words])

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

# ---------- database ----------
def open_db():
    db = sqlite3.connect(DB_FILE)
    db.execute("CREATE TABLE IF NOT EXISTS examples (id INTEGER PRIMARY KEY, question TEXT, answer TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS weights  (answer TEXT, word TEXT, value REAL)")
    return db

def load_examples(db):
    return db.execute("SELECT question, answer FROM examples ORDER BY id").fetchall()

def add_example(db, q, a):
    db.execute("INSERT INTO examples (question, answer) VALUES (?, ?)", (q, a))
    db.commit()

def save_weights(db, W, words, answers):
    db.execute("DELETE FROM weights")
    db.executemany("INSERT INTO weights VALUES (?, ?, ?)",
                   [(a, w, float(W[i, j])) for i, a in enumerate(answers)
                                           for j, w in enumerate(words)])
    db.commit()

def load_weights(db):
    rows = db.execute("SELECT answer, word, value FROM weights").fetchall()
    if not rows:
        return None
    answers = sorted({r[0] for r in rows})
    words   = sorted({r[1] for r in rows})
    W = np.zeros((len(answers), len(words)))
    for a, w, v in rows:
        W[answers.index(a), words.index(w)] = v
    return W, words, answers

def answer(prompt, W, words, answers):
    prompt = clean(prompt)
    x = to_vector(prompt, words)
    scores = W @ x
    p = softmax(scores)

    known   = [w for w in prompt.split() if w in words]
    unknown = [w for w in prompt.split() if w not in words]
    print()
    if unknown:
        print(f"  unknown words (ignored): {unknown}")
    if not known:
        print("  I don't know any of these words -> no idea.\n")
        return
    print("  " + "word".ljust(12) + "".join(a.rjust(13) for a in answers))
    for w in known:
        j = words.index(w)
        print("  " + w.ljust(12) + "".join(f"{W[i, j]:+13.2f}" for i in range(len(answers))))
    print("  " + "SUM".ljust(12) + "".join(f"{s:+13.2f}" for s in scores))
    print("  " + "probability".ljust(12) + "".join(f"{v:13.0%}" for v in p))
    print(f"\n  >> {answers[p.argmax()]}  ({p.max():.0%} sure)\n")

# 2. Load the model from the database (or create it the first time)
db = open_db()
qa = load_examples(db)
if not qa:                                # first start: fill with default examples
    for q, a in default_qa:
        add_example(db, clean(q), a)
    qa = load_examples(db)

model = load_weights(db)
if model:
    W, words, answers = model
    print(f"Loaded model from {DB_FILE}: {len(qa)} examples, "
          f"{len(answers)} answers x {len(words)} words = {W.size} weights\n")
else:
    W, words, answers = train(qa)
    save_weights(db, W, words, answers)
    print(f"Trained new model and saved it to {DB_FILE}\n")

print("Ask me something (e.g. 'wie heißt du?').")
print("Teach me:  <question> = <answer>")
print("Show data: data")
print("Quit:      exit\n")

while True:
    prompt = input("prompt> ").strip()
    if prompt.lower() in ("exit", "quit", ""):
        break
    if prompt.lower() in ("data", "daten"):   # show + test all examples
        print(f"\n  {len(qa)} examples:\n")
        correct = 0
        for n, (q, a) in enumerate(qa, 1):
            p = softmax(W @ to_vector(q, words))
            guess = answers[p.argmax()]
            ok = guess == a
            correct += ok
            mark = "✓" if ok else f"✗ model says {guess}"
            print(f"  {n:2d}. {q:35s} = {a:12s} {p.max():4.0%}  {mark}")
        print(f"\n  {correct}/{len(qa)} correct\n")
        continue
    if "=" in prompt:                       # "question = answer" -> learn it
        if prompt.lower().startswith("lerne:"):
            prompt = prompt[len("lerne:"):]  # old style still works
        q, a = prompt.split("=", 1)
        if not clean(q) or not a.strip():
            print("  please write: question = answer\n")
            continue
        a = a.strip()
        for existing in answers:            # "berlin" = "Berlin": reuse known spelling
            if existing.casefold() == a.casefold():
                a = existing
        add_example(db, clean(q), a)
        qa = load_examples(db)
        W, words, answers = train(qa)
        save_weights(db, W, words, answers)
        print(f"  learned: '{q.strip()}' -> {a}  (retrained, saved to {DB_FILE})\n")
        continue
    answer(prompt, W, words, answers)
