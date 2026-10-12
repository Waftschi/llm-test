#!/usr/bin/env python3
"""Barkeeper web server: chat + teaching in the browser. Run, then open http://localhost:8765"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, os, random, sqlite3, sys
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SIMPLE    = os.environ.get("BARKEEPER_PROFILE") == "simple"
DB_FILE   = "barkeeper_simple.db" if SIMPLE else "barkeeper2.db"
SEED_FILE = "barkeeper_simple_seed.json" if SIMPLE else "barkeeper_seed.json"
HIDDEN    = 24

INTENTS = {
    "gruss":           (["Servus! Was darf's sein?"], 0, None),
    "bier":            (["Ein Bier, gern!"], 4.50, "bier"),
    "wein":            (["Ein Glas Wein, bitte schön."], 6.00, "wein"),
    "zahlen":          (["Das macht {bill:.2f} €. Danke dir!"], 0, "start"),
    "nichts_bestellt": (["Du hast doch noch gar nichts bestellt!"], 0, None),
    "tschuess":        (["Mach's gut, bis bald!"], 0, "start"),
}
if os.path.exists(SEED_FILE):
    seed = json.load(open(SEED_FILE, encoding="utf-8"))
    INTENTS = {k: (v["replies"], v["price"], v["context"]) for k, v in seed["intents"].items()}
    HIDDEN = seed.get("hidden", HIDDEN)

# ---------- model (same maths as barkeeper2.py) ----------
def clean(text):
    text = text.casefold()
    for ch in "?!.,;:":
        text = text.replace(ch, "")
    return " ".join(text.split())

def features(context, sentence):
    return clean(sentence).split() + [f"[{context}]"]

def softmax(Z):
    E = np.exp(Z - Z.max(axis=-1, keepdims=True))
    return E / E.sum(axis=-1, keepdims=True)

def forward(m, X):
    Hd = np.tanh(X @ m["W1"].T + m["b1"])
    return Hd, softmax(Hd @ m["W2"].T + m["b2"])

def to_matrix(examples, vocab, intents):
    col = {f: j for j, f in enumerate(vocab)}
    X = np.zeros((len(examples), len(vocab)))
    Y = np.zeros((len(examples), len(intents)))
    for n, (c, s, i) in enumerate(examples):
        for f in features(c, s):
            if f in col:
                X[n, col[f]] = 1
        Y[n, intents.index(i)] = 1
    return X, Y

def train(examples, rounds=1500, lr=0.5):
    vocab   = sorted({f for c, s, _ in examples for f in features(c, s)})
    intents = sorted({i for _, _, i in examples})
    X, Y = to_matrix(examples, vocab, intents)
    np.random.seed(0)
    m = {"vocab": vocab, "intents": intents,
         "W1": np.random.randn(HIDDEN, len(vocab)) * 0.5, "b1": np.zeros(HIDDEN),
         "W2": np.random.randn(len(intents), HIDDEN) * 0.5, "b2": np.zeros(len(intents))}
    v = {k: np.zeros_like(m[k]) for k in ("W1", "b1", "W2", "b2")}
    for _ in range(rounds):
        Hd, P = forward(m, X)
        D2 = (P - Y) / len(examples)
        D1 = (D2 @ m["W2"]) * (1 - Hd**2)
        grads = {"W2": D2.T @ Hd, "b2": D2.sum(0), "W1": D1.T @ X, "b1": D1.sum(0)}
        for k in grads:
            v[k] = 0.9 * v[k] - lr * grads[k]
            m[k] += v[k]
    return m

# ---------- database ----------
def db_connect():
    db = sqlite3.connect(DB_FILE)
    db.execute("CREATE TABLE IF NOT EXISTS examples (id INTEGER PRIMARY KEY, context TEXT, sentence TEXT, intent TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS model (name TEXT PRIMARY KEY, value TEXT)")
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

print("Loading model...", flush=True)
_db = db_connect()
m = load_model(_db)
if m is None:
    print("No model yet - run 'python3 barkeeper2.py' (or 'python3 barkeeper2.py simple') once first.", flush=True)
    sys.exit(1)
print(f"Model loaded: {len(load_examples(_db))} examples", flush=True)
_db.close()

# ---------- conversation ----------
S = {"context": "start", "bill": 0.0, "last": None, "pending": None}

def choices():
    return list(INTENTS) + [i for i in m["intents"] if i not in INTENTS]

def menu():
    return ["Was hätte ich tun sollen? Nummer oder neue Antwort eintippen ('skip' = überspringen):"] + \
           [f"{n:2d}. {c}" for n, c in enumerate(choices(), 1)]

def learn(ctx, sentence, intent):
    global m
    for known in choices():
        if known.casefold() == intent.casefold():
            intent = known
    db = db_connect()
    db.execute("DELETE FROM examples WHERE context=? AND sentence=?", (ctx, clean(sentence)))
    db.execute("INSERT INTO examples (context, sentence, intent) VALUES (?, ?, ?)", (ctx, clean(sentence), intent))
    db.commit()
    m = train(load_examples(db))
    save_model(db, m)
    db.close()
    return intent

def act(intent):
    before = (S["context"], S["bill"])
    replies, price, new_ctx = INTENTS.get(intent, ([intent], 0, None))
    if intent == "tschuess" and S["bill"] > 0:
        return f"Halt! Erst zahlen, {S['bill']:.2f} € bitte!", before
    reply = random.choice(replies).format(bill=S["bill"])
    if intent == "zahlen":
        S["bill"] = 0.0
    S["bill"] += price
    if new_ctx:
        S["context"] = new_ctx
    return reply, before

def build_log(feats, x, h, p):
    vocab, intents = m["vocab"], m["intents"]
    top = np.argsort(p)[::-1][:2]
    strong = np.argsort(-np.abs(h))[:4]
    sums = []
    for i in top:
        shown = float(sum(m["W2"][i, k] * h[k] for k in strong))
        others = float(m["W2"][i] @ h) - shown
        sums.append({"shown": shown, "others": others, "bias": float(m["b2"][i]),
                     "score": shown + others + float(m["b2"][i])})
    scores = m["W2"] @ h + m["b2"]
    E = np.exp(scores - scores.max())
    soft = {"best": float(scores.max()), "total": float(E.sum()),
            "rest": float(E.sum() - sum(E[i] for i in top)), "rest_n": len(intents) - len(top),
            "rows": [{"intent": intents[i], "score": float(scores[i]), "exp": float(E[i])} for i in top]}
    concepts = []
    for k in strong:
        drivers = sorted([(float(m["W1"][k, j]), vocab[j]) for j in np.nonzero(x)[0]],
                         key=lambda t: -abs(t[0]))[:3]
        word_sum = float(m["W1"][k] @ x)
        concepts.append({"id": int(k), "activation": float(h[k]),
                         "word_sum": word_sum, "bias": float(m["b1"][k]), "z": word_sum + float(m["b1"][k]),
                         "drivers": [{"word": w, "weight": v} for v, w in drivers],
                         "votes": [float(m["W2"][i, k] * h[k]) for i in top]})
    return {"unknown": [f for f in feats if f not in vocab],
            "softmax": soft,
            "top": [{"intent": intents[i], "prob": float(p[i]), **sums[n]} for n, i in enumerate(top)],
            "concepts": concepts}

def handle(said):
    S["log"] = None
    cmd = said.lower()
    p = S["pending"]

    if p:                                            # waiting for a teaching answer
        S["pending"] = None
        if cmd in ("skip", "weiter", "abbrechen"):
            return ["ok, übersprungen"]
        opts = choices()
        pick = opts[int(cmd) - 1] if cmd.isdigit() and 1 <= int(cmd) <= len(opts) else said.strip()
        if p["kind"] == "fix":
            S["context"], S["bill"] = p["before"]
        intent = learn(p["ctx"], p["sentence"], pick)
        reply, before = act(intent)
        S["last"] = (p["ctx"], clean(p["sentence"]), intent, before)
        return [f"✅ gelernt: [{p['ctx']}] '{clean(p['sentence'])}' → {intent}", reply]

    if cmd in ("reset", "neu"):
        S.update(context="start", bill=0.0, last=None, pending=None)
        return ["(ein neuer Gast kommt herein)"]

    if cmd in ("falsch", "wrong"):
        if not S["last"]:
            return ["Es gibt noch nichts zu korrigieren."]
        ctx, sentence, wrong, before = S["last"]
        S["pending"] = {"kind": "fix", "ctx": ctx, "sentence": sentence, "before": before}
        return [f"Du sagtest '{sentence}' in [{ctx}], ich tat: {wrong}"] + menu()

    if "=" in said:                                  # direct teaching: [ctx] satz = antwort
        left, intent = [s.strip() for s in said.split("=", 1)]
        ctx = S["context"]
        if left.startswith("[") and "]" in left:
            ctx, left = left[1:left.index("]")].strip().lower(), left[left.index("]") + 1:]
        if not clean(left) or not intent:
            return ["Bitte so schreiben: satz = antwort"]
        intent = learn(ctx, left, intent)
        return [f"✅ gelernt: [{ctx}] '{clean(left)}' → {intent}"]

    feats = features(S["context"], said)
    x = np.array([1.0 if f in feats else 0.0 for f in m["vocab"]])
    h_out, p_out = forward(m, x)
    intent = m["intents"][p_out.argmax()]
    conf = float(p_out.max())
    S["log"] = build_log(feats, x, h_out, p_out)
    known = [f for f in feats[:-1] if f in m["vocab"]]
    if not known or len(known) < len(feats[:-1]) / 2 or conf < 0.4:
        S["pending"] = {"kind": "teach", "ctx": S["context"], "sentence": said}
        return [f"Hä? Wie bitte? (beste Vermutung: {intent}, {conf:.0%})"] + menu()
    ctx = S["context"]
    reply, before = act(intent)
    S["last"] = (ctx, clean(said), intent, before)
    return [f"{reply}   ({intent}, {conf:.0%})", "(falsch? tippe: falsch)"]

# ---------- HTTP ----------
class Handler(BaseHTTPRequestHandler):
    def json_response(self, code, data):
        self.send_response(code)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path == "/":
            page = open("barkeeper_chat.html", "rb").read()
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page)
        elif self.path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(b"OK")
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        try:
            if self.path == "/chat":
                length = int(self.headers.get("Content-Length", 0))
                message = json.loads(self.rfile.read(length).decode()).get("message", "").strip()
                if not message:
                    return self.json_response(400, {"error": "Empty"})
                lines = handle(message)
                self.json_response(200, {"response": "\n".join(lines), "context": S["context"],
                                         "bill": float(S["bill"]), "teaching": bool(S["pending"]),
                                         "log": S.get("log")})
            elif self.path == "/reset":
                S.update(context="start", bill=0.0, last=None, pending=None)
                self.json_response(200, {"context": "start", "bill": 0.0})
            else:
                self.send_response(404); self.end_headers()
        except Exception as e:
            self.json_response(500, {"error": str(e)})

    def log_message(self, *args):
        pass

if __name__ == "__main__":
    print("Barkeeper server: http://localhost:8765  (Ctrl+C to stop)", flush=True)
    try:
        HTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
