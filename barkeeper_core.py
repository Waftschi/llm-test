"""Barkeeper core: the two-layer neural network, without any input()/print().

    h = tanh(W1 · x + b1)      layer 1: words + context -> hidden concepts
    p = softmax(W2 · h + b2)   layer 2: concepts -> what the barkeeper does

Used by the console (barkeeper2.py) and the web server (server.py).
Everything that was printed before (decision log, blame) is returned as plain dicts.
"""
import numpy as np
import sqlite3, random, json, os

HERE = os.path.dirname(os.path.abspath(__file__))

# training parameters (shown in the web UI)
ROUNDS, LR, MOMENTUM, SEED, INIT_SCALE = 1500, 0.5, 0.9, 0, 0.5

# ---------- 1. Intents and seed examples ----------
DEFAULT_HIDDEN = 24                            # number of hidden concepts (layer 1 size)
DEFAULT_INTENTS = {
    "gruss":           (["Servus! Was darf's sein?"], 0, None),
    "bier":            (["Ein Bier, gern!"], 4.50, "bier"),
    "wein":            (["Ein Glas Wein, bitte schön."], 6.00, "wein"),
    "zahlen":          (["Das macht {bill:.2f} €. Danke dir!"], 0, "start"),
    "nichts_bestellt": (["Du hast doch noch gar nichts bestellt!"], 0, None),
    "tschuess":        (["Mach's gut, bis bald!"], 0, "start"),
}
DEFAULT_SUGGESTIONS = ["kein bier", "heute kein wein", "noch eins", "zahlen bitte", "tschüss"]
DEFAULT_FALLBACK = {"reply": "Hä? Wie bitte?", "min_confidence": 0.4}   # when nothing is understood
DEFAULT_EXAMPLES = [
    ("start", "hallo", "gruss"), ("start", "ein bier bitte", "bier"), ("start", "ein wein", "wein"),
    ("bier", "noch eins", "bier"), ("start", "noch eins", "nichts_bestellt"),
    ("bier", "zahlen bitte", "zahlen"), ("start", "zahlen bitte", "nichts_bestellt"),
    ("start", "tschüss", "tschuess"),
]

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

def train_one_matrix(examples, rounds=ROUNDS, lr=LR):  # the old model, for "compare"
    vocab   = sorted({f for c, s, _ in examples for f in features(c, s)})
    intents = sorted({i for _, _, i in examples})
    X, Y = to_matrix(examples, vocab, intents)
    W, v = np.zeros((len(intents), len(vocab))), 0
    for _ in range(rounds):
        P = softmax(X @ W.T)
        v = MOMENTUM * v - lr * (P - Y).T @ X / len(examples)
        W += v
    return lambda X_: softmax(X_ @ W.T), vocab, intents


class Session:
    """One guest at the bar: where the conversation is, what they owe."""
    def __init__(self):
        self.context, self.bill = "start", 0.0
        self.last = None        # (context, sentence, intent, state before) of the last answer
        self.pending = None     # waiting for the teacher: {"kind", "ctx", "sentence", "old"}

    def state(self):
        return {"context": self.context, "bill": round(self.bill, 2)}


class Barkeeper:
    """The trained model + its database. Shared by all sessions (learning is global)."""
    def __init__(self, profile=""):
        self.profile = profile
        self.simple = profile.startswith("simple")       # "simple" (German) or "simple-en" (English)
        stem = "barkeeper_" + profile.replace("-", "_") if self.simple else None
        self.db_file = os.path.join(HERE, stem + ".db" if stem else "barkeeper2.db")
        seed_file = os.path.join(HERE, stem + "_seed.json" if stem else "barkeeper_seed.json")
        self.hidden, self.intent_defs = DEFAULT_HIDDEN, DEFAULT_INTENTS
        self.fallback, self.seed_examples = DEFAULT_FALLBACK, list(DEFAULT_EXAMPLES)
        self.suggestions = DEFAULT_SUGGESTIONS               # example phrases shown as buttons / hints
        if os.path.exists(seed_file):
            seed = json.load(open(seed_file, encoding="utf-8"))
            self.intent_defs = {k: (v["replies"], v["price"], v["context"]) for k, v in seed["intents"].items()}
            self.seed_examples = [tuple(e) for e in seed["examples"]]
            self.hidden = seed.get("hidden", self.hidden)
            self.suggestions = seed.get("suggestions", self.suggestions)
            self.fallback = {**self.fallback, **seed.get("fallback", {})}
        if not self.simple:
            self.seed_examples += KEIN_EXAMPLES

        self.db = self.open_db()
        self.examples = self.load_examples()
        self.m = self.load_model()
        self.trained_on_start = self.m is None
        if self.trained_on_start:
            self.m = self.train(self.examples)
            self.save_model(self.m)

    # ---------- training ----------
    def train(self, examples, rounds=ROUNDS, lr=LR):
        vocab   = sorted({f for c, s, _ in examples for f in features(c, s)})
        intents = sorted({i for _, _, i in examples})
        X, Y = to_matrix(examples, vocab, intents)
        np.random.seed(SEED)                             # same start every time -> repeatable
        m = {"vocab": vocab, "intents": intents,
             "W1": np.random.randn(self.hidden, len(vocab)) * INIT_SCALE, "b1": np.zeros(self.hidden),
             "W2": np.random.randn(len(intents), self.hidden) * INIT_SCALE, "b2": np.zeros(len(intents))}
        # "momentum": keep a bit of the last step, so learning rolls faster downhill
        v = {k: np.zeros_like(m[k]) for k in ("W1", "b1", "W2", "b2")}
        for _ in range(rounds):
            Hd, P = forward(m, X)                        # guess (for all examples at once)
            D2 = (P - Y) / len(examples)                 # error at the output
            D1 = (D2 @ m["W2"]) * (1 - Hd**2)            # error passed back to layer 1
            grads = {"W2": D2.T @ Hd, "b2": D2.sum(0),   # blame for every weight
                     "W1": D1.T @ X,  "b1": D1.sum(0)}
            for k in grads:                              # correct
                v[k] = MOMENTUM * v[k] - lr * grads[k]
                m[k] += v[k]
        return m

    # ---------- database: model stored as readable JSON ----------
    def open_db(self):
        db = sqlite3.connect(self.db_file)
        db.execute("CREATE TABLE IF NOT EXISTS examples (id INTEGER PRIMARY KEY, context TEXT, sentence TEXT, intent TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS model (name TEXT PRIMARY KEY, value TEXT)")
        if not db.execute("SELECT 1 FROM examples").fetchone():
            db.executemany("INSERT INTO examples (context, sentence, intent) VALUES (?, ?, ?)",
                           [(c, clean(s), i) for c, s, i in self.seed_examples])
            db.commit()
        return db

    def load_examples(self):
        return self.db.execute("SELECT context, sentence, intent FROM examples ORDER BY id").fetchall()

    def save_model(self, m):
        self.db.execute("DELETE FROM model")
        self.db.executemany("INSERT INTO model VALUES (?, ?)",
                            [(k, json.dumps(v.tolist() if isinstance(v, np.ndarray) else v, ensure_ascii=False))
                             for k, v in m.items()])
        self.db.commit()

    def load_model(self):
        rows = dict(self.db.execute("SELECT name, value FROM model").fetchall())
        if not rows:
            return None
        m = {k: json.loads(v) for k, v in rows.items()}
        for k in ("W1", "b1", "W2", "b2"):
            m[k] = np.array(m[k])
        return m

    # ---------- prediction ----------
    def predict(self, context, sentence):
        feats = features(context, sentence)
        x = np.array([1.0 if f in feats else 0.0 for f in self.m["vocab"]])
        h, p = forward(self.m, x)
        return feats, x, h, p

    def choices(self):
        return list(self.intent_defs) + [i for i in self.m["intents"] if i not in self.intent_defs]

    def n_weights(self):
        m = self.m
        return int(m["W1"].size + m["W2"].size + m["b1"].size + m["b2"].size)

    # ---------- decision log through both layers (as data) ----------
    def decision_log(self, feats, x, h, p):
        m = self.m
        vocab, intents = m["vocab"], m["intents"]
        top_i = np.argsort(p)[::-1][:2]                   # the 2 best answers
        # the 4 concepts with the biggest vote for the winning answer (|W2 · h|), not just the most active ones
        strong = np.argsort(-np.abs(m["W2"][top_i[0]] * h))[:4]

        layer1 = []
        for k in strong:
            drivers = sorted([(m["W1"][k, j], vocab[j]) for j in np.nonzero(x)[0]],
                             key=lambda t: -abs(t[0]))[:3]
            word_sum = float(m["W1"][k] @ x)              # all active word weights added up
            z = word_sum + float(m["b1"][k])
            act = float(h[k])
            layer1.append({"concept": int(k), "activation": act,
                           "label": "strong" if act > 0.5 else ("weak" if act < -0.5 else "neutral"),
                           "drivers": [{"word": w, "weight": float(v)} for v, w in drivers],
                           "word_sum": word_sum, "bias": float(m["b1"][k]), "z": z, "tanh": float(np.tanh(z))})

        scores = m["W2"] @ h + m["b2"]                    # score of every answer
        E = np.exp(scores - scores.max())                 # softmax step 1 (shifted by the best score)
        layer2 = []
        for i in top_i:
            votes = [float(m["W2"][i, k] * h[k]) for k in strong]
            shown = sum(votes)
            others = float(m["W2"][i] @ h) - shown        # concepts not listed above
            layer2.append({"intent": intents[i], "votes": votes, "shown": shown, "others": others,
                           "bias": float(m["b2"][i]), "score": shown + others + float(m["b2"][i])})
        softmax_rows = [{"intent": intents[i], "score": float(scores[i]),
                         "shifted": float(scores[i] - scores.max()), "exp": float(E[i]),
                         "share": float(E[i] / E.sum())} for i in top_i]
        return {"unknown": [f for f in feats if f not in vocab],
                "layer1": {"top_intent": intents[top_i[0]], "concepts": layer1},
                "layer2": {"concepts": [int(k) for k in strong], "answers": layer2},
                "softmax": {"best_score": float(scores.max()), "answers": softmax_rows,
                            "others_count": len(intents) - len(top_i),
                            "others_exp": float(E.sum() - sum(E[i] for i in top_i)),
                            "total_exp": float(E.sum())},
                "final": [{"intent": intents[i], "p": float(p[i])} for i in top_i]}

    # ---------- blame before retraining (as data) ----------
    def blame(self, ctx, sentence, intent):
        """How wrong is the current model for this example, and who is to blame?"""
        m = self.m
        feats, x, h, p = self.predict(ctx, sentence)
        intents = m["intents"]
        y = np.zeros(len(intents))
        if intent in intents:
            y[intents.index(intent)] = 1.0
        D2 = p - y                                        # error of every answer
        D1 = (m["W2"].T @ D2) * (1 - h**2)                # blame passed back to the concepts
        answers = [{"intent": intents[i], "p": float(p[i]), "right": float(y[i]), "d2": float(D2[i])}
                   for i in np.argsort(-np.abs(D2))[:3]]
        concepts = []
        for k in np.argsort(-np.abs(D1))[:3]:
            drivers = sorted([(m["W1"][k, j], m["vocab"][j]) for j in np.nonzero(x)[0]],
                             key=lambda t: -abs(t[0]))[:3]
            concepts.append({"concept": int(k), "blame": float(D1[k]), "h": float(h[k]),
                             "drivers": [{"word": w, "weight": float(v)} for v, w in drivers]})
        return {"sentence": clean(sentence), "context": ctx, "intent": intent,
                "new_words": [f for f in feats if f not in m["vocab"]],
                "new_intent": intent if intent not in intents else None,
                "answers": answers, "concepts": concepts}

    # ---------- learning ----------
    def learn(self, ctx, sentence, intent):
        for known in list(self.intent_defs) + self.m["intents"]:   # reuse known intent spelling
            if known.casefold() == intent.casefold():
                intent = known
        blame = self.blame(ctx, sentence, intent)
        old = self.db.execute("DELETE FROM examples WHERE context=? AND sentence=?",
                              (ctx, clean(sentence))).rowcount
        self.db.execute("INSERT INTO examples (context, sentence, intent) VALUES (?, ?, ?)",
                        (ctx, clean(sentence), intent)); self.db.commit()
        self.examples = self.load_examples()
        self.m = self.train(self.examples); self.save_model(self.m)
        return {"context": ctx, "sentence": clean(sentence), "intent": intent,
                "replaced": old, "examples": len(self.examples), "blame": blame}

    # ---------- the barkeeper acts ----------
    def act(self, s, intent, prob):
        before = (s.context, s.bill)
        replies, price, new_ctx = self.intent_defs.get(intent, ([intent], 0, None))
        if intent == "tschuess" and s.bill > 0:
            return {"intent": intent, "prob": prob, "blocked": True, "before": before,
                    "reply": f"Halt! Erst zahlen, {s.bill:.2f} € bitte!"}
        reply = random.choice(replies).format(bill=s.bill)
        if intent == "zahlen":
            s.bill = 0.0
        s.bill += price
        if new_ctx:
            s.context = new_ctx
        return {"intent": intent, "prob": prob, "blocked": False, "before": before, "reply": reply}

    def say(self, s, said):
        feats, x, h, p = self.predict(s.context, said)
        intent = self.m["intents"][int(p.argmax())]
        known_words = [f for f in feats[:-1] if f in self.m["vocab"]]
        res = {"type": "say", "text": said, "context_before": s.context, "features": feats,
               "log": self.decision_log(feats, x, h, p), "intent": intent, "confidence": float(p.max())}
        too_few = len(known_words) < len(feats[:-1]) / 2
        if not known_words or too_few or p.max() < self.fallback["min_confidence"]:
            s.pending = {"kind": "unknown", "ctx": s.context, "sentence": said, "old": (s.context, s.bill)}
            res.update(understood=False, reply=self.fallback["reply"], choices=self.choices())
        else:
            ctx = s.context
            a = self.act(s, intent, float(p.max()))
            s.last = (ctx, clean(said), intent, a["before"])
            res.update(understood=True, act=a, reply=a["reply"])
        res.update(s.state())
        return res

    def teach(self, s, intent):
        """The teacher answered the pending question (unknown sentence or 'falsch')."""
        pend, s.pending = s.pending, None
        if not pend:
            return {"type": "error", "message": "nobody asked me anything"}
        if pend["kind"] == "wrong":
            s.context, s.bill = pend["old"]
        learned = self.learn(pend["ctx"], pend["sentence"], intent)
        a = self.act(s, learned["intent"], 1.0)
        s.last = (pend["ctx"], clean(pend["sentence"]), learned["intent"], a["before"])
        return {"type": "taught", "learned": learned, "act": a, "reply": a["reply"], **s.state()}

    def handle(self, s, said):
        """One line of chat, like the console: reset / falsch / 'sentence = answer' / normal talking."""
        cmd = said.lower().strip()
        if cmd == "reset":
            s.__init__()
            return {"type": "reset", **s.state()}
        if cmd in ("falsch", "wrong"):
            if not s.last:
                return {"type": "error", "message": "nothing to correct yet"}
            ctx, sentence, wrong, old = s.last
            s.pending = {"kind": "wrong", "ctx": ctx, "sentence": sentence, "old": old}
            return {"type": "wrong", "context": ctx, "sentence": sentence, "wrong_intent": wrong,
                    "choices": self.choices()}
        if said.count("=") > 1:   # two examples in one line would become one wrong answer
            return {"type": "error", "message": "one example per line, please use only one '=': sentence = answer"}
        if "=" in said:
            left, intent = [t.strip() for t in said.split("=", 1)]
            ctx = s.context
            if left.startswith("[") and "]" in left:
                ctx, left = left[1:left.index("]")].strip().lower(), left[left.index("]") + 1:]
            if not clean(left) or not intent:
                return {"type": "error", "message": "please write: sentence = answer"}
            return {"type": "learned", "learned": self.learn(ctx, left, intent), **s.state()}
        return self.say(s, said)

    # ---------- inspection ----------
    def data(self):
        X, _ = to_matrix(self.examples, self.m["vocab"], self.m["intents"])
        _, P = forward(self.m, X)
        rows = []
        for n, ((c, s, i), p) in enumerate(zip(self.examples, P), 1):
            guess = self.m["intents"][int(p.argmax())]
            rows.append({"n": n, "context": c, "sentence": s, "intent": i, "guess": guess, "ok": guess == i})
        return {"rows": rows, "correct": sum(r["ok"] for r in rows), "total": len(rows)}

    def compare(self):
        """One matrix vs two layers, same data."""
        m, ex = self.m, self.examples
        one, v1, i1 = train_one_matrix(ex)
        X1, _ = to_matrix(ex, v1, i1)
        guess1 = [i1[k] for k in one(X1).argmax(1)]
        X2, _ = to_matrix(ex, m["vocab"], m["intents"])
        guess2 = [m["intents"][k] for k in forward(m, X2)[1].argmax(1)]
        tests = [(c, clean(s), i) for c, s, i in TEST_SENTENCES]
        T1, _ = to_matrix(tests, v1, i1)
        T2, _ = to_matrix(tests, m["vocab"], m["intents"])
        t1 = [i1[k] for k in one(T1).argmax(1)]
        t2 = [m["intents"][k] for k in forward(m, T2)[1].argmax(1)]
        return {"total": len(ex),
                "one_ok": sum(g == i for g, (_, _, i) in zip(guess1, ex)),
                "two_ok": sum(g == i for g, (_, _, i) in zip(guess2, ex)),
                "tests": [{"sentence": s, "expected": i, "one": a, "two": b}
                          for (c, s, i), a, b in zip(tests, t1, t2)],
                "one_test_ok": sum(g == t[2] for g, t in zip(t1, tests)),
                "two_test_ok": sum(g == t[2] for g, t in zip(t2, tests))}

    def params(self):
        m = self.m
        return {"profile": self.profile or "default", "suggestions": self.suggestions, "hidden": self.hidden,
                "vocab_size": len(m["vocab"]), "n_intents": len(m["intents"]),
                "n_examples": len(self.examples), "vocab": m["vocab"], "intents": m["intents"],
                "w1_shape": list(m["W1"].shape), "w2_shape": list(m["W2"].shape),
                "w1_size": int(m["W1"].size), "w2_size": int(m["W2"].size),
                "b1_size": int(m["b1"].size), "b2_size": int(m["b2"].size),
                "n_weights": self.n_weights(),
                "rounds": ROUNDS, "learning_rate": LR, "momentum": MOMENTUM,
                "seed": SEED, "init_scale": INIT_SCALE,
                "min_confidence": self.fallback["min_confidence"], "fallback_reply": self.fallback["reply"],
                "intent_defs": {k: {"replies": v[0], "price": v[1], "context": v[2]}
                                for k, v in self.intent_defs.items()}}
