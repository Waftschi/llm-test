"""One matrix vs. two layers.

A guest who only drinks beer or wine:
  "bier bitte"  -> bier      "kein bier" -> wein
  "wein bitte"  -> wein      "kein wein" -> bier

"kein" alone means nothing. It only matters in COMBINATION with the drink.
One matrix can only ADD word votes, so it can never learn this.
A second layer (hidden layer) can learn combinations.
"""
import numpy as np
np.random.seed(1)

data = [
    ("bier bitte", "bier"), ("ein bier", "bier"),
    ("wein bitte", "wein"), ("ein wein", "wein"),
    ("kein bier", "wein"),  ("bitte kein bier", "wein"),
    ("kein wein", "bier"),  ("bitte kein wein", "bier"),
]
words   = sorted({w for q, _ in data for w in q.split()})
answers = sorted({a for _, a in data})

def vec(q):
    return np.array([1.0 if w in q.split() else 0.0 for w in words])

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

def onehot(a):
    t = np.zeros(len(answers)); t[answers.index(a)] = 1
    return t

def test(predict, title):
    print(f"\n{title}")
    ok = 0
    for q, a in data:
        p = predict(vec(q))
        guess = answers[p.argmax()]
        ok += guess == a
        print(f"   {q:18s} -> {guess:5s} {p.max():4.0%}   {'✓' if guess == a else '✗ should be ' + a}")
    print(f"   {ok}/{len(data)} correct")

# =====================================================================
# A) ONE MATRIX (like all our programs so far)
#    p = softmax(W · x)
# =====================================================================
W = np.zeros((len(answers), len(words)))
for _ in range(2000):
    for q, a in data:
        x = vec(q)
        p = softmax(W @ x)
        W -= 0.1 * np.outer(p - onehot(a), x)

test(lambda x: softmax(W @ x), "A) ONE MATRIX  (answers x words)")

# =====================================================================
# B) TWO LAYERS
#    h = tanh(W1 · x + b1)     layer 1: words -> hidden "concepts"
#    p = softmax(W2 · h + b2)  layer 2: concepts -> answers
# =====================================================================
H = 3                                            # number of hidden neurons
W1 = np.random.randn(H, len(words)) * 0.5        # concepts x words
b1 = np.zeros(H)
W2 = np.random.randn(len(answers), H) * 0.5      # answers x concepts
b2 = np.zeros(len(answers))

for _ in range(3000):
    for q, a in data:
        x = vec(q)
        # forward: guess
        h = np.tanh(W1 @ x + b1)
        p = softmax(W2 @ h + b2)
        # backward: who is to blame for the error? (chain rule)
        d2 = p - onehot(a)                       # error at the output (as before)
        d1 = (W2.T @ d2) * (1 - h**2)            # error passed back to the hidden layer
        # correct both layers
        W2 -= 0.1 * np.outer(d2, h); b2 -= 0.1 * d2
        W1 -= 0.1 * np.outer(d1, x); b1 -= 0.1 * d1

def two_layers(x):
    return softmax(W2 @ np.tanh(W1 @ x + b1) + b2)

test(two_layers, f"B) TWO LAYERS  (words -> {H} concepts -> answers)")

# =====================================================================
# Decision log through both layers
# =====================================================================
print("\n" + "=" * 62)
print("DECISION LOG (two layers)")
print("=" * 62)
print("\nLayer 1: how much each word excites each concept")
print("   " + "word".ljust(10) + "".join(f"concept{k}".rjust(10) for k in range(H)))
for j, w in enumerate(words):
    print("   " + w.ljust(10) + "".join(f"{W1[k, j]:+10.2f}" for k in range(H)))
print("\nLayer 2: what each active concept votes for")
print("   " + "answer".ljust(10) + "".join(f"concept{k}".rjust(10) for k in range(H)))
for i, a in enumerate(answers):
    print("   " + a.ljust(10) + "".join(f"{W2[i, k]:+10.2f}" for k in range(H)))

for q in ["bier bitte", "kein bier", "kein wein"]:
    h = np.tanh(W1 @ vec(q) + b1)
    p = softmax(W2 @ h + b2)
    print(f"\n'{q}'")
    print("   concepts:   " + "  ".join(f"concept{k}={h[k]:+.2f}" for k in range(H)))
    print("   answer:     " + "  ".join(f"{a}={v:.0%}" for a, v in zip(answers, p)))
