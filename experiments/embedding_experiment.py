"""Embedding experiment: why can an LLM guess that "kein merlot" means "bier" without being told?

Separate from the barkeeper (nothing here touches barkeeper2.py or its databases).

    python3 experiments/embedding_experiment.py        # the comparison table
    python3 experiments/embedding_experiment.py chat   # type your own sentences

Two small networks learn the SAME tiny task ("Bier oder Wein?", the one from the slides):

    wein -> wein   bier -> bier   kein bier -> wein   kein wein -> bier   (+ a few variants)

Both are trained ONLY with the words bier, wein and kein. Then both get words they never saw
in training (merlot, pils, ...). The only difference is how a word enters the network:

  A  one-hot   one column per known word (what the barkeeper does)      -> unknown word = nothing
  B  embedding a short vector per word, learned from a mini text corpus -> unknown word = a vector
                                                                           close to similar words

The embeddings come from a little hand-made text (CORPUS below), using the simplest method:
count which words appear next to each other, then squeeze the table into a few numbers per word.
Real LLMs do the same idea with billions of words and far more numbers per word.

Be aware: this corpus is tidy on purpose (every wine sentence looks like every other one), so words of
the same group get almost identical vectors. Real text is messy, and the embeddings are only as good
as the text they come from. Try changing the corpus or the seed and see what breaks.
"""
import itertools
import numpy as np

# ---------------------------------------------------------------- 1. a tiny text corpus
WINES = ["wein", "rotwein", "weisswein", "merlot", "lambrusco", "chianti", "riesling"]
BEERS = ["bier", "pils", "helles", "weizen", "export"]
WINE_TEXT = ["ein glas {w} schmeckt nach trauben", "{w} wird aus trauben gemacht",
             "{w} kommt aus der flasche und wird im weinglas serviert", "zum essen passt {w} aus dem weinglas"]
BEER_TEXT = ["ein glas {w} schmeckt nach malz", "{w} wird aus gerste und hopfen gemacht",
             "{w} kommt vom fass und wird im bierglas serviert", "zum essen passt {w} aus dem bierglas"]
KEIN_TEXT = ["ich will kein {w}", "heute gibt es kein {w}", "bitte kein {w}"]       # "kein" sits next to every drink

CORPUS = []
for w in WINES:
    CORPUS += [t.format(w=w) for t in WINE_TEXT + KEIN_TEXT]
for w in BEERS:
    CORPUS += [t.format(w=w) for t in BEER_TEXT + KEIN_TEXT]
CORPUS = [s.split() for s in CORPUS]

# ---------------------------------------------------------------- 2. embeddings: co-occurrence -> few numbers
def make_embeddings(corpus, dim=8, window=2):
    words = sorted({w for s in corpus for w in s})
    idx = {w: i for i, w in enumerate(words)}
    C = np.zeros((len(words), len(words)))
    for s in corpus:                                          # count neighbours within the window
        for i, w in enumerate(s):
            for j in range(max(0, i - window), min(len(s), i + window + 1)):
                if i != j:
                    C[idx[w], idx[s[j]]] += 1
    total = C.sum()
    pw, pc = C.sum(1, keepdims=True) / total, C.sum(0, keepdims=True) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        ppmi = np.maximum(np.log((C / total) / (pw * pc)), 0)   # "more together than by chance"
    ppmi = np.nan_to_num(ppmi, nan=0.0, posinf=0.0, neginf=0.0)
    U, S, _ = np.linalg.svd(ppmi)                              # squeeze into `dim` numbers per word
    E = U[:, :dim] * np.sqrt(S[:dim])
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9      # every word vector has length 1
    return {w: E[i] for w, i in idx.items()}

def neighbours(emb, word, n=4):
    sims = {w: float(emb[word] @ v) for w, v in emb.items() if w != word}
    return sorted(sims.items(), key=lambda t: -t[1])[:n]

# ---------------------------------------------------------------- 3. the tiny task
ANSWERS = ["bier", "wein"]
TRAIN = [("bier", "bier"), ("ein bier", "bier"), ("bier bitte", "bier"), ("ich will bier", "bier"),
         ("wein", "wein"), ("ein wein", "wein"), ("wein bitte", "wein"), ("ich will wein", "wein"),
         ("kein bier", "wein"), ("kein bier bitte", "wein"), ("ich will kein bier", "wein"),
         ("kein wein", "bier"), ("kein wein bitte", "bier"), ("ich will kein wein", "bier")]
TEST = [("merlot", "wein"), ("ein merlot bitte", "wein"), ("kein merlot", "bier"), ("kein merlot bitte", "bier"),
        ("lambrusco", "wein"), ("kein lambrusco", "bier"), ("kein chianti", "bier"),
        ("pils", "bier"), ("kein pils", "wein"), ("kein helles", "wein"), ("kein weizen", "wein")]

def softmax(Z):
    E = np.exp(Z - Z.max(1, keepdims=True)); return E / E.sum(1, keepdims=True)

def train(X, Y, hidden=8, rounds=3000, lr=0.3, seed=0):
    rng = np.random.RandomState(seed)
    W1, b1 = rng.randn(X.shape[1], hidden) * 0.5, np.zeros(hidden)
    W2, b2 = rng.randn(hidden, Y.shape[1]) * 0.5, np.zeros(Y.shape[1])
    v = [np.zeros_like(a) for a in (W1, b1, W2, b2)]
    for _ in range(rounds):
        H = np.tanh(X @ W1 + b1); P = softmax(H @ W2 + b2)
        D2 = (P - Y) / len(X); D1 = (D2 @ W2.T) * (1 - H ** 2)
        for k, (p, g) in enumerate(zip((W1, b1, W2, b2), (X.T @ D1, D1.sum(0), H.T @ D2, D2.sum(0)))):
            v[k] = 0.9 * v[k] - lr * g; p += v[k]
    return lambda Xn: softmax(np.tanh(Xn @ W1 + b1) @ W2 + b2)

known = sorted({w for s, _ in TRAIN for w in s.split()})              # words the network sees in training

def onehot(sentence):                                                  # A: one column per known word
    x = np.zeros(len(known))
    for w in sentence.split():
        if w in known: x[known.index(w)] = 1
    return x

def embedded(sentence):                                                # B: sum of the word vectors
    return sum((emb[w] for w in sentence.split() if w in emb), np.zeros(len(next(iter(emb.values())))))

Y = np.array([[1.0 if a == t else 0.0 for a in ANSWERS] for _, t in TRAIN])
emb, nets = {}, {}

def build(corpus):
    """(Re)make the embeddings from the corpus and train both networks on the same 14 sentences."""
    global emb, nets
    emb = make_embeddings(corpus)
    nets = {}
    for name, enc in (("one-hot", onehot), ("embedding", embedded)):
        nets[name] = (enc, train(np.array([enc(s) for s, _ in TRAIN]), Y))

def answer(sentence):
    out = {}
    for name, (enc, net) in nets.items():
        p = net(np.array([enc(sentence)]))[0]
        out[name] = (ANSWERS[p.argmax()], p.max())
    return out

def chat():
    corpus = [list(s) for s in CORPUS]
    build(corpus)
    print("Chat with both networks (type a sentence, they both answer bier or wein).")
    print("  near merlot            words closest to a word")
    print("  text primitivo ist ein roter wein aus apulien   add a sentence to the corpus (the embeddings are rebuilt)")
    print("  exit                   leave")
    print("Try: merlot / kein merlot / kein pils / kein primitivo (unknown) -> text ... -> kein primitivo\n")
    while True:
        said = input("you> ").strip().lower()
        if said in ("", "exit", "quit"):
            break
        if said.startswith("near "):
            w = said[5:].strip()
            if w not in emb:
                print(f"  '{w}' is not in the corpus\n"); continue
            print("  " + "   ".join(f"{n} ({s:+.2f})" for n, s in neighbours(emb, w, 6)) + "\n"); continue
        if said.startswith("text "):
            corpus.append(said[5:].split()); build(corpus)
            print(f"  corpus now has {len(corpus)} sentences, embeddings rebuilt, both networks trained again\n"); continue
        words = said.split()
        print(f"  one-hot sees:   {[w for w in words if w in known] or 'nothing'}"
              f"   embedding sees: {[w for w in words if w in emb] or 'nothing'}")
        no_vec = [w for w in words if w not in emb and w not in known]
        if no_vec:
            print(f"  unknown to both (no vector, not in the corpus): {', '.join(no_vec)}")
        res = answer(said)
        for name, (a, pr) in res.items():
            print(f"  {name:10s} -> {a} ({pr:.0%})")
        print()

build(CORPUS)

if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "chat":
    chat()
elif __name__ == "__main__":
    print(f"Mini corpus: {len(CORPUS)} sentences, {len(emb)} words, {len(next(iter(emb.values())))} numbers per word\n")
    print("Which words are close to 'merlot' (cosine, 1.0 = same direction)?")
    for w, s in neighbours(emb, "merlot"):
        print(f"   {w:12s} {s:+.2f}")
    print("\nAnd to 'pils'?")
    for w, s in neighbours(emb, "pils"):
        print(f"   {w:12s} {s:+.2f}")
    print(f"\n   merlot vs wein: {float(emb['merlot'] @ emb['wein']):+.2f}     merlot vs pils: {float(emb['merlot'] @ emb['pils']):+.2f}"
          f"     merlot vs kein: {float(emb['merlot'] @ emb['kein']):+.2f}")
    print(f"\nTraining examples: {len(TRAIN)}, only with the drink words bier / wein (and 'kein').")
    for name, (enc, net) in nets.items():
        ok = sum(ANSWERS[net(np.array([enc(s)]))[0].argmax()] == t for s, t in TRAIN)
        print(f"   {name:10s} {ok}/{len(TRAIN)} training sentences right")
    print("\nNEW sentences with words the network never saw in training:\n")
    print(f"   {'sentence':20s}{'should be':11s}{'one-hot':16s}{'embedding'}")
    score = {k: 0 for k in nets}
    for s, t in TEST:
        row = f"   {s:20s}{t:11s}"
        for name, (enc, net) in nets.items():
            p = net(np.array([enc(s)]))[0]; a = ANSWERS[p.argmax()]
            score[name] += a == t
            row += f"{a:6s}{p.max():4.0%} {'✓' if a == t else '✗'}   "
        print(row)
    print(f"\n   {'':31s}{score['one-hot']:2d}/{len(TEST)}{'':11s}{score['embedding']:2d}/{len(TEST)}")
    print("\nHow to read this:")
    print("  - one-hot ignores every unknown word, so merlot and pils look the same to it (as an empty input).")
    print("    Its 'right' answers for the wine words are luck: with nothing to go on it falls back to its default.")
    print("  - the embedding network never saw merlot either, but the vector of 'merlot' is close to 'wein',")
    print("    so everything it learned about wein also applies to merlot (and 'kein merlot' flips to bier).")
    print("  - the network did not learn this ability from the 14 examples, it comes from the corpus text.")
    print("  - all drink words share a big common part (merlot vs pils is still +0.95), only a small difference tells")
    print("    wine from beer. The network picks up exactly that difference, which is why it can tell them apart.")
