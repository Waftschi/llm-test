import numpy as np
np.random.seed(0)

# ---------- 1. Data: words -> numbers ----------
text = "ich mag katzen . du magst hunde . ich mag hunde . du magst katzen .".split()
vocab = sorted(set(text))
V = len(vocab)                      # vocabulary size
idx = {w: i for i, w in enumerate(vocab)}
pairs = [(idx[a], idx[b]) for a, b in zip(text, text[1:])]   # (current word, next word)

# ---------- 2. Parameters (the "unknowns" of the system) ----------
d = 8                               # embedding size
E = np.random.randn(d, V) * 0.1     # embedding matrix
W = np.random.randn(V, d) * 0.1     # output matrix

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

# ---------- 3. Training = solving the equations approximately ----------
lr = 0.5
for epoch in range(301):
    loss = 0
    for x, y in pairs:
        h = E[:, x]                 # (1) h = E · x        embedding
        z = W @ h                   # (2) z = W · h        scores (logits)
        p = softmax(z)              # (3) p = softmax(z)   probabilities
        loss += -np.log(p[y])       # (4) L = -log p_y     error

        dz = p.copy(); dz[y] -= 1   # (5) dL/dz = p - y_onehot
        dW = np.outer(dz, h)        # (6) dL/dW = dz · h^T
        dh = W.T @ dz               # (7) dL/dh = W^T · dz
        W -= lr * dW                # (8) gradient descent step
        E[:, x] -= lr * dh
    if epoch % 50 == 0:
        print(f"epoch {epoch:3d}  loss {loss / len(pairs):.3f}")

# ---------- 4. Use the model ----------
def predict(word):
    p = softmax(W @ E[:, idx[word]])
    top = np.argsort(p)[::-1][:3]
    return [(vocab[i], round(float(p[i]), 2)) for i in top]

for w in ["ich", "du", "mag", "magst"]:
    print(w, "->", predict(w))

word, sentence = "du", ["du"]
for _ in range(6):                  # generate text word by word
    p = softmax(W @ E[:, idx[word]])
    word = vocab[np.random.choice(V, p=p)]
    sentence.append(word)
print("generated:", " ".join(sentence))

# ---------- 5. Bonus: one attention step (what makes it a Transformer) ----------
seq = ["ich", "mag", "hunde"]
H = np.stack([E[:, idx[w]] for w in seq])          # 3 x d
Wq, Wk, Wv = (np.random.randn(d, d) * 0.5 for _ in range(3))
Q, K, Vv = H @ Wq, H @ Wk, H @ Wv                  # Q = HWq, K = HWk, V = HWv
scores = Q @ K.T / np.sqrt(d)                      # S = QK^T / sqrt(d)
scores += np.triu(np.full((3, 3), -1e9), 1)        # mask: no peeking at the future
A = np.apply_along_axis(softmax, 1, scores)        # A = softmax(S)
print("attention weights (row = word, col = which words it looks at):")
print(np.round(A, 2))
out = A @ Vv                                       # output = A · V
