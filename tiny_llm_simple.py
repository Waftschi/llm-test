import numpy as np

# 1. Text -> numbers
words = "ich mag katzen . du magst hunde . ich mag hunde .".split()
vocab = sorted(set(words))
n = len(vocab)
num = {w: i for i, w in enumerate(vocab)}

# 2. The unknowns: a table W[current word, next word], start with zeros
W = np.zeros((n, n))

def probs(row):                       # softmax: scores -> probabilities
    e = np.exp(row)
    return e / e.sum()

# 3. Learning: for every word pair, push the correct answer up
for step in range(200):
    for a, b in zip(words, words[1:]):
        p = probs(W[num[a]])          # what the model guesses now
        target = np.zeros(n); target[num[b]] = 1   # what it should guess
        W[num[a]] -= 0.1 * (p - target)            # the whole learning rule

# 4. Ask the model
for w in ["ich", "du", "mag"]:
    p = probs(W[num[w]])
    print(w, "->", vocab[p.argmax()], f"({p.max():.0%})")
