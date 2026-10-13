"""Backpropagation step by step, with a tiny network.

2 words (bier, kein) -> 2 concepts (c0, c1) -> 2 answers (bier, wein)
Training example: "kein bier" -> wein
"""
import numpy as np
np.set_printoptions(precision=3, suppress=True)

words, answers = ["bier", "kein"], ["bier", "wein"]
x    = np.array([1.0, 1.0])               # "kein bier": both words present
ziel = np.array([0.0, 1.0])               # correct answer: wein

W1 = np.array([[1.0, -0.5],               # c0:  bier +1.0, kein -0.5
               [0.2,  0.8]])              # c1:  bier +0.2, kein +0.8
W2 = np.array([[ 1.0, 0.5],               # answer bier: from c0 +1.0, from c1 +0.5
               [-0.5, 0.5]])              # answer wein: from c0 -0.5, from c1 +0.5
lr = 0.5

def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()

def forward(W1, W2):
    h = np.tanh(W1 @ x)
    p = softmax(W2 @ h)
    return h, p

print("=== FORWARD: the guess ===")
h, p = forward(W1, W2)
print(f"concepts  h = tanh(W1·x) = {h}")
print(f"answers   p = softmax(W2·h) = bier {p[0]:.0%}, wein {p[1]:.0%}")
print(f"error     -log(p_wein) = {-np.log(p[1]):.3f}")

print("\n=== BACKWARD: who is to blame? ===")
d2 = p - ziel
print(f"1) blame of the answers      d2 = p - ziel           = {d2}")
g2 = np.outer(d2, h)
print(f"2) blame of the W2 weights   outer(d2, h)            =\n{g2}")
back = W2.T @ d2
print(f"3) blame passed to concepts  W2ᵀ·d2                  = {back}")
slope = 1 - h**2
print(f"4) through tanh              × (1 - h²) = × {slope}")
d1 = back * slope
print(f"   blame of the concepts     d1                      = {d1}")
g1 = np.outer(d1, x)
print(f"5) blame of the W1 weights   outer(d1, x)            =\n{g1}")

print("\n=== UPDATE: weights -= 0.5 × blame ===")
W2n, W1n = W2 - lr * g2, W1 - lr * g1
print(f"W2 new =\n{W2n}\nW1 new =\n{W1n}")

h2, p2 = forward(W1n, W2n)
print(f"\n=== CHECK: same sentence again ===")
print(f"before: bier {p[0]:.0%}, wein {p[1]:.0%}")
print(f"after:  bier {p2[0]:.0%}, wein {p2[1]:.0%}")
