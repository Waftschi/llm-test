# Barkeeper – a small neural network for learning how it works

A bartender who understands sentences like "ein bier", "kein wein" or "zahlen bitte" (German) and keeps learning.
Not a real LLM, but a neural network with just under 4000 weights. It uses the same basic
ideas as an LLM (numbers, weights, softmax, gradient descent), but it is small enough
to follow every single step.

German version: [README.de.md](README.de.md)

## Files

| File | Purpose |
|---|---|
| `barkeeper2.py` | Two layers (words → concepts → answer), console |
| `show_matrix.py` | Shows the learned weights (words, concepts, answers, CSV) |
| `barkeeper_seed.json` | Answers, prices, contexts, starter examples and `fallback` (reply and minimum confidence for when nothing is understood) |
| `barkeeper2.db` | SQLite: training examples and the trained model (as JSON) |

Requirements: Python 3 with `numpy`. Otherwise only the standard library is used.

## Getting started

Console:

```
python3 barkeeper2.py            # large model
python3 barkeeper2.py simple     # mini model (see README_SIMPLE.md)
python3 show_matrix.py           # look at the learned matrix
```

There is deliberately no web interface yet, see "Possible next steps".

## Usage

| Input | Effect |
|---|---|
| `ein bier`, `kein wein`, `zahlen bitte` | order normally |
| `noch eins = bier` | teach directly (context = current context) |
| `[wein] noch eins = wein` | teach with an explicit context |
| `falsch` | correct the last answer (pick from a list, a number, or type a new answer) |
| `skip` | skip the teaching question |
| `reset` | new guest (reset context and bill) |

If the barkeeper does not understand a sentence ("Hä? Wie bitte?"), it automatically asks what it should have done.

More commands: `data` (check training data), `vergleich` (one matrix vs. two layers),
`log` (decision log on/off).

The decision log is printed under every answer and shows the most active concepts, their drivers,
the votes, the sums, tanh and the softmax calculation.

## How it works

### 1. Text becomes numbers
"ein bier" in the context `start` becomes the features `ein`, `bier`, `[start]`.
Each known feature (currently 133) gets a 1 if it occurs, otherwise a 0.
This vector of numbers is the input `x`.

### 2. Two layers

```
h = tanh(W1 · x + b1)       layer 1: words + context → 24 concepts
p = softmax(W2 · h + b2)    layer 2: concepts → probability per answer
```

- `W1` is a 24 × 133 table. Each concept adds up the weighted votes of the active words.
- `tanh` squeezes the sum to a value between -1 and +1 (the "activation" in the log).
- `W2` collects the votes of the concepts for each answer. `softmax` turns them into percentages (sum 100 %).
- The highest answer wins. Below 40 % (or with too many unknown words) the barkeeper asks.

### 3. Why two layers?
A single matrix only adds up independent word votes: `kein` and `bier` cannot work together,
so "kein bier" would still vote for beer. A hidden concept can be built to fire only when
`kein` and `bier` occur together, and then vote for wine. The console command `vergleich` shows the difference.

### 4. Training
Repeated (1500 rounds, over all examples at once):

1. Guess the answer.
2. Measure the error: `P - Y` (guess minus correct answer).
3. Distribute the blame backwards to every weight (backpropagation).
4. Move every weight a little against its blame (gradient descent, with momentum).

### 5. How the error finds the right place in the matrix
Every weight has an address (output `i`, input `j`):

```
change of W[i,j]  =  error at output i  ×  value of input j
```

In the code: `D2.T @ Hd` for `W2` and `D1.T @ X` for `W1`.

- **Layer 2:** gradient = error of the answer × activation of the concept. Active concepts that fit the correct answer vote for it more strongly from now on.
- **Passing it on to layer 1:** `D1 = (D2 @ W2) * (1 - Hd**2)`.
  `D2 @ W2` asks each concept how much it contributed to the wrong answer.
  `(1 - Hd**2)` is the slope of `tanh`; a concept that is already stuck at ±1 hardly changes.
- **Layer 1:** gradient = blame of the concept × `x[j]`. Only words that occurred in the sentence (`x[j] = 1`) are changed; all other columns get exactly 0.

So one example only moves the columns of its own words, and only a little (averaged over all examples).
"bier" and "kein bier" push the column `bier` in opposite directions. That forces concepts
to specialise in the combination.

### 6. Where do the concepts come from?
Nobody designs them. Each of the 24 concepts is a row in `W1` that starts with random numbers
(`np.random.randn`, fixed seed = repeatable). Training turns the randomness into specialists.

- A random start is necessary because otherwise all rows would get the same gradient and stay identical copies forever.
- The meaning ("ordering beer") is our interpretation based on the driving words in the log. The model only knows numbers.
- The numbers are arbitrary and can shift after retraining. A concept often mixes several patterns.
- Only the count (`HIDDEN = 24`) and the activation function (`tanh`) are fixed by us.

### 7. Teaching
`learn()` deletes an old example with the same context and sentence, inserts the new one and retrains everything
(same seed). The columns of the new sentence are moved until it is right; the other examples hold everything else in place.

## Possible next steps

Today the Python core runs in the console only. The web frontend comes later, on top of it:

```
Angular (chat UI, log panel, dashboard)
   │  REST/JSON
NestJS (gateway: auth, sessions, validation, rate limiting)
   │  REST/JSON (internal only)
Python core (FastAPI): model, training, teaching, explanation, SQLite
```

Sensible order:

1. Move the model code into an importable `barkeeper_core.py` (currently it lives in `barkeeper2.py` together with the console).
2. FastAPI with state per session.
3. Angular UI against this API.
4. NestJS gateway as soon as login or public hosting is needed.
