# Barkeeper – a small neural network for learning how it works

A bartender who understands sentences like "ein bier", "kein wein" or "zahlen bitte" (German) and keeps learning.
Not a real LLM, but a neural network with just under 4000 weights. It uses the same basic
ideas as an LLM (numbers, weights, softmax, gradient descent), but it is small enough
to follow every single step.

German version: [README.de.md](README.de.md)

**Short guide for colleagues** (explainable in 20–30 minutes): [docs/README.md](docs/README.md)

**Presentation:** [English](https://waftschi.github.io/llm-test/praesentation_en/presentation.html) · [German](https://waftschi.github.io/llm-test/praesentation/presentation.html) (slides that explain the network step by step)

## Files

| File | Purpose |
|---|---|
| `barkeeper_core.py` | The model: two layers (words → concepts → answer), training, teaching, decision log and blame as data. No `print`/`input` |
| `barkeeper2.py` | Console on top of the core |
| `server.py` | Web UI: small JSON API (standard library only) that also serves `web/index.html` |
| `web/index.html` | The web page (plain HTML/JS, no build step): chat on the left, network internals on the right |
| `show_matrix.py` | Shows the learned weights (words, concepts, answers, CSV) |
| `barkeeper_seed.json` | Answers, prices, contexts, starter examples and `fallback` (reply and minimum confidence for when nothing is understood) |
| `barkeeper_simple_seed.json`, `barkeeper_simple_en_seed.json` | Mini models in German and English (3 answers, 4 concepts). Also contain `suggestions`: example phrases shown as buttons in the web UI and as "Try:" in the console |
| `barkeeper2.db`, `barkeeper_simple*.db` | SQLite: training examples and the trained model (as JSON), one database per profile. Created on the first start, not part of the repo |
| `praesentation/`, `praesentation_en/` | Slides in German and English (`build_presentation.py` builds them) |

Requirements: Python 3 with `numpy`. Otherwise only the standard library is used.

```
pip3 install -r requirements.txt
```

## Getting started

### Profiles (language and size)

| Profile | Model | Language |
|---|---|---|
| *(none)* | large model, 24 concepts | German |
| `simple` | mini model, 4 concepts (see README_SIMPLE.md) | German |
| `simple-en` | mini model, 4 concepts | English |

Each profile has its own seed file and its own database, so they never mix. A profile can be given as an argument or via the environment variable `BARKEEPER_PROFILE`.

### Console

```
python3 barkeeper2.py              # large model
python3 barkeeper2.py simple       # mini model, German
python3 barkeeper2.py simple-en    # mini model, English
python3 show_matrix.py             # look at the learned matrix
```

### Web UI

```
python3 server.py                  # http://localhost:8000
python3 server.py simple-en        # profile as argument
python3 server.py simple-en 8080   # and/or another port
```

- **Left:** the chat, like the console: quick-phrase buttons, `falsch`, `reset`, `sentence = answer`. If the barkeeper does not understand, a teacher panel offers the known answers (or type a new one).
- **Right (tabs):** *Decision* (the log through both layers, with a red banner for unknown words), *Blame* (shown while teaching), *Parameters* (sizes, learning rate, momentum, rounds, seed, fallback threshold, answers, vocabulary), *Training data* and *1 matrix vs 2 layers*.
- Learning is shared by all browser tabs (one model, one database). Only context and bill are per tab.

Not yet in the web UI: a tab with the concept explanations from the slides.

## Usage

| Input | Effect |
|---|---|
| `ein bier`, `kein wein`, `zahlen bitte` | order normally |
| `noch eins = bier` | teach directly (context = current context). One example per line, only one `=`: type `lambrusco = wein` and `kein lambrusco = bier` as two separate lines |
| `[wein] noch eins = wein` | teach with an explicit context |
| `falsch` | correct the last answer (pick from a list, a number, or type a new answer) |
| `skip` | skip the teaching question |
| `reset` | new guest (reset context and bill) |

If the barkeeper does not understand a sentence ("Hä? Wie bitte?"), it automatically asks what it should have done.

More commands: `data` (check training data), `vergleich` (one matrix vs. two layers),
`log` (decision log on/off).

The decision log is printed under every answer and shows the 4 concepts with the biggest vote for the winning answer, their drivers,
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

Today: `barkeeper_core.py` (model) → console and `server.py` + `web/index.html` (plain JS). The API stays the same if the layers grow:

```
Angular (chat UI, log panel, dashboard)        ← replaces web/index.html if the UI outgrows plain JS
   │  REST/JSON
NestJS (gateway: auth, sessions, validation, rate limiting)   ← only when login or public hosting is needed
   │  REST/JSON (internal only)
Python core: barkeeper_core.py + server.py
```

Ideas: a concepts tab with the explanations from the slides, editing training data in the UI, a language switch in the UI.

## License

[MIT](LICENSE): free to use, change and share, as long as the copyright notice is kept.
