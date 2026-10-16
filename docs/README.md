# Barkeeper in 25 minutes

A short guide for colleagues: what the Barkeeper is, how it works, and how to explain it in 20–30 minutes.
No maths background needed. Details are in the main [README](../README.md); the slides are the speaking aid.

German version: [README.de.md](README.de.md)

## What is it?

A small chatbot that plays a bartender. You type a sentence ("ein bier", "kein wein", "zahlen bitte"),
and it answers. It is a **neural network with 4,258 numbers**, not a language model.
It uses the same building blocks as an LLM (numbers, weights, softmax, learning from errors),
but it is small enough to look at every step.

**Not** in scope: generating free text, understanding unknown words, a web interface (that comes later).

## The story in five ideas

1. **Text becomes numbers.** Every known word gets a 1 if it is in the sentence, otherwise a 0.
   The context (what the guest ordered last, e.g. `[wein]`) counts as one more word.
2. **Layer 1: words → concepts.** 24 "concepts" are yes/no questions about the sentence,
   for example "does *kein* occur together with *bier*?". Nobody designs them, the network finds them itself.
3. **Layer 2: concepts → answer.** Each concept votes for or against each of the 34 possible answers.
   Softmax turns the scores into percentages. The highest wins. Below 40 % the barkeeper says "Hä? Wie bitte?".
4. **Learning = guess, measure, blame, nudge.** The network guesses, measures the error, passes the blame
   backwards to every weight and moves each weight a tiny step. Repeated 1,500 times.
   The weights are the only thing that is learned.
5. **Why two layers?** "kein bier" needs a *combination* of two words. One layer can only add up single-word votes.
   `tanh` between the layers is what makes combinations possible.

## Plan for 25 minutes

Slide numbers refer to the current deck (`praesentation_en/presentation.html`, 27 slides). They shift if slides are added.

| Time | Topic | Slides | Key message |
|---|---|---|---|
| 0–3 min | The task | 1–3 | A bartender who understands "kein bier" |
| 3–11 min | **Part 1: Answers** | 4–11 | How a sentence becomes an answer (numbers → layer 1 → layer 2 → softmax) |
| 11–18 min | **Part 2: Learning** | 13–19 | Where the weights come from: guess, measure, blame, nudge |
| 18–21 min | **Part 3: Context** | 22, 25–27 | Why two layers, and what an LLM does differently |
| 21–25 min | **Live demo** | – | See below |

**Short route (20 minutes):** skip slides 7–8 (concept origin, W2/b2), 18–19 (backpropagation detail) and 23–24 (hurdle example, tanh proof).
Those are the "dive deeper" slides for questions.

## Live demo (about 5 minutes)

Start: `pip3 install -r requirements.txt`, then `python3 barkeeper2.py`.

| Type | What to expect | Point to make |
|---|---|---|
| `ein bier` | bier, 100 % | The decision log shows concepts, votes and softmax |
| `reset`, then `kein bier` | wein | "kein" flips the drink, a combination of two words |
| `reset`, then `kein wein` | bier | The same rule works the other way |
| `log` | decision log off/on | Switch it off for a calmer demo |
| `asdf qwer` | "Hä? Wie bitte?" and a teaching question | It asks instead of guessing. Press **Enter** to skip |
| `vergleich` | one matrix 8/12, two layers 12/12 on new sentences | The proof that two layers are needed |
| `python3 show_matrix.py konzept 15` | the words and answers driving concept 15 | Concepts are just rows of numbers |

**Context matters:** `ein bier` followed directly by `kein bier` (without `reset`) is less sure of wein than
`kein bier` for a new guest, because the context `[bier]` pulls the other way. A good talking point.

**Teaching:** `noch eins = bier` teaches an example. This retrains the *whole* model from random weights
(a few seconds) and changes `barkeeper2.db`. For a demo use the mini model (`python3 barkeeper2.py simple`) or copy the database first.

## Key numbers

| | Big model | Meaning |
|---|---|---|
| Training examples | 230 | sentence + context → answer |
| Features | 141 | known words and context tokens |
| Concepts | 24 | hidden layer |
| Answers | 34 | possible replies |
| Weights and biases | 4,258 | 24×141 + 24 + 34×24 + 34 |
| Training | 1,500 rounds | learning rate 0.5, momentum 0.9, seed 0 |

## Glossary

| Term | In plain words |
|---|---|
| **Weight** | A learned number: how strongly one input pushes a result up or down |
| **Bias** | A learned base value that applies no matter what the input is |
| **Concept** | One of the 24 hidden yes/no questions of layer 1 (a row of numbers in `W1`) |
| **tanh** | Squeezes a sum into −1 … +1 (−1 = no, +1 = yes) |
| **Softmax** | Turns scores into percentages that add up to 100 % |
| **Error (P − Y)** | Guess minus correct answer. Positive = too strong, negative = too weak |
| **Backpropagation** | Passing the error backwards to find out how much each weight is to blame |
| **Training round** | One pass of guess, measure, blame, nudge over all examples |
| **Seed** | Fixed start for the random numbers, so training is repeatable |

## Questions you may get

- **Is this an LLM?** No. It has the same ingredients, but only 4,258 weights, a fixed vocabulary and a list of prepared answers.
- **What does an LLM do differently?** It predicts the next word, has billions of weights, uses attention to relate words to each other
  and is trained on huge amounts of text. The learning principle is the same.
- **Why 24 concepts?** Our choice. Too few are too coarse, too many memorise the examples.
- **Does it understand new words?** No. Unknown words are ignored. If more than half are unknown, it asks.
- **Why retrain everything when teaching?** It keeps the code simple and repeatable. A faster way (a few steps on the new example) risks forgetting old knowledge.

## Where to find what

| What | Where |
|---|---|
| Console app | `barkeeper2.py` (`python3 barkeeper2.py`, mini model: `python3 barkeeper2.py simple`) |
| Answers, prices, examples, fallback | `barkeeper_seed.json` |
| Look at learned weights | `python3 show_matrix.py` |
| Slides | [English](https://waftschi.github.io/llm-test/praesentation_en/presentation.html) · [German](https://waftschi.github.io/llm-test/praesentation/presentation.html) |
| Full explanation | [README](../README.md) |
