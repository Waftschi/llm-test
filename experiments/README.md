# Experiments

Small, separate experiments. They do not touch `barkeeper2.py` or its databases.

## embedding_experiment.py

Why can an LLM work out that "kein merlot" means "bier" without being taught? The barkeeper cannot: it has one
column per known word, so a new word like "merlot" has nothing in common with "wein". An LLM gives every word an
**embedding**, a short vector of numbers learned from lots of text, where words used in similar places get similar vectors.

```
python3 experiments/embedding_experiment.py
```

Two small networks learn the same tiny task ("Bier oder Wein?"), trained only with the words bier, wein and kein:

| Network | How a word enters | Unknown word `merlot` |
|---|---|---|
| A, one-hot (like the barkeeper) | one column per known word | ignored, looks like nothing |
| B, embedding | a vector per word, taken from a mini text corpus | a vector close to `wein` |

Then both get sentences with words they never saw (merlot, lambrusco, chianti, pils, helles, ...).
In the run the embedding network gets all 11 new sentences right, the one-hot network 7 of 11, and its right
answers are luck (it gives the same answer for every unknown word).

What to take away:
- The generalisation does not come from the 14 training examples but from the **text** the embeddings were made from.
- The corpus here is tiny and tidy on purpose, so it works very cleanly. Real text is messy, and embeddings are only as good as their text.
- Results vary a little with the seed and the size of the embedding (10 to 11 of 11 in a quick sweep).
- Method: count which words stand next to each other (PPMI), then squeeze that table into 8 numbers per word (SVD). Only numpy is needed.

Things to try: change `WINES` / `BEERS` / the templates, reduce `dim`, or train the embedding network with `kein merlot` removed from `TEST`.

## Try it in a chat

```
python3 experiments/embedding_experiment.py chat
```

Type a sentence and both networks answer (bier or wein). Commands: `near merlot` (closest words), `text <sentence>` (add a sentence to the corpus, the embeddings are rebuilt), `exit`.

Teach a new word only through text, no labels. Paste these one after another, asking `kein primitivo` before and after:

```
kein primitivo
text ein glas primitivo schmeckt nach trauben
text primitivo wird aus trauben gemacht
text primitivo kommt aus der flasche und wird im weinglas serviert
text zum essen passt primitivo aus dem weinglas
text ich will kein primitivo
text heute gibt es kein primitivo
text bitte kein primitivo
kein primitivo
```

Before: `primitivo` is unknown to both networks. After the seven sentences the embedding network answers `kein primitivo` with bier.
With only one or two sentences it still gets it wrong: a word needs enough text, used like the other wines, to land in the right place.
