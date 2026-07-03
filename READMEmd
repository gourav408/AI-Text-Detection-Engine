# AI Text Detection Engine

Classifying **AI-generated** vs **human-written** text with self-trained **Word2Vec** embeddings and two custom **Keras** models (a feed-forward ANN and a Bidirectional LSTM).

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.x-orange">
  <img alt="Gensim" src="https://img.shields.io/badge/Gensim-Word2Vec-green">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-lightgrey">
</p>

---

## Overview

This project is an end-to-end NLP pipeline that takes raw text all the way to a working AI-text detector. It downloads a labelled human-vs-AI corpus, analyses the lexical fingerprints that separate the two classes, learns semantic embeddings, and trains and compares two deep-learning models under a **leakage-safe** evaluation protocol.

**Headline results:** ~**97.7% test accuracy** and **0.997 ROC-AUC** on a held-out set.

---

## Results

| Model | Accuracy | ROC-AUC |
|-------|:--------:|:-------:|
| ANN (Word2Vec mean-pooled vectors) | 0.977 | 0.997 |
| Bi-LSTM (Word2Vec sequences) | 0.977 | 0.997 |

**Key finding:** on this vocabulary-separable corpus the lightweight **ANN matches the Bi-LSTM**, showing that a cheap baseline is highly competitive and that word order adds little signal here.

---

## What it does — step by step

| # | Step | What happens | Why it matters |
|---|------|--------------|----------------|
| 1 | **Data** | Loads a balanced 8,000-document human-vs-AI corpus from public sources, with an offline fallback. | Reproducible, class-balanced input that always runs. |
| 2 | **Integrity** | De-duplicates documents *before* splitting. | Stops repeated texts from leaking labels across train/test. |
| 3 | **Preprocessing** | Cleans, normalises and tokenises; keeps function words. | Consistent input while preserving a strong stylistic signal. |
| 4 | **Feature analysis** | Computes interpretable lexical stats (type-token ratio, long-word & stop-word ratios) with plots. | Explains *how* AI and human text differ — not a black box. |
| 5 | **Leakage-safe split** | Splits first, then fits Word2Vec + tokenizer on **training rows only**. | Delivers honest, uninflated metrics — the methodological core. |
| 6 | **Embeddings** | Trains skip-gram **Word2Vec** and uses it to warm-start the LSTM. | Transfers learned semantics; sanity-checkable via nearest neighbours. |
| 7 | **Modelling** | Builds a custom **ANN** baseline and a **Bi-LSTM**. | Quantifies the cost/benefit of a sequence model vs a cheap baseline. |
| 8 | **Evaluation** | Accuracy, precision/recall/F1, confusion matrices and ROC/AUC. | Rigorous, multi-metric comparison on unseen data. |
| 9 | **Inference** | Scores new text, including an out-of-domain stress test. | Shows real-world use and is honest about failure modes. |

---

## Project structure

```
.
├── AI_Text_Detection_Engine_FINAL.ipynb   # Main notebook (data → analysis → models → evaluation → inference)
└── README.md
```

---

## Getting started

### Option A — Google Colab (recommended)

1. Open `AI_Text_Detection_Engine_FINAL.ipynb` in Colab.
2. Set the runtime to GPU: **Runtime → Change runtime type → GPU**.
3. **Runtime → Run all.**

### Option B — Local

```bash
# clone
git clone https://github.com/<your-username>/ai-text-detection-engine.git
cd ai-text-detection-engine

# (optional) virtual environment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# install dependencies
pip install tensorflow gensim datasets scikit-learn seaborn matplotlib pandas numpy

# launch
jupyter notebook AI_Text_Detection_Engine_FINAL.ipynb
```

> A GPU is recommended for the Bi-LSTM. On CPU it still runs, just slower.

### Use your own data

Set `LOCAL_CSV` at the top of the data cell to a CSV with two columns — `text` and `label` (`0` = human, `1` = AI) — to train on your own corpus.

---

## How it works

- **Embeddings.** A skip-gram Word2Vec model (100-dim) is trained on the training split only and used both to build mean-pooled document vectors (for the ANN) and to initialise the LSTM's embedding layer.
- **Model A — ANN.** A compact fully-connected network over mean-pooled Word2Vec vectors. Fast, hard to overfit, and a strong baseline.
- **Model B — Bi-LSTM.** Reads the padded token sequence in both directions to capture word-order and long-range dependencies, with the embedding layer fine-tuned from Word2Vec.
- **Leakage control.** The test set is held out *before* any representation is learned, and de-duplication is done up front, so reported metrics are not inflated by leakage.

---

## Tech stack

Python · TensorFlow / Keras · Gensim (Word2Vec) · scikit-learn · pandas · NumPy · Matplotlib · Seaborn

---

## Limitations

- **Domain generalisation.** Trained on formal student essays, so it learns *this domain's* cues rather than a universal "AI-ness" signal. Very casual text (slang, chat, tweets) can be mislabelled — the notebook includes a deliberate out-of-domain stress test.
- **No sentence-level features.** The source corpus arrives lower-cased and de-punctuated, so sentence-length style features are not recoverable; the analysis relies on word-level statistics.
- **Compute.** The Bi-LSTM needs a GPU and enough epochs to reach its potential.

---

## Future work

- Swap Word2Vec for contextual embeddings (FastText or a Transformer such as DistilBERT).
- Add an attention layer on top of the LSTM.
- Ensemble the ANN + LSTM.
- Train on multiple domains for better generalisation.
- Feed the hand-crafted lexical features into the classifier alongside the embeddings.

---

## 📄 License

Released under the MIT License. See [`LICENSE`](LICENSE) for details.

---

*AI Text Detection Engine — a deep-learning NLP project.*
