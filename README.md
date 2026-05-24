# Neural Machine Translation — Urdu to Roman Urdu

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![BiLSTM](https://img.shields.io/badge/BiLSTM-Encoder--Decoder-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=flat-square&logo=streamlit)
![Language](https://img.shields.io/badge/Language-Urdu%20%E2%86%92%20Roman%20Urdu-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

**A seq2seq BiLSTM encoder-decoder model for transliterating Urdu script into Roman Urdu**

[📝 Medium Blog](#) · [💼 LinkedIn Post](#) · [🤗 Live Demo](#)

</div>

---

## 📌 Overview

This project builds a **sequence-to-sequence Neural Machine Translation (NMT)** system using a **Bidirectional LSTM encoder** and an **LSTM decoder** to transliterate Urdu script (اردو) into Roman Urdu. The model is trained on the `urdu_ghazals_rekhta` dataset — poetic Urdu Ghazals — pushing BiLSTM-based NMT on low-resource, morphologically rich text.

```
Urdu Script Input  →  BiLSTM Encoder  →  Context Vector
      اردو                                      ↓
                                        LSTM Decoder
                                               ↓
                                     Roman Urdu Output
                                          "urdu"
```

---

## ✨ Features

- ✅ Seq2seq BiLSTM encoder (2 layers) + LSTM decoder (4 layers) in PyTorch
- ✅ Urdu text normalization and subword tokenization (BPE / WordPiece)
- ✅ Roman Urdu target extraction / rule-based conversion from dataset
- ✅ Minimum 3 controlled experiments with varying hyperparameters
- ✅ BLEU, Perplexity, CER, and Levenshtein distance evaluation
- ✅ Qualitative translation examples vs ground truth
- ✅ Streamlit live demo for real-time transliteration

---

## 🗂️ Repository Structure

```
📦 urdu-to-roman-nmt/
├── 📁 data/
│   ├── preprocess.py              # Urdu normalization, Roman Urdu extraction
│   └── tokenizer.py               # BPE / WordPiece tokenization
├── 📁 model/
│   ├── encoder.py                 # BiLSTM encoder (2 layers)
│   ├── decoder.py                 # LSTM decoder (4 layers)
│   └── seq2seq.py                 # Full seq2seq model
├── 📁 training/
│   ├── train.py                   # Training loop
│   └── experiments.py             # 3 controlled hyperparameter experiments
├── 📁 evaluation/
│   └── metrics.py                 # BLEU, Perplexity, CER, Levenshtein
├── 📁 app/
│   └── app.py                     # Streamlit transliteration demo
├── 📓 urdu_roman_nmt.ipynb        # Full pipeline notebook
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🧠 Model Architecture

### Encoder — Bidirectional LSTM
- Reads the Urdu input sequence in **both forward and backward directions**
- 2 BiLSTM layers — captures richer contextual representations than a unidirectional LSTM
- Produces a context vector by concatenating forward and backward hidden states

### Decoder — LSTM
- 4 LSTM layers — deeper decoder for complex transliteration patterns
- Takes the encoder context vector as its initial hidden state
- Generates Roman Urdu tokens one at a time

### Why BiLSTM for Urdu?
Urdu is written right-to-left and has rich morphology — a bidirectional encoder captures context from both directions of the script simultaneously, which significantly improves the quality of the context vector for transliteration.

---

## 📊 Dataset

**urdu_ghazals_rekhta**
- Source: [github.com/amir9ume/urdu_ghazals_rekhta](https://github.com/amir9ume/urdu_ghazals_rekhta)
- Content: Urdu Ghazals (poetic works) in Urdu script, English transliteration, and Hindi script
- Extraction: Urdu script (source) → Roman Urdu / English transliteration (target) pairs
- Note: If Roman Urdu is not directly present, rule-based transliteration conversion is applied

**Dataset Split:**

| Split | Percentage |
|---|---|
| Train | 50% |
| Validation | 25% |
| Test | 25% |

---

## 🔤 Preprocessing Pipeline

- **Urdu normalization** — character standardization, extraneous punctuation removal
- **Roman Urdu extraction** — extract or derive Roman Urdu targets from the dataset
- **Rule-based conversion** — apply transliteration rules where Roman Urdu is not directly available
- **Tokenization** — subword tokenization using BPE or WordPiece for both source and target
- **Vocabulary building** — separate vocabularies for Urdu and Roman Urdu with `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>` tokens

---

## ⚗️ Experiments

At least **3 controlled experiments** are conducted by varying one parameter at a time:

| Experiment | Parameter Varied | Values Tested |
|---|---|---|
| Exp 1 | Embedding Dimension | 128, 256, 512 |
| Exp 2 | Hidden Size | 256, 512 |
| Exp 3 | Learning Rate | 1e-3, 5e-4, 1e-4 |

All other parameters are held constant per experiment for fair comparison.

### Full Hyperparameter Search Space

| Parameter | Values |
|---|---|
| Embedding Dimension | 128, 256, 512 |
| LSTM Hidden Size | 256, 512 |
| BiLSTM Encoder Layers | 1, 2, 3, 4 |
| LSTM Decoder Layers | 2, 3, 4 |
| Dropout Rate | 0.1, 0.3, 0.5 |
| Learning Rate | 1e-3, 5e-4, 1e-4 |
| Batch Size | 32, 64, 128 |
| Optimizer | Adam |
| Loss | Cross-Entropy |

---

## 📈 Evaluation

### Automatic Metrics

| Metric | What It Measures |
|---|---|
| BLEU | N-gram overlap between generated and reference transliteration |
| Perplexity | Model confidence in predicting next token (lower = better) |
| CER | Character Error Rate — fraction of characters incorrectly transliterated |
| Levenshtein Distance | Edit distance between generated and reference string |

### Qualitative Evaluation
Side-by-side comparison of model output vs ground truth transliterations across multiple test samples.

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/urdu-to-roman-nmt.git
cd urdu-to-roman-nmt
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Clone the dataset

```bash
git clone https://github.com/amir9ume/urdu_ghazals_rekhta.git data/urdu_ghazals_rekhta
```

### 4. Preprocess and train

```bash
python data/preprocess.py
python training/train.py
```

### 5. Run experiments

```bash
python training/experiments.py
```

### 6. Launch the Streamlit app

```bash
streamlit run app/app.py
```

---

## 🖥️ Live Demo

The Streamlit app allows real-time transliteration:

- Input Urdu text (اردو script)
- Get instant Roman Urdu transliteration output
- Deployed live — no setup needed

---

## ✅ Tasks Completed

- [x] Urdu text normalization and Roman Urdu extraction
- [x] Subword tokenization (BPE / WordPiece)
- [x] BiLSTM encoder (2 layers) + LSTM decoder (4 layers) in PyTorch
- [x] Training with cross-entropy loss and Adam optimizer
- [x] 3 controlled hyperparameter experiments
- [x] BLEU, Perplexity, CER, Levenshtein evaluation
- [x] Qualitative output vs ground truth comparison
- [x] Streamlit live demo deployment
- [x] Medium blog post
- [x] LinkedIn post

---

## 🏆 Bonus

- [ ] Dataset augmentation via back-transliteration or noise injection
- [ ] Replace BiLSTM + LSTM with xLSTM architecture

---

## 🔗 Links

| Resource | Link |
|---|---|
| Medium Blog Post | [Read on Medium](#) |
| LinkedIn Post | [View on LinkedIn](#) |
| Live Demo | [Streamlit App](#) |
| Dataset | [urdu_ghazals_rekhta](https://github.com/amir9ume/urdu_ghazals_rekhta) |

---

## 👤 Author

**Muhammad Naeem** — FAST-NUCES 

---

## 📄 License

This project is for academic purposes under the MIT License.
