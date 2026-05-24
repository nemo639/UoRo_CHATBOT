# Urdu Conversational Chatbot with Transformer Multi-Head Attention

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch)
![Transformer](https://img.shields.io/badge/Transformer-From%20Scratch-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?style=flat-square&logo=streamlit)
![Language](https://img.shields.io/badge/Language-Urdu%20%D8%A7%D8%B1%D8%AF%D9%88-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

**A custom Transformer encoder-decoder chatbot built from scratch for Urdu conversational AI**

[📝 Medium Blog](#) · [💼 LinkedIn Post](#) · [🤗 Live Demo](#)

</div>

---

## 📌 Overview

This project implements a fully custom **Transformer encoder-decoder architecture from scratch** (no pre-trained models) for building a conversational chatbot in Urdu. The model uses multi-head attention to capture contextual relationships in Urdu text and generates fluent, context-aware responses.

```
Urdu Input  →  Encoder (Multi-Head Attention)  →  Context Representation
                                                          ↓
                                              Decoder (Masked MHA + Cross-Attention)
                                                          ↓
                                               Generated Urdu Response
```

---

## ✨ Features

- ✅ Full Transformer encoder-decoder implemented from scratch in PyTorch
- ✅ Multi-Head Attention, Positional Encoding, and Feed-Forward layers
- ✅ Urdu text normalization (diacritics removal, Alef/Yeh standardization)
- ✅ Teacher forcing during training
- ✅ Greedy and Beam search decoding strategies
- ✅ BLEU, ROUGE-L, chrF, and Perplexity evaluation
- ✅ Human evaluation — Fluency, Relevance, Adequacy (1–5 scale)
- ✅ Streamlit / Gradio interface with right-to-left Urdu rendering
- ✅ Deployed on Streamlit Cloud / Gradio public link

---

## 🗂️ Repository Structure

```
📦 urdu-transformer-chatbot/
├── 📁 data/
│   └── preprocess.py              # Urdu normalization, tokenization, vocab
├── 📁 model/
│   ├── attention.py               # Multi-Head Attention
│   ├── positional_encoding.py     # Sinusoidal positional encoding
│   ├── encoder.py                 # Transformer encoder stack
│   ├── decoder.py                 # Transformer decoder stack
│   └── transformer.py             # Full encoder-decoder model
├── 📁 training/
│   ├── train.py                   # Training loop with teacher forcing
│   └── evaluate.py                # BLEU, ROUGE-L, chrF, Perplexity
├── 📁 inference/
│   └── generate.py                # Greedy & Beam search decoding
├── 📁 app/
│   └── app.py                     # Streamlit / Gradio chatbot UI
├── 📓 urdu_chatbot.ipynb          # Full pipeline notebook
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🧠 Model Architecture

The entire Transformer is built from scratch using base PyTorch — no HuggingFace model weights used.

### Encoder
- Embeds the Urdu input tokens
- Applies sinusoidal positional encoding
- Passes through N stacked encoder layers, each containing:
  - Multi-Head Self-Attention
  - Position-wise Feed-Forward Network
  - Layer Normalization + Residual Connections

### Decoder
- Embeds the target tokens
- Applies positional encoding
- Passes through N stacked decoder layers, each containing:
  - Masked Multi-Head Self-Attention (prevents attending to future tokens)
  - Cross-Attention over encoder output
  - Feed-Forward Network + Layer Norm + Residuals

### Hyperparameters

| Parameter | Suggested Value |
|---|---|
| Embedding Dimensions | 256 / 512 |
| Attention Heads | 2 |
| Encoder Layers | 2 |
| Decoder Layers | 2 |
| Dropout | 0.1 – 0.3 |
| Batch Size | 32 / 64 |
| Learning Rate | 1e-4 – 5e-4 (Adam) |

---

## 📊 Dataset

**Urdu Conversational Dataset (20,000 samples)**
- Source: [Kaggle — muhammadahmedansari/urdu-dataset-20000](https://www.kaggle.com/datasets/muhammadahmedansari/urdu-dataset-20000)
- Content: Urdu question-answer conversational pairs
- Split: **80% Train / 10% Validation / 10% Test**

---

## 🔤 Preprocessing Pipeline

Urdu text requires specialised preprocessing before tokenization:

- **Diacritics removal** — strip Harakat (زیر، زبر، پیش) that vary between writers
- **Alef normalization** — standardize ا، آ، أ، إ to a single form
- **Yeh normalization** — standardize ی، ے، ئ forms
- **Tokenization** — whitespace and subword tokenization
- **Vocabulary building** — build source and target vocab with special tokens (`<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`)

---

## 🏋️ Training

- **Teacher forcing** — during training the decoder receives ground truth tokens as input at each step, speeding up convergence
- **Best model saving** — checkpoint saved based on highest validation BLEU score
- **Optimizer** — Adam with learning rate scheduling

---

## 📈 Evaluation

### Automatic Metrics

| Metric | What It Measures |
|---|---|
| BLEU | N-gram overlap between generated and reference response |
| ROUGE-L | Longest common subsequence recall |
| chrF | Character-level F-score — robust for morphologically rich languages like Urdu |
| Perplexity | How confidently the model predicts the next token (lower = better) |

### Human Evaluation

Three annotators rated a random sample of responses on a 1–5 scale:

| Dimension | Description |
|---|---|
| Fluency | Is the generated Urdu grammatically natural? |
| Relevance | Does the response address the input? |
| Adequacy | Does the response convey the correct meaning? |

Qualitative examples comparing model output vs ground truth are included in the notebook.

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/urdu-transformer-chatbot.git
cd urdu-transformer-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Add the Kaggle dataset to `data/` or run:

```bash
kaggle datasets download -d muhammadahmedansari/urdu-dataset-20000
```

### 4. Preprocess and train

```bash
python data/preprocess.py
python training/train.py
```

### 5. Launch the chatbot app

```bash
streamlit run app/app.py
```

---

## 🖥️ Chatbot Interface

The Streamlit / Gradio app includes:

- **Urdu input box** with right-to-left text rendering
- **Generated reply display** in Urdu script
- **Conversation history** panel
- **Decoding strategy selector** — Greedy or Beam Search

---

## ✅ Tasks Completed

- [x] Urdu text normalization and preprocessing
- [x] Vocabulary building with special tokens
- [x] Transformer encoder-decoder from scratch (PyTorch)
- [x] Multi-Head Attention, Positional Encoding, Feed-Forward layers
- [x] Training with teacher forcing
- [x] Validation with BLEU-based checkpoint saving
- [x] BLEU, ROUGE-L, chrF, Perplexity evaluation
- [x] Human evaluation (Fluency, Relevance, Adequacy)
- [x] Greedy and Beam search decoding
- [x] Streamlit / Gradio chatbot interface with RTL rendering
- [x] Deployment on Streamlit Cloud / Gradio public link

---

## 🔗 Links

| Resource | Link |
|---|---|
| Medium Blog Post | [Read on Medium](#) |
| LinkedIn Post | [View on LinkedIn](#) |
| Live Demo | [Streamlit / Gradio](#) |
| Dataset | [Kaggle — Urdu Dataset 20000](https://www.kaggle.com/datasets/muhammadahmedansari/urdu-dataset-20000) |

---

## 👤 Author

**Muhammad Naeem** — FAST-NUCES 

---

## 📄 License

This project is for academic purposes under the MIT License.
