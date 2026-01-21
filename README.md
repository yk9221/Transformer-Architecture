# Transformer Architecture & BERT Fine-Tuning

## 📖 Overview
This project explores the mathematical foundations of Large Language Models (LLMs) through a dual approach:
1.  **Low-Level Implementation:** Building core Transformer components (RoPE, Multi-Head Attention, ALiBi) from scratch in PyTorch to validate mathematical correctness.
2.  **Applied Optimization:** Fine-tuning BERT variants on the **AG News** dataset to evaluate trade-offs between model size, accuracy, and inference speed.

## 🚀 Key Features
* **Custom Layers:** Implemented **Rotary Positional Embeddings (RoPE)**, **RMSNorm**, and Attention mechanisms without high-level wrappers.
* **Model Optimization:** Fine-tuned **TinyBERT (4L)** vs. **BERT-Base**, achieving 94% of base performance with 7x fewer parameters.
* **Embedding Analysis:** Benchmarked full fine-tuning against **Linear Probing** using fixed Sentence Transformer embeddings.

## 🛠️ Tech Stack
* **Languages:** Python 3.9+, PyTorch
* **Libraries:** Hugging Face (Transformers, Datasets), Scikit-Learn, NumPy
* **Concepts:** NLP, Transfer Learning, Matrix Calculus

## 📊 Results (AG News Test Set)

| Model | Method | Accuracy | Params |
| :--- | :--- | :--- | :--- |
| **BERT Base** | Full Fine-Tuning | **94.2%** | 110M |
| **TinyBERT** | Full Fine-Tuning | **88.5%** | 14.5M |
| **Sentence-BERT** | Linear Probe | 86.1% | ~4K |

## 💻 Usage

**1. Install Dependencies**
```bash
pip install torch transformers datasets scikit-learn
