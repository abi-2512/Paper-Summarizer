
---

# 🔍 Research Paper Summarizer

A powerful tool for analyzing long-form research papers. This app uses cutting-edge NLP models to **summarize full papers**, **rank sentences by ethical relevance**, and **detect bias or inconsistencies** — all in a clean Streamlit interface.

---

## 🚀 Features

### 🧠 AI-Powered Summarization

* Utilizes **Long-T5 (`google/long-t5-tglobal-base`)** to handle long documents up to \~16,000 tokens.
* Extracts a concise, high-quality summary from the full paper.

### 💬 Ethical Sentence Ranking

* Ranks sentences by relevance to ethical concerns using **semantic similarity** with `all-MiniLM-L6-v2`.

### ⚠️ Bias Detection

* Runs **zero-shot classification** (`facebook/bart-large-mnli`) to label sentences as:

  * `BIAS`
  * `INCONSISTENT`
  * `SOUND`

### 🖥 Clean UI via Streamlit

* Upload `.txt` research papers or choose from provided samples
* See summary, top ethical sentences, and sentence-level bias analysis
* Download the generated summary

---

## 📁 Sample Papers Included

* **AlexNet**
* **Attention is All You Need**
* **Generative Adversarial Networks (GANs)**

All located in the `samples/` directory as `.txt` files.

---

## 🛠 Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/research-paper-analyzer.git
   cd research-paper-analyzer
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**

   ```bash
   streamlit run app.py
   ```

---

## 📦 Requirements

* Python 3.8+
* PyTorch + Transformers
* Streamlit
* SentenceTransformers
* GPU (optional but recommended for speed)

---

## 🧪 Known Issues

* ❗ Large files >16,000 tokens are truncated — proper chunking support is not yet implemented (this is why it can only analyze abstracts and conclusions)
* ❗ Bias classification is experimental and should not be used for formal reviews
* ❗ Sentence segmentation may be imperfect for poorly formatted `.txt` files

---

## 🌱 Future Improvements

* [ ] Chunk long documents and stitch multi-part summaries
* [ ] Improve bias classification with fine-tuned models
* [ ] Add visual heatmaps for ethical sentence relevance
* [ ] PDF upload + parsing support
* [ ] Save full analysis reports (PDF/Markdown)

---

## ✍️ Authors

Made with ❤️ by Abi
If you use this tool, drop a star ⭐ and share your feedback!

---
