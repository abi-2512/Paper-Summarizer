import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import re
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Long-T5 tokenizer and model
@st.cache_resource
def load_models():
    summarizer_tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("google/long-t5-tglobal-base").to(device)
    summarizer = pipeline("summarization", model=summarizer_model, tokenizer=summarizer_tokenizer, device=0 if torch.cuda.is_available() else -1)

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    return summarizer, classifier, st_model, summarizer_tokenizer

summarizer, classifier, st_model, tokenizer = load_models()

def split_into_sentences(text):
    return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]

def summarize_paper(text, max_input_tokens=15000):
    tokens = tokenizer.encode(text, truncation=False)
    if len(tokens) > max_input_tokens:
        st.warning(f"Text is too long ({len(tokens)} tokens). Truncating to {max_input_tokens}.")
        tokens = tokens[:max_input_tokens]
        text = tokenizer.decode(tokens, skip_special_tokens=True)

    summary = summarizer(text, max_length=512, min_length=150, do_sample=False)[0]['summary_text']
    return summary

def rank_sentences_by_similarity(sentences, query):
    query_embedding = st_model.encode(query, convert_to_tensor=True)
    sentence_embeddings = st_model.encode(sentences, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
    sorted_indices = torch.argsort(cosine_scores, descending=True)
    return [(sentences[i], cosine_scores[i].item()) for i in sorted_indices]

def detect_bias(sentence):
    labels = ["biased", "inconsistent", "sound"]
    result = classifier(sentence, labels)
    return result['labels'][0], result['scores'][0]

# Sidebar: About
st.sidebar.title("📘 About This Demo")
st.sidebar.markdown("""
**Research Paper Analyzer**  
Built using:
- 🤗 `long-t5-tglobal-base` for summarization (handles up to 16,000 tokens)
- 🤗 `facebook/bart-large-mnli` for zero-shot classification
- 🧠 `all-MiniLM-L6-v2` for semantic similarity

Designed to analyze **full research papers** for key insights and ethical concerns.
""")

# Page Title
st.title("🔍 Research Paper Analyzer")
st.caption("Upload a research paper or try a sample. See a smart summary + ethical/bias review.")

# Sample files
sample_texts = {
    "Sample: AlexNet": open("samples/AlexNet.txt").read(),
    "Sample: Attention is All You Need": open("samples/Attention.txt").read(),
    "Sample: Generative Adversarial Networks": open("samples/GANS.txt").read(),
}

selected_sample = st.selectbox("📁 Try a sample paper (or upload your own):", ["-- Select --"] + list(sample_texts.keys()))
uploaded_file = st.file_uploader("Or upload your .txt file", type="txt")

paper_text = None
if selected_sample != "-- Select --":
    paper_text = sample_texts[selected_sample]
elif uploaded_file is not None:
    try:
        paper_text = uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        st.stop()

if not paper_text:
    st.info("Please select a sample or upload a .txt file to begin.")
    st.stop()

# Summarize
with st.spinner("🔬 Summarizing..."):
    summary = summarize_paper(paper_text)
    st.subheader("📄 Summary")
    st.write(summary)

# Sentence-level analysis
sentences = split_into_sentences(summary)

with st.spinner("📊 Ranking by Ethical Relevance..."):
    query = "ethical issues, bias, and methodological inconsistencies"
    ranked = rank_sentences_by_similarity(sentences, query)
    st.subheader("💬 Top Ranked Sentences")
    for sent, score in ranked[:5]:
        st.markdown(f"**Score:** {score:.4f}  \n{sent}")

with st.spinner("🧠 Analyzing for Bias/Inconsistency..."):
    st.subheader("⚠️ Bias & Consistency Detection")
    for sent in sentences:
        label, score = detect_bias(sent)
        emoji = {"biased": "❌", "inconsistent": "⚠️", "sound": "✅"}[label]
        st.write(f"{emoji} **{label.upper()}** ({score:.2f}): {sent}")

# Download
st.download_button("💾 Download Summary", summary, file_name="summary.txt")
