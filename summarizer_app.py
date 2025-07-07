import streamlit as st
import nltk
import re
import numpy as np
import networkx as nx
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from gtts import gTTS
import os
import base64
import datetime

# Download stopwords if not present
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Preprocessing function
def preprocess_sentences(text):
    sentences = text.split(".")
    cleaned = []
    for sentence in sentences:
        cleaned_text = re.sub("[^a-zA-Z]", " ", sentence).lower()
        words = cleaned_text.split()
        if words:
            cleaned.append(words)
    return cleaned

# Similarity logic
def sentence_similarity(sent1, sent2, stop_words=None):
    if stop_words is None:
        stop_words = []
    sent1 = [w for w in sent1 if w not in stop_words]
    sent2 = [w for w in sent2 if w not in stop_words]
    all_words = list(set(sent1 + sent2))
    v1 = [0] * len(all_words)
    v2 = [0] * len(all_words)

    for w in sent1:
        v1[all_words.index(w)] += 1
    for w in sent2:
        v2[all_words.index(w)] += 1

    return 1 - cosine_distance(v1, v2)

# Build similarity matrix
def build_similarity_matrix(sentences, stop_words):
    size = len(sentences)
    sim_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i != j:
                sim_matrix[i][j] = sentence_similarity(sentences[i], sentences[j], stop_words)
    return sim_matrix

# Final summarization logic
def summarize(text, top_n=3):
    original_sentences = [s.strip() for s in text.split(".") if s.strip()]
    sentences = preprocess_sentences(text)
    stop_words = stopwords.words('english')

    sim_matrix = build_similarity_matrix(sentences, stop_words)
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph)

    ranked = sorted(((scores[i], s) for i, s in enumerate(original_sentences)), reverse=True)
    top_sentences = [s for _, s in ranked[:top_n]]
    return top_sentences

# === Streamlit UI ===
st.set_page_config(page_title="📝 Text Summarizer", layout="centered")
st.title("📝 AI-Powered Text Summarizer")

# Info
today = datetime.datetime.now().strftime("%d %B %Y")
st.caption(f"📅 Date: {today}")
st.markdown("Upload a `.txt` file or paste your article below to generate a summary.")

# File upload
uploaded_file = st.file_uploader("📄 Upload a text file", type="txt")

# Or paste text
input_text = st.text_area("✏️ Or paste your article here:", height=250)

# Options
top_n = st.slider("🎯 Number of sentences in summary:", 1, 10, 3)
style = st.radio("🧾 Format summary as:", ["Paragraph", "Bullet Points", "Numbered List"])
lang = st.selectbox("🔈 Choose voice language:", ["en", "hi", "kn"])

# Summary generation
if st.button("🚀 Generate Summary"):
    text = ""
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
    elif input_text.strip():
        text = input_text.strip()
    else:
        st.warning("❗ Please upload a file or paste text.")

    if text:
        summary_list = summarize(text, top_n=top_n)
        summary = ". ".join(summary_list)

        # Format output
        if style == "Bullet Points":
            summary_display = "\n".join(["• " + s for s in summary_list])
        elif style == "Numbered List":
            summary_display = "\n".join([f"{i+1}. {s}" for i, s in enumerate(summary_list)])
        else:
            summary_display = summary

        st.success("✅ Summary:")
        st.text_area("📝 Summary Output:", summary_display, height=200)

        # Word/Character count
        st.markdown(f"📃 **Original**: {len(text.split())} words / {len(text)} characters")
        st.markdown(f"✂️ **Summary**: {len(summary.split())} words / {len(summary)} characters")

        # Download
        b64 = base64.b64encode(summary.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="summary.txt">📥 Download Summary</a>'
        st.markdown(href, unsafe_allow_html=True)

        # TTS
        if st.button("🔊 Listen to Summary"):
            tts = gTTS(summary, lang=lang)
            tts.save("summary.mp3")
            audio_file = open("summary.mp3", "rb")
            st.audio(audio_file.read(), format='audio/mp3')

# Footer
st.markdown("---")
st.caption("🚀 Made with 💙 by Anusha using Streamlit, NLTK, NetworkX, and gTTS.")

# Background styling
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://source.unsplash.com/1600x900/?notebook,paper,writing');
            background-size: cover;
        }
    </style>
""", unsafe_allow_html=True)
