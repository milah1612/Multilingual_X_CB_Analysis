import streamlit as st
import pandas as pd
import re, html, emoji
import sqlite3, os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import plotly.express as px
from collections import Counter
from deep_translator import GoogleTranslator
from langdetect import detect

# ==============================
# Language Mapping
# ==============================
LANG_MAP = {
    "ar": "arabic",
    "fr": "french",
    "en": "english",
    "es": "spanish",
    "hi": "hindi",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "unknown": "unknown"
}

# ==============================
# DB Functions
# ==============================
DB_FILE = "tweets.db"
CSV_FILE = "tweet_data.csv"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tweets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            language TEXT,
            binary_label INTEGER,
            sentiment TEXT,
            model_clean TEXT,
            eda_clean TEXT,
            translated_tweet TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def migrate_csv_to_sqlite():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM tweets")
    count = cursor.fetchone()[0]
    conn.close()

    if count == 0 and os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        if "translated_tweet" not in df.columns:
            df["translated_tweet"] = "[not translated]"
        conn = sqlite3.connect(DB_FILE)
        df.to_sql("tweets", conn, if_exists="append", index=False)
        conn.close()
        print("✅ Migrated CSV into SQLite (first time only)")
    else:
        print("➡️ DB already has data, skipping migration") 

def load_tweets():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM tweets ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def insert_tweet(text, language, binary_label, sentiment, model_clean, eda_clean, translated_tweet):
    timestamp = datetime.now().isoformat()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO tweets (text, language, binary_label, sentiment, model_clean, eda_clean, translated_tweet, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (text, language, binary_label, sentiment, model_clean, eda_clean, translated_tweet, timestamp))
    conn.commit()
    conn.close()

    new_row = pd.DataFrame([{
        "id": None,
        "text": text,
        "language": language,
        "binary_label": binary_label,
        "sentiment": sentiment,
        "model_clean": model_clean,
        "eda_clean": eda_clean,
        "translated_tweet": translated_tweet,
        "timestamp": timestamp
    }])
    return new_row

# ==============================
# Init and Cache
# ==============================
init_db()
migrate_csv_to_sqlite()

if "df" not in st.session_state:
    st.session_state.df = load_tweets()

# ==============================
# Load Hugging Face Model
# ==============================
MODEL_PATH = "Mila1612/mdeberta-cyberbullying"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model()

# ==============================
# Cleaning Functions
# ==============================
def clean_for_model(text):
    text = str(text).lower()
    text = re.sub(r"(https?://\S+|www\.\S+|\bhttp\b)", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\brt\b", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_for_eda(text):
    text = str(text).lower()
    text = html.unescape(text)
    text = re.sub(r"(https?://\S+|www\.\S+|\bhttp\b)", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\brt\b", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[•…]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==============================
# Prediction
# ==============================
def predict(text, threshold=0.35):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        cb_prob = probs[0][1].item()
        pred = 1 if cb_prob > threshold else 0
    return pred, cb_prob

# ==============================
# Helper: Extract hashtags & emojis
# ==============================
def extract_hashtags(text):
    if isinstance(text, str):
        return re.findall(r"#\w+", text)
    return []

def extract_emojis(text):
    """Return list of emojis in text (safe for NaN)."""
    if isinstance(text, str):
        return [c for c in text if c in emoji.EMOJI_DATA]
    return []


# ==============================
# Dashboard Layout
# ==============================
st.set_page_config(page_title="Cyberbullying Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center;'>🚨 Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)

tabs = st.tabs(["All 🌍", "Cyberbullying 🚨", "Non-Cyberbullying 🙂"])

# ==============================
# All Tab
# ==============================
with tabs[0]:
    df = st.session_state.df
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("📊 Sentiment Distribution")
        sentiment_counts = df["sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["sentiment", "count"]
        fig_pie = px.pie(sentiment_counts, values="count", names="sentiment", color="sentiment",
                         height=500, color_discrete_map={"Cyberbullying": "#FF6F61", "Non Cyberbullying": "#4C9AFF"})
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.subheader("🌍 Language Distribution by Sentiment")
        lang_dist = df.groupby(["language", "sentiment"]).size().reset_index(name="count")
        fig_bar = px.bar(lang_dist, x="language", y="count", color="sentiment", barmode="group",
                         text="count", height=500,
                         color_discrete_map={"Cyberbullying": "#FF6F61", "Non Cyberbullying": "#4C9AFF"})
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("📝 All Tweets")
    st.dataframe(df[["language", "sentiment", "model_clean", "translated_tweet"]],
                 use_container_width=True, height=400)

# ==============================
# Cyberbullying Tab
# ==============================
with tabs[1]:
    df_cb = st.session_state.df[st.session_state.df["sentiment"] == "Cyberbullying"].copy()
    df_cb["hashtags"] = df_cb["text"].apply(extract_hashtags)

    # --- KPIs
    total_cb = len(df_cb)
    avg_len_cb = df_cb["eda_clean"].str.len().mean()
    pct_cb = (total_cb / len(st.session_state.df) * 100) if len(st.session_state.df) > 0 else 0

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Tweets", f"{total_cb:,}")
    kpi2.metric("Avg. Length", f"{avg_len_cb:.1f}")
    kpi3.metric("% of Dataset", f"{pct_cb:.1f}%")

    st.markdown("---")

    # --- Row 2: Top Words + Hashtags
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔤 Top Words")
        words = " ".join(df_cb["eda_clean"].astype(str)).split()
        word_freq = Counter(words)
        top_words = pd.DataFrame(word_freq.most_common(15), columns=["word", "count"])
        fig = px.bar(top_words, x="word", y="count", text="count", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏷️ Top Hashtags")
        hashtags = [h for tags in df_cb["hashtags"] for h in tags]
        hash_freq = Counter(hashtags)
        top_hash = pd.DataFrame(hash_freq.most_common(15), columns=["hashtag", "count"])
        if not top_hash.empty:
            fig2 = px.bar(top_hash, x="hashtag", y="count", text="count", height=400)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No hashtags found.")

    st.markdown("---")

    # --- Top Emojis (table format)
    st.subheader("😊 Top Emojis")
    emojis = [e for em in df_cb["eda_clean"].apply(extract_emojis) for e in em]
    emoji_freq = Counter(emojis).most_common(10)
    if emoji_freq:
        emoji_df = pd.DataFrame(emoji_freq, columns=["emoji", "count"])
        st.table(emoji_df)
    else:
        st.info("No emojis found.")

    st.markdown("---")

    # --- Table with Language Filter
    st.subheader("📋 Cyberbullying Tweets")
    lang_options = ["All"] + sorted(df_cb["language"].dropna().unique())
    selected_lang = st.selectbox("🌍 Filter by Language", lang_options)

    filtered_cb = df_cb.copy()
    if selected_lang != "All":
        filtered_cb = filtered_cb[filtered_cb["language"] == selected_lang]

    st.dataframe(
        filtered_cb[["language", "sentiment", "model_clean", "translated_tweet"]],
        use_container_width=True, height=400
    )

    # ✅ Report download (with full cols)
    export_cb = filtered_cb[["id", "language", "binary_label", "sentiment", "model_clean", "translated_tweet"]]
    csv = export_cb.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("⬇ Download Cyberbullying Report", csv, "cyberbullying_report.csv", "text/csv")


# ==============================
# Sidebar
# ==============================
st.sidebar.header("🔍 X Cyberbullying Detection")
tweet_input = st.sidebar.text_area("✍️ Enter a tweet for analysis:")
if st.sidebar.button("Analyze Tweet"):
    if tweet_input.strip():
        model_cleaned = clean_for_model(tweet_input)
        eda_cleaned = clean_for_eda(tweet_input)
        label, cb_prob = predict(model_cleaned)
        sentiment = "Cyberbullying" if label == 1 else "Non Cyberbullying"
        try:
            detected_code = detect(tweet_input)
            lang = LANG_MAP.get(detected_code, detected_code)
        except:
            lang = "unknown"
        try:
            translated = GoogleTranslator(source="auto", target="en").translate(tweet_input)
        except Exception:
            translated = "[translation error]"

        new_row = insert_tweet(tweet_input, lang, label, sentiment, model_cleaned, eda_cleaned, translated)
        st.session_state.df = pd.concat([new_row, st.session_state.df], ignore_index=True)

        st.sidebar.success(f"✅ Prediction: {sentiment}")
        st.sidebar.write(f"🌍 Language: {lang}")
        st.sidebar.write(f"🌐 Translated: {translated}")
