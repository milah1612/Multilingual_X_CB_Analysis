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

    # --- KPI Metrics
    st.subheader("📌 Cyberbullying Insights")
    kpi1, kpi2, kpi3 = st.columns(3)
    total_cb = len(df_cb)
    avg_len = df_cb["eda_clean"].str.len().mean()
    perc = (total_cb / len(st.session_state.df)) * 100
    kpi1.metric("Total CB Tweets", total_cb)
    kpi2.metric("Avg. Tweet Length", f"{avg_len:.1f}")
    kpi3.metric("% of Dataset", f"{perc:.1f}%")

    # --- Language filter
    languages = ["All"] + sorted(df_cb["language"].dropna().unique())
    selected_lang = st.selectbox("🌍 Filter by Language", languages, key="cb_lang")
    if selected_lang != "All":
        df_cb = df_cb[df_cb["language"] == selected_lang]

    # --- CB Distribution by Language
    st.subheader("🌍 CB Distribution by Language")
    cb_lang_dist = df_cb["language"].value_counts().reset_index()
    cb_lang_dist.columns = ["language", "count"]
    fig_cb_lang = px.bar(cb_lang_dist, x="language", y="count", color="language",
                         text="count", height=500,
                         color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig_cb_lang, use_container_width=True)

    # --- Distinctive Hashtags (Bubble Chart)
    hashtags = [h for tags in df_cb["hashtags"] for h in tags]
    top_hashtags = Counter(hashtags).most_common(15)
    if top_hashtags:
        st.subheader("#️⃣ Distinctive Hashtags")
        hashtags_df = pd.DataFrame(top_hashtags, columns=["hashtag", "count"])
        fig_bubble = px.scatter(hashtags_df, x="hashtag", y="count", size="count",
                                color="hashtag", hover_name="hashtag",
                                size_max=60, height=500)
        st.plotly_chart(fig_bubble, use_container_width=True)
    else:
        st.info("No hashtags found for this filter.")

    # --- Distinctive Emojis (Interactive Table)
    emojis = [e for em in df_cb["eda_clean"].apply(extract_emojis) for e in em]
    top_emojis = Counter(emojis).most_common(15)
    if top_emojis:
        st.subheader("😊 Distinctive Emojis")
        emojis_df = pd.DataFrame(top_emojis, columns=["emoji", "count"])
        emojis_df["percentage"] = (emojis_df["count"] / emojis_df["count"].sum() * 100).round(1)
        st.dataframe(emojis_df, use_container_width=True, height=300)
    else:
        st.info("No emojis found for this filter.")

    # --- Average Tweet Length by Class
    st.subheader("📏 Average Tweet Length by Class")
    avg_len_class = st.session_state.df.groupby("sentiment")["eda_clean"].apply(lambda x: x.str.len().mean()).reset_index()
    avg_len_class.columns = ["sentiment", "avg_length"]
    fig_len = px.bar(avg_len_class, x="sentiment", y="avg_length", color="sentiment",
                     text="avg_length", height=500,
                     color_discrete_map={"Cyberbullying": "#FF6F61", "Non Cyberbullying": "#4C9AFF"})
    st.plotly_chart(fig_len, use_container_width=True)

    # --- Hashtag Clustering (Treemap)
    st.subheader("🧩 Hashtag Clustering (Experimental)")
    if top_hashtags:
        fig_cluster = px.treemap(hashtags_df, path=["hashtag"], values="count",
                                 color="count", color_continuous_scale="Viridis",
                                 height=500)
        st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        st.info("No hashtags found for clustering.")

    # --- CB Prevalence by Language
    st.subheader("📊 CB Prevalence by Language")
    lang_total = st.session_state.df.groupby("language").size().reset_index(name="total")
    lang_cb = df_cb.groupby("language").size().reset_index(name="cb_count")
    prevalence = pd.merge(lang_cb, lang_total, on="language", how="left")
    prevalence["cb_percent"] = (prevalence["cb_count"] / prevalence["total"] * 100).round(1)
    fig_prev = px.bar(prevalence, x="language", y="cb_percent", color="language",
                      text="cb_percent", height=500,
                      color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_prev, use_container_width=True)

    # --- Cyberbullying Tweets Table
    st.subheader("📋 Cyberbullying Tweets")
    st.dataframe(df_cb[["language", "sentiment", "model_clean", "translated_tweet"]],
                 use_container_width=True, height=400)

    # --- Report Download
    export_df = df_cb[["id", "language", "binary_label", "sentiment", "model_clean"]]
    csv = export_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("⬇ Download Cyberbullying Report", csv,
                       "cyberbullying_report.csv", "text/csv")

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
