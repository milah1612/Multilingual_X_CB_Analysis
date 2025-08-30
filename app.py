import streamlit as st
import pandas as pd
import re, html
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import plotly.express as px
from collections import Counter
from deep_translator import GoogleTranslator
from langdetect import detect
import io, os
from sqlalchemy import create_engine, text

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

LANG_COLORS = {
    "arabic": "#FF6F61",
    "french": "#6B5B95",
    "english": "#88B04B",
    "spanish": "#F7CAC9",
    "hindi": "#92A8D1",
    "german": "#955251",
    "italian": "#B565A7",
    "portuguese": "#009B77",
    "unknown": "#DD4124"
}

# ==============================
# Database Setup
# ==============================
DATABASE_URL = st.secrets["DATABASE_URL"]
engine = create_engine(DATABASE_URL)

def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS tweets (
                id SERIAL PRIMARY KEY,
                text TEXT,
                language TEXT,
                binary_label INTEGER,
                sentiment TEXT,
                model_clean TEXT,
                eda_clean TEXT,
                translated_tweet TEXT,
                source_file TEXT,
                timestamp TEXT
            )
        """))

def migrate_csv_to_postgres():
    if os.path.exists("tweet_data.csv"):
        df = pd.read_csv("tweet_data.csv")

        # Ensure required schema
        required_cols = [
            "text", "language", "binary_label", "sentiment",
            "model_clean", "eda_clean", "translated_tweet",
            "source_file", "timestamp"
        ]

        # Add missing columns if not present
        for col in required_cols:
            if col not in df.columns:
                if col == "binary_label":
                    df[col] = 0
                elif col == "source_file":
                    df[col] = "initial_csv"
                elif col == "timestamp":
                    df[col] = datetime.now().isoformat()
                else:
                    df[col] = ""

        # Reorder and fill NaN
        df = df[required_cols].fillna("")

        with engine.begin() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM tweets")).scalar()
            if count == 0:
                df.to_sql("tweets", engine, if_exists="replace", index=False, method="multi")
                print("✅ Migrated CSV into Postgres (first time only)")
            else:
                print("➡️ DB already has data, skipping migration")


def load_tweets():
    with engine.connect() as conn:
        return pd.read_sql("SELECT * FROM tweets ORDER BY timestamp DESC", conn)


def insert_tweet(text, language, binary_label, sentiment, model_clean, eda_clean,
                 translated_tweet, source_file="manual"):
    timestamp = datetime.now().isoformat()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO tweets (
                text, language, binary_label, sentiment,
                model_clean, eda_clean, translated_tweet,
                source_file, timestamp
            )
            VALUES (
                :text, :language, :binary_label, :sentiment,
                :model_clean, :eda_clean, :translated_tweet,
                :source_file, :timestamp
            )
        """), {
            "text": text, "language": language, "binary_label": binary_label,
            "sentiment": sentiment, "model_clean": model_clean, "eda_clean": eda_clean,
            "translated_tweet": translated_tweet, "source_file": source_file,
            "timestamp": timestamp
        })

    # Return as DataFrame for UI concat
    return pd.DataFrame([{
        "text": text, "language": language, "binary_label": binary_label,
        "sentiment": sentiment, "model_clean": model_clean, "eda_clean": eda_clean,
        "translated_tweet": translated_tweet, "source_file": source_file,
        "timestamp": timestamp
    }])


# ==============================
# Init DB + Session
# ==============================
init_db()
migrate_csv_to_postgres()

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
# Helper: Extract hashtags
# ==============================
def extract_hashtags(text):
    if isinstance(text, str):
        return re.findall(r"#\w+", text)
    return []

# ==============================
# Pagination Helper
# ==============================
def render_paginated_table(df, key_prefix, columns=None, rows_per_page=20):
    if columns:
        df = df[columns].rename(columns={"model_clean": "tweet"})
    total_rows = len(df)
    total_pages = (total_rows // rows_per_page) + (1 if total_rows % rows_per_page else 0)
    page = st.number_input("Page", min_value=1, max_value=max(total_pages, 1),
                           value=1, key=f"{key_prefix}_page")
    start_idx = (page - 1) * rows_per_page
    end_idx = start_idx + rows_per_page
    st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True, height=400)
    st.caption(f"Page {page} of {total_pages} — showing {rows_per_page} rows per page")

# ==============================
# Dashboard Layout
# ==============================
st.set_page_config(page_title="Cyberbullying Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center;'>🚨 SENTIMENT ANALYSIS DASHBOARD</h1>", unsafe_allow_html=True)

tabs = st.tabs(["All 🌍", "Cyberbullying 🚨", "Non-Cyberbullying 🙂"])

# ==============================
# All Tab
# ==============================
with tabs[0]:
    df = st.session_state.df
    if not df.empty:
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
        render_paginated_table(df, key_prefix="all", columns=["language", "sentiment", "model_clean", "translated_tweet"])
    else:
        st.info("No data available.")

# ==============================
# Cyberbullying Tab
# ==============================
with tabs[1]:
    df_cb = st.session_state.df[st.session_state.df["sentiment"] == "Cyberbullying"].copy()
    if not df_cb.empty:
        df_cb["hashtags"] = df_cb["text"].apply(extract_hashtags)

        st.subheader("📌 Cyberbullying Insights")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total CB Tweets", len(df_cb))
        kpi2.metric("Avg. Tweet Length", f"{df_cb['eda_clean'].str.len().mean():.1f}")
        kpi3.metric("% of Dataset", f"{(len(df_cb) / len(st.session_state.df)) * 100:.1f}%")

        st.subheader("🌍 CB Distribution by Language")
        cb_lang_dist = df_cb["language"].value_counts().reset_index()
        cb_lang_dist.columns = ["language", "count"]
        fig_cb_lang = px.bar(cb_lang_dist, x="language", y="count", color="language",
                             text="count", height=500, color_discrete_map=LANG_COLORS)
        st.plotly_chart(fig_cb_lang, use_container_width=True)

        st.subheader("📋 Cyberbullying Tweets")
        render_paginated_table(df_cb, key_prefix="cb", columns=["language", "sentiment", "model_clean", "translated_tweet"])
    else:
        st.info("No cyberbullying tweets yet.")

# ==============================
# Non-Cyberbullying Tab
# ==============================
with tabs[2]:
    df_ncb = st.session_state.df[st.session_state.df["sentiment"] == "Non Cyberbullying"].copy()
    if not df_ncb.empty:
        df_ncb["hashtags"] = df_ncb["text"].apply(extract_hashtags)

        st.subheader("📌 Non-Cyberbullying Insights")
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total NCB Tweets", len(df_ncb))
        kpi2.metric("Avg. Tweet Length", f"{df_ncb['eda_clean'].str.len().mean():.1f}")
        kpi3.metric("% of Dataset", f"{(len(df_ncb) / len(st.session_state.df)) * 100:.1f}%")

        st.subheader("🌍 NCB Distribution by Language")
        ncb_lang_dist = df_ncb["language"].value_counts().reset_index()
        ncb_lang_dist.columns = ["language", "count"]
        fig_ncb_lang = px.bar(ncb_lang_dist, x="language", y="count", color="language",
                              text="count", height=500, color_discrete_map=LANG_COLORS)
        st.plotly_chart(fig_ncb_lang, use_container_width=True)

        st.subheader("📋 Non-Cyberbullying Tweets")
        render_paginated_table(df_ncb, key_prefix="ncb", columns=["language", "sentiment", "model_clean", "translated_tweet"])
    else:
        st.info("No non-cyberbullying tweets yet.")

# ==============================
# Sidebar
# ==============================
st.sidebar.image("twitter_icon.png", use_container_width=True)
st.sidebar.header("🔍 X Cyberbullying Detection")
st.sidebar.markdown("""
**X CYBERBULLYING DETECTION**  
This application detects cyberbullying in tweets across multiple languages.  
Supports **English, Arabic, French, German, Hindi, Italian, Portuguese, and Spanish**.  
""")

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
    else:
        st.sidebar.warning("Please enter some text.")

# ---- Bulk Upload Analysis ----
st.sidebar.subheader("📤 Upload Tweets for Auto Analysis")
uploaded_file = st.sidebar.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        new_df = pd.read_csv(uploaded_file)
    else:
        new_df = pd.read_excel(uploaded_file)

    if "text" not in new_df.columns:
        st.sidebar.error("❌ File must have a 'text' column containing tweets.")
    else:
        results = []
        for _, row in new_df.iterrows():
            raw_text = str(row["text"])
            model_cleaned = clean_for_model(raw_text)
            eda_cleaned = clean_for_eda(raw_text)
            label, cb_prob = predict(model_cleaned)
            sentiment = "Cyberbullying" if label == 1 else "Non Cyberbullying"

            try:
                detected_code = detect(raw_text)
                lang = LANG_MAP.get(detected_code, detected_code)
            except:
                lang = "unknown"

            try:
                translated = GoogleTranslator(source="auto", target="en").translate(raw_text)
            except Exception:
                translated = "[translation error]"

            new_row = insert_tweet(raw_text, lang, label, sentiment,
                                   model_cleaned, eda_cleaned, translated,
                                   source_file=uploaded_file.name)
            results.append(new_row)

        if results:
            st.session_state.df = pd.concat([pd.concat(results), st.session_state.df], ignore_index=True)
            st.sidebar.success("✅ Uploaded tweets analyzed and added to dashboard!")
