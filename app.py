import streamlit as st
import pandas as pd
import re, html, io, sqlite3, os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import plotly.express as px
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

LANG_COLORS = {
    "arabic": "#FF6F61",
    "french": "#6B5B95",
    "english": "#88B04B",
    "spanish": "#F7CAC9",
    "hindi": "#92A8D1",
    "german": "#955251",
    "italian": "#B565A7",
    "portuguese": "#009B77",
}

SUPPORTED_LANGS = set(LANG_MAP.keys()) - {"unknown"}  # ISO codes

# ==============================
# DB Functions
# ==============================
DB_FILE = "tweets.db"
CSV_FILE = "tweet_data.csv"

def init_db(force_reset=False):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    if force_reset:
        cursor.execute("DROP TABLE IF EXISTS tweets")
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
            timestamp TEXT,
            source_file TEXT
        )
    """)
    conn.commit()
    conn.close()

def migrate_csv_to_sqlite(force=False):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM tweets")
    count = cursor.fetchone()[0]
    conn.close()

    if force or (count == 0 and os.path.exists(CSV_FILE)):
        df = pd.read_csv(CSV_FILE)

        if "translated_tweet" not in df.columns:
            df["translated_tweet"] = "[not translated]"
        if "timestamp" not in df.columns:
            df["timestamp"] = datetime.now().isoformat()
        if "binary_label" not in df.columns and "sentiment" in df.columns:
            df["binary_label"] = df["sentiment"].apply(lambda x: 1 if "Cyberbullying" in str(x) else 0)
        if "source_file" not in df.columns:
            df["source_file"] = "initial_csv"

        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        for _, row in df.iterrows():
            cur.execute("""
                INSERT INTO tweets (text, language, binary_label, sentiment, model_clean, eda_clean, translated_tweet, timestamp, source_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row.get("text", ""),
                row.get("language", "unknown"),
                int(row.get("binary_label", 0)),
                row.get("sentiment", "Non Cyberbullying"),
                row.get("model_clean", ""),
                row.get("eda_clean", ""),
                row.get("translated_tweet", "[not translated]"),
                row.get("timestamp", datetime.now().isoformat()),
                row.get("source_file", "initial_csv"),
            ))
        conn.commit()
        conn.close()
        print("✅ Migrated CSV into SQLite (schema-safe)")

def load_tweets():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM tweets ORDER BY timestamp DESC", conn)
    conn.close()
    df["language_display"] = df["language"].map(LANG_MAP).fillna(df["language"])
    return df

def insert_tweet(text, language, binary_label, sentiment, model_clean, eda_clean, translated_tweet, source_file="manual"):
    timestamp = datetime.now().isoformat()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO tweets (text, language, binary_label, sentiment, model_clean, eda_clean, translated_tweet, timestamp, source_file)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (text, language, binary_label, sentiment, model_clean, eda_clean, translated_tweet, timestamp, source_file))
    conn.commit()
    conn.close()
    return pd.DataFrame([{
        "text": text,
        "language": language,
        "binary_label": binary_label,
        "sentiment": sentiment,
        "model_clean": model_clean,
        "eda_clean": eda_clean,
        "translated_tweet": translated_tweet,
        "timestamp": timestamp,
        "source_file": source_file,
        "language_display": LANG_MAP.get(language, language)
    }])

# ==============================
# Arabic Backfill
# ==============================
def is_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', str(text)))

def backfill_arabic():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    rows = cursor.execute("SELECT id, text, translated_tweet FROM tweets WHERE language='ar'").fetchall()

    updates = []
    for rid, raw_text, translated in rows:
        if (
            translated is None
            or translated.strip() in ["", "[not translated]", "[translation error]"]
            or is_arabic(translated)
        ):
            try:
                fixed = GoogleTranslator(source="ar", target="en").translate(raw_text)
                updates.append((fixed, rid))
                print(f"✅ Fixed Arabic row {rid}: {fixed[:50]}...")
            except Exception as e:
                print(f"⚠️ Arabic row {rid} failed: {e}")

    if updates:
        cursor.executemany("UPDATE tweets SET translated_tweet=? WHERE id=?", updates)
        conn.commit()
    conn.close()
    print(f"✨ Arabic backfill done: {len(updates)} rows updated")

# ==============================
# Init
# ==============================
init_db()
migrate_csv_to_sqlite()
if "df" not in st.session_state:
    st.session_state.df = load_tweets()
backfill_arabic()
st.session_state.df = load_tweets()

# ==============================
# Model
# ==============================
MODEL_PATH = "Mila1612/mdeberta-cyberbullying"
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model
tokenizer, model = load_model()

# ==============================
# Cleaning & Prediction
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

def predict(text, threshold=0.35):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        cb_prob = probs[0][1].item()
        pred = 1 if cb_prob > threshold else 0
    return pred, cb_prob

# ==============================
# Helpers
# ==============================
def language_filter_ui(df, key):
    langs_available = sorted([l for l in LANG_MAP.values() if l != "unknown"])
    options = ["All"] + langs_available
    choice = st.selectbox("Filter by language", options=options, key=key)
    if choice == "All":
        return df
    else:
        return df[df["language_display"] == choice]

def render_paginated_table(df, key_prefix, columns=None, rows_per_page=20):
    if columns:
        df = df[columns]
    total_rows = len(df)
    total_pages = (total_rows // rows_per_page) + (1 if total_rows % rows_per_page else 0)
    page = st.number_input("Page", min_value=1, max_value=max(total_pages, 1),
                           value=1, key=f"{key_prefix}_page")
    start_idx = (page - 1) * rows_per_page
    end_idx = start_idx + rows_per_page
    st.dataframe(df.iloc[start_idx:end_idx], width="stretch", height=400)
    st.caption(f"Page {page} of {total_pages} — showing {rows_per_page} rows per page")

# ==============================
# Dashboard Layout
# ==============================
st.set_page_config(page_title="Cyberbullying Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center;'>🚨 SENTIMENT ANALYSIS DASHBOARD</h1>", unsafe_allow_html=True)

tabs = st.tabs(["All 🌍", "Cyberbullying 🚨", "Non-Cyberbullying 🙂", "Tools 🛠️"])

# (Keep your tab logic here like before — unchanged for charts, tools, delete, etc.)
