import streamlit as st
import pandas as pd
import re, html
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
        if "timestamp" not in df.columns:
            df["timestamp"] = datetime.now().isoformat()
        conn = sqlite3.connect(DB_FILE)
        df.to_sql("tweets", conn, if_exists="append", index=False)
        conn.close()
        print("✅ Migrated CSV into SQLite (first time only)")

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

    return pd.DataFrame([{
        "text": text,
        "language": language,
        "binary_label": binary_label,
        "sentiment": sentiment,
        "model_clean": model_clean,
        "eda_clean": eda_clean,
        "translated_tweet": translated_tweet,
        "timestamp": timestamp
    }])

# ==============================
# Init
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
# Cleaning & Translation
# ==============================
def clean_for_model(text):
    text = str(text).lower()
    text = re.sub(r"(https?://\S+|www\.\S+|\bhttp\b)", "", text)
    text = re
