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
    "unknown": "#DD4124"
}

SUPPORTED_LANGS = set(LANG_MAP.keys()) - {"unknown"}  # ISO codes

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
            timestamp TEXT,
            source_file TEXT
        )
    """)
    conn.commit()
    conn.close()

def migrate_csv_to_sqlite(force=True):
    """Always reload CSV → SQLite (ensures data loads)"""
    if not os.path.exists(CSV_FILE):
        print("⚠️ CSV file not found")
        return

    df = pd.read_csv(CSV_FILE)
    if df.empty:
        print("⚠️ CSV is empty!")
        return

    if "translated_tweet" not in df.columns:
        df["translated_tweet"] = "[not translated]"

    if "timestamp" not in df.columns:
        df["timestamp"] = datetime.now().isoformat()

    if "binary_label" not in df.columns and "sentiment" in df.columns:
        df["binary_label"] = df["sentiment"].apply(lambda x: 1 if "Cyberbullying" in str(x) else 0)

    df["source_file"] = "initial_csv"

    conn = sqlite3.connect(DB_FILE)
    df.to_sql("tweets", conn, if_exists="replace", index=False)
    conn.close()
    print(f"✅ Migrated {len(df)} rows from CSV into SQLite")

def load_tweets():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM tweets ORDER BY timestamp DESC", conn)
    conn.close()
    if df.empty:
        print("⚠️ No tweets found in DB after load")
    else:
        print(f"📊 Loaded {len(df)} rows from DB")
    df["language_display"] = df["language"].map(LANG_MAP).fillna(df["language"])
    return df

# ==============================
# Translation Helpers
# ==============================
def is_arabic(text):
    """Check if a string contains Arabic characters."""
    return bool(re.search(r'[\u0600-\u06FF]', str(text)))

def safe_translate(text, lang_code, row_id=None, context="general"):
    try:
        if lang_code == "ar":
            translated = GoogleTranslator(source="ar", target="en").translate(text)
        else:
            translated = GoogleTranslator(source="auto", target="en").translate(text)
        print(f"✅ [{context}] Row {row_id if row_id else '-'} | {lang_code} → {translated[:60]}")
        return translated
    except Exception as e:
        print(f"⚠️ [{context}] Row {row_id if row_id else '-'} | {lang_code} | Error: {e}")
        return "[translation error]"

def backfill_arabic_translations():
    """Fix untranslated Arabic rows at startup"""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT id, text, translated_tweet FROM tweets WHERE language='ar'", conn)

    updates = []
    for _, row in df.iterrows():
        if (
            pd.isna(row["translated_tweet"])
            or str(row["translated_tweet"]).strip() in ["", "[not translated]"]
            or is_arabic(row["translated_tweet"])
        ):
            translated = safe_translate(str(row["text"]), "ar", row_id=row["id"], context="backfill-ar")
            updates.append((translated, row["id"]))

    if updates:
        cursor = conn.cursor()
        cursor.executemany("UPDATE tweets SET translated_tweet=? WHERE id=?", updates)
        conn.commit()
        print(f"✨ Backfilled {len(updates)} Arabic translations")
    else:
        print("⏭ No Arabic rows needed fix")
    conn.close()

# ==============================
# Init
# ==============================
init_db()
migrate_csv_to_sqlite(force=True)
backfill_arabic_translations()
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

# ==============================
# Sidebar
# ==============================
st.sidebar.image("twitter_icon.png", width="stretch")
st.sidebar.header("🔍 X Cyberbullying Detection")
st.sidebar.markdown("""
**X CYBERBULLYING DETECTION**  
Detects cyberbullying in tweets across multiple languages.  
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
            lang = detected_code if detected_code in LANG_MAP else "unknown"
        except:
            lang = "unknown"
        translated = safe_translate(tweet_input, lang, context="sidebar")
        new_row = pd.DataFrame([{
            "text": tweet_input,
            "language": lang,
            "binary_label": label,
            "sentiment": sentiment,
            "model_clean": model_cleaned,
            "eda_clean": eda_cleaned,
            "translated_tweet": translated,
            "timestamp": datetime.now().isoformat(),
            "source_file": "manual",
            "language_display": LANG_MAP.get(lang, lang)
        }])
        st.session_state.df = pd.concat([new_row, st.session_state.df], ignore_index=True)
        st.session_state.analysis_result = {
            "sentiment": sentiment,
            "lang": LANG_MAP.get(lang, lang),
            "translated": translated
        }
        st.rerun()
    else:
        st.sidebar.warning("Please enter some text.")

if "analysis_result" in st.session_state:
    result = st.session_state.analysis_result
    st.sidebar.success(f"✅ Prediction: {result['sentiment']}")
    st.sidebar.write(f"🌍 Language: {result['lang']}")
    st.sidebar.write(f"🌐 Translated: {result['translated']}")
