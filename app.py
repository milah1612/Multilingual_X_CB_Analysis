import streamlit as st
import pandas as pd
import re, html
import sqlite3, os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import plotly.express as px
from langdetect import detect
from deep_translator import GoogleTranslator   

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

    # Update CSV as backup
    new_row = pd.DataFrame([{
        "text": text,
        "language": language,
        "binary_label": binary_label,
        "sentiment": sentiment,
        "model_clean": model_clean,
        "eda_clean": eda_clean,
        "translated_tweet": translated_tweet,
        "timestamp": timestamp
    }])
    if os.path.exists(CSV_FILE):
        new_row.to_csv(CSV_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        new_row.to_csv(CSV_FILE, mode="w", header=True, index=False, encoding="utf-8-sig")

    return new_row  # ✅ return new row for session update


# ==============================
# Init and Cache
# ==============================
init_db()
migrate_csv_to_sqlite()

if "df" not in st.session_state:
    st.session_state.df = load_tweets()   # ✅ cache dataset


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
# Prediction Function
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
# Sentiment Explorer with Tabs at Top
# ==============================
st.set_page_config(page_title="Cyberbullying Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center;'>🚨 Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)

tabs = st.tabs(["All", "Cyberbullying 🚨", "Non-Cyberbullying 🛡️"])

for i, sentiment_filter in enumerate(["All", "Cyberbullying", "Non Cyberbullying"]):
    with tabs[i]:
        df = st.session_state.df

        # Filter dataset
        if sentiment_filter == "All":
            filtered_df = df.copy()
        else:
            filtered_df = df[df["sentiment"] == sentiment_filter]

        # ====== KPI Metrics ======
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tweets", len(filtered_df))
        col2.metric("Avg. Length", int(filtered_df["model_clean"].str.len().mean()))
        col3.metric("% of Dataset", f"{len(filtered_df)/len(df)*100:.1f}%")

        # ====== Only show Pie + Bar in ALL ======
        if sentiment_filter == "All":
            col1, col2 = st.columns([1, 1.2])
            with col1:
                st.subheader("📊 Sentiment Distribution")
                sentiment_counts = df["sentiment"].value_counts().reset_index()
                sentiment_counts.columns = ["sentiment", "count"]
                fig_pie = px.pie(sentiment_counts, values="count", names="sentiment", color="sentiment",
                                 height=500, color_discrete_map={"Cyberbullying": "#FF6F61", "Non Cyberbullying": "#4C9AFF"})
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                st.subheader("🌍 Language Distribution by Sentiment")
                lang_dist = df.groupby(["language", "sentiment"]).size().reset_index(name="count")
                fig_bar = px.bar(lang_dist, x="language", y="count", color="sentiment", barmode="group",
                                 text="count", height=500,
                                 color_discrete_map={"Cyberbullying": "#FF6F61", "Non Cyberbullying": "#4C9AFF"})
                st.plotly_chart(fig_bar, use_container_width=True)

        # ====== Table + Download ======
        st.subheader("📝 Tweets")
        export_df = filtered_df.rename(columns={"model_clean": "tweet"})[
            ["id", "language", "binary_label", "sentiment", "tweet"]
        ]

        csv = export_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"{sentiment_filter.replace(' ','_').lower()}_tweets.csv",
            mime="text/csv",
            key=f"download_{i}"
        )

        st.dataframe(export_df, use_container_width=True, height=400)


# ==============================
# Sidebar
# ==============================
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
        except Exception as e:
            translated = f"(Translation failed: {e})"

        new_row = insert_tweet(tweet_input, lang, label, sentiment, model_cleaned, eda_cleaned, translated)
        st.session_state.df = pd.concat([new_row, st.session_state.df], ignore_index=True)

        st.sidebar.success(f"✅ Prediction: {sentiment}")
        st.sidebar.write(f"🌍 Language: {lang}")
        st.sidebar.write(f"🌐 Translated: {translated}")
    else:
        st.sidebar.warning("Please enter some text.")
