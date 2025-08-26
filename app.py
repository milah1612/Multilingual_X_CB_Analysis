import streamlit as st
import pandas as pd
import re, html
import sqlite3, os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import plotly.express as px
from langdetect import detect
from googletrans import Translator

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
    """Migrate CSV into SQLite if DB empty"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM tweets")
    count = cursor.fetchone()[0]
    conn.close()

    if count == 0 and os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        df["timestamp"] = datetime.now().isoformat()
        df["translated_tweet"] = "[not translated]"  # default
        conn = sqlite3.connect(DB_FILE)
        df.to_sql("tweets", conn, if_exists="append", index=False)
        conn.close()
        print("✅ Migrated CSV into SQLite")

def load_tweets():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM tweets ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def insert_tweet(text, language, binary_label, sentiment, model_clean, eda_clean, translated_tweet):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO tweets (text, language, binary_label, sentiment, model_clean, eda_clean, translated_tweet, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (text, language, binary_label, sentiment, model_clean, eda_clean, translated_tweet, datetime.now().isoformat()))
    conn.commit()
    conn.close()

# Init DB and migrate if needed
init_db()
migrate_csv_to_sqlite()
df = load_tweets()

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
# Translator
# ==============================
translator = Translator()

def detect_and_translate(text):
    try:
        lang = detect(text)
    except:
        lang = "unknown"
    try:
        translated = translator.translate(text, dest="en").text
    except:
        translated = "[translation error]"
    return lang, translated

# ==============================
# Dashboard Layout
# ==============================
st.set_page_config(page_title="Cyberbullying Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center;'>🚨 Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)

# ---- Charts Section
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment", "count"]

    fig_pie = px.pie(
        sentiment_counts,
        values="count",
        names="sentiment",
        color="sentiment",
        height=500,
        color_discrete_map={"Cyberbullying": "#FF6F61", "Not Cyberbullying": "#4C9AFF"}
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("🌍 Language Distribution by Sentiment")
    lang_dist = df.groupby(["language", "sentiment"]).size().reset_index(name="count")

    fig_bar = px.bar(
        lang_dist,
        x="language",
        y="count",
        color="sentiment",
        barmode="group",
        text="count",
        height=500,
        color_discrete_map={"Cyberbullying": "#FF6F61", "Not Cyberbullying": "#4C9AFF"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ==============================
# Tweets Table
# ==============================
st.subheader("📝 Sentiment and Processed Tweets")
df_display = df.rename(columns={"eda_clean": "tweet"})

# Filters
languages = ["All"] + sorted(df_display["language"].dropna().unique())
selected_lang = st.selectbox("🌍 Filter by Language", languages)
rows_to_show = st.slider("📊 Number of rows to display", 10, 100, 20)

filtered_df = df_display.copy()
if selected_lang != "All":
    filtered_df = filtered_df[filtered_df["language"] == selected_lang]

page_size = rows_to_show
total_pages = (len(filtered_df) // page_size) + 1
page = st.number_input("📑 Page", min_value=1, max_value=total_pages, step=1)

start_idx = (page - 1) * page_size
end_idx = start_idx + page_size

st.dataframe(filtered_df[["language", "sentiment", "tweet", "translated_tweet"]].iloc[start_idx:end_idx],
             use_container_width=True, height=400)

st.caption(f"Showing {start_idx+1}–{min(end_idx, len(filtered_df))} of {len(filtered_df)} tweets")

# ==============================
# Sidebar: Tweet Search
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

        # Predict
        label, cb_prob = predict(model_cleaned)
        sentiment = "Cyberbullying" if label == 1 else "Not Cyberbullying"

        # Detect + Translate
        lang, translated = detect_and_translate(tweet_input)

        # Save to DB
        insert_tweet(tweet_input, lang, label, sentiment, model_cleaned, eda_cleaned, translated)

        # Reload dataframe
        df = load_tweets()
        df_display = df.rename(columns={"eda_clean": "tweet"})

        st.sidebar.success(f"✅ Prediction: {sentiment}")
        st.sidebar.write(f"🌍 Detected Language: {lang}")
        st.sidebar.write(f"🌐 Translated: {translated}")
    else:
        st.sidebar.warning("Please enter some text.")
