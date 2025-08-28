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
import io   # ✅ moved here (not duplicated later)

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
# Helper: Extract hashtags
# ==============================
def extract_hashtags(text):
    if isinstance(text, str):
        return re.findall(r"#\w+", text)
    return []

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
    rows_per_page = 20
    total_rows = len(df)
    total_pages = (total_rows // rows_per_page) + (1 if total_rows % rows_per_page else 0)

    if "all_page" not in st.session_state:
        st.session_state.all_page = 1

    col1, col2, col3 = st.columns([4,1,1])
    with col2:
        if st.button("⬅ Prev", key="prev_all") and st.session_state.all_page > 1:
            st.session_state.all_page -= 1
    with col3:
        if st.button("Next ➡", key="next_all") and st.session_state.all_page < total_pages:
            st.session_state.all_page += 1

    start_idx = (st.session_state.all_page - 1) * rows_per_page
    end_idx = start_idx + rows_per_page

    st.dataframe(
        df[["language", "sentiment", "model_clean"]]
        .rename(columns={"model_clean": "tweet"})
        .iloc[start_idx:end_idx],
        use_container_width=True,
        height=400
    )
    st.caption(f"Page {st.session_state.all_page} of {total_pages} — showing {rows_per_page} rows per page")

# ==============================
# Cyberbullying Tab
# ==============================
with tabs[1]:
    df_cb = st.session_state.df[st.session_state.df["sentiment"] == "Cyberbullying"].copy()
    df_cb["hashtags"] = df_cb["text"].apply(extract_hashtags)

    st.subheader("📌 Cyberbullying Insights")
    kpi1, kpi2, kpi3 = st.columns(3)
    total_cb = len(df_cb)
    avg_len = df_cb["eda_clean"].str.len().mean()
    perc = (total_cb / len(st.session_state.df)) * 100
    kpi1.metric("Total CB Tweets", total_cb)
    kpi2.metric("Avg. Tweet Length", f"{avg_len:.1f}")
    kpi3.metric("% of Dataset", f"{perc:.1f}%")

    languages = ["All"] + sorted(df_cb["language"].dropna().unique())
    selected_lang = st.selectbox("🌍 Filter by Language", languages, key="cb_lang")
    if selected_lang != "All":
        df_cb = df_cb[df_cb["language"] == selected_lang]

    st.subheader("🌍 CB Distribution by Language")
    cb_lang_dist = df_cb["language"].value_counts().reset_index()
    cb_lang_dist.columns = ["language", "count"]
    fig_cb_lang = px.bar(cb_lang_dist, x="language", y="count", color="language",
                         text="count", height=500,
                         color_discrete_map=LANG_COLORS)
    st.plotly_chart(fig_cb_lang, use_container_width=True)

    hashtags = [h for tags in df_cb["hashtags"] for h in tags]
    top_hashtags = Counter(hashtags).most_common(15)
    if top_hashtags:
        st.subheader("#️⃣ Distinctive Hashtags")
        hashtags_df = pd.DataFrame(top_hashtags, columns=["hashtag", "count"])
        fig_bubble = px.scatter(hashtags_df, x="hashtag", y="count", size="count",
                                color="hashtag", hover_name="hashtag",
                                size_max=60, height=500)
        st.plotly_chart(fig_bubble, use_container_width=True)

        st.subheader("🧩 Hashtag Clustering")
        fig_cluster = px.treemap(hashtags_df, path=["hashtag"], values="count",
                                 color="count", color_continuous_scale="Viridis",
                                 height=500)
        st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        st.info("No hashtags found.")

    st.subheader("📋 Cyberbullying Tweets")
    rows_per_page = 20
    total_rows = len(df_cb)
    total_pages = (total_rows // rows_per_page) + (1 if total_rows % rows_per_page else 0)

    if "cb_page" not in st.session_state:
        st.session_state.cb_page = 1

    col1, col2, col3 = st.columns([4,1,1])
    with col2:
        if st.button("⬅ Prev", key="prev_cb") and st.session_state.cb_page > 1:
            st.session_state.cb_page -= 1
    with col3:
        if st.button("Next ➡", key="next_cb") and st.session_state.cb_page < total_pages:
            st.session_state.cb_page += 1

    start_idx = (st.session_state.cb_page - 1) * rows_per_page
    end_idx = start_idx + rows_per_page

    st.dataframe(
        df_cb[["language", "sentiment", "model_clean"]]
        .rename(columns={"model_clean": "tweet"})
        .iloc[start_idx:end_idx],
        use_container_width=True,
        height=400
    )
    st.caption(f"Page {st.session_state.cb_page} of {total_pages} — showing {rows_per_page} rows per page")

    export_df = df_cb[["id", "language", "binary_label", "sentiment", "model_clean"]].rename(
        columns={"model_clean": "tweet"}
    )
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Cyberbullying")
    st.download_button(
        "⬇ Download Cyberbullying Report (Excel)",
        data=output.getvalue(),
        file_name="cyberbullying_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ==============================
# Non-Cyberbullying Tab
# ==============================
with tabs[2]:
    df_ncb = st.session_state.df[st.session_state.df["sentiment"] == "Non Cyberbullying"].copy()
    df_ncb["hashtags"] = df_ncb["text"].apply(extract_hashtags)

    st.subheader("📌 Non-Cyberbullying Insights")
    kpi1, kpi2, kpi3 = st.columns(3)
    total_ncb = len(df_ncb)
    avg_len = df_ncb["eda_clean"].str.len().mean()
    perc = (total_ncb / len(st.session_state.df)) * 100
    kpi1.metric("Total NCB Tweets", total_ncb)
    kpi2.metric("Avg. Tweet Length", f"{avg_len:.1f}")
    kpi3.metric("% of Dataset", f"{perc:.1f}%")

    languages = ["All"] + sorted(df_ncb["language"].dropna().unique())
    selected_lang = st.selectbox("🌍 Filter by Language", languages, key="ncb_lang")
    if selected_lang != "All":
        df_ncb = df_ncb[df_ncb["language"] == selected_lang]

    st.subheader("🌍 NCB Distribution by Language")
    ncb_lang_dist = df_ncb["language"].value_counts().reset_index()
    ncb_lang_dist.columns = ["language", "count"]
    fig_ncb_lang = px.bar(ncb_lang_dist, x="language", y="count", color="language",
                          text="count", height=500,
                          color_discrete_map=LANG_COLORS)
    st.plotly_chart(fig_ncb_lang, use_container_width=True)

    hashtags = [h for tags in df_ncb["hashtags"] for h in tags]
    top_hashtags = Counter(hashtags).most_common(15)
    if top_hashtags:
        st.subheader("#️⃣ Distinctive Hashtags")
        hashtags_df = pd.DataFrame(top_hashtags, columns=["hashtag", "count"])
        fig_bubble = px.scatter(hashtags_df, x="hashtag", y="count", size="count",
                                color="hashtag", hover_name="hashtag",
                                size_max=60, height=500)
        st.plotly_chart(fig_bubble, use_container_width=True)

        st.subheader("🧩 Hashtag Clustering")
        fig_cluster = px.treemap(hashtags_df, path=["hashtag"], values="count",
                                 color="count", color_continuous_scale="Viridis",
                                 height=500)
        st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        st.info("No hashtags found.")

    st.subheader("📋 Non-Cyberbullying Tweets")
    rows_per_page = 20
    total_rows = len(df_ncb)
    total_pages = (total_rows // rows_per_page) + (1 if total_rows % rows_per_page else 0)

    if "ncb_page" not in st.session_state:
        st.session_state.ncb_page = 1

    col1, col2, col3 = st.columns([4,1,1])
    with col2:
        if st.button("⬅ Prev", key="prev_ncb") and st.session_state.ncb_page > 1:
            st.session_state.ncb_page -= 1
    with col3:
        if st.button("Next ➡", key="next_ncb") and st.session_state.ncb_page < total_pages:
            st.session_state.ncb_page += 1

    start_idx = (st.session_state.ncb_page - 1) * rows_per_page
    end_idx = start_idx + rows_per_page

    st.dataframe(
        df_ncb[["language", "sentiment", "model_clean"]]
        .rename(columns={"model_clean": "tweet"})
        .iloc[start_idx:end_idx],
        use_container_width=True,
        height=400
    )
    st.caption(f"Page {st.session_state.ncb_page} of {total_pages} — showing {rows_per_page} rows per page")

    export_df = df_ncb[["id", "language", "binary_label", "sentiment", "model_clean"]].rename(
        columns={"model_clean": "tweet"}
    )
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Non-Cyberbullying")
    st.download_button(
        "⬇ Download Non-Cyberbullying Report (Excel)",
        data=output.getvalue(),
        file_name="non_cyberbullying_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

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
