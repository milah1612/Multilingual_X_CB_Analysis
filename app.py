import streamlit as st
import pandas as pd
import re, html, io, sqlite3, os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import plotly.express as px 
from collections import Counter 
from deep_translator import GoogleTranslator
from langdetect import detect_langs

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
# DB Setup
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

def migrate_csv_to_sqlite():
    """Seed DB from CSV only if DB is empty"""
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
        if "binary_label" not in df.columns and "sentiment" in df.columns:
            df["binary_label"] = df["sentiment"].apply(lambda x: 1 if "Cyberbullying" in str(x) else 0)
        df["source_file"] = "initial_csv"
        conn = sqlite3.connect(DB_FILE)
        df.to_sql("tweets", conn, if_exists="append", index=False)
        conn.close()
        print("✅ Migrated CSV into SQLite (first time only)")

def load_tweets():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM tweets ORDER BY timestamp DESC", conn)
    conn.close()
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
        "source_file": source_file
    }])

def delete_rows_by_ids(ids):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.executemany("DELETE FROM tweets WHERE id = ?", [(i,) for i in ids])
    conn.commit()
    conn.close()

def delete_rows_by_source(source_file):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tweets WHERE source_file = ?", (source_file,))
    conn.commit()
    conn.close()

# ==============================
# Translation Helpers
# ==============================
def is_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', str(text)))

def safe_translate(text, lang_code, row_id=None, context="general"):
    try:
        if lang_code == "ar":
            translated = GoogleTranslator(source="ar", target="en").translate(text)
        else:
            translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated
    except:
        return "[translation error]"

def backfill_missing_arabic_translations():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT id, text, language, translated_tweet FROM tweets", conn)
    updates = []
    for _, row in df.iterrows():
        if row["language"] == "arabic":
            if (
                pd.isna(row["translated_tweet"])
                or str(row["translated_tweet"]).strip() in ["", "[not translated]"]
                or is_arabic(row["translated_tweet"])
            ):
                translated = safe_translate(row["text"], "ar", row_id=row["id"], context="backfill")
                updates.append((translated, row["id"]))
    if updates:
        cursor = conn.cursor()
        cursor.executemany("UPDATE tweets SET translated_tweet=? WHERE id=?", updates)
        conn.commit()
    conn.close()

# ==============================
# Language Detection
# ==============================
from langdetect import detect_langs

def detect_language(text):
    try:
        langs = detect_langs(text)
        # langs looks like: [hi:0.55, en:0.45]
        top = max(langs, key=lambda x: x.prob)
        # If Hindi/Arabic/etc appears with decent probability, override English
        for lang in langs:
            if lang.lang in ["hi", "ar", "es", "fr", "de", "it", "pt"] and lang.prob > 0.25:
                return lang.lang
        return top.lang
    except:
        return "unknown"

# ==============================
# Init + Seed
# ==============================
init_db()
migrate_csv_to_sqlite()
if "df" not in st.session_state:
    st.session_state.df = load_tweets()
backfill_missing_arabic_translations()
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
    langs_available = sorted([l for l in df["language"].unique() if l != "unknown"])
    options = ["All"] + langs_available
    choice = st.selectbox("Filter by language", options=options, key=key)
    if choice == "All":
        return df
    else:
        return df[df["language"] == choice]

def render_paginated_table(df, key_prefix, columns=None, rows_per_page=20, selectable=False):
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
# All Tab
# ==============================
with tabs[0]:
    st.subheader("📊 Overall Insights")
    df = language_filter_ui(st.session_state.df, key="all_filter")
    col1, col2 = st.columns([1, 1.2])
    with col1:
        sentiment_counts = df["sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["sentiment", "count"]
        fig_pie = px.pie(sentiment_counts, values="count", names="sentiment", color="sentiment",
                         height=500, color_discrete_map={"Cyberbullying": "#FF6F61", "Non Cyberbullying": "#4C9AFF"})
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        lang_dist = df.groupby(["language", "sentiment"]).size().reset_index(name="count")
        fig_bar = px.bar(lang_dist, x="language", y="count", color="sentiment", barmode="group",
                         text="count", height=500,
                         color_discrete_map={"Cyberbullying": "#FF6F61", "Non Cyberbullying": "#4C9AFF"})
        st.plotly_chart(fig_bar, use_container_width=True)
    st.subheader("📝 All Tweets")
    render_paginated_table(df, key_prefix="all", columns=["language", "sentiment", "model_clean", "translated_tweet"])

# ==============================
# Cyberbullying Tab
# ==============================
with tabs[1]:
    st.subheader("📌 Cyberbullying Insights")
    df_cb = st.session_state.df[st.session_state.df["sentiment"] == "Cyberbullying"].copy()
    df_cb = language_filter_ui(df_cb, key="cb_filter")

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total CB Tweets", len(df_cb))
    if not df_cb.empty:
        kpi2.metric("Avg. Tweet Length", f"{df_cb['eda_clean'].str.len().mean():.1f}")
    kpi3.metric("% of Dataset", f"{(len(df_cb) / len(st.session_state.df)) * 100:.1f}%")

    if not df_cb.empty:
        cb_lang_dist = df_cb["language"].value_counts().reset_index()
        cb_lang_dist.columns = ["language", "count"]
        fig_cb_lang = px.bar(
            cb_lang_dist, x="language", y="count", color="language",
            text="count", height=500, color_discrete_map=LANG_COLORS
        )
        st.plotly_chart(fig_cb_lang, use_container_width=True)

        # --- Hashtag Analysis ---
        df_cb["hashtags"] = df_cb["text"].apply(lambda x: re.findall(r"#\w+", str(x)))
        hashtags = [h for tags in df_cb["hashtags"] for h in tags]
        top_hashtags = Counter(hashtags).most_common(15)

        if top_hashtags:
            st.subheader("#️⃣ Distinctive Hashtags")
            hashtags_df = pd.DataFrame(top_hashtags, columns=["hashtag", "count"])

            # Bubble chart
            fig_bubble = px.scatter(
                hashtags_df, x="hashtag", y="count", size="count", color="hashtag",
                hover_name="hashtag", size_max=60, height=500
            )
            st.plotly_chart(fig_bubble, use_container_width=True)

            # Treemap
            st.subheader("🧩 Hashtag Clustering")
            fig_cluster = px.treemap(
                hashtags_df, path=["hashtag"], values="count", color="count",
                color_continuous_scale="Viridis", height=500
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

    st.subheader("📋 Cyberbullying Tweets")
    render_paginated_table(df_cb, key_prefix="cb",
                           columns=["language", "sentiment", "model_clean", "translated_tweet"])

# ==============================
# Non-Cyberbullying Tab
# ==============================
with tabs[2]:
    st.subheader("📌 Non-Cyberbullying Insights")
    df_ncb = st.session_state.df[st.session_state.df["sentiment"] == "Non Cyberbullying"].copy()
    df_ncb = language_filter_ui(df_ncb, key="ncb_filter")

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total NCB Tweets", len(df_ncb))
    if not df_ncb.empty:
        kpi2.metric("Avg. Tweet Length", f"{df_ncb['eda_clean'].str.len().mean():.1f}")
    kpi3.metric("% of Dataset", f"{(len(df_ncb) / len(st.session_state.df)) * 100:.1f}%")

    if not df_ncb.empty:
        ncb_lang_dist = df_ncb["language"].value_counts().reset_index()
        ncb_lang_dist.columns = ["language", "count"]
        fig_ncb_lang = px.bar(
            ncb_lang_dist, x="language", y="count", color="language",
            text="count", height=500, color_discrete_map=LANG_COLORS
        )
        st.plotly_chart(fig_ncb_lang, use_container_width=True)

        # --- Hashtag Analysis ---
        df_ncb["hashtags"] = df_ncb["text"].apply(lambda x: re.findall(r"#\w+", str(x)))
        hashtags = [h for tags in df_ncb["hashtags"] for h in tags]
        top_hashtags = Counter(hashtags).most_common(15)

        if top_hashtags:
            st.subheader("#️⃣ Distinctive Hashtags")
            hashtags_df = pd.DataFrame(top_hashtags, columns=["hashtag", "count"])

            # Bubble chart
            fig_bubble = px.scatter(
                hashtags_df, x="hashtag", y="count", size="count", color="hashtag",
                hover_name="hashtag", size_max=60, height=500
            )
            st.plotly_chart(fig_bubble, use_container_width=True)

            # Treemap
            st.subheader("🧩 Hashtag Clustering")
            fig_cluster = px.treemap(
                hashtags_df, path=["hashtag"], values="count", color="count",
                color_continuous_scale="Viridis", height=500
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

    st.subheader("📋 Non-Cyberbullying Tweets")
    render_paginated_table(df_ncb, key_prefix="ncb",
                           columns=["language", "sentiment", "model_clean", "translated_tweet"])

# ==============================
# Tools Tab
# ==============================
with tabs[3]:
    st.subheader("🛠️ Tools")
    tool_choice = st.radio("Choose Tool:", ["Download Data", "Upload Data", "Delete Data"])
    df_all = st.session_state.df.copy()

    # --- Download ---
    if tool_choice == "Download Data":
        sentiments = ["All"] + sorted(df_all["sentiment"].unique())
        sentiment_choice = st.selectbox("Filter by Sentiment", options=sentiments, index=0)
        langs_available = sorted([l for l in df_all["language"].unique() if l != "unknown"])
        lang_options = ["All"] + langs_available
        lang_choice = st.selectbox("Filter by Language", options=lang_options, index=0)
        df_filtered = df_all.copy()
        if sentiment_choice != "All":
            df_filtered = df_filtered[df_filtered["sentiment"] == sentiment_choice]
        if lang_choice != "All":
            df_filtered = df_filtered[df_filtered["language"] == lang_choice]
        if "timestamp" in df_filtered.columns:
            df_filtered["timestamp"] = pd.to_datetime(df_filtered["timestamp"], errors="coerce")
            if not df_filtered["timestamp"].isna().all():
                min_date, max_date = df_filtered["timestamp"].min(), df_filtered["timestamp"].max()
                date_range = st.date_input("Select Date Range",
                                           value=(min_date.date(), max_date.date()),
                                           min_value=min_date.date(), max_value=max_date.date())
                df_filtered = df_filtered[
                    (df_filtered["timestamp"].dt.date >= date_range[0]) &
                    (df_filtered["timestamp"].dt.date <= date_range[1])
                ]
        base_cols = ["language", "sentiment", "text", "translated_tweet"]
        include_timestamp = st.checkbox("Include Timestamp", value=True)
        selected_cols = base_cols + (["timestamp"] if include_timestamp else [])
        if not df_filtered.empty and selected_cols:
            df_out = df_filtered[selected_cols]
            st.write("📊 Preview", df_out.head(10))
            csv_buffer = io.StringIO()
            df_out.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
            st.download_button("⬇️ Download CSV", data=csv_buffer.getvalue(),
                               file_name="tweets_filtered.csv", mime="text/csv")
        else:
            st.info("⚠️ No data matches your filter.")

    # --- Upload ---
    elif tool_choice == "Upload Data":
        st.write("📤 Upload CSV/XLSX (must contain a 'text' column)")
        uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                new_df = pd.read_csv(uploaded_file)
            else:
                new_df = pd.read_excel(uploaded_file)

            if "text" not in new_df.columns:
                st.error("❌ File must contain 'text' column")
            else:
                results = []
                for _, row in new_df.iterrows():
                    raw_text = str(row["text"]).strip()
                    if not raw_text:
                        continue
                    model_cleaned = clean_for_model(raw_text)
                    eda_cleaned = clean_for_eda(raw_text)

                    # Predict sentiment
                    if "binary_label" in new_df.columns and not pd.isna(row.get("binary_label", None)):
                        label = int(row["binary_label"])
                        sentiment = "Cyberbullying" if label == 1 else "Non Cyberbullying"
                    else:
                        label, _ = predict(model_cleaned)
                        sentiment = "Cyberbullying" if label == 1 else "Non Cyberbullying"

                    # Language detection + mapping
                    lang_code = detect_language(raw_text)
                    lang = LANG_MAP.get(lang_code, "unknown")

                    # Translation
                    translated = safe_translate(raw_text, lang_code, context="upload")

                    # Insert into DB
                    new_row = insert_tweet(raw_text, lang, label, sentiment,
                                           model_cleaned, eda_cleaned, translated,
                                           source_file=f"upload:{uploaded_file.name}")
                    results.append(new_row)

                if results:
                    new_data = pd.concat(results, ignore_index=True)
                    st.session_state.df = pd.concat([new_data, st.session_state.df], ignore_index=True)

                    # ✅ Show preview immediately
                    st.success("✅ Uploaded data analyzed and saved!")
                    st.write("📊 Preview of Uploaded Data")
                    st.dataframe(new_data[["language", "sentiment", "text", "translated_tweet"]].head(10))


    # --- Delete ---
elif tool_choice == "Delete Data":
    st.write("🗑 Delete tweets from DB")
    sentiments = ["All"] + sorted(df_all["sentiment"].unique())
    sentiment_choice = st.selectbox("Filter by Sentiment", options=sentiments, index=0, key="del_sent")
    langs_available = sorted([l for l in df_all["language"].unique() if l != "unknown"])
    lang_options = ["All"] + langs_available
    lang_choice = st.selectbox("Filter by Language", options=lang_options, index=0, key="del_lang")

    df_filtered = df_all.copy()
    if sentiment_choice != "All":
        df_filtered = df_filtered[df_filtered["sentiment"] == sentiment_choice]
    if lang_choice != "All":
        df_filtered = df_filtered[df_filtered["language"] == lang_choice]

    st.write("📊 Preview of Data (with ID + Source)")
    st.dataframe(df_filtered[["id", "source_file", "language", "sentiment", "text", "translated_tweet"]].head(20))

    ids_to_delete = st.multiselect("Select rows by ID to delete", df_filtered["id"].tolist())
    if st.button("Delete Selected Rows") and ids_to_delete:
        delete_rows_by_ids(ids_to_delete)
        st.session_state.df = load_tweets()
        st.success(f"✅ Deleted {len(ids_to_delete)} rows.")
        st.rerun()   # ✅ force refresh after delete

    sources = df_all["source_file"].dropna().unique().tolist()
    if sources:
        source_choice = st.selectbox("Delete by Source File", ["None"] + sources, key="del_source")
        if source_choice != "None" and st.button("Delete All from Source"):
            delete_rows_by_source(source_choice)
            st.session_state.df = load_tweets()
            st.success(f"✅ Deleted all rows from source: {source_choice}")
            st.rerun()   # ✅ force refresh after delete



# ==============================
# Sidebar - Single Tweet Analysis
# ==============================
st.sidebar.image("twitter_icon.png", width="stretch")
st.sidebar.header("🔍 X Cyberbullying Detection")
st.sidebar.markdown("""  
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

        # ✅ new language detection
        lang_code = detect_language(tweet_input)
        lang = LANG_MAP.get(lang_code, lang_code)
        translated = safe_translate(tweet_input, lang_code, context="sidebar")

        new_row = insert_tweet(tweet_input, lang, label, sentiment,
                               model_cleaned, eda_cleaned, translated, source_file="manual")
        st.session_state.df = pd.concat([new_row, st.session_state.df], ignore_index=True)
        st.session_state.analysis_result = {
            "sentiment": sentiment,
            "lang": lang,
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
