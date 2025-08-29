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
import io

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

# ==============================
# Cyberbullying Tab
# ==============================
with tabs[1]:
    df_cb = st.session_state.df[st.session_state.df["sentiment"] == "Cyberbullying"].copy()
    df_cb["hashtags"] = df_cb["text"].apply(extract_hashtags)

    st.subheader("📌 Cyberbullying Insights")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total CB Tweets", len(df_cb))
    kpi2.metric("Avg. Tweet Length", f"{df_cb['eda_clean'].str.len().mean():.1f}")
    kpi3.metric("% of Dataset", f"{(len(df_cb) / len(st.session_state.df)) * 100:.1f}%")

    # ✅ Distribution by language
    st.subheader("🌍 CB Distribution by Language")
    if not df_cb.empty:
        cb_lang_dist = df_cb["language"].value_counts().reset_index()
        cb_lang_dist.columns = ["language", "count"]
        fig_cb_lang = px.bar(cb_lang_dist, x="language", y="count", color="language",
                             text="count", height=500, color_discrete_map=LANG_COLORS)
        st.plotly_chart(fig_cb_lang, use_container_width=True)

    # ✅ Hashtag analysis
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
                                 color="count", color_continuous_scale="Viridis", height=500)
        st.plotly_chart(fig_cluster, use_container_width=True)

    st.subheader("📋 Cyberbullying Tweets")
    render_paginated_table(df_cb, key_prefix="cb",
                           columns=["language", "sentiment", "model_clean", "translated_tweet"])

# ==============================
# Non-Cyberbullying Tab
# ==============================
with tabs[2]:
    df_ncb = st.session_state.df[st.session_state.df["sentiment"] == "Non Cyberbullying"].copy()
    df_ncb["hashtags"] = df_ncb["text"].apply(extract_hashtags)

    st.subheader("📌 Non-Cyberbullying Insights")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total NCB Tweets", len(df_ncb))
    kpi2.metric("Avg. Tweet Length", f"{df_ncb['eda_clean'].str.len().mean():.1f}")
    kpi3.metric("% of Dataset", f"{(len(df_ncb) / len(st.session_state.df)) * 100:.1f}%")

    # ✅ Distribution by language
    st.subheader("🌍 NCB Distribution by Language")
    if not df_ncb.empty:
        ncb_lang_dist = df_ncb["language"].value_counts().reset_index()
        ncb_lang_dist.columns = ["language", "count"]
        fig_ncb_lang = px.bar(ncb_lang_dist, x="language", y="count", color="language",
                              text="count", height=500, color_discrete_map=LANG_COLORS)
        st.plotly_chart(fig_ncb_lang, use_container_width=True)

    # ✅ Hashtag analysis
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
                                 color="count", color_continuous_scale="Viridis", height=500)
        st.plotly_chart(fig_cluster, use_container_width=True)

    st.subheader("📋 Non-Cyberbullying Tweets")
    render_paginated_table(df_ncb, key_prefix="ncb",
                           columns=["language", "sentiment", "model_clean", "translated_tweet"])   


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

        # ✅ Insert new row
        new_row = insert_tweet(tweet_input, lang, label, sentiment, model_cleaned, eda_cleaned, translated)
        st.session_state.df = pd.concat([new_row, st.session_state.df], ignore_index=True)

        # ✅ Store results in session state so they persist after rerun
        st.session_state.analysis_result = {
            "sentiment": sentiment,
            "lang": lang,
            "translated": translated
        }

        # ✅ Rerun so charts + tables refresh
        st.rerun()

    else:
        st.sidebar.warning("Please enter some text.")

# ==============================
# Show analysis result if available
# ==============================
if "analysis_result" in st.session_state:
    result = st.session_state.analysis_result
    st.sidebar.success(f"✅ Prediction: {result['sentiment']}")
    st.sidebar.write(f"🌍 Language: {result['lang']}")
    st.sidebar.write(f"🌐 Translated: {result['translated']}")


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
                                   model_cleaned, eda_cleaned, translated)
            results.append(new_row)

        if results:
            # ✅ Merge uploaded results into dashboard data
            st.session_state.df = pd.concat([pd.concat(results), st.session_state.df],
                                            ignore_index=True)

            # ✅ Store a flag to show success once
            st.session_state.upload_success = True
            st.rerun()

# ✅ Show upload success message (persists after rerun)
if "upload_success" in st.session_state and st.session_state.upload_success:
    st.sidebar.success("✅ Uploaded tweets analyzed and added to dashboard!")
    st.session_state.upload_success = False


# ==============================
# Admin / DB Maintenance (optional)
# ==============================
with st.expander("🛠️ DB Maintenance (Admin)"):
    import sqlite3, os
    from datetime import datetime

    db_path = os.path.abspath(DB_FILE)
    st.write(f"**DB path:** `{db_path}`")
    try:
        size = os.path.getsize(DB_FILE)
        st.write(f"**DB size:** {size/1024:.1f} KB")
    except Exception:
        st.warning("DB file not found in current working directory.")

    # Show basic stats
    try:
        with sqlite3.connect(DB_FILE) as conn:
            total = pd.read_sql("SELECT COUNT(*) AS n FROM tweets", conn).iloc[0,0]
            st.write(f"**Total rows:** {total}")

            # Preview
            st.write("### Preview (most recent first)")
            preview_n = st.number_input("Rows to preview", 5, 1000, 20, key="admin_preview_n")
            preview_df = pd.read_sql(
                f"""
                SELECT id, timestamp, language, sentiment, model_clean, translated_tweet
                FROM tweets
                ORDER BY timestamp DESC
                LIMIT {int(preview_n)}
                """, conn
            )
            st.dataframe(preview_df, use_container_width=True, height=300)

            # Backup to CSV
            if st.button("⬇️ Backup entire table to CSV"):
                dump = pd.read_sql("SELECT * FROM tweets ORDER BY timestamp DESC", conn)
                csv_bytes = dump.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    "Download tweets_backup.csv",
                    data=csv_bytes,
                    file_name="tweets_backup.csv",
                    mime="text/csv"
                )

            st.markdown("---")

            # Remove duplicates (keep latest by timestamp)
            st.write("### Remove duplicate rows")
            st.caption("Duplicates are considered rows with the same (text, language, sentiment). Keeps the newest timestamp.")
            if st.button("🧹 De-duplicate"):
                # Uses a window function; SQLite ≥3.25 supports this on Streamlit Cloud.
                conn.execute("""
                    DELETE FROM tweets
                    WHERE id IN (
                      SELECT id FROM (
                        SELECT
                          id,
                          ROW_NUMBER() OVER (
                            PARTITION BY text, language, sentiment
                            ORDER BY datetime(timestamp) DESC
                          ) AS rn
                        FROM tweets
                      )
                      WHERE rn > 1
                    );
                """)
                conn.commit()
                new_total = pd.read_sql("SELECT COUNT(*) AS n FROM tweets", conn).iloc[0,0]
                st.success(f"De-duplication done. New total rows: {new_total}")

            st.markdown("---")

            # Delete by date range
            st.write("### Delete by date range")
            col_from, col_to = st.columns(2)
            with col_from:
                date_from = st.date_input("From (inclusive)")
            with col_to:
                date_to = st.date_input("To (inclusive)")

            if st.button("🗑️ Delete rows in date range"):
                # Convert to ISO constraints; timestamp stored as ISO text in your app
                ts_from = datetime.combine(date_from, datetime.min.time()).isoformat()
                ts_to   = datetime.combine(date_to,   datetime.max.time()).isoformat()

                # Confirm count first
                count_df = pd.read_sql(
                    "SELECT COUNT(*) AS n FROM tweets WHERE timestamp BETWEEN ? AND ?",
                    conn, params=(ts_from, ts_to)
                )
                to_delete = int(count_df.iloc[0,0])
                if to_delete == 0:
                    st.info("No rows in that range.")
                else:
                    conn.execute("DELETE FROM tweets WHERE timestamp BETWEEN ? AND ?", (ts_from, ts_to))
                    conn.commit()
                    st.success(f"Deleted {to_delete} rows between {ts_from} and {ts_to}.")

            st.markdown("---")

            # Delete by sentiment or language (quick filters)
            st.write("### Quick delete by filters")
            langs = ["(Any)"] + sorted([r[0] for r in conn.execute("SELECT DISTINCT language FROM tweets").fetchall() if r[0]])
            sentiments = ["(Any)"] + sorted([r[0] for r in conn.execute("SELECT DISTINCT sentiment FROM tweets").fetchall() if r[0]])
            c1, c2 = st.columns(2)
            with c1:
                sel_lang = st.selectbox("Language filter", langs)
            with c2:
                sel_sent = st.selectbox("Sentiment filter", sentiments)

            if st.button("🗑️ Delete by filters"):
                where = []
                params = []
                if sel_lang != "(Any)":
                    where.append("language = ?")
                    params.append(sel_lang)
                if sel_sent != "(Any)":
                    where.append("sentiment = ?")
                    params.append(sel_sent)
                if where:
                    where_sql = " WHERE " + " AND ".join(where)
                else:
                    where_sql = ""

                count_df = pd.read_sql(f"SELECT COUNT(*) AS n FROM tweets{where_sql}", conn, params=params)
                to_delete = int(count_df.iloc[0,0])
                if to_delete == 0:
                    st.info("No matching rows.")
                else:
                    conn.execute(f"DELETE FROM tweets{where_sql}", params)
                    conn.commit()
                    st.success(f"Deleted {to_delete} rows matched by filters.")

            st.markdown("---")

            # Truncate all rows (danger)
            st.write("### Danger zone")
            if st.checkbox("I understand this will delete ALL rows permanently."):
                if st.button("🧨 Delete ALL rows"):
                    conn.execute("DELETE FROM tweets")
                    conn.commit()
                    st.success("All rows deleted.")

            # VACUUM to reclaim space
            if st.button("🧰 VACUUM (reclaim file space)"):
                conn.execute("VACUUM")
                st.success("VACUUM completed.")

    except Exception as e:
        st.error(f"DB maintenance error: {e}")

