import streamlit as st
import pandas as pd
import re, html
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt
import seaborn as sns  
import plotly.express as px


# ==============================
# Load Dataset
# ==============================
@st.cache_data
def load_data():
    return pd.read_csv("tweet_data.csv")

df = load_data()

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
def predict(text, threshold=0.35):  # default threshold = 0.35
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        cb_prob = probs[0][1].item()  # probability of Cyberbullying
        pred = 1 if cb_prob > threshold else 0
    return pred, cb_prob

# ==============================
# Dashboard Layout
# ==============================
st.set_page_config(page_title="Cyberbullying Dashboard", layout="wide")

st.markdown("<h1 style='text-align: center;'>🚨 Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)

# ---- Charts Section (wider layout)
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
        height=500,  # bigger
        color_discrete_map={
            "Cyberbullying": "#FF6F61",
            "Not Cyberbullying": "#4C9AFF"
        }
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(
        legend=dict(orientation="h", y=-0.2, x=0.3),
        font=dict(size=14)
    )
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
        height=500,  # match pie chart
        color_discrete_map={
            "Cyberbullying": "#FF6F61",
            "Not Cyberbullying": "#4C9AFF"
        }
    )
    fig_bar.update_layout(
        xaxis_title="Language",
        yaxis_title="Number of Tweets",
        legend=dict(orientation="h", y=-0.3, x=0.3),
        font=dict(size=14)
    )
    st.plotly_chart(fig_bar, use_container_width=True)


st.subheader("📝 Sentiment and Processed Tweets")

# Make table wide and scrollable
st.dataframe(
    df[["language", "sentiment", "eda_clean"]].head(50),  # show more + include language
    use_container_width=True,  # full width
    height=400  # adjust height for visibility
)


# ==============================
# Sidebar: Tweet Search & Prediction
# ==============================
st.sidebar.header("🔍 X Cyberbullying Detection")

st.sidebar.markdown("""
**X CYBERBULLYING DETECTION**  
This application detects cyberbullying in tweets across multiple languages.  
It supports **English, Arabic, French, German, Hindi, Italian, Portuguese, and Spanish**.  

""")

tweet_input = st.sidebar.text_area("✍️ Enter a tweet for analysis:")

if st.sidebar.button("Analyze Tweet"):
    if tweet_input.strip():
        # Clean
        model_cleaned = clean_for_model(tweet_input)
        eda_cleaned = clean_for_eda(tweet_input)

        # Predict
        label, cb_prob = predict(model_cleaned)
        sentiment = "Cyberbullying" if label == 1 else "Not Cyberbullying"

        # (Stub translation - replace later with real translator)
        translated = f"[English Translation Placeholder] {eda_cleaned}"

        # Language detection (basic)
        lang = "unknown"
        if re.search(r"[اأإء-ي]", tweet_input): lang = "arabic"
        elif re.search(r"[àâçéèêëîïôûùüÿñæœ]", tweet_input): lang = "french"
        elif re.search(r"[а-яА-Я]", tweet_input): lang = "russian"
        else: lang = "english"

        # Append to dataframe
        new_row = {
            "text": tweet_input,
            "language": lang,
            "binary_label": label,
            "sentiment": sentiment,
            "model_clean": model_cleaned,
            "eda_clean": eda_cleaned
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        st.sidebar.success(f"✅ Prediction: {sentiment} (Confidence: {cb_prob:.2f})")
        st.sidebar.write(f"🌍 Language: {lang}")
        st.sidebar.write(f"🌐 Translated: {translated}")
    else:
        st.sidebar.warning("Please enter some text.")
