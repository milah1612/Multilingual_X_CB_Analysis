import streamlit as st
import pandas as pd
import re, html
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt
import seaborn as sns

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
st.title("🚨 Cyberbullying Detection Dashboard")

# ---- Charts Section
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = df["sentiment"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%")
    st.pyplot(fig)

with col2:
    st.subheader("🌍 Language Distribution")
    lang_counts = df["language"].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=lang_counts.index, y=lang_counts.values, ax=ax, color="skyblue")
    plt.xticks(rotation=45)
    st.pyplot(fig)

st.subheader("📝 Sentiment and Processed Tweets")
st.dataframe(df[["sentiment", "eda_clean"]].head(20))

# ==============================
# Sidebar: Tweet Search & Prediction
# ==============================
st.sidebar.header("🔍 Tweet Search & Prediction")
tweet_input = st.sidebar.text_area("Enter a tweet:")

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

        # Language detection (very simple - improve later)
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
