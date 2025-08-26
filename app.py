import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# Load Data
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("tweet_data.csv", encoding="utf-8-sig")

    # Map binary labels to readable sentiment
    df["Sentiment"] = df["binary_label"].map({0: "Not Cyberbullying", 1: "Cyberbullying"})
    return df

df = load_data()

# ===============================
# Dashboard Title
# ===============================
st.set_page_config(page_title="Cyberbullying Dashboard", layout="wide")
st.title("🚨 Cyberbullying Detection Dashboard (Overview)")
st.markdown("This dashboard shows an overview of the cleaned dataset before prediction.")

# ===============================
# Layout
# ===============================
col1, col2 = st.columns(2)

# --- Pie Chart: Sentiment Distribution
with col1:
    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = df["Sentiment"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
           startangle=90, colors=["#1f77b4", "#ff7f0e"])
    ax.axis("equal")
    st.pyplot(fig)

# --- Bar Chart: Language Distribution
with col2:
    st.subheader("🌍 Language Distribution")
    lang_counts = df["language"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots()
    lang_counts.plot(kind="bar", color="skyblue", ax=ax)
    plt.ylabel("Count")
    st.pyplot(fig)

# ===============================
# Table View
# ===============================
st.subheader("📝 Sentiment and Processed Tweets")
st.dataframe(
    df[["Sentiment", "eda_clean"]].rename(columns={"eda_clean": "Processed Text"}).head(100)
)
