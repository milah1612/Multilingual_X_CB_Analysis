import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ==============================
# Load Model & Tokenizer
# ==============================
MODEL_PATH = "Mila1612/mdeberta-cyberbullying"  # Hugging Face Hub repo

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model()

# ==============================
# Prediction Function
# ==============================
def predict(text):
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
        pred = torch.argmax(probs).item()
    return pred, probs[0][1].item()

# ==============================
# Streamlit UI
# ==============================
st.title("🚨 Cyberbullying Detection App")
st.markdown("Enter a tweet or text, and the model will classify it as **Cyberbullying (CB)** or **Not Cyberbullying (NCB)**.")

tweet = st.text_area("✍️ Enter a tweet:")

if st.button("Predict"):
    if tweet.strip():
        label, cb_prob = predict(tweet)
        if label == 1:
            st.error(f"🔴 Cyberbullying Detected (Confidence: {cb_prob:.2f})")
        else:
            st.success(f"✅ Not Cyberbullying (Confidence: {1 - cb_prob:.2f})")
    else:
        st.warning("Please enter some text.")
