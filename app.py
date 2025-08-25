import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.title("🚨 Debugging Model Load")

MODEL_PATH = "Mila1612/mdeberta-cyberbullying"

try:
    st.write("⏳ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    st.success("✅ Tokenizer loaded")

    st.write("⏳ Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Model load failed: {e}")
