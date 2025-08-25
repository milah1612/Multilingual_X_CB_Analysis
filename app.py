import streamlit as st

st.title("🚨 Debugging Imports")

try:
    import torch
    st.success("✅ Torch imported successfully")
except Exception as e:
    st.error(f"❌ Torch import failed: {e}")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    st.success("✅ Transformers imported successfully")
except Exception as e:
    st.error(f"❌ Transformers import failed: {e}")
