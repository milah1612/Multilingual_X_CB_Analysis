# Multilingual_X_CB_analysis
This repository contains the deployment files for multilingual cyberbullying detection on X (Twitter) using the fine-tuned mDeBERTa model. Several multilingual transformer models were evaluated in the study, but mDeBERTa was selected as the best-performing model and deployed in this Streamlit application for real-time prediction.

#Dataset

The dataset used is the CardiffNLP Tweet Sentiment Multilingual dataset, which is publicly available on Hugging Face:
👉 CardiffNLP/tweet_sentiment_multilingual

The processed sample (tweet_data.csv) is included in this repository to support the deployed application.

#Tools

Google Colab with GPU

Python (Hugging Face Transformers, PyTorch, pandas, NumPy, scikit-learn)

Streamlit for deployment
