# app.py

import streamlit as st
import pandas as pd
import re
import string
import joblib

def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def output_label(n):
    return "True News" if n == 1 else "Fake News"

# Load models and vectorizer
LR = joblib.load("lr_model.pkl")
DT = joblib.load("dt_model.pkl")
GB = joblib.load("gb_model.pkl")
RF = joblib.load("rf_model.pkl")
vectorization = joblib.load("vectorizer.pkl")

st.title("Fake News Detector")

news_input = st.text_area("Enter news to verify:")

if st.button("Predict"):
    if not news_input.strip():
        st.warning("Please enter some news text.")
    else:
        clean_text = wordopt(news_input)
        vect_text = vectorization.transform([clean_text])
        st.write("Logistic Regression:", output_label(LR.predict(vect_text)[0]))
        st.write("Decision Tree:", output_label(DT.predict(vect_text)[0]))
        st.write("Gradient Boosting:", output_label(GB.predict(vect_text)[0]))
        st.write("Random Forest:", output_label(RF.predict(vect_text)[0]))