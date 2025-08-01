# fake_news_detector.py

import pandas as pd
import numpy as np
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# ---------- Text Preprocessing ----------
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

# ---------- Label Decoder ----------
def output_label(n):
    return "True News" if n == 1 else "Fake News"

# ---------- Manual Test Function ----------
def manual_testing(news):
    new_data = pd.DataFrame({"text": [news]})
    new_data["text"] = new_data["text"].apply(wordopt)
    new_xv = vectorization.transform(new_data["text"])

    print("\nManual Test Result:")
    print("Logistic Regression:", output_label(LR.predict(new_xv)[0]))
    print("Decision Tree:      ", output_label(DT.predict(new_xv)[0]))
    print("Gradient Boosting:  ", output_label(GB.predict(new_xv)[0]))
    print("Random Forest:      ", output_label(RF.predict(new_xv)[0]))

# ---------- Load or Train Models ----------
try:
    print("Loading saved models...")
    LR = joblib.load("lr_model.pkl")
    DT = joblib.load("dt_model.pkl")
    GB = joblib.load("gb_model.pkl")
    RF = joblib.load("rf_model.pkl")
    vectorization = joblib.load("vectorizer.pkl")

except:
    print("No saved models found — training now...")

    # Load data
    data_fake = pd.read_csv('Fake.csv')
    data_true = pd.read_csv('True.csv')

    data_fake['class'] = 0
    data_true['class'] = 1

    data_fake = data_fake.iloc[:-10]
    data_true = data_true.iloc[:-10]

    data = pd.concat([data_fake, data_true], axis=0)
    data = data.drop(['title', 'subject', 'date'], axis=1)
    data = data.sample(frac=1).reset_index(drop=True)

    data['text'] = data['text'].apply(wordopt)

    # Split and vectorize
    x = data['text']
    y = data['class']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)

    # Train models
    LR = LogisticRegression()
    DT = DecisionTreeClassifier()
    GB = GradientBoostingClassifier(random_state=0)
    RF = RandomForestClassifier(random_state=0)

    LR.fit(xv_train, y_train)
    DT.fit(xv_train, y_train)
    GB.fit(xv_train, y_train)
    RF.fit(xv_train, y_train)

    # Evaluate once
    print("\nModel Evaluation:")
    print("Logistic Regression:\n", classification_report(y_test, LR.predict(xv_test)))
    print("Decision Tree:\n", classification_report(y_test, DT.predict(xv_test)))
    print("Gradient Boosting:\n", classification_report(y_test, GB.predict(xv_test)))
    print("Random Forest:\n", classification_report(y_test, RF.predict(xv_test)))

    # Save models
    joblib.dump(LR, "lr_model.pkl")
    joblib.dump(DT, "dt_model.pkl")
    joblib.dump(GB, "gb_model.pkl")
    joblib.dump(RF, "rf_model.pkl")
    joblib.dump(vectorization, "vectorizer.pkl")

# ---------- Run Manual Test ----------
if __name__ == "__main__":
    news = input("\nEnter news to verify: ")
    manual_testing(news)